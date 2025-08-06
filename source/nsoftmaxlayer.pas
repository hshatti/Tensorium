unit nSoftmaxLayer;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
{$pointermath on}
interface

uses
  SysUtils, Math, ntypes, nBaseLayer, nTensors

  {$if defined(USE_CUDART)}
  , nnCuda
  {$endif}
  ;

type

  { TSoftmaxLayer }

  TSoftmaxLayer = class(TBaseLayer)
    loss        : TSingleTensor;
    noLoss       : boolean;
    softmaxTree : TArray<TTree>;
    temperature : Single;
    constructor Create(const aBatch, aInputs:SizeInt; const aGroups:SizeInt=1);
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
{$if defined(USE_OPENCL)}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
{$elseif defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState);  override;
    procedure backwardGPU(var state: TNNetState); override;
{$endif}
    class procedure softmaxBatch(const input: PSingle; const n:SizeInt; const batch, batch_size, groups, group_size, stride: SizeInt; const temp: single; const output: PSingle); static;
    class procedure softmaxCrossEntropy(const pred, truth: TSingleTensor; var delta, error: TSingleTensor);  static;

  end;

procedure softmax(const n: SizeInt; const input: PSingle; const temp: single; const stride: SizeInt; const output: PSingle);

implementation


{ TSoftmaxLayer }

constructor TSoftmaxLayer.Create(const aBatch, aInputs: SizeInt;
  const aGroups: SizeInt);
begin
  assert(aInputs mod aGroups = 0);
  LayerType      := ltSOFTMAX;
  ActivationType := acSOFTMAX;
  batch          := Abatch;
  groups         := Agroups;
  inputs         := Ainputs;
  inputShape     := [batch, inputs];
  temperature    := 1;
  outputs        := Ainputs;
  loss           := TSingleTensor.Create([batch, Inputs], batch);
  output         := TSingleTensor.Create([batch, Inputs], batch);
  delta          := TSingleTensor.Create([batch, Inputs], batch);
  cost           := [0];
end;

procedure TSoftmaxLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0]     := batch;
  loss.resize([batch, Inputs], batch);
  output.resize([batch, Inputs], batch);
  delta.resize([batch, Inputs], batch);
end;

procedure TSoftmaxLayer.setTrain(ATrain: boolean);
begin
  if FTrain = ATrain then exit;
  FTrain := ATrain
end;

procedure softmax(const n: SizeInt; const input: PSingle; const temp: single; const stride: SizeInt; const output: PSingle);
var i:SizeInt;
    sum, largest, e : single;
    o:PSingle;
begin
  // todo [Softmax] SIMDIfy & GPU
  if n=0 then exit;
  sum := 0;
  largest := input[0];
  for i := 1 to n-1 do
      if input[i*stride] > largest then
          largest := input[i*stride];
  for i := 0 to n-1 do  begin
      //e := exp(ensureRange((input[i*stride] - largest)/temp, minSingleExp, maxSingleExp));
      e := exp((input[i*stride] - largest)/temp);
      sum := sum + e;
      output[i*stride] := e;
  end;
  for i := 0 to n-1 do  begin
      o:=@output[i*stride];
      o^ := o^ / sum;
  end;

end;

class procedure TSoftmaxLayer.softmaxBatch(const input: PSingle;
  const n: SizeInt; const batch, batch_size, groups, group_size,
  stride: SizeInt; const temp: single; const output: PSingle);
var g, b:SizeInt;
begin
  // todo [TSoftmaxLayer::softmaxBatch] SIMDfy & GPU
  for b := 0 to batch-1 do
      for g := 0 to groups-1 do
          softmax(n
          , input + b*batch_size + g*group_size
          , temp
          , stride
          , output + b*batch_size + g*group_size);
end;

class procedure TSoftmaxLayer.softmaxCrossEntropy(const pred, truth: TSingleTensor; var delta, error: TSingleTensor);
var i:SizeInt;
    t,p :single;
begin
  //todo [TSoftmaxLayer::softmaxCrossEntropy] simdfy & GPU
  for i := 0 to Delta.size() -1 do begin
      t := truth.data[i];
      p := pred.data[i];
      if t<>0 then
          error.data[i] := -ln(max(p , sEPSILON))
      else
          error.data[i] := 0;
      delta.data[i] := t - p;
  end
end;

procedure TSoftmaxLayer.forward(var state: TNNetState);
var
    i, count, group_size: SizeInt;
begin
    {$ifdef USE_TELEMETRY}
     if benchmark then metrics.forward.start(layerType);
    {$endif}

    if assigned(softmaxTree) then
        begin     // todo [TSoftmaxLayer::forward SIMDfy & GPU
            count := 0;
            for i := 0 to softmaxTree[0].groups -1 do
                begin
                    group_size := softmaxTree[0].group_size[i];
                    softmaxBatch(pointer(state.input.data+count), group_size, batch, inputs, 1, 0, 1, temperature, pointer(output.data+count));
                    inc(count , group_size)
                end
        end
    else
        softmaxBatch(pointer(state.input.data), inputs div groups, batch, inputs, groups, inputs div groups, 1, temperature, pointer(output.Data));
    if assigned(state.truth.data) and not noloss then
        begin
            softmaxCrossEntropy(output, state.truth, delta, loss);
            cost[0] := loss.Sum();
        end;
    {$ifdef USE_TELEMETRY}
     if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;

procedure TSoftmaxLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.backward.start(layerType);
  {$endif}

  //axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, net.delta, 1)
  state.delta.add(delta.Data);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

{$if defined(USE_OPENCL)}
procedure TSoftmaxLayer.forwardGPU(var state: TNNetState);
var t1, t2, t3 :TSingleTensor;
var
  i, count, group_size: SizeInt;
  t:TSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.forward.start(layerType);
  {$endif}

  output.setOCL();
  loss.setOCL;
  delta.setOCL;

  if assigned(softmaxTree) then begin
      count := 0;
      for i := 0 to softmaxTree[0].groups -1 do begin
          group_size := softmaxTree[0].group_size[i];
          ocl.softmaxBatch(group_size, state.input.devData, count, batch, inputs, 1, 0, 1, temperature, output.devData, count
            {$IFDEF CL_EVENTS}
            , 1, @state.events[i mod batch], @state.events[i mod batch]);
            {$ELSE}
            );
            {$ENDIF}

          inc(count , group_size)
      end
  end else
      ocl.softmaxBatch(inputs div groups, state.input.devData, 0, batch, inputs, groups, inputs div groups, 1, temperature, output.devData, 0
        {$IFDEF CL_EVENTS}
        , 1, pointer(state.events), pointer(state.events));
        {$ELSE}
        );
        {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  if assigned(state.truth.data) and not noloss then begin
    if not state.truth.wasGPU() then
      state.truth.pushToDevice();
    ocl.crossEntropySoftmax(output.size(), output.devData, state.truth.devData, delta.devData, loss.devData
      {$IFDEF CL_EVENTS}
      , 1, pointer(state.events), pointer(state.events));
      {$ELSE}
      );
      {$ENDIF}

    //ocl.finish();
    //ocl.waitForEvents(batch, pointer(events));
    //softmaxCrossEntropy(output, state.truth, delta, loss);
    //delta.pullFromDevice(t);
    //writeln(state.index,' FW SOFTMAX sumSqrDelta : ', t.sumSqrDiff(delta):1:6);
    //readln;
    loss.pullFromDevice();
    cost[0] := loss.Sum();
  end;

  //output.pullFromDevice(t1);
  //forward(state);
  //t1.printStat();
  //output.printStat();

  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TSoftmaxLayer.backwardGPU(var state: TNNetState);
var t: TSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.backward.start(layerType);
  {$endif}

  //if not state.delta.wasGPU() then state.delta.pushToDevice();
  if not delta.wasGPU() then delta.pushToDevice();

  ocl.addvv(delta.size(), delta.devData, 0, 1, state.delta.devData, 0, 1, state.delta.devData, 0, 1
    {$IFDEF CL_EVENTS}
    , 1, pointer(state.events), pointer(state.events));
    {$ELSE}
    );
    {$ENDIF}

  //backward(state);
  //state.delta.pullFromDevice(t);
  //writeln(state.index,' BW SOFTMAX sumSqrDiff state.delta : ', t.sumSqrDiff(delta):1:6);
  //readln;
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$elseif defined(USE_CUDART)}
procedure TSoftmaxLayer.forwardGPU(var state: TNNetState);
var t1, t2, t3 :TSingleTensor;
var
  i, count, group_size: SizeInt;
  t:TSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.forward.start(layerType);
  {$endif}

  output.setCUDA();
  loss.setCUDA;
  delta.setCUDA;

  if assigned(softmaxTree) then begin
      count := 0;
      for i := 0 to softmaxTree[0].groups -1 do begin
          group_size := softmaxTree[0].group_size[i];
          cuda.softmaxBatch(group_size, state.input.devData, count, batch, inputs, 1, 0, 1, temperature, output.devData, count);

          inc(count , group_size)
      end
  end else begin
      cuda.softmaxBatch(inputs div groups, state.input.devData, 0, batch, inputs, groups, inputs div groups, 1, temperature, output.devData, 0);
//softmaxBatch(pointer(state.input.data), inputs div groups, batch, inputs, groups, inputs div groups, 1, temperature, pointer(output.Data));
  end;

  if assigned(state.truth.data) and not noloss then begin
    if not state.truth.wasGPU() then
      state.truth.pushToDevice();
    cuda.crossEntropySoftmax(output.size(), output.devData, state.truth.devData, delta.devData, loss.devData);
//softmaxCrossEntropy(output, state.truth, delta, loss);

    loss.pullFromDevice();
    cost[0] := loss.Sum();
  end;

//output.printGpuSumSqrDiff();
//delta.printGpuSumSqrDiff();
//loss.printGpuSumSqrDiff();

  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TSoftmaxLayer.backwardGPU(var state: TNNetState);
var t: TSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.backward.start(layerType);
  {$endif}
//delta.printGpuSumSqrDiff();
  //if not state.delta.wasGPU() then state.delta.pushToDevice();
  if not delta.wasGPU() then delta.pushToDevice();

  cuda.addvv(delta.size(), delta.devData, 0, 1, state.delta.devData, 0, 1, state.delta.devData, 0, 1);
//backward(state);
//state.delta^.printGpuSumSqrDiff();
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$endif}

end.

