unit nConnectedlayer;
{$ifdef fpc}
{$mode Delphi}{$H+}
{$endif}

interface

uses
  SysUtils, Math
  ,ntensors, NTypes, nBaseLayer
{$if defined (USE_OPENCL)}
  , OpenCL
  {$ifdef CL_BLAST} , clblast {$endif}
{$elseif defined(USE_CUDART)}
  , cudarttypes, cudart, cublas_api, nnCuda
{$endif}
  {$ifdef USE_TELEMETRY}
  , nOpMetrics
  {$endif}
  ;

type

  { TConnectedLayer }

  TConnectedLayer = class(TBaseLayer)
    constructor Create(const ABatch, ASteps, AInputs, aOutputs: SizeInt; const AActivationType:TActivationType= acLINEAR; AIsBatchNormalized:boolean=false);
    procedure setBatch(ABatch:SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure DeNormalize;
    procedure update(const args: TUpdateArgs); override;
    function getWorkspaceShape: TArray<SizeInt>; override;
    {$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState);  override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}
  end;

implementation

{ TConnectedLayer }

constructor TConnectedLayer.Create(const ABatch, ASteps, AInputs,
  aOutputs: SizeInt; const AActivationType: TActivationType;
  AIsBatchNormalized: boolean);
var randomRange:Single;
begin
  batch                 := ABatch; // note split to steps in case of RNN
  Steps                 := ASteps;
  layerType             := ltCONNECTED;
  ActivationType        := AActivationType;
  inputs                := AInputs;
  inputShape            := [steps * batch, inputs];
  outputs               := aOutputs;
  isBatchNormalized     := AIsBatchNormalized;

  output                := TSingleTensor.Create([steps * batch , outputs], steps * Batch);
  weights               := TSingleTensor.Create([outputs , inputs]);
  randomRange           := sqrt(2/inputs);
  weights.UniformDistribution(-randomRange, randomRange);
  biases                := TSingleTensor.Create([outputs]);

  if isBatchNormalized then begin
    //if train then begin
    //    scale_updates       := TSingleTensor.Create([outputs]);
    //    scales.fill(1.0);
    //    mean                := TSingleTensor.Create([outputs]);
    //    mean_delta          := TSingleTensor.Create([outputs]);
    //    variance            := TSingleTensor.Create([outputs]);
    //    variance_delta      := TSingleTensor.Create([outputs]);
    //    x                   := TSingleTensor.Create([steps * batch, outputs], steps * Batch);
    //    x_norm              := TSingleTensor.Create([steps * batch, outputs], steps * Batch)
    //end;
    rolling_mean        := TSingleTensor.Create([outputs]);
    rolling_variance    := TSingleTensor.Create([outputs]);
    scales              := TSingleTensor.Create([outputs]);
    scales.fill(1.0);
  end;
  //writeln('FC ', steps,' X ', batch, ' X ', inputs);
  //readln
end;

procedure TConnectedLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := steps * ABatch;

  output.reSize([steps * batch , outputs], steps * Batch) ;

  if FTrain then
      delta.reSize([steps * batch , outputs], steps * Batch) ;

  if isBatchNormalized and train then begin
    x.reSize([steps * batch, outputs], steps * batch);
    x_norm.reSize([steps * batch, outputs], steps * Batch);
  end;
end;

procedure TConnectedLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  FTrain := ATrain;

  if FTrain then begin
    delta                 := TSingleTensor.Create([steps * batch , outputs], steps * Batch);
    weight_updates        := TSingleTensor.Create([inputs , outputs]);
    bias_updates          := TSingleTensor.Create([outputs]);
    if isBatchNormalized then begin
        x                     := TSingleTensor.Create([steps * batch , outputs], steps * Batch);
        x_norm                := TSingleTensor.Create([steps * batch , outputs], steps * Batch);
        mean                  := TSingleTensor.Create([outputs]);
        variance              := TSingleTensor.Create([outputs]);
        mean_delta            := TSingleTensor.Create([outputs]);
        variance_delta        := TSingleTensor.Create([outputs]);
        scale_updates         := TSingleTensor.Create([outputs]);
    end;
  end else begin
    delta          .free;
    weight_updates .free;
    bias_updates   .free;
    if isBatchNormalized then begin
        x                .free;
        x_norm           .free;
        mean             .free;
        variance         .free;
        mean_delta       .free;
        variance_delta   .free;
        scale_updates    .free;
    end;
  end;
end;

procedure TConnectedLayer.forward(var state: TNNetState);
var outputStep, inputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and (metrics.forward.isSubPropagation()=0)
     then
         metrics.forward.start(layerType);
  {$endif}

  //output.fill(0);

  inputStep  := batch * inputs;
  outputStep := batch * outputs;
  offset   := state.step*outputStep;
  assert(state.input^.size() >= (state.inputStep+1)*inputStep, '[TConnectedLayer.Forward] inputStep out of range!');
  assert(output.size() >= (state.step+1)*outputStep, '[TConnectedLayer.Forward] step out of range!');
  assert(weights.Size() = inputs*outputs, '[TConnectedLayer.Forward] incorrect weights shape!');

  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.start(opGemm);
  {$endif}


  TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, outputs, inputs, 1
    , state.input^.Data + state.inputStep*inputStep, inputs
    , weights.Data, inputs
    , 0, output.data + offset, outputs);
  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.finish(opGemm);
  {$endif}
    //state.input.matMul(weights, output, CblasNoTrans, CblasTrans);
  if isBatchNormalized then begin
      if state.isTraining then begin
          //mean_cpu(output.data , batch, outputs, 1, mean.data);
          //variance_cpu(output.data , mean, batch, outputs, 1, variance.data);
          output.MeansAndVars(mean, variance, offset, outputStep);
          //scal_cpu(outputs, 0.95, rolling_mean, 1);
          //axpy_cpu(outputs, 0.05, mean, 1, rolling_mean, 1);
          rolling_mean.Multiply(0.95);
          rolling_mean.axpy(0.05, mean);

          //scal_cpu(outputs, 0.95, rolling_variance, 1);
          //axpy_cpu(outputs, 0.05, variance, 1, rolling_variance, 1);
          rolling_variance.Multiply(0.95);
          rolling_variance.axpy(0.05, variance);

          //copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
          output.CopyTo(x, offset, 1, offset, 1, outputStep);
          //normalize_cpu(output, mean, variance, batch, outputs, 1);
          output.Normalize(mean, variance, offset, outputStep);
          //copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1)
          output.copyTo(x_norm, offset, 1, offset, 1, outputStep) ;
      end else
          //normalize each column
          //normalize_cpu(output, rolling_mean, rolling_variance, batch, outputs, 1);
          output.Normalize(rolling_mean, rolling_variance, offset, outputStep);

      //scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
      output.forwardScale(scales, offset, outputStep);
      output.forwardBias(biases, offset, outputStep);
      //output.FusedMultiplyAdd(scales, biases);
  end else

  //for i := 0 to batch -1 do
  //    TSingleTensor.axpysvv(outputs, 1, biases.data, 1, output.data+i * outputs, 1);
    output.forwardBias(biases, offset, outputStep);

  //activate_array(l.output, l.outputs * l.batch, l.activation);
  activate(offset);
  //write(#13, 'FW [',state.index,'] ', state.step);
  //write(#$1b'[2J'#$1b'[H');
  //output.print(psGray);
  //readln;



  {$ifdef USE_TELEMETRY}
  if benchmark
  //and (metrics.forward.isSubPropagation()<=1)
     then
     metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.backward(var state: TNNetState);
var outOffset, outStepSize, inStepSize:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and (metrics.backward.isSubPropagation()=0)
     then
     metrics.backward.start(layerType);
  {$endif}
  outStepSize := batch * outputs;
  inStepSize := batch * inputs;
  outOffset   := state.step * outStepSize;
  //write(#13, 'BW [',state.index,'] ', state.step);
  assert((state.inputStep>=0) and (state.step>=0), '[TConnectedLayer.backward] state step values must be positive!');
  assert(delta.size() >= (state.step + 1) * outStepSize, '[TConnectedLayer.backward] step out of range!');
  assert(state.input.size() >= (state.inputStep + 1)*inStepSize, '[TConnectedLayer.backward] inputStep out of range!');

  //gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
  Derivative(outOffset);
  //for i := 0 to batch -1 do
  //    TSingleTensor.axpysvv(outputs, 1, delta.data+i * outputs, 1, bias_updates.data, 1);
  //bias_updates.add(delta, outOffset, outputs);

  bias_updates.addSums(delta, outOffset, outStepSize);
  //if l.batch_normalize then
  if isBatchNormalized {and (batch > 1)} then begin
      // spatial dot (x_norm . delta) then add to scale_updates
      //backward_scale_cpu(x_norm, delta, batch, outputs, 1, scale_updates);
      scale_updates.addDots(x_norm, delta, outOffset, outStepSize);

      // add scales to all delta batches
      //scale_bias(delta, scales, batch, outputs, 1);
      delta.forwardScale(scales, outOffset,  outStepSize);

      //mean_delta_cpu(delta, variance, batch, outputs, 1, mean_delta);
      //variance_delta_cpu(x, delta, mean, variance, batch, outputs, 1, variance_delta);
      delta.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta, outOffset, outStepSize);

      //normalize_delta_cpu(x, mean, variance, mean_delta, variance_delta, batch, outputs, 1, delta)
      delta.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta, outOffset, outStepSize);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.start(opGemm);
  {$endif}
  //delta.matMul(state.input, weight_updates, CblasTrans, CblasNoTrans);

  //if assigned(state.delta) then
      //sgemm(0, 0, l.batch, l.inputs, l.outputs, 1, l.delta, l.outputs, l.weights, l.inputs, 1, state.delta, l.inputs)



  TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, outputs, inputs, batch, 1
  , delta.Data + outOffset, outputs
  , state.input^.Data + state.inputStep*inStepSize, inputs
  , 1, weight_updates.Data, inputs);

  if assigned(state.delta) and assigned(state.delta.Data) then begin
      assert((state.deltaStep>=0) and (state.delta.size() >= (state.deltaStep + 1)*inStepSize), '[TConnectedLayer.backward] inputStep out of range!');

      TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, inputs, outputs, 1
      , delta.Data + outOffset, outputs
      , weights.Data, inputs
      , 1, state.delta^.Data + state.deltaStep*inStepSize, inputs);
  end;
  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.finish(opGemm);
  {$endif}
      //delta.matMul(weights, state.delta)

  {$ifdef USE_TELEMETRY}
  if benchmark
     //and (metrics.forward.isSubPropagation()=1)
     then
     metrics.backward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.DeNormalize;
var
    i, j: SizeInt;
    _scale: single;
begin
    // tofdo SIMDfy and GPU
    for i := 0 to outputs -1 do
        begin
            _scale := scales.data[i] / max(sqrt(rolling_variance.data[i]), sEPSILON);
            for j := 0 to inputs -1 do
                weights.data[i * inputs+j] := weights.data[i * inputs+j] * _scale;
            biases.data[i] := biases.data[i] - (rolling_mean.data[i] * _scale);
            scales.data[i] := 1;
            rolling_mean.data[i] := 0;
            rolling_variance.data[i] := 1
        end
end;

procedure TConnectedLayer.update(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}
  //axpy_cpu(l.outputs, args.learning_rate / args.batch, l.bias_updates, 1, l.biases, 1);
  //TSingleTensor.axpysvv(outputs, args.learningRate / args.batch, bias_updates, 1, biases, 1);
  biases.axpy(args.learningRate / args.batch, bias_updates);

  //scal_cpu(l.outputs, args.momentum, l.bias_updates, 1);
  bias_updates.Multiply(args.momentum);
  if isBatchNormalized then begin
      //axpy_cpu(l.outputs, args.learning_rate / args.batch, l.scale_updates, 1, l.scales, 1);
      scales.axpy(args.learningRate / args.batch, scale_updates);

      //scal_cpu(l.outputs, args.momentum, l.scale_updates, 1)
      scale_updates.Multiply(args.momentum);
  end;

  //axpy_cpu(l.inputs * l.outputs, -args.decay * args.batch, l.weights, 1, l.weight_updates, 1);
  //TSingleTensor.axpysvv(weight_updates.Size(), -args.decay * args.batch, weights, 1, weight_updates, 1);
  weight_updates.axpy(-args.decay * args.batch, weights);

  //axpy_cpu(l.inputs * l.outputs, args.learning_rate / args.batch, l.weight_updates, 1, l.weights, 1);
  //TSingleTensor.axpysvv(weights.size(), args.learningRate / args.batch, weight_updates, 1, weights, 1);
  weights.axpy(args.learningRate / args.batch, weight_updates);

  //scal_cpu(l.inputs * l.outputs, args.momentum, l.weight_updates, 1)
  weight_updates.Multiply(args.momentum);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

function TConnectedLayer.getWorkspaceShape: TArray<SizeInt>;
begin
  result := nil
end;

{$if defined(USE_OPENCL)}
//const clearscr=#$1B'[2J'#$1B'[2;1H';
procedure TConnectedLayer.forwardGPU(var state: TNNetState);
var outputStep, inputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then
     metrics.forward.start(layerType);
  {$endif}

  inputStep  := batch * inputs;
  outputStep := batch * outputs;
  offset   := state.step*outputStep;
  assert(state.input^.size() >= (state.inputStep+1)*inputStep, '[TConnectedLayer.ForwardGPU] inputStep out of range!');
  assert(output.size() >= (state.step+1)*outputStep, '[TConnectedLayer.ForwardGPU] step out of range!');
  assert(weights.Size() = inputs*outputs, '[TConnectedLayer.ForwardGPU] incorrect weights shape!');

  if not state.input.wasGPU() then  begin
      state.input.pushToDevice;
  end;
  if not weights.wasGPU() then  begin
      weights.pushToDevice;
  end;
  if not biases.wasGPU() then  begin
      biases.pushToDevice;
  end;
  output.setOCL;

  {$IFDEF CL_BLAST}
  ocl.FErr := integer(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes
          , batch, outputs, inputs, 1
          , state.input.devData, 0, inputs
          , weights.devData    , 0, inputs
          , 0, output.devData  , 0, outputs
          , @ocl.ActiveQueue
          {$IFDEF CL_EVENTS}
          , pointer(state.events));
          {$ELSE}
           , nil));
          {$ENDIF}
          ocl.checkError();
  {$ELSE}
  ocl.gemm(false, true
          , batch, outputs, inputs, 1
          , state.input.devData, state.inputStep*inputStep, inputs
          , weights.devData    , 0, inputs
          , 0, output.devData  , offset, outputs
          {$IFDEF CL_EVENTS}
          , 1, pointer(state.events), pointer(state.events));
          {$ELSE}
          );
          {$ENDIF}
  {$ENDIF}

  if isBatchNormalized then begin
      {$ifdef USE_TELEMETRY}
      //ocl.waitForEvents(batch, pointer(events));
      if benchmark then metrics.forward.start(ltBATCHNORM);
      {$endif}
      if not scales.wasGPU() then scales.pushToDevice;
      if not rolling_mean.wasGPU() then rolling_mean.pushToDevice;
      if not rolling_variance.wasGPU() then rolling_variance.pushToDevice;

      if state.isTraining then begin
          ocl.meanAndVars(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData, variance.devData);
          ocl.scale(rolling_mean.size(), 0.95, rolling_mean.devData, 1);
          ocl.axpy(rolling_mean.Size(), 0.05, mean.devData, 0, 1, rolling_mean.devData, 0, 1);
          ocl.scale(rolling_variance.Size(), 0.95, rolling_variance.devData, 1);
          ocl.axpy(rolling_variance.size(), 0.05, variance.devData, 0, 1, rolling_variance.devData, 0, 1);
          ocl.copy(outputStep, output.devData, offset, 1, x.devData, offset, 1);
          ocl.normalize(mean.Size(), outputStep, output.groups, mean.devData, 1, variance.devData, 1, output.devData, offset);
          ocl.copy(outputStep, output.devData, offset, 1, x_norm.devData, offset, 1);
      end else begin
          ocl.normalize(rolling_mean.Size(), outputStep, output.Groups, rolling_mean.devData, 1, rolling_variance.devData, 1, output.devData, offset);
      end;
      //ocl.forwardScale(outputStep, output.devData, offset, scales.size(), scales.devData, 1, output.Groups);
      //ocl.forwardBias(outputStep, output.devData, offset, biases.size(), biases.devData, 1, output.Groups);
      ocl.forwardScaleAdd(outputStep, output.devData, offset, scales.size(), scales.devData, biases.devData, 1, output.Groups);

      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.forward.finish(ltBATCHNORM);
      {$endif}
  end else begin
    ocl.forwardBias(outputStep, output.devData, offset, biases.size(), biases.devData, 1, output.groups);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.act.start(ActivationType);
  {$endif}
  ocl.ActivateArray(outputStep, output.devData, offset, longint(ActivationType));
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.act.finish(ActivationType);
  {$endif}

  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.backwardGPU(var state: TNNetState);
var// t:TSingleTensor;
  outStepSize, inStepSize, outOffset: SizeInt;
  t1, t2 :TSingleTensor;
  n1 :TSizeIntTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then
     metrics.backward.start(layerType);
  {$endif}
  outStepSize := batch * outputs;
  inStepSize  := batch * inputs;
  outOffset   := state.step * outStepSize;

  assert((state.inputStep>=0) and (state.step>=0), '[TConnectedLayer.backwardGPU] state step values must be positive!');
  assert(delta.size() >= (state.step + 1) * outStepSize, '[TConnectedLayer.backwardGPU] step out of range!');
  assert(state.input.size() >= (state.inputStep + 1)*inStepSize, '[TConnectedLayer.backwardGPU] inputStep out of range!');

  if not delta.wasGPU() then delta.pushToDevice;
  if not state.input.wasGPU() then state.input.pushToDevice;
  if not output.wasGPU() then output.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not weight_updates.wasGPU() then weight_updates.pushToDevice;


  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.grad.start(ActivationType);
  {$endif}
  ocl.DeriveArray(outStepSize, output.devData, outOffset, longint(ActivationType), delta.devData
  {$IFDEF CL_EVENTS}
  , 1, pointer(state.events), pointer(state.events));
  {$ELSE}
  );
  {$ENDIF}
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.grad.finish(ActivationType);
  {$endif}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.backwardBias(bias_updates.size(), bias_updates.devData, outStepSize, delta.devData, outOffset, 1, batch
  {$IFDEF CL_EVENTS}
  , 1, pointer(state.events), pointer(state.events));
  {$ELSE}
  );
  {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));

  if isBatchNormalized {and (batch > 1)} then begin
      {$ifdef USE_TELEMETRY}
      metrics.backward.start(ltBATCHNORM);
      {$endif}
      //scale_updates.addDots(x_norm, delta);
      //delta.add(scales);
      //TSingleTensor.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);
      //TSingleTensor.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);

      ocl.addDots(outStepSize, scale_updates.Size(), delta.groups, x_norm.devData, delta.devData, outOffset, scale_updates.devData);
      ocl.forwardScale(outStepSize, delta.devData, outOffset, scales.Size(), scales.devData, 1, delta.groups);
      ocl.meansAndVarsDelta(outStepSize, mean.size(), delta.groups, delta.devData, x.devData, outOffset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
      ocl.normalizeDelta(outStepSize, mean.size(), delta.groups, delta.devData, x.devData, outOffset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      metrics.backward.finish(ltBATCHNORM);
      {$endif}
  end;

//
  {$IFDEF CL_BLAST}
  ocl.FErr := integer(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo
      , outputs, inputs, batch, 1
      , delta.devData, 0, outputs
      , state.input.devData, 0, inputs
      , 1, weight_updates.devData, 0, inputs, @ocl.ActiveQueue
      {$IFDEF CL_EVENTS}
      , pointer(state.events));
      {$ELSE}
       , nil));
      {$ENDIF}
      ocl.checkError();
  {$ELSE}
  ocl.gemm(true, false
    , outputs, inputs, batch, 1
    , delta.devData, outOffset, outputs
    , state.input.devData, state.inputStep*inStepSize, inputs
    , 1, weight_updates.devData, 0, inputs
    {$IFDEF CL_EVENTS}
    , 1, pointer(state.events), pointer(state.events));
    {$ELSE}
    );
    {$ENDIF}
  {$ENDIF}
  //weight_updates.pullFromDevice(t1);
  //n1 := t1.findNaNs;
  //if n1.size>0 then begin
  //   state.input.pullFromDevice(t2, inputs*batch, state.inputStep*inputs*batch);
  //   beep;
  //end;
  if assigned(state.delta) and assigned(state.delta^.devData) then begin
      assert(state.delta.size() >= (state.deltaStep + 1)*inStepSize, '[TConnectedLayer.backwardGPU] inputStep out of range!');

      if not weights.wasGPU then weights.pushToDevice;
      //if not state.delta.wasGPU() then state.delta.pushToDevice;
   {$IFDEF CL_BLAST}
   ocl.FErr := integer(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo
          , batch, inputs, outputs, 1
          , delta.devData, 0, outputs
          , weights.devData, 0, inputs
          , 1, state.delta.devData, 0, inputs, @ocl.ActiveQueue
          {$IFDEF CL_EVENTS}
          , pointer(state.events));
          {$ELSE}
           , nil));
          {$ENDIF}
      ocl.checkError();
   {$ELSE}
      ocl.gemm( false, false
          , batch, inputs, outputs, 1
          , delta.devData, outOffset, outputs
          , weights.devData, 0, inputs
          , 1, state.delta.devData, state.deltaStep*inStepSize, inputs
          {$IFDEF CL_EVENTS}
          , 1, pointer(state.events), pointer(state.events));
          {$ELSE}
          );
          {$ENDIF}
   {$ENDIF}
      //ocl.waitForEvents(batch, pointer(events));
      //ocl.finish();

  end ;

  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then
     metrics.backward.finish(layerType);
  {$endif}

end;

procedure TConnectedLayer.updateGPU(const args: TUpdateArgs);
var t : TSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

  if not biases.wasGPU() then biases.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not weights.wasGPU() then weights.pushToDevice;
  if not weight_updates.wasGPU() then weight_updates.pushToDevice;

  ocl.axpy(biases.size(), args.learningRate / args.batch, bias_updates.devData, 0, 1, biases.devData, 0, 1
  {$IFDEF CL_EVENTS}
  , 1, pointer(events), pointer(events) );
  {$ELSE}
  );
  {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.scale(bias_updates.size(), args.momentum, bias_updates.devData, 1
  {$IFDEF CL_EVENTS}
  , 1, pointer(events), pointer(events) );
  {$ELSE}
  );
  {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.axpy(weight_updates.size(), -args.decay * args.batch, weights.devData, 0, 1, weight_updates.devData, 0, 1
  {$IFDEF CL_EVENTS}
  , 1, pointer(events), pointer(events) );
  {$ELSE}
  );
  {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.axpy(weights.size(), args.learningRate / args.batch, weight_updates.devData, 0, 1, weights.devData, 0, 1
  {$IFDEF CL_EVENTS}
  , 1, pointer(events), pointer(events) );
  {$ELSE}
  );
  {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.scale(weight_updates.size(), args.momentum, weight_updates.devData, 1
  {$IFDEF CL_EVENTS}
  , 1, pointer(events), pointer(events) );
  {$ELSE}
  );
  {$ENDIF}
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  if isBatchNormalized {and (batch > 1)} then begin
      //scales.axpy(args.learningRate / args.batch, scale_updates);
      //scale_updates.Multiply(args.momentum);
      ocl.axpy(scales.size(), args.learningRate / args.batch, scale_updates.devData, 0, 1, scales.devData, 0, 1);
      ocl.scale(scale_updates.size(), args.momentum, scale_updates.devData, 1);
  end;

  //update(args);
  inherited ;

  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

{$elseif defined(USE_CUDART)}
procedure TConnectedLayer.forwardGPU(var state: TNNetState);
var outputStep, inputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then
     metrics.forward.start(layerType);
  {$endif}

  inputStep  := batch * inputs;
  outputStep := batch * outputs;
  offset   := state.step*outputStep;
  assert(state.input^.size() >= (state.inputStep+1)*inputStep, '[TConnectedLayer.ForwardGPU] inputStep out of range!');
  assert(output.size() >= (state.step+1)*outputStep, '[TConnectedLayer.ForwardGPU] step out of range!');
  assert(weights.Size() = inputs*outputs, '[TConnectedLayer.ForwardGPU] incorrect weights shape!');

  if not state.input.wasGPU() then  begin
      state.input.pushToDevice;
  end;
  if not weights.wasGPU() then  begin
      weights.pushToDevice;
  end;
  if not biases.wasGPU() then  begin
      biases.pushToDevice;
  end;
  output.setCUDA;

  cuda.gemm(false, true
    , batch, outputs, inputs, 1.0
    , state.input.devData, state.inputStep*inputStep, inputs
    , weights.devData    , 0, inputs
    , 0.0, output.devData  , offset, outputs
  );

//TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, outputs, inputs, 1
//  , state.input^.Data + state.inputStep*inputStep, inputs
//  , weights.Data, inputs
//  , 0, output.data + offset, outputs);

  if isBatchNormalized then begin
      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.forward.start(ltBATCHNORM);
      {$endif}
      if not scales.wasGPU() then scales.pushToDevice;
      if not rolling_mean.wasGPU() then rolling_mean.pushToDevice;
      if not rolling_variance.wasGPU() then rolling_variance.pushToDevice;

      if state.isTraining then begin
          //cuda.meansAndVars(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData, variance.devData);
          cuda.means(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData);
          cuda.variances(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData, variance.devData);

//output.MeansAndVars(mean, variance, offset, outputStep);

          cuda.scale(rolling_mean.size(), 0.95, rolling_mean.devData, 1);
//rolling_mean.Multiply(0.95);
          cuda.axpy(rolling_mean.Size(), 0.05, mean.devData, 0, 1, rolling_mean.devData, 0, 1);
//rolling_mean.axpy(0.05, mean);
          cuda.scale(rolling_variance.Size(), 0.95, rolling_variance.devData, 1);
//rolling_variance.Multiply(0.95);
          cuda.axpy(rolling_variance.size(), 0.05, variance.devData, 0, 1, rolling_variance.devData, 0, 1);
//rolling_variance.axpy(0.05, variance);
          cuda.copy(outputStep, output.devData, offset, 1, x.devData, offset, 1);
//output.CopyTo(x, offset, 1, offset, 1, outputStep);
          cuda.normalize(mean.Size(), outputStep, output.groups, mean.devData, 1, variance.devData, 1, output.devData, offset);
//output.Normalize(mean, variance, offset, outputStep);
          cuda.copy(outputStep, output.devData, offset, 1, x_norm.devData, offset, 1);
//output.copyTo(x_norm, offset, 1, offset, 1, outputStep) ;
      end else begin
          cuda.normalize(rolling_mean.Size(), outputStep, output.Groups, rolling_mean.devData, 1, rolling_variance.devData, 1, output.devData, offset);
//output.Normalize(rolling_mean, rolling_variance, offset, outputStep);
      end;
      cuda.forwardScaleAdd(outputStep, output.devData, offset, scales.size(), scales.devData, biases.devData, 1, output.Groups);
//output.forwardScale(scales, offset, outputStep);
//output.forwardBias(biases, offset, outputStep);
      {$ifdef USE_TELEMETRY}
      cuda.finish();
      if benchmark then metrics.forward.finish(ltBATCHNORM);
      {$endif}
  end else begin
    cuda.forwardBias(outputStep, output.devData, offset, biases.size(), biases.devData, 1, output.groups);
//output.forwardBias(biases, offset, outputStep);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.act.start(ActivationType);
  {$endif}
  cuda.ActivateArray(outputStep, output.devData, offset, longint(ActivationType));
//activate();
  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.act.finish(ActivationType);
  {$endif}
//output.printGpuSumSqrDiff();
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.backwardGPU(var state: TNNetState);
var// t:TSingleTensor;
  outStepSize, inStepSize, outOffset: SizeInt;
  t1, t2 :TSingleTensor;
  n1 :TSizeIntTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then
     metrics.backward.start(layerType);
  {$endif}
  outStepSize := batch * outputs;
  inStepSize  := batch * inputs;
  outOffset   := state.step * outStepSize;

  assert((state.inputStep>=0) and (state.step>=0), '[TConnectedLayer.backwardGPU] state step values must be positive!');
  assert(delta.size() >= (state.step + 1) * outStepSize, '[TConnectedLayer.backwardGPU] step out of range!');
  assert(state.input.size() >= (state.inputStep + 1)*inStepSize, '[TConnectedLayer.backwardGPU] inputStep out of range!');

  if not delta.wasGPU() then delta.pushToDevice;
  if not state.input.wasGPU() then state.input.pushToDevice;
  if not output.wasGPU() then output.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not weight_updates.wasGPU() then weight_updates.pushToDevice;


  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.grad.start(ActivationType);
  {$endif}
  cuda.DeriveArray(outStepSize, output.devData, outOffset, longint(ActivationType), delta.devData);

//Derivative(outOffset);
//output.printGpuSumSqrDiff();
//delta.printGpuSumSqrDiff();

  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.grad.finish(ActivationType);
  {$endif}

  cuda.backwardBias(bias_updates.size(), bias_updates.devData, outStepSize, delta.devData, outOffset, 1, batch);
//bias_updates.addSums(delta, outOffset, outStepSize);
//bias_updates.printGpuSumSqrDiff();

  if isBatchNormalized {and (batch > 1)} then begin
      {$ifdef USE_TELEMETRY}
      metrics.backward.start(ltBATCHNORM);
      {$endif}

      cuda.addDots(outStepSize, scale_updates.Size(), delta.groups, x_norm.devData, delta.devData, outOffset, scale_updates.devData);
//scale_updates.addDots(x_norm, delta);
//x_norm.printGpuSumSqrDiff();

      cuda.forwardScale(outStepSize, delta.devData, outOffset, scales.Size(), scales.devData, 1, delta.groups);
//delta.add(scales);
//delta.printGpuSumSqrDiff();

      cuda.meansAndVarsDelta(outStepSize, mean.size(), delta.groups, delta.devData, x.devData, outOffset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
//TSingleTensor.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);
//mean_delta.printGpuSumSqrDiff();
//variance_delta.printGpuSumSqrDiff();

      cuda.normalizeDelta(outStepSize, mean.size(), delta.groups, delta.devData, x.devData, outOffset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
//TSingleTensor.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
//delta.printGpuSumSqrDiff();

      {$ifdef USE_TELEMETRY}
      cuda.finish();
      metrics.backward.finish(ltBATCHNORM);
      {$endif}
  end;

//
  cuda.gemm(true, false
    , outputs, inputs, batch, 1
    , delta.devData, outOffset, outputs
    , state.input.devData, state.inputStep*inStepSize, inputs
    , 1, weight_updates.devData, 0, inputs
    );

//TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, outputs, inputs, batch, 1
//  , delta.Data + outOffset, outputs
//  , state.input^.Data + state.inputStep*inStepSize, inputs
//  , 1, weight_updates.Data, inputs);

  if assigned(state.delta) and assigned(state.delta^.devData) then begin
      assert(state.delta.size() >= (state.deltaStep + 1)*inStepSize, '[TConnectedLayer.backwardGPU] inputStep out of range!');

      if not weights.wasGPU then weights.pushToDevice;
      //if not state.delta.wasGPU() then state.delta.pushToDevice;
      cuda.gemm( false, false
          , batch, inputs, outputs, 1
          , delta.devData, outOffset, outputs
          , weights.devData, 0, inputs
          , 1, state.delta.devData, state.deltaStep*inStepSize, inputs
          );

//TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, inputs, outputs, 1
//  , delta.Data + outOffset, outputs
//  , weights.Data, inputs
//  , 1, state.delta^.Data + state.deltaStep*inStepSize, inputs);

  end ;

//bias_updates.printGpuSumSqrDiff();
//weight_updates.printGpuSumSqrDiff();
//state.delta^.printGpuSumSqrDiff();

  {$ifdef USE_TELEMETRY}
  if benchmark
     //and not metrics.forward.isSubPropagation()
     then
     metrics.backward.finish(layerType);
  {$endif}

end;

procedure TConnectedLayer.updateGPU(const args: TUpdateArgs);
var t : TSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

  if not biases.wasGPU() then biases.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not weights.wasGPU() then weights.pushToDevice;
  if not weight_updates.wasGPU() then weight_updates.pushToDevice;

  cuda.axpy(biases.size(), args.learningRate / args.batch, bias_updates.devData, 0, 1, biases.devData, 0, 1);
  //cuda.waitForEvents(batch, pointer(events));
  //cuda.finish();

  cuda.scale(bias_updates.size(), args.momentum, bias_updates.devData, 1);
  //cuda.waitForEvents(batch, pointer(events));
  //cuda.finish();

  cuda.axpy(weight_updates.size(), -args.decay * args.batch, weights.devData, 0, 1, weight_updates.devData, 0, 1);
  //cuda.waitForEvents(batch, pointer(events));
  //cuda.finish();

  cuda.axpy(weights.size(), args.learningRate / args.batch, weight_updates.devData, 0, 1, weights.devData, 0, 1);
  //cuda.waitForEvents(batch, pointer(events));
  //cuda.finish();

  cuda.scale(weight_updates.size(), args.momentum, weight_updates.devData, 1);
  //cuda.waitForEvents(batch, pointer(events));
  //cuda.finish();

  if isBatchNormalized {and (batch > 1)} then begin
      //scales.axpy(args.learningRate / args.batch, scale_updates);
      //scale_updates.Multiply(args.momentum);
      cuda.axpy(scales.size(), args.learningRate / args.batch, scale_updates.devData, 0, 1, scales.devData, 0, 1);
      cuda.scale(scale_updates.size(), args.momentum, scale_updates.devData, 1);
  end;

  //update(args);
  inherited ;

  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;
{$endif}

end.

