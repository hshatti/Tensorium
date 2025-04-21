unit nRNNLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  sysutils, nTypes, nTensors, nBaseLayer, nConnectedlayer;

type

  { TRNNLayer }

  TRNNLayer = class(TBaseLayer)
    steps, hidden :Sizeint;
    state : TSingleTensor;
    InputLayer, selfLayer, outputLayer : TConnectedLayer;
    isShortcut : boolean;
    constructor Create(aBatch:SizeInt; const AInputs, aHidden, aOutputs, aSteps: SizeInt; const aActivation: TActivationType; const aBatchNormalized:boolean; const log: SizeInt);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
    {$ifdef USE_OPENCL}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}
    destructor Destroy; override;
  private
    procedure rnnStepForward(var s, state: TNNetState; const hiddenStep, i: SizeInt);
    procedure rnnStepBackward(var s, state: TNNetState; const hiddenStep, i: SizeInt);
  end;

implementation
uses math, termesc;
{ TRNNLayer }


constructor TRNNLayer.Create(aBatch: SizeInt; const AInputs, aHidden, aOutputs,
  aSteps: SizeInt; const aActivation: TActivationType;
  const aBatchNormalized: boolean; const log: SizeInt);
begin
  layerType := ltRNN;
  steps := aSteps;
  batch := aBatch div steps;
  hidden := aHidden;
  inputs := aInputs;
  outputs := aOutputs;
  inputShape := [steps*batch, inputs];
  state := TSingleTensor.Create([(steps+1)* batch, hidden], (steps+1)* batch);
  ActivationType := aActivation;
  isBatchNormalized := aBatchNormalized;
  //writeln('rand_seed ',rand_seed);
  InputLayer := TConnectedLayer.Create(batch, steps, inputs, hidden, ActivationType, isBatchNormalized);
  //InputLayer.weights.printStat();;
  //writeln('rand_seed : ', rand_seed);
  //InputLayer.batch := batch;
  //if workspaceSize < inputLayer.workspaceSize then
      //workspaceSize := inputLayer.workspaceSize;

  //writeln('rand_seed ',rand_seed);
  selfLayer := TConnectedLayer.Create(batch, steps, hidden, hidden, TActivationType(ifthen((log = 2), ord(acLOGGY), (ifthen(log = 1, ord(acLOGISTIC), ord(ActivationType) ) ) ) ), isBatchNormalized);
  //selfLayer.weights.printStat;
  //writeln('rand_seed : ', rand_seed);
  //selfLayer.batch := batch;
  //if workspaceSize < selfLayer.workspaceSize then
      //workspaceSize := selfLayer.workspaceSize;
  //writeln('rand_seed ',rand_seed);
  outputLayer := TConnectedLayer.Create(batch, steps, hidden, outputs, ActivationType, isBatchNormalized);
  //outputLayer.weights.printStat();
  //writeln('rand_seed : ', rand_seed);
  //readln();
  //outputLayer.batch := batch;
  //if workspaceSize < outputLayer.workspaceSize then
      //workspaceSize := outputLayer.workspaceSize;

  output := outputLayer.output;
  //delta := outputLayer.delta;

  //InputLayer.weights.printStat();
  //selfLayer.weights.printStat();
  //outputLayer.weights.printStat();
  //readln();

end;

procedure TRNNLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch div steps;
  inputShape[0] := ABatch*steps;

  state.reSize([(steps+1)* batch, hidden], batch*(steps+1));
  InputLayer.setBatch(batch);
  //InputLayer.batch := batch;
  //if workspaceSize < inputLayer.workspaceSize then
      //workspaceSize := inputLayer.workspaceSize;



  selfLayer.setBatch(batch);
  //selfLayer.batch := batch;
  //if workspaceSize < selfLayer.workspaceSize then
      //workspaceSize := selfLayer.workspaceSize;

  outputLayer.setBatch(batch);
  output := outputLayer.output;

  delta := outputLayer.delta;
  //outputLayer.batch := batch;
  //if workspaceSize < outputLayer.workspaceSize then
      //workspaceSize := outputLayer.workspaceSize;

end;

procedure TRNNLayer.setTrain(ATrain: boolean);
begin
  if (ATrain=FTrain) then exit();
  InputLayer.setTrain(ATrain);
  selfLayer.setTrain(ATrain);
  outputLayer.setTrain(ATrain);
  output := outputLayer.output;
  delta := outputLayer.delta;
  FTrain := ATrain
end;

procedure fill_cpu(const N:SizeInt; const val:single; P:PSingle; const stride:SizeInt);
var i:SizeInt;
begin
  if stride=1 then begin
      FillDWord(p^, N, DWord(val));
      exit
  end;
  for i:=0 to N-1 do
      P[i*stride] := val;
end;

type
  TRNNState = record
    input, output, Delta: TSingleTensor
  end;

procedure TRNNLayer.rnnStepForward(var s, state: TNNetState; const hiddenStep,
  i: SizeInt);
var j:SizeInt;
begin
    j := i;
    s.step := i;
    s.inputStep:=i;
    s.input := state.input;
    inputLayer.forward(s);

    s.input := @self.state;
    selfLayer.forward(s);

    if state.isTraining then begin
        inc(j);
    end;
    if isShortcut then
      self.state.copyTo(self.state, j*hiddenStep, 1, i*hiddenStep, 1, hiddenStep)
    else
      self.state.FillExt(0, j*hiddenStep, hiddenStep);

    self.state.add(inputLayer.output, j*hiddenStep, i*hiddenStep, hiddenStep);
    self.state.add(selfLayer.output, j*hiddenStep, i*hiddenStep, hiddenStep);
    //s.input := @state;

    s.inputStep:=j;
    outputLayer.forward(s);
end;

procedure TRNNLayer.rnnStepBackward(var s, state: TNNetState; const hiddenStep,
  i: SizeInt);
var
  offset: SizeInt;
begin
  offset := i*hiddenStep;
  TSingleTensor.addvv(hiddenStep, inputLayer.output.data+offset, 1, selfLayer.output.data+offset, 1, self.state.data+(i+1)*hiddenStep, 1);
  //inputLayer.output.CopyTo(self.state, (i+1)*hiddenStep, 1, offset, 1, hiddenStep);
  //self.state.add(selfLayer.output, (i+1)*hiddenStep, offset, hiddenStep);

  s.step := i;
  s.inputStep := i+1;
  s.deltaStep := i;
  s.input := @self.state;
  s.delta := @selfLayer.delta;

  outputLayer.backward(s);

  s.inputStep := i;
  s.deltaStep := i-1;
  if i = 0 then
      s.delta := nil;
  selfLayer.backward(s);

  selfLayer.delta.CopyTo(inputLayer.delta, offset, 1, offset, 1, hiddenStep);
  if (i > 0) and isShortcut then
      selfLayer.delta.add(selfLayer.delta, (i-1)*hiddenStep, offset, hiddenStep);

  s.input := state.input;
  if assigned(state.delta) then begin
      s.delta := state.delta
  end else
      s.delta := nil;

  s.inputStep := i;
  s.deltaStep := i;
  inputLayer.backward(s);
end;

procedure TRNNLayer.forward(var state: TNNetState);
var
    s: TNNetState;
    i, j, inputStep, hiddenStep, outputStep: SizeInt;
    //old_state: PSingle;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}
    s := default(TNNetState);
    s.isTraining := state.isTraining;
    s.workspace := state.workspace;
    s.net := state.net;
    s.index := state.index;

    inputStep := inputs*batch;
    hiddenStep := hidden*batch;
    outputStep := outputs*batch;

    InputLayer .reGroup(batch);
    selfLayer  .reGroup(batch);
    outputLayer.reGroup(batch);

    outputLayer.delta.fill(0);
    selfLayer.delta.fill(0);
    inputLayer.delta.fill(0);
    if state.isTraining then
        self.state.FillExt(0, 0, hiddenStep);
    for i := 0 to steps -1 do
      rnnStepForward(s, state, hiddenStep, i);
    if state.isTraining then
        write(#13, 'FW RNN [',state.index,'] ', 100*i/steps:3:0,'%');
    InputLayer .reGroup(steps*batch);
    selfLayer  .reGroup(steps*batch);
    outputLayer.reGroup(steps*batch);
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;


(*
procedure backward_rnn_layer(var l: TRNNLayer; const state: TNNetState);
var
    s: TNNetState;
    r: TRNNState;
    i, inputStep, hiddenStep, outputStep: SizeInt;
    inputLayer, selfLayer, outputLayer : TConnectedLayer;
begin
    s := default(TNNetState);
    s.isTraining := state.isTraining;
    s.workspace := state.workspace;
    s.net := state.net;

    inputStep  := l.batch*l.inputs;
    hiddenStep := l.Batch*l.hidden;
    outputStep := l.Batch*l.outputs;

    inputLayer := l.InputLayer;
    selfLayer :=l.selfLayer;
    outputlayer := l.outputLayer;

    increment_layer(InputLayer, l.steps-1);
    increment_layer(selfLayer, l.steps-1);
    increment_layer(outputLayer, l.steps-1);

    if pointer(l.state.data)<>pointer(l.state.DynData) then
        l.state.resetReference;
    l.state.data := l.state.data + (hiddenStep * l.steps);
    try
    for i := l.steps-1 downto 0 do begin
        //copy_cpu(hiddenStep, l.inputLayer.output.data, 1, l.state.data, 1);
        inputLayer.output.CopyTo(l.state, 0, 1, 0, 1, hiddenStep);
        //axpy_cpu(hiddenStep, 1, l.selfLayer.output.data, 1, l.state.data, 1);
        l.state.add(selfLayer.output.data, 0, hiddenStep);
        //s.step := i;
        s.step := 0;
        s.inputStep := 0;
        s.deltaStep := 0;
        s.input := @r.input;
        s.delta := @r.delta;

        r.input := l.state;
        r.Delta := selfLayer.delta;

        outputLayer.backward(s);

        //l.state.data := l.state.data - hiddenStep;
        dec(l.state.data, hiddenStep);
        r.input := l.state;
        //s.delta.data := l.selfLayer.delta.data - hiddenStep;
        dec(r.Delta.data, hiddenStep);
        if i = 0 then
            s.delta := nil;
        l.selfLayer.backward(s);

        copy_cpu(hiddenStep, selfLayer.delta.data, 1, inputLayer.delta.data, 1);
        if (i > 0) and l.isShortcut then
            axpy_cpu(hiddenStep, 1, selfLayer.delta.data, 1, selfLayer.delta.data - hiddenStep, 1);

        r.input := state.input^;
        r.input.data := r.input.data + i*inputStep;
        if assigned(state.delta) then begin
            r.delta := state.delta^;
            r.delta.data := r.delta.data + i*inputStep;
            s.delta := @r.delta
        end
        else
            s.delta := nil;

        inputLayer.backward(s);

        increment_layer(inputLayer, -1);
        increment_layer(selfLayer, -1);
        increment_layer(outputLayer, -1);
        //write(#13, 'BW RNN [',state.index,'] ', 100*i/l.steps:3:0,'%')

    end;
    finally
      reset_layer(InputLayer);
      reset_layer(selfLayer);
      reset_layer(outputLayer);
    end;
end;
*)

procedure TRNNLayer.backward(var state: TNNetState);
var
    s: TNNetState;
    i, inputStep, hiddenStep, outputStep: SizeInt;
begin
{$ifdef USE_TELEMETRY}
if benchmark then metrics.backward.start(layerType);
{$endif}
    s := default(TNNetState);
    s.isTraining := state.isTraining;
    s.workspace := state.workspace;
    s.net := state.net;
    s.index:= state.index;;

    inputStep  := batch*inputs;
    hiddenStep := Batch*hidden;
    outputStep := Batch*outputs;

    InputLayer .reGroup(batch);
    selfLayer  .reGroup(batch);
    outputLayer.reGroup(batch);
    //increment_layer(l.InputLayer, l.steps-1);
    //increment_layer(l.selfLayer, l.steps-1);
    //increment_layer(l.outputLayer, l.steps-1);
    //
    //if pointer(l.state.data)<>pointer(l.state.DynData) then
    //    l.state.resetReference;
    //l.state.data := l.state.data + (hiddenStep * l.steps);
    try
    for i := steps-1 downto 0 do begin
      rnnStepBackward(s, state, hiddenStep, i);
    end;
    finally
      write(#13, 'BW RNN [',state.index,'] ', 100*i/steps:3:0,'%');
      InputLayer .reGroup(steps*batch);
      selfLayer  .reGroup(steps*batch);
      outputLayer.reGroup(steps*batch);
    end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TRNNLayer.update(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}
  InputLayer.update(args);
  selfLayer.update(args);
  outputLayer.update(args);
  inherited update(args);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

destructor TRNNLayer.Destroy;
begin
  freeAndNil(InputLayer);
  freeAndNil(selfLayer);
  freeAndNil(outputLayer);

  inherited Destroy;
end;

{$ifdef USE_OPENCL}
procedure TRNNLayer.forwardGPU(var state: TNNetState);
var
    s: TNNetState;
    i, j, inputStep, hiddenStep, outputStep: SizeInt;

    gpuERR : Single;
    tmp :TSingleTensor;
    //t1, t2, t3 :TSingleTensor;
    //old_state: PSingle;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}
  if not state.input.wasGPU() then state.input.pushToDevice;

  if not inputLayer.weights.wasGPU() then inputLayer.weights.pushToDevice;
  if not inputLayer.biases.wasGPU() then inputLayer.biases.pushToDevice;

  if not selfLayer.weights.wasGPU() then selfLayer.weights.pushToDevice;
  if not selfLayer.biases.wasGPU() then selfLayer.biases.pushToDevice;

  if not outputLayer.weights.wasGPU() then outputLayer.weights.pushToDevice;
  if not outputLayer.biases.wasGPU() then outputLayer.biases.pushToDevice;
  if not self.state.wasGPU() then self.state.pushToDevice;

  output.setOCL;


  s := default(TNNetState);
  s.isTraining := state.isTraining;
  s.workspace := state.workspace;
  s.net := state.net;
  s.index := state.index;

  inputStep  := inputs *batch;
  hiddenStep := hidden *batch;
  outputStep := outputs*batch;

  InputLayer .reGroup(batch);
  selfLayer  .reGroup(batch);
  outputLayer.reGroup(batch);

  ocl.fill(InputLayer.delta.Size(), InputLayer.delta.devData, 0, 0, 1);
  ocl.fill(selfLayer.delta.Size(), selfLayer.delta.devData, 0, 0, 1);
  ocl.fill(outputLayer.delta.Size(), outputLayer.delta.devData, 0, 0, 1);

  if state.isTraining then begin
      ocl.fill(hiddenStep, self.state.devData, 0, 0, 1);
  end;

  for i := 0 to steps -1 do begin

          j := i;
          s.step := i;
          s.inputStep:=i;
          s.input := state.input;
          inputLayer.forwardGPU(s);


          s.input := @self.state;
          selfLayer.forwardGPU(s);

          if state.isTraining then begin
              inc(j);
          end;
          if isShortcut then begin
            ocl.copy(hiddenStep, self.state.devData, i*hiddenStep, 1, self.state.devData, j*hiddenStep, 1);
          end else begin
            ocl.fill(hiddenStep, self.state.devData, j*hiddenStep, 0, 1);
          end;


          ocl.addvv(hiddenStep, inputLayer.output.devData, i*hiddenStep, 1, self.state.devData, j*hiddenStep, 1, self.state.devData, j*hiddenStep, 1);
          ocl.addvv(hiddenStep, selfLayer.output.devData, i*hiddenStep, 1, self.state.devData, j*hiddenStep, 1, self.state.devData, j*hiddenStep, 1);

          //s.input := @self.state;
          s.inputStep:=j;
          outputLayer.forwardGPU(s);

          if state.isTraining then write(#13, 'FW RNN [',state.index,'] ', 100*i/steps:3:0,'%')
          //if l.steps = 1 then break;

          //state.input.data := state.input.data + inputStep;
          //increment_layer(l.inputLayer, 1);
          //increment_layer(l.selfLayer, 1);
          //increment_layer(l.outputLayer, 1)
  end;

//output.printGpuSumSqrDiff();
  InputLayer .reGroup(steps*batch);
  selfLayer  .reGroup(steps*batch);
  outputLayer.reGroup(steps*batch);

  //l.output.printStat
    //state.input.resetReference;
    //reset_layer(l.InputLayer);
    //reset_layer(l.selfLayer);
    //reset_layer(l.outputLayer);
    //l.output.printStat();
    //readLn()

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TRNNLayer.backwardGPU(var state: TNNetState);
var
    s: TNNetState;
    i, inputStep, hiddenStep, outputStep: SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  s := default(TNNetState);
  s.isTraining := state.isTraining;
  s.workspace := state.workspace;
  s.net := state.net;

  inputStep  := batch*inputs;
  hiddenStep := Batch*hidden;
  outputStep := Batch*outputs;

  InputLayer .reGroup(batch);
  selfLayer  .reGroup(batch);
  outputLayer.reGroup(batch);
  //increment_layer(l.InputLayer, l.steps-1);
  //increment_layer(l.selfLayer, l.steps-1);
  //increment_layer(l.outputLayer, l.steps-1);
  //
  //if pointer(l.state.data)<>pointer(l.state.DynData) then
  //    l.state.resetReference;
  //l.state.data := l.state.data + (hiddenStep * l.steps);

  //try
  for i := steps-1 downto 0 do begin
      ocl.copy(hiddenStep, InputLayer.output.devData, i*hiddenStep, 1, self.state.devData, (1+i)*hiddenStep, 1);
      //inputLayer.output.CopyTo(self.state, (i+1)*hiddenStep, 1, i*hiddenStep, 1, hiddenStep);
  //InputLayer.output.printStat;
  //
  //InputLayer.output.pullFromDevice(t1);
  //t1.printStat;

      ocl.addvv(hiddenStep, selfLayer.output.devData, i*hiddenStep, 1, self.state.devData, (i+1)*hiddenStep, 1, self.state.devData, i*hiddenStep, 1);
      //self.state.add(selfLayer.output, (i+1)*hiddenStep, i*hiddenStep, hiddenStep);

      s.step := i;
      s.inputStep := i+1;
      s.deltaStep := i;
      //s.step := 0;

      s.input := @self.state;
      s.Delta := @selfLayer.delta;
      outputLayer.backwardGPU(s);
      //outputLayer.backward(s);


      s.inputStep := i;
      s.deltaStep := i-1;
      if i = 0 then
          s.delta := nil;
      selfLayer.backwardGPU(s);
      //selfLayer.backward(s);

      ocl.copy(hiddenStep, selfLayer.delta.devData, i*hiddenStep, 1, inputLayer.delta.devData, i*hiddenStep, 1);
      //selfLayer.delta.CopyTo(inputLayer.delta, i*hiddenStep, 1, i*hiddenStep, 1, hiddenStep);

      if (i > 0) and isShortcut then
          ocl.addvv(hiddenStep, selfLayer.delta.devData, i*hiddenStep, 1, selfLayer.delta.devData, (i-1)*hiddenStep, 1, selfLayer.delta.devData, (i-1)*hiddenStep, 1);
          //selfLayer.delta.add(selfLayer.delta, (i-1)*hiddenStep, i*hiddenStep, hiddenStep);

      s.input := state.input;
      //r.input.data := r.input.data + i*inputStep;
      if assigned(state.delta) then begin
          //r.delta.data := r.delta.data + i*inputStep;
          s.delta := state.delta
      end
      else
          s.delta := nil;

      s.inputStep := i;
      s.deltaStep := i;
      inputLayer.backwardGPU(s);
      //inputLayer.backward(s);

      //increment_layer(l.inputLayer, -1);
      //increment_layer(l.selfLayer, -1);
      //increment_layer(l.outputLayer, -1);
      write(#13, 'BW RNN [',state.index,'] ', 100*i/steps:3:0,'%')

  end;
  //finally
    InputLayer .reGroup(steps*Batch);
    selfLayer  .reGroup(steps*Batch);
    outputLayer.reGroup(steps*Batch);
  //end;


  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TRNNLayer.updateGPU(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}
  InputLayer.updateGPU(args);
  selfLayer.updateGPU(args);
  outputLayer.updateGPU(args);

  inherited updateGPU(args);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;
{$endif}

end.

