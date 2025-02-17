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
    constructor Create(aBatch:SizeInt; const AInputs, aHidden, aOutputs, aSteps: SizeInt; const aActivation: TActivationType; const aBatchNormalize:boolean; const log: SizeInt);
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
  end;

implementation
uses math;
{ TRNNLayer }


constructor TRNNLayer.Create(aBatch: SizeInt; const AInputs, aHidden, aOutputs,
  aSteps: SizeInt; const aActivation: TActivationType;
  const aBatchNormalize: boolean; const log: SizeInt);
begin
  layerType := ltRNN;
  steps := aSteps;
  batch := aBatch ;
  hidden := aHidden;
  inputs := aInputs;
  outputs := aOutputs;
  inputShape := [steps, batch, inputs];
  state := TSingleTensor.Create([steps+1, batch, hidden], batch, steps+1);
  ActivationType := aActivation;
  isBatchNormalized := aBatchNormalize;
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
  delta := outputLayer.delta;

  //InputLayer.weights.printStat();
  //selfLayer.weights.printStat();
  //outputLayer.weights.printStat();
  //readln();

end;

procedure TRNNLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := ABatch;

  state.reSize([steps+1, batch, hidden], batch, steps+1);
  InputLayer.setBatch(batch);
  //InputLayer.batch := batch;
  //if workspaceSize < inputLayer.workspaceSize then
      //workspaceSize := inputLayer.workspaceSize;



  selfLayer.setBatch(batch);
  //selfLayer.batch := batch;
  //if workspaceSize < selfLayer.workspaceSize then
      //workspaceSize := selfLayer.workspaceSize;

  outputLayer.setBatch(batch);
  //outputLayer.batch := batch;
  //if workspaceSize < outputLayer.workspaceSize then
      //workspaceSize := outputLayer.workspaceSize;

end;

procedure TRNNLayer.setTrain(ATrain: boolean);
begin
  InputLayer.setTrain(ATrain);
  selfLayer.setTrain(ATrain);
  outputLayer.setTrain(ATrain);
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

procedure increment_layer(const l: TConnectedLayer; const steps: SizeInt);
var
    num: SizeInt;
begin
    num := l.outputs * l.batch * steps;
    l.output.Data := l.output.Data + num;
    l.delta.Data  := l.delta.Data  + num;
    l.x.Data      := l.x.Data      + num;
    l.x_norm.Data := l.x_norm.Data + num;
{$ifdef GPU}
    l.output_gpu := l.output_gpu + num;
    l.delta_gpu := l.delta_gpu + num;
    l.x_gpu := l.x_gpu + num;
    l.x_norm_gpu := l.x_norm_gpu + num
{$endif}
end;

procedure reset_layer(const l:TConnectedLayer);
begin
  l.output.resetReference;
  l.delta.resetReference;
  l.x.resetReference;
  l.x_norm.resetReference;
end;

procedure axpy_cpu(const N:SizeInt; const a:Single; const X:PSingle; const incx:SizeInt; const Y:PSingle; const incy:SizeInt);
begin
  TSingleTensor.axpysvv(N, a, X, incx, Y, incy);
end;

procedure copy_cpu(const N:SizeInt; src:PSingle; const incSrc:SizeInt; const dst:PSingle; const incDst:SizeInt);
var i:SizeInt;begin
  if (incSrc=1) and (incDst=1) then begin
      move(src^, dst^, N*Sizeof(Single));
      exit
  end;
  for i:=0 to N-1 do
      dst[i*incDst] := src[i*incSrc]
end;

procedure forward_rnn_layer(var l: TRNNLayer; const state: TNNetState);
var
    s: TNNetState;
    i, inputStep, hiddenStep, outputStep: SizeInt;
    input_layer: TConnectedLayer;
    self_layer: TConnectedLayer;
    output_layer: TConnectedLayer;
    old_state: PSingle;
begin

    s := default(TNNetState);
    s.isTraining := state.isTraining;
    s.workspace := state.workspace;
    s.net := state.net;

    input_layer :=  l.InputLayer;
    self_layer :=  l.selfLayer;
    output_layer :=  l.outputLayer;

    inputStep := l.inputs*l.batch;
    hiddenStep := l.hidden*l.batch;
    outputStep := l.outputs*l.batch;

  if state.isTraining then begin
      output_layer.delta.fill(0);
      self_layer.delta.fill(0);
      input_layer.delta.fill(0);
      l.state.FillExt(0, 0, hiddenStep);
  end;
  for i := 0 to l.steps -1 do
      begin
          s.step := i;
          s.input := state.input;
          input_layer.forward(s);

          s.input := @l.state;
          self_layer.forward(s);

          if state.isTraining then begin
              //old_state := l.state.data;
              //l.state.data := l.state.data + hiddenStep;
              if l.isShortcut then
                  l.state.copyTo(l.state, (i+1)*hiddenStep, 1, i*hiddenStep, 1, hiddenStep);
          end;
          if not l.isShortcut then
              l.state.FillExt(0, i*hiddenStep, hiddenStep);
          //if l.isShortcut then
          //    copy_cpu(hiddenStep, old_state, 1, l.state.Data, 1)
          //else
          //    l.state.FillExt(0, 0, hiddenStep);
          l.state.add(input_layer.output.Data, i*hiddenStep, hiddenStep);
          l.state.add(self_layer.output.Data, i*hiddenStep, hiddenStep);
          //s.input := @l.state;
          output_layer.forward(s);

          //if l.steps = 1 then break;

          //state.input.data := state.input.data + inputStep;
          //increment_layer(input_layer, 1);
          //increment_layer(self_layer, 1);
          //increment_layer(output_layer, 1)
      end;
    //state.input.resetReference;
    //reset_layer(l.InputLayer);
    //reset_layer(l.selfLayer);
    //reset_layer(l.outputLayer);
    //l.output.printStat();
    //readLn()
end;


procedure TRNNLayer.forward(var state: TNNetState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    forward_rnn_layer(self, state);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;

procedure backward_rnn_layer(var l: TRNNLayer; const state: TNNetState);
var
    s: TNNetState;
    i, inputStep, hiddenStep, outputStep: SizeInt;
    input_layer: TConnectedLayer;
    self_layer: TConnectedLayer;
    output_layer: TConnectedLayer;
begin
    s := default(TNNetState);
    s.isTraining := state.isTraining;
    s.workspace := state.workspace;

    input_layer :=  l.InputLayer;
    self_layer :=  l.selfLayer;
    output_layer :=  l.outputLayer;

    inputStep  := l.batch*l.inputs;
    hiddenStep := l.Batch*l.hidden;
    outputStep := l.Batch*l.outputs;

    increment_layer(input_layer, l.steps-1);
    increment_layer(self_layer, l.steps-1);
    increment_layer(output_layer, l.steps-1);
    l.state.data := l.state.data + (hiddenStep * l.steps);
    for i := l.steps-1 downto 0 do begin
        //copy_cpu(hiddenStep, input_layer.output.data, 1, l.state.data, 1);
        input_layer.output.CopyTo(l.state, (i+1)*hiddenStep, 1, i*hiddenStep, 1, hiddenStep);
        //axpy_cpu(hiddenStep, 1, self_layer.output.data, 1, l.state.data, 1);
        l.state.add(self_layer.output.data + (i+1)*hiddenStep, i*hiddenStep, hiddenStep);
        s.step := i;
        s.input := @l.state;
        s.delta := @self_layer.delta;
        output_layer.backward(s);

        l.state.data := l.state.data - hiddenStep;
        s.input := @l.state;
        s.delta.data := self_layer.delta.data - hiddenStep;
        if i = 0 then
            s.delta.Data := nil;
        self_layer.backward(s);

        copy_cpu(hiddenStep, self_layer.delta.data, 1, input_layer.delta.data, 1);
        if (i > 0) and l.isShortcut then
            axpy_cpu(hiddenStep, 1, self_layer.delta.data, 1, self_layer.delta.data - hiddenStep, 1);
        s.input.data := state.input.data + i*inputStep;
        if assigned(state.delta.data) then
            s.delta.data := state.delta.data + i*inputStep
        else
            s.delta.data := nil;
        input_layer.backward(s);

        increment_layer(input_layer, -1);
        increment_layer(self_layer, -1);
        increment_layer(output_layer, -1);
    end
end;

procedure TRNNLayer.backward(var state: TNNetState);
begin
  backward_rnn_layer(self, state);
end;

procedure TRNNLayer.update(const args: TUpdateArgs);
begin
  InputLayer.update(args);
  selfLayer.update(args);
  outputLayer.update(args);
  inherited update(args);
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
begin

end;

procedure TRNNLayer.backwardGPU(var state: TNNetState);
begin

end;

procedure TRNNLayer.updateGPU(const args: TUpdateArgs);
begin
  inherited updateGPU(args);
end;
{$endif}

end.

