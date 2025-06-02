unit nDropOutLayer;
{$ifdef fpc}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;

type

  { TDropoutLayer }

  TDropoutLayer = class(TBaseImageLayer)
    probability            : single;
    scale                  : single;
    dropBlock              : boolean;
    dropBlockSizeRel       : single;
    rand                   : TSingleTensor;
    dropBlockSizeAbs       : SizeInt;
    constructor Create(const aBatch:SizeInt; const aProbability: single; const aInputs:SizeInt; const aChannels:SizeInt=0; const aHeight:SizeInt = 0; const aWidth: SizeInt=0; const aDropblock: boolean= false; const aDropblock_size_rel: single=0; const aDropblock_size_abs: sizeInt = 0);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
{$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState);override;
    procedure backwardGPU(var state: TNNetState);override;
{$endif}
end;

implementation

{ TDropoutLayer }

constructor TDropoutLayer.Create(const aBatch: SizeInt;
  const aProbability: single; const aInputs: SizeInt; const aChannels: SizeInt;
  const aHeight: SizeInt; const aWidth: SizeInt; const aDropblock: boolean;
  const aDropblock_size_rel: single; const aDropblock_size_abs: sizeInt);
begin
  layerType := ltDROPOUT;
  probability := aProbability;
  dropBlock := aDropblock;
  dropBlockSizeRel := aDropblock_size_rel;
  dropBlockSizeAbs := aDropblock_size_abs;
  if dropblock then
      begin
          w := aWidth;
          outW := w;
          h := aHeight;
          outH := h;
          c := aChannels;
          outC := c;
          if (w <= 0) or (h <= 0) or (c <= 0) then
               raise Exception.Create(format(' Error: DropBlock - there must be positive values for: result.w=%d, result.h=%d, result.c=%d ',[ w, h, c]));
      end;
  inputs := Ainputs;
  outputs := Ainputs;
  batch := Abatch;
  inputShape := [batch, inputs];
  rand := TSingleTensor.Create([batch, inputs], batch);

  scale := 1 / (1.0-probability);
end;

procedure TDropoutLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := batch;
  rand.resize([batch, inputs], batch);
end;

procedure TDropoutLayer.setTrain(ATrain: boolean);
begin
  if FTrain = ATrain then exit();
  FTrain := ATrain
end;

procedure TDropoutLayer.forward(var state: TNNetState);
var
    i: SizeInt;
    r: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  output := state.input^;
  if not state.isTraining then
      exit();
  //rand.UniformDistribution(0,1);
  for i := 0 to batch * inputs -1 do
      begin
          r := random();//rand_uniform(0, 1);
          rand.data[i] := r;
          if r < probability then
          //if rand.Data[i] < probability then
              state.input.Data[i] := 0
          else
              state.input.Data[i] := state.input.Data[i] * scale
      end ;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TDropoutLayer.backward(var state: TNNetState);
var
    i: SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  if not assigned(state.delta.Data) then
      exit();
  for i := 0 to batch *  inputs -1 do
      begin
          if rand.Data[i] < probability then
              state.delta.Data[i] := 0
          else
              state.delta.Data[i] := state.delta.Data[i] * scale
      end;
  delta := state.delta^;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;


{$if defined(USE_OPENCL)}
procedure TDropoutLayer.forwardGPU(var state: TNNetState);
begin
  if not state.input.wasGPU() then state.input.pushToDevice;
  //random(1000); // next randSeed
  output := state.input^;
  ocl.forwardDropout(output.size(), output.devData, probability, scale, rand.devData, output.devData);
  //rand.pullFromDevice();
  //rand.print(psGray);
  //readln
end;

procedure TDropoutLayer.backwardGPU(var state: TNNetState);
begin
  if not assigned(state.delta.devData) then
      exit();
  if not state.delta.wasGPU() then state.delta.pushToDevice;
  ocl.backwardDropout(state.delta.Size(), state.delta.devData, probability, scale, rand.devData, state.delta.devData);
  delta := state.delta^;
end;
{$elseif defined(USE_CUDART)}
procedure TDropoutLayer.forwardGPU(var state: TNNetState);
begin
  if not state.input.wasGPU() then state.input.pushToDevice;
  output := state.input^;
  cuda.forwardDropout(output.size(), output.devData, probability, scale, rand.devData, output.devData);
  //rand.pullFromDevice();
  //rand.print(psGray);
  //readln
end;

procedure TDropoutLayer.backwardGPU(var state: TNNetState);
begin
  if not assigned(state.delta.devData) then
      exit();
  if not state.delta.wasGPU() then state.delta.pushToDevice;
  cuda.backwardDropout(state.delta.Size(), state.delta.devData, probability, scale, rand.devData, state.delta.devData);
  delta := state.delta^;
end;
{$endif}

end.

