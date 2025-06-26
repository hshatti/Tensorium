unit nAttentionLayer;
{$ifdef FPC}
  {$mode Delphi}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$SafeFPUExceptions off}
{$endif}
{$PointerMath on}

interface

uses
  SysUtils
  , math
  , nTypes, nBaseLayer, nTensors, nConnectedlayer, termesc, nActivation
  , steroids
  ;

type

  { TAttentionLayer }

  TAttentionLayer = class(TBaseLayer)
    maxSeq, Heads, KVHeads, headLen, KVHeadLen: SizeInt;
    wq, wk, wv, wo : TConnectedLayer;
    mask : TSingleTensor;
    constructor Create(const aBatch, aInputs, aMaxSeq, aHeads: SizeInt; aKVHeads: SizeInt = 0);
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
    {$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}

  end;

implementation

{ TAttentionLayer }

constructor TAttentionLayer.Create(const aBatch, aInputs, aMaxSeq,
  aHeads: SizeInt; aKVHeads: SizeInt);
begin
  if aKVHeads=0 then aKVHeads := aHeads;
  assert(ainputs mod aHeads=0, '[TAttention.Create] <heads> must e a multiples of <KVHeads>');
  assert(aHeads mod aKVHeads=0, '[TAttention.Create] <heads> must e a multiples of <KVHeads>');
  batch  := aBatch;
  inputs := aInputs;
  Heads  := aHeads;
  KVHeads := aKVHeads;
  headLen := inputs div heads;
  KVheadLen := inputs div KVHeads;
  wq := TConnectedLayer.Create(batch, 1, inputs, inputs, acLINEAR, false, false);
  wk := TConnectedLayer.Create(batch, 1, inputs, kvHeads*headLen, acLINEAR, false, false);
  wv := TConnectedLayer.Create(batch, 1, inputs, kvHeads*headLen, acLINEAR, false, false);
  wo := TConnectedLayer.Create(batch, 1, inputs, inputs, acLINEAR, false, false);
  mask.resize([maxSeq, maxSeq]);
  mask.triangularFill(-infinity, true);

end;

procedure TAttentionLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  wq.setTrain(ATrain);
  wk.setTrain(ATrain);
  wv.setTrain(ATrain);
  wo.setTrain(ATrain);
  output   := wo.output;
  delta    := wo.delta;
  FTrain   := ATrain;
end;

procedure TAttentionLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit;
  wq.setBatch(ABatch);
  wk.setBatch(ABatch);
  wv.setBatch(ABatch);
  wo.setBatch(ABatch);
  output   := wo.output;
  delta    := wo.delta;
  Batch   := aBatch;
end;

procedure TAttentionLayer.forward(var state: TNNetState);
begin

end;

procedure TAttentionLayer.backward(var state: TNNetState);
begin

end;

procedure TAttentionLayer.update(const args: TUpdateArgs);
begin
  inherited update(args);
end;

{$if defined(USE_OPENCL) or defined(USE_CUDART)}
procedure TAttentionLayer.forwardGPU(var state: TNNetState);
begin

end;

procedure TAttentionLayer.backwardGPU(var state: TNNetState);
begin

end;

procedure TAttentionLayer.updateGPU(const args: TUpdateArgs);
begin
  inherited updateGPU(args);
end;
{$endif}

var s :single;
initialization

  s := exp(NegInfinity);

end.

