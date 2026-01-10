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
  , windows
  ;

type

  { TAttentionLayer }

  TAttentionLayer = class(TBaseLayer)
  private const
         QUERY = 0;
         KEY   = 1;
         VALUE = 2;
  public
    maxSeq, Heads, KVHeads, headLen : SizeInt;
    //KVHeadLen: SizeInt;
    wq, wk, wv, wo : TConnectedLayer;
    mask, attnWeights : TNeuralTensor;
    attnOut : TNeuralTensor;
    isFuture : boolean;
    constructor Create(const aBatch, aInputs, aMaxSeq, aHeads: SizeInt; const aIsFuture:boolean = false);
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    function splitHeads(const x:TNeuralTensor):TNeuralTensor;
    function combineHeads(const x:TNeuralTensor):TNeuralTensor;
    function ScaledDotProductAttention(const Q, K, V:TNeuralTensor;const mask:TNeuralTensor.PT=nil):TNeuralTensor; // SDPA
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
//uses nNet;

{ TAttentionLayer }

constructor TAttentionLayer.Create(const aBatch, aInputs, aMaxSeq, aHeads: SizeInt; const aIsFuture: boolean);
begin
  //if aKVHeads=0 then aKVHeads := aHeads;
  assert(ainputs mod aHeads=0, '[TAttention.Create] <heads> must e a multiples of <KVHeads>');
  //assert(aHeads mod aKVHeads=0, '[TAttention.Create] <heads> must e a multiples of <KVHeads>');
  batch  := aBatch;
  inputs := aInputs;
  Heads  := aHeads;
  //KVHeads := aKVHeads;
  headLen := inputs div heads;   //d_model
  //KVheadLen := inputs div KVHeads;
  maxSeq := aMaxSeq;
  wq := TConnectedLayer.Create(batch*maxSeq, 1, inputs, inputs, acLINEAR, false, false);
  wk := TConnectedLayer.Create(batch*maxSeq, 1, inputs, {kvHeads*headLen} inputs, acLINEAR, false, false);
  wv := TConnectedLayer.Create(batch*maxSeq, 1, inputs, {kvHeads*headLen} inputs, acLINEAR, false, false);
  wo := TConnectedLayer.Create(batch*maxSeq, 1, inputs, inputs, acLINEAR, false, false);
  output := wo.output;
  attnOut := TNeuralTensor.Create([batch*maxSeq, inputs]);
  //attnOut.resize([batch, input, input], batch);
  isFuture:=aIsFuture;
  if isFuture then begin
    mask.resize([maxSeq, maxSeq]);
    mask.triangularFill(-infinity, true)
  end;

end;

procedure TAttentionLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  wq.setTrain(ATrain);
  wk.setTrain(ATrain);
  wv.setTrain(ATrain);
  wo.setTrain(ATrain);
  //attnDelta.resize([batch, input, input], batch);
  output   := wo.output;
  delta    := wo.delta;
  FTrain   := ATrain;
end;

procedure TAttentionLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit;
  wq.setBatch(ABatch*maxSeq);
  wk.setBatch(ABatch*maxSeq);
  wv.setBatch(ABatch*maxSeq);
  wo.setBatch(ABatch*maxSeq);
  attnOut.resize([aBatch*maxSeq, inputs]);
  if isFuture and (mask.size()<>maxSeq*maxSeq) then begin
    mask.resize([maxSeq, maxSeq]);
    mask.triangularFill(-infinity, true)
  end;

  output   := wo.output;
  delta    := wo.delta;
  Batch   := aBatch;
end;

function TAttentionLayer.splitHeads(const x: TNeuralTensor): TNeuralTensor;
var tmp : TSingleTensor;
begin
   tmp := x;
   tmp.reshape([batch, tmp.shape[1], Heads, headLen]);
   result := tmp.Permute([0, 2, 1, 3]);
end;

function TAttentionLayer.combineHeads(const x: TNeuralTensor): TNeuralTensor;
var tmp:TSingleTensor;
begin
  result := x.permute([0, 2, 1, 3]);
  result.reShape([batch, tmp.shape[1], heads*headLen {inputs}]);
end;

function TAttentionLayer.ScaledDotProductAttention(const Q, K,
  V: TNeuralTensor; const mask: TNeuralTensor.PT): TNeuralTensor;
var
  d_k : SizeInt;
  scores:TSingleTensor;
begin
  d_k := Q.w;
  scores := Q.matMul(K.permute([0, 1, 3, 2]));
  scores.Divide(sqrt(d_k));
  if assigned(mask) then scores.add(mask);
  attnWeights.reSize(scores.shape, scores.groups);
  scores.copyTo(attnWeights);
  scores.Softmax();
  result := scores.matmul(V);
end;

procedure TAttentionLayer.forward(var state: TNNetState);
var //Q, K, V,
  _mask : PSingle;
  s:TNNetState;
begin
  assert(length(state.inputs)=3, '[TAttentionLayer.forward]  <state.inputs> must have two inputs!');

  s := state;

  s.input := state.inputs[QUERY];
  Wq.forward(s);

  s.input := state.inputs[KEY];
  Wk.forward(s);

  s.input := state.inputs[VALUE];
  Wv.forward(s);

  _mask := nil;
  if isFuture then _mask:=mask.Data;

  attnOut := combineHeads(ScaledDotProductAttention(splitHeads(wq.output), splitHeads(wk.output), splitHeads(wv.output), _mask));

  s.input := @attnOut;
  Wo.forward(s);
  //output:= wo.output;
end;

procedure TAttentionLayer.backward(var state: TNNetState);
var
  attnDelta:TSingleTensor;
  s:TNNetState;
begin
  assert((length(state.inputs)=3), '[TAttentionLayer.backward] <state.deltas> and <state.deltas> must have three values!');
  s:= state;
  attnDelta.resize([batch, inputs, inputs], batch);
  s.input:=@attnOut;
  s.delta:=@attnDelta;
  wo.backward(state);

  wq.delta.add(attnDelta);
  wk.delta.add(attnDelta);
  wv.delta.add(attnDelta);

  //attnDelta := splitHeads(state.delta^);
  //
  //attnDelta := combineHeads(attnDelta);
  s.input:=state.inputs[QUERY];
  s.delta:=@wk.delta;
  wq.backward(s);

  s.input:=state.inputs[KEY];
  s.delta:=@wv.delta;
  wk.backward(s);

  s.input:=state.inputs[VALUE];
  s.delta:=@state.delta;
  wv.backward(s);
end;

procedure TAttentionLayer.update(const args: TUpdateArgs);
begin
  inherited update(args);
  wo.update(args);
  wq.update(args);
  wk.update(args);
  wv.update(args);
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

function sum(const a, b:single):single;
begin
  result := a+b
end;

var s,s1 :TSingleTensor;
initialization

//sDigits := 0;
  //s.linSpace(1, 4*4*4*4, 4*4*4*4);
  //s.reshape([4,4,4,4]);
  ////s1.resize([4,4]);
  //s.reduce(sum, [1,3], s1);
  //s1.print;
  ////s.permute([3,0,1,2]).print;
  //readln


end.

