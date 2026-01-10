{$ifdef FPC}
  {$mode Delphi}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$SafeFPUExceptions off}
{$endif}
{$PointerMath on}
unit nTransformers;

interface

uses
  nTypes, nAttentionLayer, nConnectedlayer, nBaseLayer, ntensors;

type

  { TNormLayer }

  TNormLayer = class(TBaseLayer)
    x_mu, dx_norm :TNeuralTensor;
    constructor Create(const aBatch, aInputs:SizeInt);
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
  end;

  { TDecoderLayer }

  TDecoderLayer = class(TBaseLayer)
  private
    const MLP_HIDDEN_LAYERS = 1;
  public
    heads: SizeInt;
    attn, enc_dec_attn: TAttentionLayer;
    ffn : TFeedForwardLayer;
    norm1, norm2, norm3 : TNormLayer;
    constructor Create(const aBatch, aInputs, aMaxSeq, aHeads, aMLPHiddenOutputs:SizeInt; const aMLPLayers: SizeInt=1);
    destructor destroy; override;
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
  end;

  { TEncoderLayer }

  TEncoderLayer = class(TBaseLayer)
  private
    const MLP_HIDDEN_LAYERS = 1;
  public
    heads: SizeInt;
    attn: TAttentionLayer;
    ffn : TFeedForwardLayer;
    norm1, norm2: TNormLayer;
    constructor Create(const aBatch, aInputs, aMaxSeq, aHeads, aMLPHiddenOutputs:SizeInt; const aMLPLayers: SizeInt=1);
    destructor destroy; override;
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
  end;


  { TTransformerLayer }

  TTransformerLayer = class(TBaseLayer)
    public
      InputVocab, OutputVocab, heads, SeqLen : SizeInt;
      encoders : TArray<TEncoderLayer>;
      decoders : TArray<TDecoderLayer>;
      outLinear   : TConnectedLayer;
      embeddingInput, embeddingOutput : TNeuralTensor;
      embeddingsIn, embeddingsOut : TIntTensor;
      positionalEncoder : TNeuralTensor;
    constructor Create(
      const aBatch, aInputs, aHeads, aEncoders, aDecoders:SizeInt;
      const aInputVocab, aOutputVocab:SizeInt;
      const aSeqLen:SizeInt;
      const aMLPHiddenOutputs :SizeInt);
    destructor destroy; override;
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
    class procedure PositionaEncoding(const seqLength:Sizeint; const dst: TNeuralTensor);static;
  end;

implementation

function max(const a,b:single):single; inline;
begin
  if a>b then
    result := a
  else
    result := b
end;
{ TNormLayer }

constructor TNormLayer.Create(const aBatch, aInputs: SizeInt);
begin
  batch := aBatch;
  inputs := aInputs;
  outputs := aInputs;
  weights := TSingleTensor.Create([Inputs]);  //gamma
  weights.fill(1);
  biases := TSingleTensor.Create([Inputs]);   //beta
  mean := TSingleTensor.create([Batch]);
  variance := TSingleTensor.create([Batch]);
  output := TSingleTensor.Create([batch, inputs]);
  x_norm := TSingleTensor.Create([batch, inputs]);
end;

procedure TNormLayer.setTrain(ATrain: boolean);
begin
  if ATrain then begin
    mean_delta := TNeuralTensor.create([Batch]);
    variance_delta := TNeuralTensor.create([Batch]);
    x_mu := TNeuralTensor.create([batch, inputs]);
    dx_norm := TNeuralTensor.Create([batch, inputs]);
  end else begin
    mean_delta.free;
    variance_delta.free;
    x_mu.free;
    dx_norm.free;
  end;
  FTrain := aTrain
end;

procedure TNormLayer.setBatch(ABatch: SizeInt);
begin
  mean.resize([aBatch]);
  variance.resize([aBatch]);
  output.resize([aBatch, inputs]);
  x_norm.resize([aBatch, inputs]);
  if FTrain then begin
    mean_delta.resize([aBatch]);
    variance_delta.resize([aBatch]);
    dx_norm.resize([aBatch, inputs]);
  end;
  batch := aBatch
end;

procedure TNormLayer.forward(var state: TNNetState);
var i,j:SizeInt;
begin
  state.input.copyTo(x_norm);
  for i:=0 to batch-1 do begin
    mean.data[i]     := state.input.sum(1, inputs, i*inputs)/inputs;
    variance.data[i] := max(state.input.sumSqrDiff(mean.data[i], 1, inputs, i*inputs), sEPSILON);
    x_norm.Normalize(mean.data[i], sqrt(variance.data[i]), 1, inputs, i*inputs);
  end;
  x_norm.CopyTo(output);
  for i:=0 to batch-1 do begin
    output.Multiply(weights, i*inputs, 0, inputs);
    output.Add(biases, i*inputs, 0, inputs);
  end
end;

procedure TNormLayer.backward(var state: TNNetState);
var N, T, D, i, j:SizeInt;
  mu1, mu2, sg, stddev, dweights, dbias : TNeuralTensor.T;
begin
  D := state.input.w;
  T := state.input.h;
  N := state.input.c;
  state.input.CopyTo(x_mu);
  x_mu.Subtract(mean);
  delta.copyTo(dx_norm);
  dx_norm.Multiply(weights);
  //dx_norm = d_output * self.gamma.data  // (N, T, D)
  //for j:=0 to inputs-1 do begin
  //  dweights := 0;
  //  dbias    := 0;
  //  for i:=0 to batch-1 do begin
  //    dweights := dweights + delta.data[i*inputs+j] * x_norm.data[i*inputs+j];
  //    dbias    := dbias   + delta.Data[i*inputs+j];
  //  end;
  //  weight_updates.data[i] := weight_updates.data[i] + dweights;
  //  bias_updates.data[i] := bias_updates.data[i] + dbias;
  //end;
  weight_updates.addDots(x_norm, delta);
  bias_updates.addSums(delta);

  // std_inv = 1. / (self.std + self.eps)
  if assigned(state.delta) and assigned (state.delta.data) then begin
    for i:=0 to batch -1 do begin
      //dvar = np.sum(dx_norm * x_mu * -0.5 * std_inv ** 3, axis=-1, keepdims=True)  # (N, T, 1)
      stddev := sqrt(variance.data[i]);
      sg := 0;
      for j := 0 to inputs-1 do
        sg := sg -0.5 * dx_norm.data[i*inputs + j] * x_norm.data[i*inputs + j] / (variance.data[i]*stddev);
      variance_delta.data[i] := sg;
       //dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2. * x_mu, axis=-1, keepdims=True)
      mu1 := 0;
      for j := 0 to inputs-1 do
        mu1 := mu1 + dx_norm.data[i*inputs + j];
      mu1 := -mu1 / variance.data[i];

      mu2 := 0;
      for j := 0 to inputs-1 do
        mu2 := mu2 + x_mu.data[i*inputs + j] ;
      mean_delta.Data[i] := -2.0 * mu1 * sg * mu2/inputs;
      //dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / inputs) + (dmean / inputs)
      for j:=0 to inputs-1 do
        state.delta.data[i*inputs+j] := (dx_norm.data[i*inputs+j]/stddev) + (sg*2*x_mu.data[i*inputs+j] / inputs) + (mean_delta.Data[i] / inputs)  // (N, T, D)
    end;
  end;

end;

procedure TNormLayer.update(const args: TUpdateArgs);
begin

  if assigned(biases.data) then begin
    biases.axpy(args.learningRate / args.batch, bias_updates);
    bias_updates.Multiply(args.momentum)
  end;

  weight_updates.axpy(-args.decay * args.batch, weights);

  weights.axpy(args.learningRate / args.batch, weight_updates);

  weight_updates.Multiply(args.momentum);

  inherited update(args);
end;


{ TDecoderLayer }

constructor TDecoderLayer.Create(const aBatch, aInputs, aMaxSeq, aHeads,
  aMLPHiddenOutputs: SizeInt; const aMLPLayers: SizeInt);
begin
  Inputs      := aInputs;
  heads       := aHeads;
  outputs     := aInputs;
  batch       := aBatch;
  attn        := TAttentionLayer.Create(Batch, inputs, aMaxSeq, Heads, true);
  enc_dec_attn:= TAttentionLayer.Create(Batch, inputs, aMaxSeq, Heads);
  ffn         := TFeedForwardLayer.Create(Batch, 1, Inputs, Inputs, MLP_HIDDEN_LAYERS, aMLPHiddenOutputs, acRELU);
  ffn.hiddenLayers[MLP_HIDDEN_LAYERS].ActivationType := acLINEAR;
  norm1       := TNormLayer.Create(batch, inputs);
  norm2       := TNormLayer.Create(batch, inputs);
  norm3       := TNormLayer.Create(batch, inputs);
  output      := norm3.output;
  x           := TNeuralTensor.Create([batch, inputs]);
end;

destructor TDecoderLayer.destroy;
begin
  norm3.free;
  norm2.free;
  norm1.free;
  ffn.free;
  enc_dec_attn.free;
  attn.Free;
  inherited destroy;
end;

procedure TDecoderLayer.setTrain(ATrain: boolean);
begin
  attn.train         := aTrain;
  enc_dec_attn.train := aTrain;
  ffn.train          := aTrain;
  norm1.train        := aTrain;
  norm2.train        := aTrain;
  norm3.train        := aTrain;
  if aTrain then
    delta            := norm3.delta;
  FTrain             := aTrain
end;

procedure TDecoderLayer.setBatch(ABatch: SizeInt);
begin
  attn.batch         := aBatch;
  ffn.batch          := aBatch;
  enc_dec_attn.Batch := aBatch;
  norm1.batch        := aBatch;
  norm2.batch        := aBatch;
  norm3.batch        := aBatch;
  x.resize([batch, inputs]);
  output             := norm3.output;
  if FTrain then
    delta            := norm3.delta;
  Batch              := aBatch;
end;

procedure TDecoderLayer.forward(var state: TNNetState);
var s: TNNetState;
begin
  assert(length(state.inputs)=2, '[TDecoderLayer.forward] <state.input> must have two inputs!');
  s := state;
  s.inputs := [state.input, state.input, state.input];
  attn.forward(s);
  if FTrain then begin
    attn.output.CopyTo(x);
    x.add(s.input^);
    s.input := @x;
  end else begin
    attn.output.add(s.input^);
    s.input := @attn.output;
  end;
  norm1.forward(s);

  s.input  := @norm1.output;
  s.inputs := [s.input, state.inputs[1], state.inputs[1]];
  enc_dec_attn.forward(s);
  if FTrain then begin
    enc_dec_attn.output.CopyTo(x);
    x.add(s.input^);
    s.input := @x;
  end else begin
    enc_dec_attn.output.add(s.input^);
    s.input := @enc_dec_attn.output;
  end;
  norm2.forward(s);

  s.input:=@norm2.output;
  ffn.forward(s);

  if FTrain then begin
    ffn.output.CopyTo(x);
    x.add(s.input^);
    s.input := @x;
  end else begin
    ffn.output.add(s.input^);
    s.input := @ffn.output;
  end;
  norm3.forward(s);

end;

procedure TDecoderLayer.backward(var state: TNNetState);
var s:TNNetState;
begin

  s := state;
  s.input := @ffn.output;
  s.delta := @ffn.delta;
  norm3.backward(s);

  s.input := @norm2.output;
  s.delta := @norm2.delta;
  ffn.backward(s);
  norm2.delta.add(ffn.delta);

  s.input := @enc_dec_attn.output;
  s.delta := @enc_dec_attn.delta;
  norm2.backward(s);


  s.input := @norm1.output;
  s.delta := @norm1.delta;
  s.inputs := [@norm1.output, state.inputs[1], state.inputs[1]];
  s.delta  := @norm1.delta;
  enc_dec_attn.backward(state);
  norm1.delta.Add(norm2.delta);

  s.input := @attn.output;
  s.delta := @attn.delta;
  norm1.backward(s);

  if assigned(state.delta) then begin
    s.inputs := [state.input, state.input, state.input];
    attn.backward(state);
    state.delta.Add(attn.delta);
  end;

end;

procedure TDecoderLayer.update(const args: TUpdateArgs);
begin
  attn.update(args);
  norm1.update(args);
  ffn.update(args);
  norm2.update(args);
  enc_dec_attn.update(args);
  norm3.update(args);
  inherited update(args);
end;

{ TEncoderLayer }

constructor TEncoderLayer.Create(const aBatch, aInputs, aMaxSeq, aHeads, aMLPHiddenOutputs: SizeInt; const aMLPLayers: SizeInt);
begin
  batch  := aBatch;
  inputs := aInputs;
  heads  := aHeads;
  attn   := TAttentionLayer.Create(batch, inputs, aMaxSeq, heads);
  ffn    := TFeedForwardLayer.Create(batch, 1, Inputs, inputs, MLP_HIDDEN_LAYERS, aMLPHiddenOutputs, acRELU);
  ffn.hiddenLayers[MLP_HIDDEN_LAYERS].ActivationType := acLINEAR;
  norm1  := TNormLayer.Create(batch, inputs);
  norm2  := TNormLayer.Create(batch, inputs);
  x      := TNeuralTensor.Create([batch, inputs]);

  output := norm2.output;

end;

destructor TEncoderLayer.destroy;
begin
  norm2.free;
  norm1.free;
  ffn.free;
  attn.Free;
  inherited destroy;
end;

procedure TEncoderLayer.setTrain(ATrain: boolean);
begin
  attn.train:=aTrain;
  ffn.train := aTrain;
  norm1.train := aTrain;
  norm2.train := aTrain;
  if aTrain then
    delta := norm2.delta;
  FTrain := aTrain
end;

procedure TEncoderLayer.setBatch(ABatch: SizeInt);
begin
  attn.batch:= aBatch;
  ffn.batch:= aBatch;
  norm1.batch:= aBatch;
  norm2.batch:= aBatch;
  x.resize([batch, inputs]);
  output := norm2.output;
  if FTrain then
    delta := norm2.delta;
  Batch := aBatch;
end;

procedure TEncoderLayer.forward(var state: TNNetState);
var
  s: TNNetState;
begin
  s := state;
  //attn_output = self.self_attn.forward(x, x, x)
  s.inputs := [state.input, state.input, state.input];
  attn.forward(s);
  //x2 = x + attn_output
  if FTrain then begin
    attn.output.copyTo(x);
    x.add(s.input^);
    s.input := @x;
  end
  else begin
    attn.output.add(s.input^);
    s.input := @attn.output
  end;
  //x_norm1 = self.norm1.forward(x2)
  norm1.forward(s);

  //# Feed-forward avec connexion résiduelle et normalisation
  //ffn_output = self.ffn.forward(x_norm1)
  s.input := @norm1.output;
  ffn.forward(s);
  //x3 = x_norm1 + ffn_output

  if FTrain then begin
    ffn.output.copyTo(x);
    x.add(norm1.output);
    s.input := @x;
  end
  else begin
    ffn.output.add(norm1.output);
    s.input := @ffn.output
  end;
  //x_norm2 = self.norm2.forward(x3)
  norm2.forward(s);
  //self.output = x_norm2
  //return x_norm2
end;

procedure TEncoderLayer.backward(var state: TNNetState);
var s:TNNetState;
begin
  s := state;
  s.input := @ffn.output;
  s.delta := @ffn.delta;
  norm2.backward(s);

  s.input := @norm1.output;
  s.delta := @norm1.delta;
  ffn.backward(s);
  norm1.delta.add(ffn.delta);

  s.input := @attn.output;
  s.delta := @attn.delta;
  norm1.backward(s);


  if assigned(state.delta) and assigned(state.delta.data) then begin
    attn.backward(state);
    state.delta.add(attn.delta)
  end;

end;

procedure TEncoderLayer.update(const args: TUpdateArgs);
begin
  attn.update(args);
  norm1.update(args);
  ffn.update(args);
  norm2.update(args);
  inherited update(args);
end;


{ TTransformerLayer }

constructor TTransformerLayer.Create(const aBatch, aInputs, aHeads, aEncoders,
  aDecoders: SizeInt; const aInputVocab, aOutputVocab: SizeInt;
  const aSeqLen: SizeInt; const aMLPHiddenOutputs: SizeInt);
var i:SizeInt;
begin
  Batch       := aBatch       ;
  Inputs      := aInputs      ;
  Outputs     := aInputs      ;
  InputVocab  := aInputVocab  ;
  OutputVocab := aOutputVocab ;
  SeqLen      := aSeqLen      ;
  heads       := aHeads       ;
  setLength(encoders, aEncoders);
  for i:=0 to high(encoders) do
    encoders[i] := TEncoderLayer.Create(batch, inputs, seqLen, Heads, aMLPHiddenOutputs);

  setLength(decoders, aDecoders);
  for i:=0 to high(encoders) do
    decoders[i] := TDecoderLayer.Create(batch, inputs, seqLen, Heads, aMLPHiddenOutputs);

  embeddingInput  := TNeuralTensor.Create([batch, InputVocab, inputs]);
  embeddingInput.NormalDistribution(0, 0.01);
  embeddingOutput := TNeuralTensor.Create([batch, OutputVocab, outputs]);
  embeddingOutput.NormalDistribution(0, 0.01);

  embeddingsIn := TIntTensor.Create([batch, SeqLen]);

  positionalEncoder :=TNeuralTensor.Create([SeqLen, inputs]);
  PositionaEncoding(SeqLen, positionalEncoder);

  outLinear       := TConnectedLayer.Create(batch, 1, inputs, OutputVocab);
  output          := outLinear.output;

end;

destructor TTransformerLayer.destroy;
begin
  inherited destroy;
end;

procedure TTransformerLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  if FTrain then
    delta.resize([Batch, Outputs])
  else
    delta.free;
end;

procedure TTransformerLayer.setBatch(ABatch: SizeInt);
var i:SizeInt;
begin
  if ABatch=batch then exit;
  for i:=0 to high(encoders) do
     encoders[i].setBatch(abatch);

  for i:=0 to high(encoders) do
     decoders[i].setBatch(Abatch);

   embeddingInput.resize([abatch, InputVocab, inputs]);
   embeddingOutput.resize([abatch, OutputVocab, outputs]);
   embeddingsIn.reSize([batch, SeqLen]);
   outLinear.setBatch(aBatch);
   output.reSize([aBatch, Outputs]);
   batch :=aBatch;
   if FTrain then
     delta.resize([aBatch, Outputs]);
end;

procedure TTransformerLayer.forward(var state: TNNetState);
var encState, decState : TNNetState;
  i:longint;

begin
{$T+}
// T+ means "@" operator produces typed pointer instead of a pointer
// preventing overloaded function confusion
  state.input.getEmbeddings(@embeddingsIn, embeddingInput);
  embeddingInput.Add(positionalEncoder);
  state.truth.getEmbeddings(@embeddingsOut, embeddingOutput);
  embeddingInput.Add(positionalEncoder);
{$T-}
  encState := state;
  encstate.input := @embeddingInput;
  for i:=0 to high(encoders) do begin
    encoders[i].forward(encState);
    encState.input := @encoders[i].output;
  end;

  decState.input  := @embeddingOutput;
  decState.inputs := [@embeddingOutput, encState.input {@encoders[high(encoders)].output}];
  for i:=0 to high(decoders) do begin
    decoders[i].forward(decState);
    decState.input := @decoders[i].output;
    decState.inputs[0] := @decoders[i].output;
  end;

  outLinear.forward(decState);
  output.Softmax()

end;

procedure TTransformerLayer.backward(var state: TNNetState);
begin

end;

procedure TTransformerLayer.update(const args: TUpdateArgs);
begin
  inherited update(args);
end;

class procedure TTransformerLayer.PositionaEncoding(const seqLength: Sizeint; const dst: TNeuralTensor);
var
  i, j, idx, _inputs:SizeInt;
  factor, v: TNeuralTensor.T;
begin
  assert(seqLength = dst.h);
  _inputs := dst.w;
  factor := -ln(10000.0)/_inputs;
  for i:=0 to seqLength-1 do
    for j:=0 to _inputs div 2-1 do begin
      idx := i*_inputs + j*2;
      v := i*exp(j*2*factor);
      dst.DynData[idx]   := sin(v);
      dst.DynData[idx+1] := cos(v);
    end;
end;

initialization


finalization

end.

