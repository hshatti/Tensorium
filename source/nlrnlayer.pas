unit nLRNLayer; // normalization layer
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface
uses NTypes, nBaseLayer, ntensors;

type

  { TLRNLayer }

  TLRNLayer = class (TBaseImageLayer)
    size : SizeInt;
    alpha, beta, kappa : Single;
    squared, norms : TSingleTensor;
    constructor Create(const aBatch, aWidth, aHeight, aChannels, aSize: SizeInt; const aAlpha, aBeta, aKappa: single);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;

    {$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
    {$endif}
  end;


implementation

{ TLRNLayer }

constructor TLRNLayer.Create(const aBatch, aWidth, aHeight, aChannels,
  aSize: SizeInt; const aAlpha, aBeta, aKappa: single);
begin
  assert(aSize div 2<=aChannels, '[TLRNLayer] aSize must be <= 2 X aChannels');
  layerType := ltNORMALIZATION;
  batch := aBatch;

  h := aHeight;
  w := aWidth;
  c := aChannels;
  outh := h;
  outw := w;
  outc := c;

  inputShape  := [batch, c, h, w];
  size    := aSize;
  kappa   := aKappa;
  alpha   := aAlpha;
  beta    := aBeta;
  inputs  := w * h * c;
  output  := TSingleTensor.Create([batch, c, h, w], batch);
  delta   := TSingleTensor.Create([batch, c, h, w], batch);
  squared := TSingleTensor.Create([batch, c, h, w], batch);
  norms   := TSingleTensor.Create([batch, c, h, w], batch);
  outputs := inputs;
end;

procedure TLRNLayer.setBatch(ABatch: SizeInt);
begin
  if batch = ABatch then exit;
  batch := ABatch;
  inputShape[0] := batch;
  output  .reSize([batch, c, h, w], batch);
  delta   .reSize([batch, c, h, w], batch);
  squared .reSize([batch, c, h, w], batch);
  norms   .reSize([batch, c, h, w], batch);
end;

procedure TLRNLayer.setTrain(ATrain: boolean);
begin

  FTrain := train;
end;

procedure TLRNLayer.forward(var state: TNNetState);
var
  //input ,
  sqr_offset, norm_offset:PSingle;
  b, k, im_size , prev, next, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  im_size := w*h;
  squared.fill(0);
  state.input.square(squared.data);

  for b := 0 to batch -1 do
      begin
          offset :=  b * inputs;
          sqr_offset   := squared.data + offset;
          norm_offset  := norms.data   + offset;
          //input := state.input.data   + b * inputs;
          //pow_cpu(w * h * c, 2, input, 1, squared, 1);
          norms.fillExt(kappa, offset, im_size);
          //const_cpu(w * h, layer.kappa, norms, 1);
          for k := 0 to size div 2 -1 do
              TSingleTensor.axpy(im_size, alpha, sqr_offset + k * im_size, 1, norm_offset, 1);
          for k := 1 to c -1 do
              begin
                  //norms.CopyTo(norms, b*outputs + k*im_size, 1, b*outputs + (k-1)*im_size, 1, im_size);
                  move((norm_offset+(k-1)*im_size)[0], (norm_offset + k*im_size)[0], im_size*sizeOf(single));
                  //axpy_cpu(im_size, norm_offset + (k-1)*im_size, 1, norm_offset + k*im_size, 1);
                  prev := k-((size-1) div 2)-1;
                  next := k+(size div 2);
                  if prev >= 0 then
                      TSingleTensor.axpy(im_size, -alpha, sqr_offset + prev*im_size, 1, norm_offset + k*im_size, 1);
                  if next < c then
                      TSingleTensor.axpy(im_size, alpha, sqr_offset + next*im_size, 1, norm_offset + k*im_size, 1)
              end
      end;
  norms.power(-beta, output.data);
  //pow_cpu(outputs * layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
  output.Multiply(state.input.data);
  //mul_cpu(outputs * layer.batch, net.input, 1, layer.output, 1);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}

end;

procedure TLRNLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  norms.power(-beta,  state.delta.data);
  //pow_cpu(w * h * c * batch, -beta, norms, 1, state.delta, 1);
  state.delta.Multiply(delta.data);
  //mul_cpu(w * h * c * batch, delta, 1, state.delta, 1)
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

{$if defined(USE_OPENCL)}
procedure TLRNLayer.forwardGPU(var state: TNNetState);
var
  //input ,
  b, k, im_size , prev, next, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  im_size := w*h;
  ocl.mulvv(batch*outputs, state.input.devData, 0, 1, state.input.devData, 0, 1, squared.devData, 0, 1);
  //squared.fill(0);
  //state.input.square(squared.data);


  for b := 0 to batch -1 do
      begin
          offset :=  b * inputs;
          //sqr_offset   := squared.data + offset;
          //norm_offset  := norms.data   + offset;
          //input := state.input.data   + b * inputs;
          //pow_cpu(w * h * c, 2, input, 1, squared, 1);
          ocl.fill(im_size, norms.devData, offset, kappa, 1);
          //norms.fillExt(kappa, offset, im_size);
          //const_cpu(w * h, layer.kappa, norms, 1);
          for k := 0 to size div 2 -1 do
              ocl.axpy(im_size, alpha, squared.devData, offset + k * im_size, 1, norms.devData, offset, 1);
              //TSingleTensor.axpy(im_size, alpha, sqr_offset + k * im_size, 1, norm_offset, 1);
          for k := 1 to c -1 do
              begin
                  ocl.copy(im_size, norms.devData, offset + (k-1)*im_size, 1, norms.devData, offset + k*im_size, 1);
                  //norms.CopyTo(norms, b*outputs + k*im_size, 1, b*outputs + (k-1)*im_size, 1, im_size);
                  //move((norm_offset+(k-1)*im_size)[0], (norm_offset + k*im_size)[0], im_size*sizeOf(single));
                  //axpy_cpu(im_size, norm_offset + (k-1)*im_size, 1, norm_offset + k*im_size, 1);
                  prev := k-((size-1) div 2)-1;
                  next := k+(size div 2);
                  if prev >= 0 then
                      ocl.axpy(im_size, -alpha, squared.devData, offset + prev*im_size, 1, norms.devData, offset + k*im_size, 1);
                      //TSingleTensor.axpy(im_size, -alpha, sqr_offset + prev*im_size, 1, norm_offset + k*im_size, 1);
                  if next < c then
                      ocl.axpy(im_size, alpha, squared.devData, offset + next*im_size, 1, norms.devData, offset + k*im_size, 1)
                      //TSingleTensor.axpy(im_size, alpha, sqr_offset + next*im_size, 1, norm_offset + k*im_size, 1)
              end
      end;

  ocl.power(outputs*batch, norms.devData, 0, 1, -beta, output.devData, 0, 1);
  //norms.power(-beta, output.data);
  //pow_cpu(outputs * layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
  ocl.mulvv(outputs*batch, state.input.devData, 0, 1, output.devData, 0, 1, output.devData, 0, 1);
  //output.Multiply(state.input.data);
  //mul_cpu(outputs * layer.batch, net.input, 1, layer.output, 1);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TLRNLayer.backwardGPU(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  ocl.power(outputs*batch, norms.devData, 0, 1, -beta, state.delta.devData, 0, 1);
  //norms.power(-beta,  state.delta.data);
  ocl.mulvv(outputs*batch, delta.devData, 0, 1, state.delta.devData, 0, 1, state.delta.devData, 0, 1);
  //state.delta.Multiply(delta.data);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$elseif defined(USE_CUDART)}
procedure TLRNLayer.forwardGPU(var state: TNNetState);
var
  //input ,
  b, k, im_size , prev, next, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  im_size := w*h;
  cuda.mulvv(batch*outputs, state.input.devData, 0, 1, state.input.devData, 0, 1, squared.devData, 0, 1);
  //squared.fill(0);
  //state.input.square(squared.data);


  for b := 0 to batch -1 do
      begin
          offset :=  b * inputs;
          //sqr_offset   := squared.data + offset;
          //norm_offset  := norms.data   + offset;
          //input := state.input.data   + b * inputs;
          //pow_cpu(w * h * c, 2, input, 1, squared, 1);
          cuda.fill(im_size, norms.devData, offset, kappa, 1);
          //norms.fillExt(kappa, offset, im_size);
          //const_cpu(w * h, layer.kappa, norms, 1);
          for k := 0 to size div 2 -1 do
              cuda.axpy(im_size, alpha, squared.devData, offset + k * im_size, 1, norms.devData, offset, 1);
              //TSingleTensor.axpy(im_size, alpha, sqr_offset + k * im_size, 1, norm_offset, 1);
          for k := 1 to c -1 do
              begin
                  cuda.copy(im_size, norms.devData, offset + (k-1)*im_size, 1, norms.devData, offset + k*im_size, 1);
                  //norms.CopyTo(norms, b*outputs + k*im_size, 1, b*outputs + (k-1)*im_size, 1, im_size);
                  //move((norm_offset+(k-1)*im_size)[0], (norm_offset + k*im_size)[0], im_size*sizeOf(single));
                  //axpy_cpu(im_size, norm_offset + (k-1)*im_size, 1, norm_offset + k*im_size, 1);
                  prev := k-((size-1) div 2)-1;
                  next := k+(size div 2);
                  if prev >= 0 then
                      cuda.axpy(im_size, -alpha, squared.devData, offset + prev*im_size, 1, norms.devData, offset + k*im_size, 1);
                      //TSingleTensor.axpy(im_size, -alpha, sqr_offset + prev*im_size, 1, norm_offset + k*im_size, 1);
                  if next < c then
                      cuda.axpy(im_size, alpha, squared.devData, offset + next*im_size, 1, norms.devData, offset + k*im_size, 1)
                      //TSingleTensor.axpy(im_size, alpha, sqr_offset + next*im_size, 1, norm_offset + k*im_size, 1)
              end
      end;

  cuda.power(outputs*batch, norms.devData, 0, 1, -beta, output.devData, 0, 1);
  //norms.power(-beta, output.data);
  //pow_cpu(outputs * layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
  cuda.mulvv(outputs*batch, state.input.devData, 0, 1, output.devData, 0, 1, output.devData, 0, 1);
  //output.Multiply(state.input.data);
  //mul_cpu(outputs * layer.batch, net.input, 1, layer.output, 1);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TLRNLayer.backwardGPU(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  cuda.power(outputs*batch, norms.devData, 0, 1, -beta, state.delta.devData, 0, 1);
  //norms.power(-beta,  state.delta.data);
  cuda.mulvv(outputs*batch, delta.devData, 0, 1, state.delta.devData, 0, 1, state.delta.devData, 0, 1);
  //state.delta.Multiply(delta.data);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$endif}

end.

