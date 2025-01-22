unit nUpSampleLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}
interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;

type

  { TUpSampleLayer }

  TUpSampleLayer = class(TBaseImageLayer)
    reverse : boolean;
    stride : SizeInt;
    scale : single;
    constructor Create(const ABatch, AWidth, AHeight, AChannels: SizeInt;
      AStride: SizeInt; const AScale :Single =1);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
{$ifdef USE_OPENCL}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
{$endif}
  end;

implementation

{ TUpSampleLayer }

constructor TUpSampleLayer.Create(const ABatch, AWidth, AHeight,
  AChannels: SizeInt; AStride: SizeInt; const AScale: Single);
begin
  layerType := ltUPSAMPLE;
  batch := Abatch;
  w := AWidth;
  h := AHeight;
  c := AChannels;
  outW := AWidth * AStride;
  outH := AHeight * AStride;
  outC := AChannels;
  if AStride < 0 then
      begin
          AStride := -AStride;
          reverse := true;
          outW := AWidth div AStride;
          outH := AHeight div AStride
      end;
  stride := AStride;
  inputShape := [batch, c, h, w];
  outputs := outW * outH * outC;
  inputs := w * h * c;
  scale := AScale;
  //delta := TSingles.Create([batch, c, h, w], batch);
  output := TSingleTensor.Create([batch, outC, outH, outW], batch);
end;

procedure TUpSampleLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch = Batch then exit;
  batch := ABatch;
  output.reSize([batch, c, h, w], batch);
  inputShape[0] := batch;;
  if FTrain then
      Delta.reSize([batch, c,h, w], batch);
end;

procedure TUpSampleLayer.setTrain(ATrain: boolean);
begin
  if FTrain=Atrain then exit;
  FTrain := ATrain;
  if FTrain then
    delta := TSingleTensor.Create([batch, c, h, w], batch)
  else
    delta.free

end;

procedure upsample(const &in: PSingle; const w, h, aChannels, batch,
  stride: SizeInt; const isForward: boolean; const scale: single;
  const &out: PSingle);
var
  //i, j, k, b,
  c, y, x, in_index, out_index:SizeInt;
begin
   //for b := 0 to batch-1 do
   //    for k := 0 to aChannels-1 do
   //        for j := 0 to h*stride-1 do
   //            for i := 0 to w*stride-1 do begin
   //                in_index := b*w*h*aChannels + k*w*h + (j div stride)*w + i div stride;
   //                out_index := b*aChannels*h*w*stride*stride + k*h*w*stride*stride + j*w*stride + i;
   //                if forward then
   //                  &out[out_index] := scale*&in[in_index]
   //                else
   //                  &in[in_index] := &in[in_index] + scale*&out[out_index]
   //            end
  for c :=0 to batch*aChannels-1 do begin
    for y :=0 to h*stride-1 do
      for x:=0 to w*stride-1 do begin
        in_index   := (c*h + (y div stride))*w + x div stride;
        out_index  := (c*h*stride + y)*stride*w + x;   // <-- why having to adjust by stride instead of remving it !!!
        if isForward then
          &out[out_index] := scale*&in[in_index]
        else
          &in[in_index] := &in[in_index] + scale*&out[out_index]; // ok seems like, it's because of trying to add adjust all input pixels in the stride a in here
      end
  end;

end;

procedure TUpSampleLayer.forward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}
  //fill_cpu(l.outputs * l.batch, 0, l.output, 1);
  if reverse then begin        // todo [forward_upsample_layer] why not using rverse as a parameter instead of [if else then]
      output.fill(0);
      upsample(output.data, outW, outH, outC, batch, stride, false, scale, state.input.data)
  end
  else
      upsample(state.input.data, w, h, c, batch, stride, true, scale, output.data);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TUpSampleLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  if reverse then  // todo [backward_upsample] why not passing l.reverse to the function instead of [if then else]
      upsample(delta.data, outW, outH, outC, batch, stride, true, scale, state.delta.data)
  else
      upsample(state.delta.data, w, h, c, batch, stride, false, scale, delta.data) ;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

{$ifdef USE_OPENCL}
procedure TUpSampleLayer.forwardGPU(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}
  output.setOCL;
  if not state.input.wasGPU() then state.input.pushToDevice;
  if reverse then begin        // todo [forward_upsample_layer] why not using rverse as a parameter instead of [if else then]
      //ocl.fill(output.as2dHeight(), output.as2dWidth(), output.devData, 0, 1);
      ocl.upsample(batch, outC, outH, outW, output.devData, stride, 0, scale, state.input.devData, true)
  end
  else
      ocl.upsample(batch, c, h, w, state.input.devData, stride, 1, scale, output.devData);
  ocl.finish();
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TUpSampleLayer.backwardGPU(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  if not delta.wasGPU() then delta.pushToDevice;
  if not state.delta.wasGPU() then state.delta.pushToDevice;
  if reverse then  // todo [backward_upsample] why not passing l.reverse to the function instead of [if then else]
      ocl.upsample(batch, outC, outH, outW, delta.devData, stride, 1, scale, state.delta.devData)
  else
      ocl.upsample(batch, c, h, w, state.delta.devData, stride, 0, scale, delta.devData) ;
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$endif}

end.

