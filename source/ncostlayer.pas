unit nCostLayer;
{$ifdef FPC}
  {$mode Delphi}
 {$H+}
{$endif}
interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;


type

  { TCostLayer }

  TCostLayer = class(TBaseLayer)
     CostType :TCostType;
     scale : single;
     constructor Create(const ABatch, AInputs: SizeInt; const ACostType: TCostType = ctSSE; const AScale: single = 1);
     procedure setBatch(ABatch: SizeInt); override;
     procedure setTrain(ATrain: boolean); override;
     procedure forward(var state: TNNetState); override;
     procedure backward(var state: TNNetState); override;
  {$ifdef USE_OPENCL}
     procedure forwardGPU(var state: TNNetState); override;
     procedure backwardGPU(var state: TNNetState); override;
  {$endif}
  private
     class procedure l1_cpu(const pred, truth:TSingleTensor; var delta, error: TSingleTensor);       static;
     class procedure l2_cpu(const pred, truth:TSingleTensor; var delta, error: TSingleTensor);       static;
     class procedure smooth_l1_cpu(const pred, truth:TSingleTensor; var delta,error: TSingleTensor); static;
  end;

implementation
uses nnet, termesc;


{ TCostLayer }

constructor TCostLayer.Create(const ABatch, AInputs: SizeInt;
  const ACostType: TCostType; const AScale: single);
begin
    layerType := ltCOST;
    scale := AScale;
    batch := ABatch;
    inputs := AInputs;
    inputShape := [batch, inputs];
    outputs := AInputs;
    CostType := ACostType;
    delta := TSingleTensor.Create([ABatch, AInputs], ABatch);
    output := TSingleTensor.Create([ABatch, AInputs], ABatch);
    cost := [0];
end;

procedure TCostLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := batch;
  delta.resize([ABatch, Inputs], ABatch);
  output.resize([ABatch, Inputs], ABatch);
end;

procedure TCostLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  FTrain := ATrain;
end;

procedure TCostLayer.forward(var state: TNNetState);
var
    i: SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}
    if not assigned(state.truth.Data) then begin
        {$ifdef USE_TELEMETRY}
        if benchmark then metrics.forward.finish(layerType);
        {$endif}
        exit();
    end ;
    if CostType = ctMASKED then
        begin  // todo [CostLayer ctMASK] SIMDIfy & gpu
            for i := 0 to state.input.Size() -1 do
                if state.truth.data[i] = SECRET_NUM then
                    state.input.data[i] := SECRET_NUM
        end;
    case CostType of
      ctSMOOTH :
        smooth_l1_cpu(state.input^, state.truth, delta, output);

      ctL1 :
        l1_cpu(state.input^, state.truth, delta, output);

    else
        l2_cpu(state.input^, state.truth, delta, output);

    end;

    //Cost[0] := sum_array(@l.output[0], l.batch * l.inputs);
    Cost[0] := output.Sum();

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}

end;

procedure TCostLayer.backward(var state: TNNetState);
begin
    //axpy_cpu(l.batch * l.inputs, l.scale, @l.delta[0], 1, @state.delta[0], 1)
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  state.delta.axpy(scale, delta);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}

end;

{$ifdef USE_OPENCL}
procedure TCostLayer.forwardGPU(var state: TNNetState);
var t:TSingleTensor; nans:TSizeIntTensor;
    i:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}
  if assigned(state.truth.Data) then begin
    output.setOCL();
    delta.setOCL;
    ocl.costL2(output.Size(), state.input.devData, state.truth.devData, delta.devData, output.devData);
    output.pullFromDevice;
    if length(output.findNaNs)>0 then begin
      cursorClearScreen;
      writeln('cost error : NaN');
      with TNNet(state.net) do
          for i:=0 to high(layers) do begin
              writeln('Layer ', layers[i].LayerTypeStr, ' [', i,'] :');
              layers[i].weights.pullFromDevice;
              nans := layers[i].weights.findNaNs;
              if nans.Size()>0 then
                  WriteLn('  weights #of NaNs : ', nans.size());
              layers[i].biases.pullFromDevice;
              nans := layers[i].biases.findNaNs;
              if nans.Size()>0 then
                  WriteLn('  biases #of NaNs : ', nans.size());
              layers[i].output.pullFromDevice;
              nans := layers[i].output.findNaNs;
              if nans.Size()>0 then
                  WriteLn('  output #of NaNs : ', nans.size());
              layers[i].delta.pullFromDevice;
              nans := layers[i].delta.findNaNs;
              if nans.Size()>0 then
                  WriteLn('  delta #of NaNs : ', nans.size());

              if layers[i].isBatchNormalized then begin
                  layers[i].mean.pullFromDevice;
                  nans := layers[i].mean.findNaNs;
                  if nans.Size()>0 then
                      WriteLn('  mean #of NaNs : ', nans.size());
                  layers[i].variance.pullFromDevice;
                  nans := layers[i].variance.findNaNs;
                  if nans.Size()>0 then
                      WriteLn('  varience #of NaNs : ', nans.size());
                  layers[i].x.pullFromDevice;
                  nans := layers[i].x.findNaNs;
                  if nans.Size()>0 then
                      WriteLn('  x #of NaNs : ', nans.size());
                  layers[i].x_norm.pullFromDevice;
                  nans := layers[i].x_norm.findNaNs;
                  if nans.Size()>0 then
                      WriteLn('  x_norm #of NaNs : ', nans.size());
                  layers[i].mean_delta.pullFromDevice;
                  nans := layers[i].mean_delta.findNaNs;
                  if nans.Size()>0 then
                      WriteLn('  mean_delta #of NaNs : ', nans.size());
                  layers[i].variance_delta.pullFromDevice;
                  nans := layers[i].variance_delta.findNaNs;
                  if nans.Size()>0 then
                      WriteLn('  varience_delta #of NaNs : ', nans.size());
              end;
          end;
      readln
    end;
    cost[0] := output.sum();
  end;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TCostLayer.backwardGPU(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  ocl.axpy(delta.size(), scale, delta.devData, 0, 1, state.Delta.devData, 0, 1);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$endif}

class procedure TCostLayer.l1_cpu(const pred, truth: TSingleTensor; var delta, error: TSingleTensor);
var i :SizeInt;
    diff :Single;
begin
  // todo [L1] simdfy & gpu
  for i := 0 to delta.Size()-1 do begin
      diff := truth.data[i] - pred.data[i];
      error.data[i] := abs(diff);
      if diff > 0 then
          delta.data[i] := 1
      else
          delta.data[i]:= -1
  end
end;

class procedure TCostLayer.l2_cpu(const pred, truth: TSingleTensor; var delta, error: TSingleTensor);
var i :SizeInt;
    diff :Single;
begin
  // todo [l2] simdfy & gpu
  truth.CopyTo(delta);
  delta.Subtract(pred.Data);
  delta.copyto(error);
  error.square();
  //for i := 0 to Truth.Size()-1 do begin
  //    diff := truth.Data[i] - pred.Data[i];
  //    error.Data[i] := diff * diff;
  //    delta.Data[i] := diff
  //end
end;

class procedure TCostLayer.smooth_l1_cpu(const pred, truth: TSingleTensor; var delta, error: TSingleTensor);
var
  i:SizeInt;
  diff, abs_val:Single;
begin
  //todo [Smooth l1] simdfy & GPU
  for i := 0 to delta.Size()-1 do begin
      diff := truth.data[i] - pred.data[i];
      abs_val := abs(diff);
      if abs_val < 1 then begin
          error.data[i] := diff * diff;
          delta.data[i] := diff;
      end
      else begin
          error.data[i] := 2*abs_val - 1;
          if diff < 0 then
              delta.data[i] := 1
          else
              delta.data[i] := -1
      end;
  end;
end;


end.

