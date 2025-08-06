unit nConvolutionLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}
{$PointerMath on}
interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer, nActivation
  {$ifdef USE_CUDNN}
  , cudnn_cnn, cudnn_ops, cudnn_graph, cudnn_adv
  {$endif}
  {$ifdef USE_OPENCL}
  , nnOpenCL
  {$endif}
  {$ifdef USE_TELEMETRY}
  , nOpMetrics
  {$endif}
  {$ifdef MSWINDOWS)} , shellApi{$endif}
  ;

type

  { TBaseConvolutionalLayer }

  TBaseConvolutionalLayer = class(TBaseImageLayer)
    n, blurStride_x, blurStride_y : SizeInt;
    stride_x, stride_y            : SizeInt;
    kernelSize, Dilation, Padding : SizeInt;

    antialiasing                  : SizeInt;
    shareLayer                    : TBaseConvolutionalLayer;
    inputLayer                    : TBaseConvolutionalLayer;
    constructor create;
    function outWidth():SizeInt;
    function outHeight():SizeInt;
    procedure fuseBatchNorm;  override;
    function getWorkspaceSize: SizeInt; override;
    function getWorkspaceShape: TArray<SizeInt>; override;
    property Stride   : SizeInt read stride_x;
    property filters  : SizeInt read n write n;
  end;

  { TConvolutionLayer }

  { TConvolutionalLayer }

  TConvolutionalLayer=class(TBaseConvolutionalLayer)
  protected
    procedure setTrain(ATrain: boolean); override;
  public

    waitStreamId                  : SizeInt;
    maxBoxes, truths              : SizeInt;
    deform, Adam                  : boolean;
    assistedExcitation            : SizeInt;
    //steps,
    ActivationInput               : TSingleTensor;

    constructor Create(const ABatch, Aheight, Awidth, Achannels,
      Afilters: SizeInt; AGroups, AKernelSize:SizeInt;
      AStride_x:SizeInt=1; AStride_y:SizeInt=1; const ADilation: SizeInt=1; APadding: SizeInt=-1;
      const AActivation: TActivationType=acSIGMOID;
      const ABatch_normalize: boolean=false; const AAdam: boolean=false;
      const AIndex: SizeInt=0; const AAntialiasing: SizeInt=0; const AShare_layer: TConvolutionalLayer=nil;
      const AAssistedExcitation: SizeInt=0; const ADeform: boolean=false;
      const ATrain: boolean=false);
    procedure setBatch( ABatch: SizeInt); override;
    procedure assistedForward(var state: TNNetState);
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
    {$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState);  override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}
  end;


implementation
uses math, nnet;

{ TBaseConvolutionalLayer }

constructor TBaseConvolutionalLayer.create;
begin
  inherited Create();
  groups := 1
end;

function TBaseConvolutionalLayer.outWidth: SizeInt;
begin
  result := (w + 2 * Padding - kernelSize) div stride_x + 1
end;

function TBaseConvolutionalLayer.outHeight: SizeInt;
begin
  result := (h + 2 * Padding - kernelSize) div stride_y + 1
end;

procedure TBaseConvolutionalLayer.fuseBatchNorm;
var
  precomputed: single;
  filter_size, f, i, w_index: SizeInt;
begin
  if shareLayer <> nil then
    shareLayer.isBatchNormalized := false;
  if isBatchNormalized then
    begin
      for f := 0 to n -1 do
          begin
              precomputed := scales.data[f] / sqrt(max(rolling_variance.data[f], sEPSILON));
              biases.data[f] := biases.data[f] - rolling_mean.data[f] * precomputed;
              filter_size := kernelSize * kernelSize * c div groups;
              for i := 0 to filter_size -1 do
                  begin
                      w_index := f * filter_size + i;
                      weights.data[w_index] := weights.data[w_index] * precomputed
                  end
          end;
      if not assigned(shareLayer) then
          freeBatchNorm;
      isBatchNormalized := false;
    end
end;

function TBaseConvolutionalLayer.getWorkspaceSize: SizeInt;
begin
  result := batch * (c div groups) * outH * outW * kernelSize * kernelSize
end;

function TBaseConvolutionalLayer.getWorkspaceShape: TArray<SizeInt>;
begin
  result := [batch, (c div groups), outH , outW , kernelSize * kernelSize]
end;

{ TConvolutionalLayer }

constructor TConvolutionalLayer.Create(const ABatch, Aheight, Awidth, Achannels,
  Afilters: SizeInt; AGroups, AKernelSize: SizeInt; AStride_x: SizeInt;
  AStride_y: SizeInt; const ADilation: SizeInt; APadding: SizeInt;
  const AActivation: TActivationType; const ABatch_normalize: boolean;
  const AAdam: boolean; const AIndex: SizeInt; const AAntialiasing: SizeInt;
  const AShare_layer: TConvolutionalLayer; const AAssistedExcitation: SizeInt;
  const ADeform: boolean; const ATrain: boolean);

var
  blur_nweights, i, blur_size, blur_pad: SizeInt;

  _scale : single;

begin
  inherited Create;
  layerType := ltCONVOLUTIONAL;
  FTrain := ATrain;
  if AGroups < 1 then
      groups := 1
  else
    groups := AGroups;
  blurStride_x := aStride_x;
  blurStride_y := aStride_y;
  antialiasing := AAntialiasing;
  if antialiasing>0 then begin
      AStride_x        := 1;
      AStride_y        := 1;
      stride_x         := 1;
      stride_y         := 1
  end;
  waitStreamId := -1;
  deform := aDeform;
  assistedExcitation := AAssistedExcitation;
  shareLayer := Ashare_layer;
  index := Aindex;
  h := Aheight;
  w := Awidth;
  c := Achannels;
  n := Afilters; // same as filters := AFilters;

  //result.use_bin_output := use_bin_output;
  batch := ABatch;

  //steps := ASteps;
  stride_x := AStride_x;
  stride_y := AStride_y;
  dilation := ADilation;
  kernelSize := AKernelSize;
  if APadding<0 then
    Padding := ADilation -1 + kernelSize div 2
  else
    Padding := APadding;
  isBatchNormalized := ABatch_normalize;
  learningRateScale := 1;
  //nweights := (c div groups) * filters * kernelSize * kernelSize;
  if assigned(shareLayer) then
      begin
          if (kernelSize <> shareLayer.kernelSize) or (weights.Size() <> shareLayer.weights.Size()) or (c <> shareLayer.c) or (n <> shareLayer.n) then
              raise Exception.create('Layer KernelSize, nweights, channels or filters don''t match for the shareLayer');
          weights := shareLayer.weights;
          weight_updates := shareLayer.weight_updates;
          biases := shareLayer.biases;
          bias_updates := shareLayer.bias_updates
      end
  else
      begin
          weights := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          biases := TSingleTensor.Create([filters]);
          if train then
              begin
                  weight_updates := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
                  bias_updates := TSingleTensor.Create([filters]);
                  weights_ema := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
                  biases_ema := TSingleTensor.Create([filters])
              end
      end;
  _scale := sqrt(2 / (kernelSize * kernelSize * c / groups));
  if ActivationType in [acNORM_CHAN, acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL] then
      weights.fill(1)
  else
      weights.UniformDistribution( - _scale, _scale);
  outW := outWidth();
  outH := outHeight();
  outC := n;
  outputs := filters * outH * outW;
  inputs := w * h * c;
  inputShape := [batch, c, h, w];
  ActivationType := AActivation;
  output := TSingleTensor.Create([batch , filters, outH, outW], batch);

  if FTrain then
      delta := TSingleTensor.Create([batch , filters, outH, outW], batch);

  if isBatchNormalized then
      begin
          if assigned(shareLayer) then
              begin
                  scales := shareLayer.scales;
                  scale_updates := shareLayer.scale_updates;
                  mean := shareLayer.mean;
                  variance := shareLayer.variance;
                  mean_delta := shareLayer.mean_delta;
                  variance_delta := shareLayer.variance_delta;
                  rolling_mean := shareLayer.rolling_mean;
                  rolling_variance := shareLayer.rolling_variance
              end
          else
              begin
                  scales := TSingleTensor.Create([filters]);
                  scales.fill(1.0);
                  if train then
                      begin
                          scales_ema := TSingleTensor.Create([filters]);
                          scale_updates := TSingleTensor.Create([filters]);
                          mean := TSingleTensor.Create([filters]);
                          variance := TSingleTensor.Create([filters]);
                          mean_delta := TSingleTensor.Create([filters]);
                          variance_delta := TSingleTensor.Create([filters])
                      end;
                  rolling_mean := TSingleTensor.Create([filters]);
                  rolling_variance := TSingleTensor.Create([filters])
              end;
          if train then
              begin
                  x := TSingleTensor.Create([batch, filters, outH, outW], batch);
                  x_norm := TSingleTensor.Create([batch, filters, outH, outW], batch)
              end
      end;
  if ActivationType in [acSWISH, acMISH, acHARD_MISH] then
      ActivationInput := TSingleTensor.Create([batch, filters, outH, outW], batch);
  if adam then
      begin
          m       := TSingleTensor.Create([(c div groups), filters, kernelSize, kernelSize]);
          v       := TSingleTensor.Create([(c div groups), filters, kernelSize, kernelSize]);
          bias_m  := TSingleTensor.Create([filters]);
          scale_m := TSingleTensor.Create([filters]);
          bias_v  := TSingleTensor.Create([filters]);
          scale_v := TSingleTensor.Create([filters])
      end;

  if speedOverSize then
    workspace.resize(getWorkspaceShape, batch);
  //workspaceSize := getWorkspaceSize();
  if antialiasing>0 then
      begin
          blur_size := 3;
          blur_pad := blur_size div 2;
          if antialiasing = 2 then
              begin
                  blur_size := 2;
                  blur_pad := 0
              end;
          InputLayer := TConvolutionalLayer.Create(batch, {steps,} outH, outW, n, n, n, blur_size, blurstride_x, blurstride_y, 1, blur_pad, acLINEAR, false, false, index, 0, nil, 0, false, train);
          blur_nweights := n * blur_size * blur_size;
          if blur_size = 2 then begin
              i := 0;
              while i < blur_nweights do begin
                  InputLayer.weights.data[i+0] := 1 / 4;
                  InputLayer.weights.data[i+1] := 1 / 4;
                  InputLayer.weights.data[i+2] := 1 / 4;
                  InputLayer.weights.data[i+3] := 1 / 4;
                  i := i + (blur_size * blur_size)
              end
          end else begin
              i := 0;
              while i < blur_nweights do begin
                  InputLayer.weights.data[i+0] := 1 / 16;
                  InputLayer.weights.data[i+1] := 2 / 16;
                  InputLayer.weights.data[i+2] := 1 / 16;
                  InputLayer.weights.data[i+3] := 2 / 16;
                  InputLayer.weights.data[i+4] := 4 / 16;
                  InputLayer.weights.data[i+5] := 2 / 16;
                  InputLayer.weights.data[i+6] := 1 / 16;
                  InputLayer.weights.data[i+7] := 2 / 16;
                  InputLayer.weights.data[i+8] := 1 / 16;
                  i := i + (blur_size * blur_size)
              end;
          end;
          inputLayer.biases.fill(0);
      end

end;

procedure TConvolutionalLayer.setBatch(ABatch: SizeInt);
//var total_batch : SizeInt;
begin
  if ABatch=Batch then exit();
  Batch := ABatch;

  inputShape[0] := batch;

  output.reSize([batch , filters, outH, outW], batch);
  if train then
      delta.reSize([batch , filters, outH, outW], batch);
  if ActivationType in [acSWISH, acMISH, acHARD_MISH] then
      ActivationInput.resize([batch, filters, outH, outW], batch);
  if train then
      begin
          x.reSize([batch, filters, outH, outW], batch);
          x_norm.reSize([batch, filters, outH, outW], batch);
      end;
  if speedOverSize then
    workspace.resize(getWorkspaceShape, batch);

  if antialiasing<>0 then
    inputLayer.setBatch(batch);
end;


procedure TConvolutionalLayer.assistedForward(var state: TNNetState);
var
    iteration_num, b, _w, _h, _c, t, left, right, top, bottom: SizeInt;
    alpha: single;
    a_avg, g: TArray<single>;
    Ps :PSingle;
    truth: TBox;
    net : TNNet;
begin
    net := TNNet(state.net);
    iteration_num := net.seen div (net.batch * net.subdivisions);
    alpha := (1+cos(nTensors.PI * iteration_num / net.maxBatches));
    if assistedExcitation > 1 then
        begin
            if iteration_num > assistedExcitation then
                alpha := 0
            else
                alpha := (1+cos(nTensors.PI * iteration_num / assistedExcitation))
        end;
    setLength(a_avg, outw * outh * batch);
    setLength(g, outW * outH * batch);
    maxboxes := net.maxBoxes;
    truths := maxBoxes * (4+1);
    for b := 0 to batch -1 do
        begin
            for t := 0 to net.maxBoxes -1 do
                begin
                    truth := TBox.fromFloat(Pointer(state.truth.data + t * (4+1)+b * truths), 1);
                    if truth.x=0 then
                        break;
                    left := floor((truth.x - truth.w / 2) * outW);
                    right := ceil((truth.x + truth.w / 2) * outW);
                    top := floor((truth.y - truth.h / 2) * outH);
                    bottom := ceil((truth.y + truth.h / 2) * outH);
                    for _w := left to right do
                        for _h := top to bottom -1 do
                            g[_w + outW * _h + outW * outH * b] := 1
                end
        end;
    for b := 0 to batch -1 do
        for _w := 0 to outW -1 do
            for _h := 0 to outH -1 do
                begin
                    Ps := @a_avg[w + outW * (h + outH * b)];
                    for _c := 0 to outC -1 do
                        Ps^ := Ps^ + output.data[_w + outW * (_h + outH * (_c + outC * b))];
                    Ps^ := Ps^ / outC
                end;
    for b := 0 to batch -1 do
        for _w := 0 to outW -1 do
            for _h := 0 to outH -1 do
                for _c := 0 to outC -1 do begin
                    Ps := @output.Data[_w + outW * (_h + outH * (_c + outC * b))];
                    Ps^ := Ps^ + alpha * g[_w + outW * (_h + outH * b)] * a_avg[_w + outW * (_h + outH * b)];
                end
end;

procedure TConvolutionalLayer.setTrain(ATrain: boolean);
//var
    //total_batch:SizeInt;
begin
  if ATrain = FTrain  then exit;
  FTrain := ATrain;
  if ATrain then begin
      delta := TSingleTensor.Create([batch , filters, outH, outW], batch);
      if not assigned(shareLayer) then begin
          weight_updates := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          bias_updates := TSingleTensor.Create([filters]);
          weights_ema := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          biases_ema := TSingleTensor.Create([filters]);
          if isBatchNormalized then begin
              scales_ema := TSingleTensor.Create([filters]);
              scale_updates := TSingleTensor.Create([filters]);
              mean := TSingleTensor.Create([filters]);
              variance := TSingleTensor.Create([filters]);
              mean_delta := TSingleTensor.Create([filters]);
              variance_delta := TSingleTensor.Create([filters])
          end;
      end;
      if isBatchNormalized then begin
          x := TSingleTensor.Create([batch, filters, outH, outW], batch);
          x_norm := TSingleTensor.Create([batch, filters, outH, outW], batch)
      end;

  end else begin
      //delta.free;
      if not assigned(shareLayer) then begin
          weight_updates.free;
          bias_updates.free;
          weights_ema.free;
          biases_ema.free;
          if isBatchNormalized then begin
              scales_ema.free;
              scale_updates.free;
              mean.free;
              variance.free;
              mean_delta.free;
              variance_delta.free
          end;
      end;
      if isBatchNormalized then begin
          x.free;
          x_norm.free
      end;
  end;

end;

procedure TConvolutionalLayer.forward(var state: TNNetState);
var
    s: TNNetState;
    workspacePtr: PSingleTensor;
    //q:string;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    ////output.fill(0);


    //outH := convolutional_out_height(l);
    //outW := convolutional_out_width(l);
    // todo TConvolutionalLayer : implement forward and backward groups and groupId for multi GPU
    //k := kernelSize * kernelSize * c div groups;
    //outImgSize := outH * outW;
    //imcolSize  := outImgSize * kernelSize * kernelSize * c ;
    //m := filters div groups;
    //nweights := weights.size();
    //u := 0;
    //inc(u);
    //for i := 0 to batch -1 do
    //    for j := 0 to groups -1 do
    //    begin
    //        _A := weights.Data;
    //        im := state.input.data +(i * groups + j)*(c div groups) * h * w;
    //        _B := pointer(state.workspace.data + (i * groups + j)*(c div groups)*imcolSize);

            //im := state.input.data + i * c * h * w + j * c * h * w div groups;
            //_B := pointer(state.workspace.data + i*c*imcolSize + j*c*imcolSize div groups);

    //        _C := output.data + i * outImgSize * m;
    //        if (kernelSize = 1) and (stride = 1) and (dilation = 1) then
    //            _B := im
    //        else
    //            im2col_cpu_ext(pointer(im), c div groups, h, w, kernelSize, kernelSize, Padding * dilation, Padding * dilation, stride_y, stride_x, dilation, dilation, pointer(_B));
    //        TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, outImgSize, k, 1, _A, k, _B, outImgSize, 1, _C, outImgSize);
    //    end;

    // todo [TConvolutionLayer] implement multi GPU groups

    //if not state.input.wasGPU() then
    //  state.input.pushToDevice;
    //if train or not weights.wasGPU() then
    //    weights.pushToDevice;
    if speedOverSize then
      workspacePtr := @workspace
    else
      workspacePtr := @state.workspace;
    state.input.Conv2D(weights, output, Padding, Padding, stride_x, stride_y, Dilation, Dilation, workspacePtr^.Data);
//output.print(psColor8,8);
//readln(q);
//if q='q' then halt(0);
    //output.SaveToImage( 'tmp'+intToStr(index)+'.bmp');
    //shellApi.ShellExecute(0, 'open', 'tmp'+intToStr(index)+'.bmp', '', '', 0);
    //readln;
    //state.input.conv2d(weights, output, Padding, padding, stride_x, stride_y, Dilation, Dilation) ;


    //if not biases.wasGPU then
    //    biases.pushToDevice;

    if isBatchNormalized then
      batchNorm(state)
    else
      output.forwardBias(biases);

    //output.pullFromDevice;
    //if output.wasGPU then output.pullFromDevice;
    //if biases.wasGPU then biases.pullFromDevice;

    case ActivationType of
       acSWISH :
          activate_array_swish(output, outputs * batch, activationInput, output);
       acMISH :
          activate_array_mish(output, outputs * batch, activationInput, output);
       acHARD_MISH :
          activate_array_hard_mish(output, outputs * batch, ActivationInput, output);
       acNORM_CHAN :
          activate_array_normalize_channels(output, outputs * batch, batch, outC, outW * outH, output);
       acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
          activate_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outH, output, activationType = acNORM_CHAN_SOFTMAX_MAXVAL);
       else begin
           //output.print(true, 3);
          activate()
       end;
    end;


    if (assistedExcitation<>0) and state.isTraining then
        assistedForward(state);
    if antialiasing<>0 then
        begin
            s := default(TNNetState);
            s.isTraining := state.isTraining;
            s.workspace := state.workspace;
            //s.net := state.net;
            s.input := @output;
            //forward_convolutional_layer( l.input_layer[0], @s);
            inputLayer.forward(s);
            //move(l.input_layer[0].output[0], l.output[0], l.input_layer[0].outputs * l.input_layer[0].batch * sizeof(single))
            inputLayer.output.copyTo(output);
        end;
    //state.input.printStat;
    //output.printStat;
    //readln;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}

end;

procedure TConvolutionalLayer.backward(var state: TNNetState);
var
    //nweights,
    b, i_m, i_n, i_k, colSize: SizeInt;
    //_A, _B, _C, im: Pointer;
    workspacePtr :PSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  i_m := filters div groups;
  i_n := kernelSize * kernelSize * c div groups;
  i_k := outH * outW;

  colSize := getWorkspaceSize div batch;

  case activationType of
    acSWISH :
      gradient_array_swish(output, outputs * batch, activationInput, delta) ;
    acMISH  :
      gradient_array_mish(outputs * batch, activationInput, delta) ;
    acHARD_MISH :
      gradient_array_hard_mish(outputs * batch, activationInput, delta) ;
    acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
      gradient_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outW, delta) ;
    acNORM_CHAN :
      gradient_array_normalize_channels(output, outputs * batch, batch, outC, outW * outW, delta) ;
    else
      Derivative();
  end;
  if isBatchNormalized then
      batchNormBack(state)
  else
      bias_updates.addSums(delta);

  //nweights := weights.size();
  //for i := 0 to batch -1 do
  //    for j := 0 to groups -1 do
  //        begin
  //            _A := delta.Data+(i * groups+j) * i_m * i_k;
  //            _B := pointer(state.workspace.data);
  //            _C := weight_updates.data  + j * nweights div groups;
  //            im := state.input.data +(i * groups+j) * (c div groups) * h * w;
  //            im2col_cpu_ext(im, c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, _B);
  //            TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_m, i_n, i_k, 1, _A, i_k, _B, i_k, 1, _C, i_n);
  //            if assigned(state.delta.Data) then
  //                begin
  //                    _A := weights.Data+j * nweights div groups;
  //                    _B := delta.Data + (i * groups+j) * i_m * i_k;
  //                    _C := pointer(state.workspace.data);
  //                    TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, i_n, i_k, i_m, 1, _A, i_n, _B, i_k, 0, _C, i_k);
  //                    col2im_cpu_ext(pointer(state.workspace.data), c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, pointer(state.delta.data +(i * groups+j) * (c div groups) * h * w))
  //                end
  //        end ;

  //for i:= 0 to batch -1 do begin
  //  _A := delta.Data+i * i_m * i_k;
  //  _B := pointer(state.workspace.data + i*imcolSize);
  //  _C := weight_updates.data ;
  //  im := state.input.data + c * h * w;
  //  im2col_cpu_ext(im, c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, _B);
  //  TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_m, i_n, i_k, 1, _A, i_k, _B, i_k, 1, _C, i_n);
  //end;

  if speedOverSize then
    workspacePtr := @workspace
  else
  begin
    workspacePtr := @state.workspace;
    state.input.im2Col(kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, workspacePtr^, 1);
  end;
  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.start(opGemm);
  {$endif}
  for b:= 0 to batch -1 do
    TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_m, i_n, i_k, 1
      , delta.data + b*i_m*i_k, i_k
      , workspacePtr^.data + b*ColSize, i_k
      , 1, weight_updates.data, i_n);
  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.finish(opGemm);
  {$endif}
  if assigned(state.delta) and assigned(state.delta.data) then begin
    {$ifdef USE_TELEMETRY}
    if benchmark then tensorMetrics.start(opGemm);
    {$endif}
    TSingleTensor.gemmStridedBatched(
      CblasRowMajor, CblasTrans, CblasNoTrans, i_n, i_k, i_m, 1
      , weights.Data, i_n, 0
      , delta.Data, i_k, i_m * i_k
      , 0, workspacePtr^.data, i_k, colSize, Batch);
    {$ifdef USE_TELEMETRY}
      if benchmark then tensorMetrics.finish(opGemm);
    {$endif}
    state.delta.col2Im(kernelSize, kernelSize, padding*Dilation, padding*Dilation, stride_x, stride_y, dilation, dilation, workspacePtr^);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TConvolutionalLayer.update(const args: TUpdateArgs);
var
    learning_rate: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

  learning_rate := args.learningRate * learningRateScale;
  //axpy_cpu(l.nweights, -args.decay * args.batch, l.weights, 1, l.weight_updates, 1);
  weight_updates.axpy(-args.decay * args.batch, weights);

  //axpy_cpu(l.nweights, learning_rate / args.batch, l.weight_updates, 1, l.weights, 1);
  weights.axpy(learning_rate / args.batch, weight_updates);

  //scal_cpu(l.nweights, args.momentum, l.weight_updates, 1);
  weight_updates.Multiply(args.momentum);

  //axpy_cpu(l.n, learning_rate / args.batch, l.bias_updates, 1, l.biases, 1);
  biases.axpy(learning_rate / args.batch, bias_updates);

  //scal_cpu(l.n, args.momentum, l.bias_updates, 1);
  bias_updates.multiply(args.momentum);

  if assigned(scales.Data) then
      begin
          //axpy_cpu(l.n, learning_rate / args.batch, l.scale_updates, 1, l.scales, 1);
          scales.axpy(learning_rate / args.batch, scale_updates);

          //scal_cpu(l.n, args.momentum, l.scale_updates, 1)
          scale_updates.multiply(args.momentum);
      end;
  inherited update(args);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

{$if defined(USE_OPENCL)}
procedure TConvolutionalLayer.forwardGPU(var state: TNNetState);
var
    b, aOffset, bOffset , outImgSize, kSize, k, imColSize,
      strideA, strideB, strideC :SizeInt;
    _A, _B, _C : TCLMemory;
    s : TNNetState;
    workspacePtr: PSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  if not state.input.wasGPU() then state.input.pushToDevice;
  if not weights.wasGPU() then weights.pushToDevice;
  if not biases.wasGPU() then biases.pushToDevice;
  output.setOCL;

  kSize := kernelSize*kernelSize;
  outImgSize := output.area();
  //filters := weights.c();
  k := c * kSize;
  imColSize := c * kSize * outImgSize;
  _C := output.devData;
  strideC := outImgSize * filters;
  if (kSize <> 1) or (Stride_y * Stride_x <> 1) or (Dilation * Dilation <> 1) then begin
    strideA := state.input.volume();
    strideB := imColSize;
    if speedOverSize then
      workspacePtr := @workspace
    else
      workspacePtr := @state.workspace;
    _B := workspacePtr.devData;
    for b := 0 to batch - 1 do
      begin
        aOffset := b * strideA;
        bOffset := b * strideB;
        ocl.im2col(c, h, w, kernelSize, kernelSize, Padding, Padding, stride_y, stride_x, Dilation, Dilation, state.input.devData, aOffset, _B, bOffset);
        //ocl.gemm(false, false, filters, outImgSize, k, 1, weights.devData, 0, k, state.workspace.devData, bOffset, outImgSize, 0, _C, b * strideC, outImgSize);
        //batchesWeights[b] := weights.devData;
        //workSpaces[b] := state.workspace.devData + bOffset;
        //batchesOut[b] := _C + b * outImgSize * filters;
      end
  end else begin
    strideB := state.input.volume();
    _B := state.input.devData;

  end;
  //for b := 0 to batch - 1 do
  //  begin
  //    bOffset := b * strideB;
  //    ocl.gemm(false, false, filters, outImgSize, k, 1, weights.devData, 0, k, _B, bOffset, outImgSize, 0, _C, b * strideC, outImgSize);
  //    //batchesWeights[b] := weights.devData;
  //    //workSpaces[b] := _B + bOffset;
  //    //batchesOut[b] := _C + b * outImgSize * filters;
  //  end;

  //ocl.WriteBuffer(workSpacesDev,     batch*sizeOf(pointer), Pointer(workSpaces));
  //ocl.WriteBuffer(batchesOutDev,     batch*sizeOf(pointer), Pointer(batchesOut));
  //ocl.WriteBuffer(batchesWeightsDev, batch*sizeOf(pointer), Pointer(batchesWeights));
  //ocl.gemmBatched(false, false, filters, outImgSize, k, 1, ppsingle(batchesWeightsDev), 0, k, ppsingle(workspacesDev), 0, outImgSize, 0, ppsingle(batchesOutDev), 0, outImgSize, batch);
  ocl.gemmStridedBatched(false, false, filters, outImgSize, k, 1, weights.devData, 0,k, 0, _B, 0, outImgSize, strideB, 0, _C, 0, outImgSize, strideC, batch
  {$ifdef GPU_TEST}, output.devTest{$endif}
  );

//state.input.Conv2D(weights, output, Padding, Padding, stride_x, stride_y, Dilation, Dilation);
//output.printGpuSumSqrDiff();

  if isBatchNormalized then
    batchNormGPU(state)
  else begin
    ocl.forwardBias(output.Size(), output.devData, 0, biases.size(), biases.devData,1, Batch);
//output.forwardBias(biases);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.act.start(ActivationType);
  {$endif}
  case ActivationType of
     acSWISH :
       if train then
         ocl.activateArraySWISH(output.size(), output.devData, 0, ActivationInput.devData, output.devData)
       else
         ocl.ActivateArray(output.size(), output.devData, 0, longint(ActivationType));
     //acMISH :
     //   activate_array_mish(output, outputs * batch, activationInput, output);
     //acHARD_MISH :
     //     activate_array_hard_mish(output, outputs * batch, ActivationInput, output);
     //acNORM_CHAN :
     //     activate_array_normalize_channels(output, outputs * batch, batch, outC, outW * outH, output);
     //acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
     //     activate_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outH, output, activationType = acNORM_CHAN_SOFTMAX_MAXVAL);
     else
       ocl.ActivateArray(output.size(), output.devData, 0, longint(ActivationType));
  end;
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.act.finish(ActivationType);
  {$endif}
//Activate();

  if (assistedExcitation<>0) and state.isTraining then
      assistedForward(state);
  if antialiasing<>0 then begin
      s := default(TNNetState);
      s.isTraining := state.isTraining;
      s.workspace := state.workspace;
      s.input := @output;
      inputLayer.forwardGPU(s);
      ocl.copy(output.Size(), inputLayer.output.devData, 0, 1, output.devData, 0, 1);
  end;

//write(LayerTypeStr,' ');
//output.printGpuSumSqrDiff();
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConvolutionalLayer.backwardGPU(var state: TNNetState);
var
    b, i_m, i_n, i_k, colSize, _vol, imSize: SizeInt;
    workspacePtr: PSingleTensor;
    diff: Single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  i_m := filters div groups;
  i_n := kernelSize * kernelSize * c div groups;
  i_k := outH * outW;

  colSize := c * outH * outW * kernelSize * kernelSize;
  //colSize := getWorkspaceSize div batch;
  _vol := state.input.Volume();
  if not delta.wasGPU() then delta.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not state.input.wasGPU() then state.input.pushToDevice;

  case activationType of
    acSWISH : ;
      //gradient_array_swish(output, outputs * batch, activationInput, delta) ;
    //acMISH  :
    //  gradient_array_mish(outputs * batch, activationInput, delta) ;
    //acHARD_MISH :
    //  gradient_array_hard_mish(outputs * batch, activationInput, delta) ;
    //acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
    //  gradient_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outW, delta) ;
    //acNORM_CHAN :
    //  gradient_array_normalize_channels(output, outputs * batch, batch, outC, outW * outW, delta) ;
    else
      ocl.DeriveArray(output.size(), output.devData, 0, longint(ActivationType), delta.devData);
  end;
//Derivative();
//delta.printGpuSumSqrDiff();
  if isBatchNormalized then
    batchNormBackGPU(state)
  else begin
    ocl.backwardBias(bias_updates.size(), bias_updates.devData, delta.size, delta.devData, 0, 1, batch);
//bias_updates.addSums(delta);
  end;

  if speedOverSize then
    workspacePtr := @workspace
  else
  begin
    workspacePtr := @state.workspace;
    for b:=0 to batch-1 do begin
      ocl.im2col(c, h, w, kernelSize, kernelSize, Padding, Padding,
        stride_y, stride_x, dilation, dilation, state.input.devData , b*_vol, workspacePtr.devData, b*colSize);
    end;
//state.input.im2Col(kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, state.workspace, 1);
  end;
  workspacePtr.setOCL;

  if not weight_updates.wasGPU() then weight_updates.pushToDevice;

  //for b:= 0 to batch -1 do begin
    //ocl.gemm(false, true,
    //        i_m, i_n, i_k, 1
    //        , delta.devData , b*i_m*i_k, i_k
    //        , state.workspace.devData, b*ColSize, i_k
    //        , 1, weight_updates.devData, 0, i_n);
  //end;

//ocl.copy(weight_updates.size(), weight_updates.devData, 0, 1, weight_updates.devTest, 0, 1);
//diff := weight_updates.gpuSumSqrDiffTest();
  ocl.gemmStridedBatched(false, true,
          i_m, i_n, i_k, 1
          , delta.devdata, 0, i_k, i_m*i_k
          , workspacePtr.devData, 0, i_k, colSize
          , 1, weight_updates.devData, 0, i_n, 0, batch
          {$ifdef GPU_TEST}, weight_updates.devTest{$endif}
          );

//diff := weight_updates.gpuSumSqrDiffTest();
//if diff>sEpsilon then begin
//  write('M : ',i_m, ', N : ',i_n, ', K : ', i_k, ', lda : ', i_k, ', ldb : ', i_k, ', ldc : ', i_n, ', diff :', diff:0:sDigits,#13);
//  readln
//end;

//for b:= 0 to batch -1 do
//TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_m, i_n, i_k, 1
//, delta.data + b*i_m*i_k, i_k
//, state.workspace.data + b*ColSize, i_k
//, 1, weight_updates.data, i_n);

  if assigned(state.delta) and assigned(state.delta.devdata) then begin
    if not weights.wasGPU() then weights.pushToDevice;
    //if not state.delta.wasGPU then state.delta.pushToDevice;

    //for b := 0 to batch -1 do begin
    //  ocl.gemm(true, false,
    //      i_n, i_k, i_m, 1,
    //      weights.devData, 0    , i_n,
    //      delta.devData  , b*i_m*i_k, i_k,
    //      0, state.workspace.devData, b*colSize, i_k);
    //end;

    ocl.gemmStridedBatched(true, false,
        i_n, i_k, i_m, 1
        , weights.devData, 0, i_n, 0
        , delta.devData, 0, i_k, i_m*i_k
        , 0, workspacePtr.devData, 0, i_k, colSize, batch
        {$ifdef GPU_TEST}, workspacePtr.devTest{$endif}
        );


//for b := 0 to batch -1 do begin
//    TSingleTensor.gemm(
//      CblasRowMajor, CblasTrans, CblasNoTrans, i_n, i_k, i_m, 1
//      , weights.Data, i_n
//      , delta.Data + b * i_m * i_k, i_k
//      , 0, state.workspace.data + b*colSize, i_k);
//end;


    imSize := state.delta.Volume();
    for b := 0 to batch-1 do begin
      ocl.col2im(state.delta.c, state.delta.h, state.delta.w, kernelSize, kernelSize, Padding, Padding
              , stride_y, stride_x, dilation, dilation, workspacePtr.devData, b*colSize, state.delta.devData, b*imSize);
    end;
//state.delta.col2Im(kernelSize, kernelSize, padding*Dilation, padding*Dilation, stride_x, stride_y, dilation, dilation, state.workspace);

  end ;

//bias_updates.printGpuSumSqrDiff();
//weight_updates.printGpuSumSqrDiff();
//state.delta.printGpuSumSqrDiff();
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TConvolutionalLayer.updateGPU(const args: TUpdateArgs);
var
    learning_rate: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

  if not biases.wasGPU() then biases.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not weights.wasGPU() then weights.pushToDevice;
  if not weight_updates.wasGPU() then weight_updates.pushToDevice;

  learning_rate := args.learningRate * learningRateScale;

  ocl.axpy(biases.size(), learning_rate / args.batch, bias_updates.devData, 0, 1, biases.devData, 0, 1);
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.scale(bias_updates.size(), args.momentum, bias_updates.devData, 1);
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.axpy(weight_updates.size(), -args.decay * args.batch, weights.devData, 0, 1, weight_updates.devData, 0, 1);
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.axpy(weights.size(), learning_Rate / args.batch, weight_updates.devData, 0, 1, weights.devData, 0, 1);
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  ocl.scale(weight_updates.size(), args.momentum, weight_updates.devData, 1);
  //ocl.waitForEvents(batch, pointer(events));
  //ocl.finish();

  if assigned(scales.Data) then begin
      //scales.axpy(learning_rate / args.batch, scale_updates);
      //scale_updates.multiply(args.momentum);
    ocl.axpy(scales.size(), learning_rate / args.batch, scale_updates.devData, 0, 1, scales.devData, 0, 1);
    ocl.scale(scale_updates.size(), args.momentum, scale_updates.devData, 1);

  end;

  //update(args);
  //ocl.waitForEvents(batch, pointer(events));

  inherited;

  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;
{$elseif defined(USE_CUDART)}
procedure TConvolutionalLayer.forwardGPU(var state: TNNetState);
var
    b, aOffset, bOffset , outImgSize, kSize, k, imColSize,
      strideA, strideB, strideC :SizeInt;
    _A, _B, _C : PSingle;
    s : TNNetState;
    workspacePtr : PSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  if not state.input.wasGPU() then state.input.pushToDevice;
  if not weights.wasGPU() then weights.pushToDevice;
  if not biases.wasGPU() then biases.pushToDevice;
  output.setCUDA;

  kSize := kernelSize*kernelSize;
  outImgSize := output.area();
  //filters := weights.c();
  k := c * kSize;
  imColSize := c * kSize * outImgSize;
  _C := output.devData;
  strideC := outImgSize * filters;

  if (kSize <> 1) or (Stride_y * Stride_x <> 1) or (Dilation * Dilation <> 1) then begin
    strideA := state.input.volume();
    strideB := imColSize;
    if speedOverSize then
      workspacePtr := @workspace
    else
      workspacePtr := @state.workspace;
    _B := workspacePtr.devData;
    for b := 0 to batch - 1 do
      begin
        aOffset := b * strideA;
        bOffset := b * strideB;
        cuda.im2col(c, h, w, kernelSize, kernelSize, Padding, Padding, stride_y, stride_x, Dilation, Dilation, state.input.devData, aOffset, pointer(_B), bOffset);
        //cuda.gemm(false, false, filters, outImgSize, k, 1, weights.devData, 0, k, state.workspace.devData, bOffset, outImgSize, 0, _C, b * strideC, outImgSize);
        //batchesWeights[b] := weights.devData;
        //workSpaces[b] := state.workspace.devData + bOffset;
        //batchesOut[b] := _C + b * outImgSize * filters;
      end
  end else begin
    strideB := state.input.volume();
    _B := state.input.devData;

  end;

  //for b := 0 to batch - 1 do
  //  begin
  //    bOffset := b * strideB;
  //    cuda.gemm(false, false, filters, outImgSize, k, 1, weights.devData, 0, k, _B, bOffset, outImgSize, 0, _C, b * strideC, outImgSize);
  //    //batchesWeights[b] := weights.devData;
  //    //workSpaces[b] := _B + bOffset;
  //    //batchesOut[b] := _C + b * outImgSize * filters;
  //  end;

//output.im2Col(kernelSize, kernelSize, Padding, Padding, stride_x, stride_y, Dilation, Dilation, state.workspace);
//state.workspace.printGpuSumSqrDiff();

  //cuda.WriteBuffer(workSpacesDev,     batch*sizeOf(pointer), Pointer(workSpaces));
  //cuda.WriteBuffer(batchesOutDev,     batch*sizeOf(pointer), Pointer(batchesOut));
  //cuda.WriteBuffer(batchesWeightsDev, batch*sizeOf(pointer), Pointer(batchesWeights));
  //cuda.gemmBatched(false, false, filters, outImgSize, k, 1, ppsingle(batchesWeightsDev), 0, k, ppsingle(workspacesDev), 0, outImgSize, 0, ppsingle(batchesOutDev), 0, outImgSize, batch);
  cuda.gemmStridedBatched(false, false, filters, outImgSize, k, 1, weights.devData, 0,k, 0, pointer(_B), 0, outImgSize, strideB, 0, pointer(_C), 0, outImgSize, strideC, batch);
//TSingleTensor.gemmStridedBatched(CblasRowMajor, CblasNoTrans, CblasNoTrans, filters, outImgSize, k, 1, weights.Data, k, 0, state.workspace.Data, outImgSize, strideB, 0, output.Data, outImgSize, strideC, batch);

{$ifdef DEBUG_GPU}
state.input.Conv2D(weights, output, Padding, Padding, stride_x, stride_y, Dilation, Dilation);
output.printGpuSumSqrDiff();
{$endif}

  if isBatchNormalized then
    batchNormGPU(state)
  else begin
    cuda.forwardBias(output.Size(), output.devData, 0, biases.size(), biases.devData,1, Batch);
{$ifdef DEBUG_GPU}
output.forwardBias(biases);
{$endif}
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.act.start(ActivationType);
  {$endif}
  case ActivationType of
     acSWISH :
       if train then
         cuda.activateArraySWISH(output.size(), output.devData, 0, ActivationInput.devData, output.devData)
       else
         cuda.ActivateArray(output.size(), output.devData, 0, longint(ActivationType));
     //acMISH :
     //   activate_array_mish(output, outputs * batch, activationInput, output);
     //acHARD_MISH :
     //     activate_array_hard_mish(output, outputs * batch, ActivationInput, output);
     //acNORM_CHAN :
     //     activate_array_normalize_channels(output, outputs * batch, batch, outC, outW * outH, output);
     //acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
     //     activate_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outH, output, activationType = acNORM_CHAN_SOFTMAX_MAXVAL);
     else
       cuda.ActivateArray(output.size(), output.devData, 0, longint(ActivationType));
  end;
  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.act.finish(ActivationType);
  {$endif}

{$ifdef DEBUG_GPU}
Activate();
{$endif}

  if (assistedExcitation<>0) and state.isTraining then
      assistedForward(state);
  if antialiasing<>0 then begin
      s := default(TNNetState);
      s.isTraining := state.isTraining;
      s.workspace := state.workspace;
      s.input := @output;
      inputLayer.forwardGPU(s);
      cuda.copy(output.Size(), inputLayer.output.devData, 0, 1, output.devData, 0, 1);
  end;

{$ifdef DEBUG_GPU}
write(LayerTypeStr,' ');
output.printGpuSumSqrDiff();
{$endif}

  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConvolutionalLayer.backwardGPU(var state: TNNetState);
var
    b, i_m, i_n, i_k, colSize, _vol, imSize: SizeInt;
    workspacePtr : PSingleTensor;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  i_m := filters div groups;
  i_n := kernelSize * kernelSize * c div groups;
  i_k := outH * outW;

  colSize := c * outH * outW * kernelSize * kernelSize;
  //colSize := getWorkspaceSize div batch;
  _vol := state.input.Volume();
  if not delta.wasGPU() then delta.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not state.input.wasGPU() then state.input.pushToDevice;

  case activationType of
    acSWISH : ;
      //gradient_array_swish(output, outputs * batch, activationInput, delta) ;
    //acMISH  :
    //  gradient_array_mish(outputs * batch, activationInput, delta) ;
    //acHARD_MISH :
    //  gradient_array_hard_mish(outputs * batch, activationInput, delta) ;
    //acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
    //  gradient_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outW, delta) ;
    //acNORM_CHAN :
    //  gradient_array_normalize_channels(output, outputs * batch, batch, outC, outW * outW, delta) ;
    else
      cuda.DeriveArray(output.size(), output.devData, 0, longint(ActivationType), delta.devData);
  end;

{$ifdef DEBUG_GPU}
Derivative();
delta.printGpuSumSqrDiff();
{$endif}

  if isBatchNormalized then
    batchNormBackGPU(state)
  else begin
    cuda.backwardBias(bias_updates.size(), bias_updates.devData, delta.size, delta.devData, 0, 1, batch);
{$ifdef DEBUG_GPU}
bias_updates.addSums(delta);
{$endif}
  end;

  if speedOverSize then
    workspacePtr := @workspace
  else begin
    workspacePtr := @state.workspace;
    for b:=0 to batch-1 do begin
      cuda.im2col(c, h, w, kernelSize, kernelSize, Padding, Padding,
        stride_y, stride_x, dilation, dilation, state.input.devData , b*_vol, workspacePtr.devData, b*colSize);
    end;
{$ifdef DEBUG_GPU}
state.input.im2Col(kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, workspacePtr^, 1);
{$endif}
  end;
  workspacePtr.setCUDA;


  if not weight_updates.wasGPU() then weight_updates.pushToDevice;

  //for b:= 0 to batch -1 do begin
    //cuda.gemm(false, true,
    //        i_m, i_n, i_k, 1
    //        , delta.devData , b*i_m*i_k, i_k
    //        , state.workspace.devData, b*ColSize, i_k
    //        , 1, weight_updates.devData, 0, i_n);
  //end;

  cuda.gemmStridedBatched(false, true,
          i_m, i_n, i_k, 1
          , delta.devdata, 0, i_k, i_m*i_k
          , workspacePtr.devData, 0, i_k, colSize
          , 1, weight_updates.devData, 0, i_n, 0, batch);


{$ifdef DEBUG_GPU}
for b:= 0 to batch -1 do
TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_m, i_n, i_k, 1
, delta.data + b*i_m*i_k, i_k
, workspacePtr.data + b*ColSize, i_k
, 1, weight_updates.data, i_n);
{$endif}

  if assigned(state.delta) and assigned(state.delta.devdata) then begin
    if not weights.wasGPU() then weights.pushToDevice;
    //if not state.delta.wasGPU then state.delta.pushToDevice;

    //for b := 0 to batch -1 do begin
    //  cuda.gemm(true, false,
    //      i_n, i_k, i_m, 1,
    //      weights.devData, 0    , i_n,
    //      delta.devData  , b*i_m*i_k, i_k,
    //      0, state.workspace.devData, b*colSize, i_k);
    //end;

    cuda.gemmStridedBatched(true, false,
        i_n, i_k, i_m, 1
        , weights.devData, 0, i_n, 0
        , delta.devData, 0, i_k, i_m*i_k
        , 0, workspacePtr.devData, 0, i_k, colSize, batch);

{$ifdef DEBUG_GPU}
for b := 0 to batch -1 do begin
    TSingleTensor.gemm(
      CblasRowMajor, CblasTrans, CblasNoTrans, i_n, i_k, i_m, 1
      , weights.Data, i_n
      , delta.Data + b * i_m * i_k, i_k
      , 0, state.workspace.data + b*colSize, i_k);
end;
{$endif}

    imSize := state.delta.Volume();
    for b := 0 to batch-1 do begin
      cuda.col2im(state.delta.c, state.delta.h, state.delta.w, kernelSize, kernelSize, Padding, Padding
              , stride_y, stride_x, dilation, dilation, workspacePtr.devData, b*colSize, state.delta.devData, b*imSize);
    end;

{$ifdef DEBUG_GPU}
state.delta.col2Im(kernelSize, kernelSize, padding*Dilation, padding*Dilation, stride_x, stride_y, dilation, dilation, workspacePtr^);
{$endif}
  end ;

{$ifdef DEBUG_GPU}
bias_updates.printGpuSumSqrDiff();
weight_updates.printGpuSumSqrDiff();
state.delta.printGpuSumSqrDiff();
{$endif}

  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TConvolutionalLayer.updateGPU(const args: TUpdateArgs);
var
    learning_rate: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

  if not biases.wasGPU() then biases.pushToDevice;
  if not bias_updates.wasGPU() then bias_updates.pushToDevice;
  if not weights.wasGPU() then weights.pushToDevice;
  if not weight_updates.wasGPU() then weight_updates.pushToDevice;

  learning_rate := args.learningRate * learningRateScale;

  cuda.axpy(biases.size(), learning_rate / args.batch, bias_updates.devData, 0, 1, biases.devData, 0, 1);

  cuda.scale(bias_updates.size(), args.momentum, bias_updates.devData, 1);

  cuda.axpy(weight_updates.size(), -args.decay * args.batch, weights.devData, 0, 1, weight_updates.devData, 0, 1);

  cuda.axpy(weights.size(), learning_Rate / args.batch, weight_updates.devData, 0, 1, weights.devData, 0, 1);

  cuda.scale(weight_updates.size(), args.momentum, weight_updates.devData, 1);

  if assigned(scales.Data) then begin
    cuda.axpy(scales.size(), learning_rate / args.batch, scale_updates.devData, 0, 1, scales.devData, 0, 1);
    cuda.scale(scale_updates.size(), args.momentum, scale_updates.devData, 1);
  end;

  inherited;

  {$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

{$endif}

initialization

end.

