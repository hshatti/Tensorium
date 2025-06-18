unit nBaseLayer;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
interface

uses
  SysUtils, ntypes, ntensors, nActivation
  {$ifdef USE_OPENCL}
  , OpenCL
  {$endif}
  {$ifdef USE_TELEMETRY}
  , nOpMetrics
  {$endif}
  ;

const
  // metrics to print :
  TELEMETRY_OPS   = 1;      //  operations
  TELEMETRY_ACT   = 2;      //  activations
  TELEMETRY_FWD   = 4;      //  forward
  TELEMETRY_GRD   = 8;      //  gradients derive
  TELEMETRY_BWD   = 16;     //  backwards
  TELEMETRY_UPD   = 32;     //  updates
  TELEMETRY_ALL   = 63;     //  everything!


type

  { TBaseLayer }

  TBaseLayer = class(TInterfacedObject)
  protected
    procedure setTrain(ATrain: boolean); virtual; abstract;
  public
    FTrain                   : boolean;
    layerType                : TLayerType;
    inputShape               : TArray<SizeInt>;
    output                   : TSingleTensor;
    Batch, Steps, Groups,
      inputs, outputs        : SizeInt;
    weights                  : TSingleTensor;
    biases                   : TSingleTensor;

    weight_updates           : TSingleTensor;
    bias_updates             : TSingleTensor;

    delta                    : TSingleTensor;

    //gradient                 : TSingleTensor;
    //activated                : TSingleTensor;
    ActivationType           : TActivationType;
    id                       : SizeInt;
    isBatchNormalized        : boolean;
    bnMomentum               : single ;
    backwardStop, forwardOnly: boolean;
    dontLoad, dontLoadScales  : boolean;
    // for batch normalization
    scales                   : TSingleTensor;
    scale_updates            : TSingleTensor;
    mean                     : TSingleTensor;
    mean_delta               : TSingleTensor;
    variance                 : TSingleTensor;
    variance_delta           : TSingleTensor;
    rolling_mean             : TSingleTensor;
    rolling_variance         : TSingleTensor;
    x                        : TSingleTensor;
    x_norm                   : TSingleTensor;
    workspace                : TSingleTensor;
    // for ADAM optimization
    m                        : TSingleTensor;
    v                        : TSingleTensor;
    bias_m                   : TSingleTensor;
    scale_m                  : TSingleTensor;
    bias_v                   : TSingleTensor;
    scale_v                  : TSingleTensor;

    // Exponential Moving Average
    weights_ema, biases_ema  : TSingleTensor;
    scales_ema               : TSingleTensor;

    // cost array must be nil or size of [1]
    cost                     : TArray<Single>;
    index                    : SizeInt;
    net                      : TObject;
    {$if defined(USE_OPENCL) and defined(CL_EVENTS)}
    events                   : TArray<cl_event>;
    ev                       : TArray<cl_int>;
    {$endif}
    constructor Create(); virtual;
    function getWorkspaceSize():SizeInt; virtual;
    procedure Activate(const offset: SizeInt =0);   virtual;
    procedure Derivative(const offset: SizeInt =0); virtual;
    procedure reGroup(const stepBatch :SizeInt);

    function LayerTypeStr:string;
    procedure setBatch(ABatch :SizeInt); virtual; abstract;
    procedure freeBatchNorm;
    //destructor Destroy();override;
    procedure forward(var state : TNNetState); virtual; abstract;
    procedure backward(var state : TNNetState); virtual; abstract;
    procedure update(const args : TUpdateArgs); virtual;
    procedure fuseBatchNorm; virtual;
    function getWorkspaceShape:TArray<SizeInt>; virtual;
    property train:boolean read FTrain write setTrain;
    property workspaceSize:SizeInt read getWorkspaceSize;

    procedure batchNorm(var state: TNNetState);
    procedure batchNormBack(var state :TNNetState);
    procedure batchNormUpdate(const args : TUpdateArgs);
    {$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state : TNNetState); virtual; abstract;
    procedure backwardGPU(var state : TNNetState); virtual; abstract;
    procedure updateGPU(const args : TUpdateArgs); virtual;

    procedure batchNormGPU(var state: TNNetState);
    procedure batchNormBackGPU(var state :TNNetState);
    procedure batchNormUpdateGPU(const args : TUpdateArgs);
    {$endif}

  end;

  { TBaseImageLayer }

  TBaseImageLayer = class(TBaseLayer)
    step                     : SizeInt;
    w, h, c                  : SizeInt;
    outW, outH, outC         : SizeInt;
    learningRateScale        : Single;
    {$ifdef USE_OPENCL}
    {$ENDIF}
    function getImage():TImageData;
    function getDelta():TImageData;
  end;

{$ifdef USE_TELEMETRY}
  { TMetrics }

  TMetrics = record
    type

      { TAct }

      TAct =record
      private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TActivationType): int64;
      public
          all:array[low(TActivationType)..high(TActivationType)] of int64;
          procedure start(const a:TActivationType);
          procedure finish(const a:TActivationType);
          function total:int64;
          property Item[i:TActivationType]:int64 read GetItem ;default;
      end;

    { TFw }
       TFw = record
       private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TLayerType): int64;
       public
          all:array[low(TLayerType)..high(TLayerType)] of int64;
          procedure start(const a:TLayerType);
          procedure finish(const a:TLayerType);
          function total():int64;
          function isSubPropagation():longint;
          property Item[i:TLayerType]:int64 read GetItem ;default;
       end;

    public

      ops: PTensorMetrics;
      act, grad : TAct;
      forward, backward, update:TFw;
      procedure reset;
      function print(const telemetry:longword = TELEMETRY_ALL):string;
  end;
{$endif}

{$ifdef USE_TELEMETRY}
var
  metrics : TMetrics;
{$endif}


implementation
uses typInfo, termesc, nChrono;

{ TBaseLayer }

constructor TBaseLayer.Create();
begin
  bnMomentum:=0.1;
end;

function TBaseLayer.getWorkspaceSize(): SizeInt;
begin
  result :=0;
end;

procedure TBaseLayer.Activate(const offset: SizeInt);
begin
  {$ifdef _USE_OPENCL}
  if output.computingDevice=cdOpenCL then begin;
    if not output.wasGPU then
      output.pushToDevice;
    ocl.ActivateArray(output.devData, batch * outputs, longint(ActivationType));
  end else
  {$endif}
  assert(output.Size() - offset - batch * outputs >= 0, '[Activation] out of range!');

  activate_array(Pointer(output.Data + offset), batch * outputs, ActivationType);
end;

procedure TBaseLayer.Derivative(const offset: SizeInt);
begin
  {$if defined(_USE_OPENCL)}
  if output.computingDevice = cdOpenCL then begin
    if not output.wasGPU then
      output.pushToDevice;
    if not delta.wasGPU() then
      delta.pushToDevice;
    ocl.DeriveArray(output.devData, batch * outputs, longint(ActivationType), delta.devData);
    //ocl.finish;
  end else
  {$endif}
  assert((IntPtr(output.data) >= IntPtr(output.DynData)) and (output.Size() - offset - batch * outputs >= 0), '[Gradient] Output out of range!');
  assert((IntPtr(delta.data)  >= IntPtr(delta.DynData))  and (delta.Size()  - offset - batch * outputs >= 0), '[Gradient] Delta out of range!');
  gradient_array(pointer(output.Data + offset), batch * outputs, ActivationType, pointer(Delta.Data + offset));
end;

procedure TBaseLayer.reGroup(const stepBatch: SizeInt);
begin
    output.groups := stepBatch;
    if isBatchNormalized then begin
        x      .groups := stepBatch;
        x_norm .groups := stepBatch;
    end;
    if FTrain then begin
        delta.Groups:= stepBatch;
    end;
end;

function TBaseLayer.LayerTypeStr: string;
begin
  case layerType of
      ltCONVOLUTIONAL:
          exit('convolutional');
      ltACTIVE:
          exit('activation');
      ltLOCAL:
          exit('local');
      ltDECONVOLUTIONAL:
          exit('deconvolutional');
      ltCONNECTED:
          exit('connected');
      ltRNN:
          exit('rnn');
      ltGRU:
          exit('gru');
      ltLSTM:
          exit('lstm');
      ltCRNN:
          exit('crnn');
      ltMAXPOOL:
          exit('maxpool');
      ltREORG:
          exit('reorg');
      ltAVGPOOL:
          exit('avgpool');
      ltSOFTMAX:
          exit('softmax');
      ltDETECTION:
          exit('detection');
      ltREGION:
          exit('region');
      ltYOLO:
          exit('yolo');
      ltGaussianYOLO:
          exit('Gaussian_yolo');
      ltDROPOUT:
          exit('dropout');
      ltCROP:
          exit('crop');
      ltCOST:
          exit('cost');
      ltROUTE:
          exit('route');
      ltSHORTCUT:
          exit('shortcut');
      ltScaleChannels:
          exit('scale_channels');
      ltSAM:
          exit('sam');
      ltNORMALIZATION:
          exit('normalization');
      ltBATCHNORM:
          exit('batchnorm');
      ltUPSAMPLE:
          exit('upsample');
      //else

  end;
  exit('none')
end;

procedure TBaseLayer.freeBatchNorm;
begin
  scales.free;
  scale_updates.free;
  mean.free;
  variance.free;
  mean_delta.free;
  variance_delta.free;
  rolling_mean.free;
  rolling_variance.free;
  x.free;
  x_norm.free
end;

//destructor TBaseLayer.Destroy;
//begin
//  inherited Destroy;
//end;

procedure TBaseLayer.update(const args: TUpdateArgs);
begin
  //
end;

procedure TBaseLayer.fuseBatchNorm;
begin

end;

function TBaseLayer.getWorkspaceShape: TArray<SizeInt>;
begin
  result:= nil
end;

procedure TBaseLayer.batchNorm(var state: TNNetState);
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(ltBATCHNORM);
{$endif}

  if LayerType = ltBATCHNORM then
      state.input.copyTo(output);

  //if l.&type = ltCONNECTED then begin
  //    outC := outputs;
  //    outH :=1;
  //    outW:=1;
  //end;

  if state.isTraining then begin
      output.MeansAndVars(mean, variance);
      rolling_mean.Multiply(1-bnMomentum);
      rolling_mean.axpy(bnMomentum, mean);
      rolling_variance.Multiply(1-bnMomentum);
      rolling_variance.axpy(bnMomentum, variance);
      output.CopyTo(x);
      output.Normalize(mean, variance);
      output.copyTo(x_norm)
  end else
      output.Normalize(rolling_mean, rolling_variance);

  //output.FusedMultiplyAdd(scales, biases);
  output.forwardScale(scales);
  output.forwardBias(biases);

{$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(ltBATCHNORM);
{$endif}
end;

procedure TBaseLayer.batchNormBack(var state: TNNetState);
var offset, stepSize : SizeInt;
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(ltBATCHNORM);
{$endif}
  stepSize := batch * outputs;
  offset := stepSize * state.step;
  //bias_updates.Add(delta); //todo [batchNormBack] should we "bias_updates.Add(delta)"  here?
  // spatial dot (x_norm . delta) then add to scale_updates
  scale_updates.addDots(x_norm, delta, offset);

  // add scales to all delta batches
  delta.forwardScale(scales, offset, stepSize);
  TSingleTensor.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta, offset);
  TSingleTensor.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta, offset);
  if layerType = ltBATCHNORM then
    delta.copyTo(state.delta^);

{$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(ltBATCHNORM);
{$endif}
end;

procedure TBaseLayer.batchNormUpdate(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
    if benchmark then metrics.update.start(ltBATCHNORM);
  {$endif}
  if layerType=ltBATCHNORM then begin
    biases.axpy(args.learningRate / args.batch, bias_updates);
    bias_updates.Multiply(args.momentum);
  end;

  scales.axpy(args.learningRate / args.batch, scale_updates);
  scale_updates.Multiply(args.momentum);
  {$ifdef USE_TELEMETRY}
    if benchmark then metrics.update.finish(ltBATCHNORM);
  {$endif}
end;

{$if defined(USE_OPENCL)}
procedure TBaseLayer.updateGPU(const args: TUpdateArgs);
begin
 //
end;

procedure TBaseLayer.batchNormGPU(var state: TNNetState);
var outputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(ltBATCHNORM);
  {$endif}
    if not biases.wasGPU() then biases.pushToDevice;
    if not scales.wasGPU() then scales.pushToDevice;
    if not rolling_mean.wasGPU() then rolling_mean.pushToDevice;
    if not rolling_variance.wasGPU() then rolling_variance.pushToDevice;

    outputStep := batch*outputs;
    offset := state.step*outputStep;

    if LayerType = ltBATCHNORM then
        //state.input.copyTo(output);
        ocl.copy(outputStep, state.input.devData, 0, 1, output.devData, 0, 1);

    //if l.&type = ltCONNECTED then begin
    //    outC := outputs;
    //    outH :=1;
    //    outW:=1;
    //end;
    if not scales.wasGPU() then scales.pushToDevice;
    if not rolling_mean.wasGPU() then rolling_mean.pushToDevice;
    if not rolling_variance.wasGPU() then rolling_variance.pushToDevice;

    if state.isTraining then begin
        //output.MeansAndVars(mean, variance);
        ocl.meansAndVars(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData, variance.devData);

        //rolling_mean.Multiply(0.9);
        ocl.scale(rolling_mean.size(), 0.9, rolling_mean.devData, 1);

  //      //rolling_mean.axpy(0.1, mean);
        ocl.axpy(rolling_mean.Size(), 0.1, mean.devData, 0, 1, rolling_mean.devData, 0, 1);

  //      rolling_variance.Multiply(0.9);
        ocl.scale(rolling_variance.Size(), 0.9, rolling_variance.devData, 1);

  //      rolling_variance.axpy(0.1, variance);
        ocl.axpy(rolling_variance.size(), 0.1, variance.devData, 0, 1, rolling_variance.devData, 0, 1);

  //      output.CopyTo(x);
        ocl.copy(outputStep, output.devData, offset, 1, x.devData, offset, 1);

  //      output.Normalize(mean, variance);
        ocl.normalize(mean.Size(), outputStep, output.groups, mean.devData, 1, variance.devData, 1, output.devData, offset);

  //      output.copyTo(x_norm) ;
        ocl.copy(outputStep, output.devData, offset, 1, x_norm.devData ,offset, 1);
    end else begin
  //      output.Normalize(rolling_mean, rolling_variance);
        ocl.normalize(rolling_mean.Size(), outputStep, output.Groups, rolling_mean.devData, 1, rolling_variance.devData, 1, output.devData, 0);
    end;



    //output.Multiply(scales);
    //output.add(biases);
    ocl.forwardScaleAdd(outputStep, output.devData, offset, scales.size(), scales.devData, biases.devData, 1, output.groups);

  {$ifdef USE_TELEMETRY}
    ocl.finish();
    if benchmark then metrics.forward.finish(ltBATCHNORM);
  {$endif}
end;

procedure TBaseLayer.batchNormBackGPU(var state: TNNetState);
var
  outputStep, offset: SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(ltBATCHNORM);
  {$endif}

  // spatial dot (x_norm . delta) then add to scale_updates
  //scale_updates.addDots(x_norm, delta);
  //
  //// add scales to all delta batches
  //delta.add(scales);
  //TSingleTensor.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);
  //TSingleTensor.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
  //if layerType = ltBATCHNORM then
  //  delta.copyTo(state.delta^);
  //ocl.backwardBias(bias_updates.Size(), bias_updates.devData, delta.size(), delta.devData, 1, delta.groups);  //todo [batchNormBack] should we "bias_updates.Add(delta)"  here?
  outputStep := batch*outputs;
  offset := state.step*outputStep;

  ocl.addDots(outputStep, scale_updates.Size(), delta.groups, x_norm.devData, delta.devData, offset, scale_updates.devData);
  ocl.forwardScale(outputStep, delta.devData, offset, scales.Size(), scales.devData, 1, delta.groups);
  ocl.meansAndVarsDelta(outputStep, mean_delta.size(), delta.groups, delta.devData, x.devData, offset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
  ocl.normalizeDelta(outputStep, mean.size(), delta.groups, delta.devData, x.devData, offset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
  if layerType = ltBATCHNORM then
    ocl.copy(outputStep, delta.devData, 0, 1, state.delta^.devData, 0, 1);
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.backward.finish(ltBATCHNORM);
  {$endif}
end;

procedure TBaseLayer.batchNormUpdateGPU(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(ltBATCHNORM);
  {$endif}
  //biases.axpy(args.learningRate / args.batch, bias_updates);
  //bias_updates.Multiply(args.momentum);
  //
  //scales.axpy(args.learningRate / args.batch, scale_updates);
  //scale_updates.Multiply(args.momentum);
  if layerType=ltBATCHNORM then begin
    ocl.axpy(biases.size(), args.learningRate / args.batch, bias_updates.devData, 0, 1, biases.devData, 0, 1);
    ocl.scale(bias_updates.size(), args.momentum, bias_updates.devData, 1);

  end;

  ocl.axpy(scales.size(), args.learningRate / args.batch, scale_updates.devData, 0, 1, scales.devData, 0, 1);
  ocl.scale(scale_updates.size(), args.momentum, scale_updates.devData, 1);

  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.update.finish(ltBATCHNORM);
  {$endif}
end;
{$elseif defined(USE_CUDART)}
procedure TBaseLayer.updateGPU(const args: TUpdateArgs);
begin
 //
end;

procedure TBaseLayer.batchNormGPU(var state: TNNetState);
var
  outputStep, offset: SizeInt;
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(ltBATCHNORM);
{$endif}
  if not biases.wasGPU() then biases.pushToDevice;
  if not scales.wasGPU() then scales.pushToDevice;
  if not rolling_mean.wasGPU() then rolling_mean.pushToDevice;
  if not rolling_variance.wasGPU() then rolling_variance.pushToDevice;

  outputStep := batch*outputs;
  offset := state.step*outputStep;

  if LayerType = ltBATCHNORM then begin
      cuda.copy(outputStep, state.input.devData, 0, 1, output.devData, 0, 1);
//state.input.copyTo(output);
  end;

  //if l.&type = ltCONNECTED then begin
  //    outC := outputs;
  //    outH :=1;
  //    outW:=1;
  //end;
  if not scales.wasGPU() then scales.pushToDevice;
  if not rolling_mean.wasGPU() then rolling_mean.pushToDevice;
  if not rolling_variance.wasGPU() then rolling_variance.pushToDevice;

  if state.isTraining then begin
      //cuda.meansAndVars(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData, variance.devData);
      cuda.means(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData);
      cuda.variances(outputStep, mean.Size(), output.Groups, output.devData, offset, mean.devData, variance.devData);
//output.MeansAndVars(mean, variance);
//mean.printGpuSumSqrDiff();
//variance.printGpuSumSqrDiff();

      cuda.scale(rolling_mean.size(), 1-bnMomentum, rolling_mean.devData, 1);
//rolling_mean.Multiply(1- bnMomentum);
//rolling_mean.printGpuSumSqrDiff();


      cuda.axpy(rolling_mean.Size(), bnMomentum, mean.devData, 0, 1, rolling_mean.devData, 0, 1);
//rolling_mean.axpy(bnMomentum, mean);
//rolling_mean.printGpuSumSqrDiff();

      cuda.scale(rolling_variance.Size(), 1-bnMomentum, rolling_variance.devData, 1);
//rolling_variance.Multiply(1-bnMomentum);
//rolling_variance.printGpuSumSqrDiff();

      cuda.axpy(rolling_variance.size(), bnMomentum, variance.devData, 0, 1, rolling_variance.devData, 0, 1);
//rolling_variance.axpy(bnMomentum, variance);
//rolling_variance.printGpuSumSqrDiff();

      cuda.copy(outputStep, output.devData, offset, 1, x.devData, offset, 1);
//output.CopyTo(x);
//x.printGpuSumSqrDiff();

      cuda.normalize(mean.Size(), outputStep, output.groups, mean.devData, 1, variance.devData, 1, output.devData, offset);
//output.Normalize(mean, variance);
//output.printGpuSumSqrDiff();

      cuda.copy(outputStep, output.devData, offset, 1, x_norm.devData ,offset, 1);
//output.copyTo(x_norm) ;
//x_norm.printGpuSumSqrDiff();
  end else begin
      cuda.normalize(rolling_mean.Size(), outputStep, output.Groups, rolling_mean.devData, 1, rolling_variance.devData, 1, output.devData, 0);
//output.Normalize(rolling_mean, rolling_variance);
//output.printGpuSumSqrDiff();
  end;



  //cuda.forwardScaleAdd(outputStep, output.devData, offset, scales.size(), scales.devData, biases.devData, 1, output.groups);
  cuda.forwardScale(outputStep, output.devData, offset, scales.size(), scales.devData, 1, output.groups);
  cuda.forwardBias(outputStep, output.devData, offset, biases.size(), biases.devData, 1, output.groups);
//output.forwardScale(scales);
//output.forwardBias(biases);

//output.printGpuSumSqrDiff();
//mean.printGpuSumSqrDiff();
//variance.printGpuSumSqrDiff();
//x.printGpuSumSqrDiff();
//x_norm.printGpuSumSqrDiff();



{$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.forward.finish(ltBATCHNORM);
{$endif}
end;

procedure TBaseLayer.batchNormBackGPU(var state: TNNetState);
var
  outputStep, offset: SizeInt;
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(ltBATCHNORM);
{$endif}

  // spatial dot (x_norm . delta) then add to scale_updates
  //scale_updates.addDots(x_norm, delta);
  //
  //// add scales to all delta batches
  //delta.add(scales);
  //TSingleTensor.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);
  //TSingleTensor.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
  //if layerType = ltBATCHNORM then
  //  delta.copyTo(state.delta^);
  //cuda.backwardBias(bias_updates.Size(), bias_updates.devData, delta.size(), delta.devData, 1, delta.groups);  //todo [batchNormBack] should we "bias_updates.Add(delta)"  here?
  outputStep := batch*outputs;
  offset := state.step*outputStep;

  cuda.addDots(outputStep, scale_updates.Size(), delta.groups, x_norm.devData, delta.devData, offset, scale_updates.devData);
//scale_updates.addDots(x_norm, delta, offset, outputStep);
//scale_updates.printGpuSumSqrDiff();

  cuda.forwardScale(outputStep, delta.devData, offset, scales.Size(), scales.devData, 1, delta.groups);
//delta.forwardScale(scales, offset, outputStep);
//delta.printGpuSumSqrDiff();

  cuda.meansAndVarsDelta(outputStep, mean.size(), delta.groups, delta.devData, x.devData, offset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
//TSingleTensor.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta, offset);
//mean_delta.printGpuSumSqrDiff();
//variance_delta.printGpuSumSqrDiff();

  cuda.normalizeDelta(outputStep, mean.size(), delta.groups, delta.devData, x.devData, offset, mean.devData, variance.devData, mean_delta.devData, variance_delta.devData);
//TSingleTensor.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta, offset);
//delta.printGpuSumSqrDiff();

  if layerType = ltBATCHNORM then
    cuda.copy(outputStep, delta.devData, 0, 1, state.delta^.devData, 0, 1);
{$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.backward.finish(ltBATCHNORM);
{$endif}
end;

procedure TBaseLayer.batchNormUpdateGPU(const args: TUpdateArgs);
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(ltBATCHNORM);
{$endif}
  //biases.axpy(args.learningRate / args.batch, bias_updates);
  //bias_updates.Multiply(args.momentum);
  //
  //scales.axpy(args.learningRate / args.batch, scale_updates);
  //scale_updates.Multiply(args.momentum);
  if layerType=ltBATCHNORM then begin
    cuda.axpy(biases.size(), args.learningRate / args.batch, bias_updates.devData, 0, 1, biases.devData, 0, 1);
    cuda.scale(bias_updates.size(), args.momentum, bias_updates.devData, 1);

  end;

  cuda.axpy(scales.size(), args.learningRate / args.batch, scale_updates.devData, 0, 1, scales.devData, 0, 1);
  cuda.scale(scale_updates.size(), args.momentum, scale_updates.devData, 1);

{$ifdef USE_TELEMETRY}
  cuda.finish();
  if benchmark then metrics.update.finish(ltBATCHNORM);
{$endif}
end;
{$endif}

{ TBaseImageLayer }

function TBaseImageLayer.getImage(): TImageData;
begin
  result.h := outH;
  result.w := outW;
  result.c := outC;
  setLength(result.data, result.c * result.h * result.w );
  Move(output.Data[0], result.Data[0], length(result.Data)*SizeOf(Single))
end;

function TBaseImageLayer.getDelta(): TImageData;
begin
  result.h := outH;
  result.w := outW;
  result.c := outC;
  setLength(result.data, result.c * result.h * result.w );
  Move(delta.Data[0], result.Data[0], length(result.Data)*SizeOf(Single))
end;

{$ifdef USE_TELEMETRY}
{ TMetrics }

procedure TMetrics.reset;
begin
  if assigned(ops) then
    fillchar(PAnsiChar(@ops.elapsed)[0], sizeOf(ops.elapsed), #0);
  if assigned(ops) then
    fillchar(PAnsiChar(@ops.counts)[0], sizeOf(ops.counts), #0);
  fillchar(PAnsiChar(@act.all)[0], sizeOf(act.all), #0);
  fillchar(PAnsiChar(@grad.all)[0], sizeOf(grad.all), #0);
  fillchar(PAnsiChar(@forward.all)[0], sizeOf(forward.all), #0);
  fillchar(PAnsiChar(@backward.all)[0], sizeOf(backward.all), #0);
  fillchar(PAnsiChar(@update.all)[0], sizeOf(backward.all), #0);

end;

function TMetrics.print(const telemetry: longword): string;
const uSecPerSec=1000000;
var
  i :TMeasureOps;
  j :TActivationType;
  k :TLayerType;
begin
  result :='';
  if not benchmark then exit;

  if (telemetry and TELEMETRY_OPS>0) and (ops.total<>0) then begin
    result := result + 'Operations :' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 12);//+ sLineBreak;
    for i:= low(ops.elapsed) to high(ops.elapsed) do
      if ops.elapsed[i]<>0 then
        result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TMeasureOps),ord(i)),3), ops.elapsed[i]/uSecPerSec] ) + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + '-----------------------------' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + format('Total          %10.3f[ms]', [ops.total()/uSecPerSec]) + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//sLineBreak + sLineBreak;
  end;

  if (telemetry and TELEMETRY_ACT>0) and (act.total<>0) then begin
    result := result + cursorMove(cmDown, 2)+{cursorMove(cmBackward, 29) +} 'Activations :'  + cursorMove(cmDown, 1)+cursorMove(cmBackward, 13);//+ sLineBreak;
    for j:= low(act.all) to high(act.all) do
      if act.all[j]<>0 then
        result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TActivationType),ord(j)),3), act.all[j]/uSecPerSec] ) + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + '-----------------------------' + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + format('Total          %10.3f[ms]', [act.total/uSecPerSec]) + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);
  end;

  if (telemetry and TELEMETRY_FWD>0) and (forward.total<>0) then begin
    result := result + cursorMove(cmDown, 2)+{cursorMove(cmBackward, 29) +} 'Forwards :' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 10);//+ sLineBreak;
    for k:= low(forward.all) to high(forward.all) do
      if forward.all[k]<>0 then
        result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TLayerType),ord(k)),3), forward.all[k]/uSecPerSec] ) + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + '-----------------------------' + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + format('Total          %10.3f[ms]', [forward.total/uSecPerSec]) + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak + sLineBreak;
  end;

  if (telemetry and TELEMETRY_GRD>0) and (grad.total<>0) then begin
    result := result + cursorMove(cmDown, 2)+{cursorMove(cmBackward, 29) +} 'Gradients :' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 13);//+ sLineBreak;
    for j:= low(grad.all) to high(grad.all) do
      if grad.all[j]<>0 then
        result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TActivationType),ord(j)),3), grad.all[j]/uSecPerSec] ) + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + '-----------------------------' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + format('Total          %10.3f[ms]', [grad.total/uSecPerSec]) + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak + sLineBreak;
  end;

  if (telemetry and TELEMETRY_BWD>0) and (backward.total<>0) then begin
    result := result + cursorMove(cmDown, 2)+{cursorMove(cmBackward, 29) +} 'Backwards :' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 11);//+ sLineBreak;
    for k:= low(backward.all) to high(backward.all) do
      if backward.all[k]<>0 then
        result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TLayerType),ord(k)),3), backward.all[k]/uSecPerSec] ) + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + '-----------------------------' + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + format('Total          %10.3f[ms]', [backward.total/uSecPerSec]) + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak + sLineBreak;
  end;

  if (telemetry and TELEMETRY_UPD>0) and (update.total<>0) then begin
    result := result + cursorMove(cmDown, 2)+{cursorMove(cmBackward, 29) +} 'Updats :' +cursorMove(cmDown, 1)+cursorMove(cmBackward, 8);//+ sLineBreak;
    for k:= low(update.all) to high(update.all) do
      if update.all[k]<>0 then
        result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TLayerType),ord(k)),3), update.all[k]/uSecPerSec] ) + cursorMove(cmDown, 1)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + '-----------------------------' + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak;
    result := result + format('Total          %10.3f[ms]', [update.total/uSecPerSec]) + cursorMove(cmDown, 2)+cursorMove(cmBackward, 29);//+ sLineBreak + sLineBreak;
  end;
end;

{ TMetrics.TAct }

function TMetrics.TAct.GetItem(i: TActivationType): int64;
begin
  result := all[i]
end;

procedure TMetrics.TAct.start(const a: TActivationType);
begin
  m[stack]:=clock;
  inc(stack)
  //all[a] := clock();
end;

procedure TMetrics.TAct.finish(const a: TActivationType);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TAct.total: int64;
var
  i: TActivationType;
begin
  result := 0;
  for i:=low(TActivationType) to high(TActivationType) do
    inc(result, all[i])
end;

{ TMetrics.TFw }

function TMetrics.TFw.GetItem(i: TLayerType): int64;
begin
  result := all[i];
end;

procedure TMetrics.TFw.start(const a: TLayerType);
begin
  m[stack]:=clock;
  inc(stack)
end;

procedure TMetrics.TFw.finish(const a: TLayerType);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TFw.total(): int64;
var
  i: TLayerType;
begin
  result := 0;
  for i:=low(TLayerType) to high(TLayerType) do
    inc(result, all[i])
end;

function TMetrics.TFw.isSubPropagation: longint;
begin
  result := stack
end;

{$endif USE_TELEMETRY}
initialization

{$ifdef USE_TELEMETRY}
  metrics.ops := @tensorMetrics;
{$endif}

end.

