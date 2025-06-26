unit nnOpenCL;
{$Z4}
{$ifdef FPC}
  {$mode Delphi}
  {$if defined(CPUX64) or defined(CPUX32)}
  {$asmmode intel}
   {$endif}
{$endif}

interface

uses
  {$if defined(MACOS) or defined(DARWIN)}
  CL
  {$else}
  OpenCL
  {$endif}
  , OpenCLHelper
  {$ifdef MSWINDOWS}
  , windows
  {$else}
  , dl
  {$endif}

{$ifdef USE_TELEMETRY}
  , nOpMetrics
{$endif}
  ;

  const
    CL_LIB_NONE  = 0;
    CL_LIB_BLAS  = 1;
    CL_LIB_BLAST = 2;
    OCL_BLOCK    = $100;

  const kernelNames : array[0..43] of PAnsiChar =(
    'vars',
    'means',
    'power',
    'fmav',
    'mulv',
    'cost_l2',
    'backward_dropout',
    'forward_dropout',
    'forward_scale_add',
    'forward_scale',
    'add_dots',
    'norm_delta',
    'means_vars_delta',
    'normblkvv',
    'normvs',
    'normvv',
    'means_vars',
    'fmavss',
    'upsample',
    'Xcol2imKernelNormal',
    'Xcol2imKernelFlip',
    'Xim2colKernelNormal',
    'Xim2colKernelFlip',
    'im2col',
    'crossEntropySoftmax',
    'softmaxBatch',
    'backward_maxpool',
    'forward_maxpool',
    'copy',
    'fill',
    'crossEntropyLogistics',
    'scale',
    'axpy',
    'subv',
    'addv',
    'backward_bias',
    'gradient_array',
    'array_avtivate_swish',
    'activate_array',
    'forward_bias',
    'sgemm1_tn',
    'sgemm1_nt',
    'sgemm2_nn',
    'sgemm1_nn'
  );

  KERNEL_vars                  = 0;
  KERNEL_means                 = 1;
  KERNEL_power                 = 2;
  KERNEL_fmav                  = 3;
  KERNEL_mulv                  = 4;
  KERNEL_cost_l2               = 5;
  KERNEL_backward_dropout      = 6;
  KERNEL_forward_dropout       = 7;
  KERNEL_forward_scale_add     = 8;
  KERNEL_forward_scale         = 9;
  KERNEL_add_dots              = 10;
  KERNEL_norm_delta            = 11;
  KERNEL_means_vars_delta      = 12;
  KERNEL_normblkvv             = 13;
  KERNEL_normvs                = 14;
  KERNEL_normvv                = 15;
  KERNEL_means_vars            = 16;
  KERNEL_fmavss                = 17;
  KERNEL_upsample              = 18;
  KERNEL_Xcol2imKernelNormal   = 19;
  KERNEL_Xcol2imKernelFlip     = 20;
  KERNEL_Xim2colKernelNormal   = 21;
  KERNEL_Xim2colKernelFlip     = 22;
  KERNEL_im2col                = 23;
  KERNEL_crossEntropySoftmax   = 24;
  KERNEL_softmaxBatch          = 25;
  KERNEL_backward_maxpool      = 26;
  KERNEL_forward_maxpool       = 27;
  KERNEL_copy                  = 28;
  KERNEL_fill                  = 29;
  KERNEL_crossEntropyLogistics = 30;
  KERNEL_scale                 = 31;
  KERNEL_axpy                  = 32;
  KERNEL_subv                  = 33;
  KERNEL_addv                  = 34;
  KERNEL_backward_bias         = 35;
  KERNEL_gradient_array        = 36;
  KERNEL_array_avtivate_swish  = 37;
  KERNEL_activate_array        = 38;
  KERNEL_forward_bias          = 39;
  KERNEL_sgemm1_tn             = 40;
  KERNEL_sgemm1_nt             = 41;
  KERNEL_sgemm2_nn             = 42;
  KERNEL_sgemm1_nn             = 43;

type

  TCLMemAccess = OpenCLHelper.TCLMemAccess;

  TCLMemory = cl_mem;

  PCLEvent = Pcl_event;
  TCLEvent = cl_event;

  TCLEvents = TArray<TCLEvent>;

  TCLDeviceType = OpenCLhelper.TCLDeviceType;


  { TNNOpenCL }

  TNNOpenCL = class(TOpenCL)
    useBLAS : integer;
    procedure ActivateArray(const N: SizeInt; const x: cl_mem; const offset: SizeInt; const activation: longint; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure activateArraySWISH(const N: SizeInt; const x: cl_mem; const offset: SizeInt; const output_sigmoid, output: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure DeriveArray(const N: SizeInt; const x: cl_mem; const offset:SizeInt; const activation: longint; delta: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure forwardBias(const dstSize: SizeInt; const dst: cl_mem; const offset:SizeInt; const srcSize: SizeInt; const src: cl_mem; const incb: SizeInt; const batch: SizeInt;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure backwardBias(const dstSize: SizeInt; const dst: cl_mem; const srcSize: SizeInt; const src: cl_mem; const srcOffset:SizeInt; const incb: SizeInt ; const batch: SizeInt;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure gemm(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:single;
      const A:cl_mem; const aOffset:SizeInt; const lda:SizeInt;
      const B:cl_mem; const bOffset:SizeInt; const ldb:SizeInt;
      const BETA: single; const C:cl_mem; const cOffset:SizeInt; const ldc:SizeInt;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure gemmStridedBatched(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:single;
      const A:cl_mem; aOffset:SizeInt; const lda, strideA:SizeInt;
      const B:cl_mem; bOffset:SizeInt; const ldb, strideB:SizeInt;
      const BETA: single; const C:cl_mem; cOffset:SizeInt; const ldc, strideC:SizeInt; const batchCount : SizeInt;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure addvv(const N:SizeInt; const src1:cl_mem; const src1Offset, inca:SizeInt; const src2:cl_mem; const src2Offset, incb:SizeInt; dst:cl_mem; const dstOffset, incc:SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure subvv(const N:SizeInt; const src1:cl_mem; const src1Offset, inca:SizeInt; const src2:cl_mem; const src2Offset, incb:SizeInt; dst:cl_mem; const dstOffset, incc:SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure mulvv(const N:SizeInt; const src1:cl_mem; const src1Offset, inca:SizeInt; const src2:cl_mem; const src2Offset, incb:SizeInt; dst:cl_mem; const dstOffset, incc:SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure fmavv(const N: SizeInt;
      const src1: cl_mem; const src1Offset, inca: SizeInt;
      const src2: cl_mem; const src2Offset, incb: SizeInt;
      const src3: cl_mem; const src3Offset, incc: SizeInt;
      dst: cl_mem; const dstOffset, incd: SizeInt;
      const events: TCLEvents= nil; event: PCLEvent = nil);
    procedure axpy(const N:SizeInt; const a:single; const x:cl_mem; const xOffset:SizeInt; const incx:SizeInt; const y:cl_mem; const yOffset:SizeInt; const incy:sizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure power(const N:SizeInt; const x:cl_mem; const xOffset:SizeInt; const incx:SizeInt; const a:single; const y:cl_mem; const yOffset:SizeInt; const incy:sizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure scale(const N:SizeInt; const a:Single; const x:cl_mem; const stride:SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure crossEntropyLogistic(const N:SizeInt; const pred, truth: cl_mem; delta, error: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure fill(const N:SizeInt; const x: cl_mem; const offset:SizeInt; const val:single; const stride :SizeInt;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure copy(const N:SizeInt; const src:cl_mem; const srcOffset, inca:SizeInt; const dst:cl_mem; const dstOffset, incb:SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure softmaxBatch(const N: SizeInt; const input: cl_mem; const iOffset: SizeInt;
      const batch, batch_size, groups, group_size, stride: SizeInt;
      const temp: single; const output: cl_mem; const oOffset: SizeInt;
      const events: TCLEvents = nil ; event: PCLEvent = nil );
    procedure crossEntropySoftmax(const N:SizeInt; const pred, truth: cl_mem; delta, error: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure forwardMaxPool(const aBatch, outC, outH, outW: SizeInt; const input: cl_mem; const c, h, w: SizeInt;
      const stride_x, stride_y, padding, kernelSize: SizeInt; indexes, output: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure backwardMaxPool(const aBatch, outC, outH, outW : SizeInt; output:cl_mem; const indexes, delta : cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure im2col(const aChannels, aHeight, aWidth
      , kernelHeight, kernelWidth, padHeight, padWidth
      , strideY, strideX, dilationY, dilationX : SizeInt
      ; const im :cl_mem; const imOffset : SizeInt
      ; const col:cl_mem; const colOffset:SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure col2im(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth,
      padHeight, padWidth, strideY, strideX, dilationY, dilationX: SizeInt;
      const col: cl_mem; const colOffset: SizeInt;
      const im: cl_mem; const imOffset: SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure upSample(const aBatch, aChannels, outHeight,
      outWidth: SizeInt; const &in: cl_mem;const stride: SizeInt; const isForward: longint;
      const scale: single; const &out: cl_mem; const zeroIn :boolean = false;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure fmavss(const N: SizeInt; const src: cl_mem; const offset: SizeInt; const scalar,
      bias: single; dst : cl_mem;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure meansAndVars(const srcSize, dstSize, groups:sizeInt; const src:cl_mem; const offset:sizeInt; means, vars:cl_mem;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure normalize(const srcSize, dstSize, groups:SizeInt; means:cl_mem; const meansStride:sizeInt; vars:cl_mem; const varsStride:SizeInt; dst:cl_mem; const dstOffset :sizeInt;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure meansAndVarsDelta(const srcSize, dstSize, groups:SizeInt; delta, x: cl_mem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: cl_mem;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure normalizeDelta(const deltaSize, meanSize, groups: SizeInt;
      const delta, x: cl_mem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: cl_mem;
      const events: TCLEvents =nil; event: PCLEvent = nil);
    procedure addDots(const N, nDst, groups:SizeInt; const src1, src2:cl_mem; const srcOffset:SizeInt; dst:cl_mem;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure forwardScale(const dstSize: SizeInt; const dst: cl_mem; const offset :SizeInt; const scaleSize: SizeInt; const scale: cl_mem; const incb: SizeInt ; const batch: SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure forwardScaleAdd(const dstSize: SizeInt; const dst: cl_mem; const offset :SizeInt; const scaleSize: SizeInt; const scales, biases: cl_mem; const incb: SizeInt ; const batch: SizeInt; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure forwardDropout(const N: SizeInt; const src: cl_mem;
      const probability, scale: single; rnd: cl_mem; dst: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure backwardDropout(const N: SizeInt; const src: cl_mem;
      const probability, scale: single; const rnd: cl_mem; dst: cl_mem; const events: TCLEvents = nil; event: PCLEvent = nil);
    procedure costL2(const N:SizeInt; const pred ,truth, delta, error: cl_mem;
      const events: TCLEvents = nil; event: PCLEvent = nil);
    //procedure halfTest(
    //  const N : SizeInt; a:cl_mem; b:cl_mem ; c:cl_mem;
    //  const events: TCLEvents = nil; event: PCLEvent = nil);
  end;

  TCLblasOrder = (
      clblasRowMajor,           (**< Every row is placed sequentially *)
      clblasColumnMajor         (**< Every column is placed sequentially *)
  );

  TCLblasTranspose = (
      clblasNoTrans,           (**< Operate with the matrix. *)
      clblasTrans,             (**< Operate with the transpose of the matrix. *)
      clblasConjTrans          (**< Operate with the conjugate transpose of the matrix. *)
  );

  TCLBlastLayout = ( CLBlastLayoutRowMajor = 101,
                                CLBlastLayoutColMajor = 102 );
  TCLBlastTranspose = ( CLBlastTransposeNo = 111, CLBlastTransposeYes = 112,
                      CLBlastTransposeConjugate = 113);

var
  clblasSgemm :function (const layout: TCLBlasOrder; const a_transpose: TCLBlasTranspose; const b_transpose: TCLBlasTranspose; const M: SizeInt; const N: SizeInt; const K: SizeInt; const alpha: single; const a_buffer: cl_mem; const a_offset: SizeInt; const lda: SizeInt; const b_buffer: cl_mem; const b_offset: SizeInt; const ldb: SizeInt; const beta: single; c_buffer: cl_mem; const c_offset: SizeInt; const ldc: SizeInt; queueCount: cl_int; queue: Pcl_command_queue; eventCount: cl_uint; const events: pcl_event; event: Pcl_event):cl_int; winapi;
  clblasDgemm :function (const layout: TCLBlasOrder; const a_transpose: TCLBlasTranspose; const b_transpose: TCLBlasTranspose; const M: SizeInt; const N: SizeInt; const K: SizeInt; const alpha: double; const a_buffer: cl_mem; const a_offset: SizeInt; const lda: SizeInt; const b_buffer: cl_mem; const b_offset: SizeInt; const ldb: SizeInt; const beta: double; c_buffer: cl_mem; const c_offset: SizeInt; const ldc: SizeInt; queueCount: cl_int; queue: Pcl_command_queue; eventCount: cl_uint; const events: pcl_event; event: Pcl_event):cl_int; winapi;
  CLBlastSgemm :function (const layout: TCLBlastLayout; const a_transpose: TCLBlastTranspose; const b_transpose: TCLBlastTranspose; const m: SizeInt; const n: SizeInt; const k: SizeInt; const alpha: single; const a_buffer: cl_mem; const a_offset: SizeInt; const a_ld: SizeInt; const b_buffer: cl_mem; const b_offset: SizeInt; const b_ld: SizeInt; const beta: single; c_buffer: cl_mem; const c_offset: SizeInt; const c_ld: SizeInt; queue: Pcl_command_queue; event: Pcl_event):cl_int; winapi;
  CLBlastDgemm :function (const layout: TCLBlastLayout; const a_transpose: TCLBlastTranspose; const b_transpose: TCLBlastTranspose; const m: SizeInt; const n: SizeInt; const k: SizeInt; const alpha: double; const a_buffer: cl_mem; const a_offset: SizeInt; const a_ld: SizeInt; const b_buffer: cl_mem; const b_offset: SizeInt; const b_ld: SizeInt; const beta: double; c_buffer: cl_mem; const c_offset: SizeInt; const c_ld: SizeInt; queue: Pcl_command_queue; event: Pcl_event):cl_int; winapi;
  CLBlastSgemmBatched :function (const layout: TCLBlastLayout; const a_transpose: TCLBlastTranspose; const b_transpose: TCLBlastTranspose; const m: SizeInt; const n: SizeInt; const k: SizeInt; const alphas: Psingle; const a_buffer: cl_mem; const a_offsets: PSizeInt; const a_ld: SizeInt; const b_buffer: cl_mem; const b_offsets: PSizeInt; const b_ld: SizeInt; const betas: Psingle; c_buffer: cl_mem; const c_offsets: PSizeInt; const c_ld: SizeInt; const batch_count: SizeInt; queue: Pcl_command_queue; event: Pcl_event):cl_int; winapi;
  CLBlastDgemmBatched :function (const layout: TCLBlastLayout; const a_transpose: TCLBlastTranspose; const b_transpose: TCLBlastTranspose; const m: SizeInt; const n: SizeInt; const k: SizeInt; const alphas: PDouble; const a_buffer: cl_mem; const a_offsets: PSizeInt; const a_ld: SizeInt; const b_buffer: cl_mem; const b_offsets: PSizeInt; const b_ld: SizeInt; const betas: PDouble; c_buffer: cl_mem; const c_offsets: PSizeInt; const c_ld: SizeInt; const batch_count: SizeInt; queue: Pcl_command_queue; event: Pcl_event):cl_int; winapi;
  CLBlastSgemmStridedBatched : function (const layout: TCLBlastLayout; const a_transpose: TCLBlastTranspose; const b_transpose: TCLBlastTranspose; const m: SizeInt; const n: SizeInt; const k: SizeInt; const alpha: single; const a_buffer: cl_mem; const a_offset: SizeInt; const a_ld: SizeInt; const a_stride: SizeInt; const b_buffer: cl_mem; const b_offset: SizeInt; const b_ld: SizeInt; const b_stride: SizeInt; const beta: single; c_buffer: cl_mem; const c_offset: SizeInt; const c_ld: SizeInt; const c_stride: SizeInt; const batch_count: SizeInt; queue: Pcl_command_queue; event: Pcl_event):cl_int; winapi;
  CLBlastDgemmStridedBatched : function (const layout: TCLBlastLayout; const a_transpose: TCLBlastTranspose; const b_transpose: TCLBlastTranspose; const m: SizeInt; const n: SizeInt; const k: SizeInt; const alpha: double; const a_buffer: cl_mem; const a_offset: SizeInt; const a_ld: SizeInt; const a_stride: SizeInt; const b_buffer: cl_mem; const b_offset: SizeInt; const b_ld: SizeInt; const b_stride: SizeInt; const beta: double; c_buffer: cl_mem; const c_offset: SizeInt; const c_ld: SizeInt; const c_stride: SizeInt; const batch_count: SizeInt; queue: Pcl_command_queue; event: Pcl_event):cl_int; winapi;

implementation

function LSize(const aSize:SizeInt):SizeInt;inline;
begin
  if aSize mod 13 = 0 then exit(13);
  if aSize mod 11 = 0 then exit(11);
  if aSize mod 8  = 0 then exit(8);
  if aSize mod 7  = 0 then exit(7);
  if aSize mod 6  = 0 then exit(6);
  if aSize mod 5  = 0 then exit(5);
  if aSize mod 4  = 0 then exit(4);
  if aSize mod 3  = 0 then exit(3);
  if aSize mod 2  = 0 then exit(2);
  result := 1;
end;

//// Solve Bezout's identity
//// a * p + b * q = r = GCD(a, b)
procedure EuclidGCD( a, b: SizeInt; var p, q, r:SizeInt);inline;
var p_1, p_2, q_1, q_2, c:SizeInt;
begin
  p := 0;
  q := 1;
  p_1 := 1;
  q_1 := 0;
  while true do begin
    c := a mod b;
    if c = 0  then
      break;
    p_2 := p_1;
    q_2 := q_1;
    p_1 := p;
    q_1 := q;
    p := p_2 - p_1 * (a div b);
    q := q_2 - q_1 * (a div b);
    a := b;
    b := c;
  end;
  r := b;
end;

{ TNNOpenCL }

procedure TNNOpenCL.ActivateArray(const N: SizeInt; const x: cl_mem;
  const offset: SizeInt; const activation: longint; const events: TCLEvents;
  event: PCLEvent);
const kernelId = KERNEL_activate_array;
//var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opActivate);
  {$endif}
  if activation <> 4{longint(acLINEAR)} then begin
    SetGlobalWorkGroupSizes(N);
    SetGlobalOffsets(0);
    //NN:=LSize(N);
    //SetLocalWorkGroupSizes(NN);
    FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)          , @x);         CheckError();
    FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(offset)     , @offset);      CheckError();
    FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(activation) , @activation);   CheckError();
    FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
      , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
      , length(events), pointer(events), event); CheckError();

  end;
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opActivate);
  {$endif}
end;

procedure TNNOpenCL.activateArraySWISH(const N: SizeInt; const x: cl_mem;
  const offset: SizeInt; const output_sigmoid, output: cl_mem;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_array_avtivate_swish;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opActivate);
  {$endif}

  //NN:=LSize(N);
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)             , @x);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(offset)        , @offset);          CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(output_sigmoid), @output_sigmoid);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(output)        , @output);          CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId],
    WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
    , length(events), pointer(events), event); CheckError();
  //inc(eventsCount);
  //FErr := clFinish(ActiveQueue); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opActivate);
  {$endif}
end;

procedure TNNOpenCL.DeriveArray(const N: SizeInt; const x: cl_mem;
  const offset: SizeInt; const activation: longint; delta: cl_mem;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_gradient_array;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opDerive);
  {$endif}
  if activation <>4{longint(acLINEAR)} then begin
    SetGlobalWorkGroupSizes(N);
    SetGlobalOffsets(0);
    //NN:=LSize(N);
    //SetLocalWorkGroupSizes(NN);
    FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)          , @x);           CheckError();
    FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(offset)     , @offset);      CheckError();
    FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(activation) , @activation);  CheckError();
    FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(delta)      , @delta);       CheckError();
    FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
    , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0]
    , nil{@LocalWorkGroupSizes[0]}
    , length(events), pointer(events), event); CheckError();
  end;
  //inc(eventsCount);
  //FErr := clFinish(ActiveQueue); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opDerive);
  {$endif}
end;

procedure TNNOpenCL.forwardBias(const dstSize: SizeInt; const dst: cl_mem;
  const offset: SizeInt; const srcSize: SizeInt; const src: cl_mem;
  const incb: SizeInt; const batch: SizeInt; const events: TCLEvents;
  event: PCLEvent);
const kernelId=KERNEL_forward_bias;
var
    blockSize, NN, MM , i, k,  bOffset:SizeInt;
    reshape:integer;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opForwardBias);
  {$endif}
  //NN:=LSize(N);
  //MM:=LSize(blockSize);
  //writeln(N, ' ',batch, ' ', blocksize);
  blockSize := dstSize div (srcSize*batch);
  reshape := integer(blockSize > srcSize);
  if reshape=0 then
    SetGlobalWorkGroupSizes(srcSize, blockSize, batch)
  else
    SetGlobalWorkGroupSizes(blockSize, srcSize, batch);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(MM, NN);
  //SetGlobalWorkGroupSizes(N, blockSize);
  //SetLocalWorkGroupSizes(NN, MM);
  bOffset :=0;
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(reshape)  , @reshape);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(dst)      , @dst);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(Offset)   , @Offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(src)      , @src);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(bOffset)  , @bOffset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb)     , @incb);      CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opForwardBias);
  {$endif}

end;

procedure TNNOpenCL.backwardBias(const dstSize: SizeInt; const dst: cl_mem;
  const srcSize: SizeInt; const src: cl_mem; const srcOffset: SizeInt;
  const incb: SizeInt; const batch: SizeInt; const events: TCLEvents;
  event: PCLEvent);
const kernelId=KERNEL_backward_bias;
var blockSize :SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opBackwardBias);
  {$endif}
  blockSize := srcSize  div (dstSize*batch);
  SetGlobalWorkGroupSizes(dstSize);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(MM, NN);
  //SetGlobalWorkGroupSizes(N, blockSize);
  //SetLocalWorkGroupSizes(NN, MM);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(dst)        , @dst);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(blockSize)  , @blockSize); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(src)        , @src);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(srcOffset)  , @srcOffset);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(batch)      , @batch);     CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
    , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
    , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opBackwardBias);
  {$endif}

  //inc(eventsCount);
  //FErr := clFinish(ActiveQueue); CheckError();
end;

var hLib : {$if defined(MSWINDOWS)}HMODULE {$else}Pointer{$endif};
    {$if defined(MSWINDOWS)}
      {$ifdef FPC}
      getProc : function(Lib : HMODULE; ProcName : LPCSTR):pointer;WINAPI;
      {$else}
      getProc : function (hModule: HMODULE; lpProcName: LPCSTR): FARPROC; winapi;
      {$endif}
    {$else}
      getProc : function(h :pointer; name:PAnsiChar):pointer;WINAPI;
    {$endif}


procedure TNNOpenCL.gemm(const transA, transB: boolean; const M, N, K: SizeInt;
  const ALPHA: single; const A: cl_mem; const aOffset: SizeInt;
  const lda: SizeInt; const B: cl_mem; const bOffset: SizeInt;
  const ldb: SizeInt; const BETA: single; const C: cl_mem;
  const cOffset: SizeInt; const ldc: SizeInt; const events: TCLEvents;
  event: PCLEvent);
var MM, KK, NN :SizeInt; kernelId:integer;
label done;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opGemm);
  {$endif}
  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]

  //MM := LSize(M);
  //NN := LSize(N);


  if useBLAS = CL_LIB_BLAS then begin
      if not assigned(clBlasSGemm) then begin
        {$if defined(MSWINDOWS)}
          hLib := LoadLibrary('clBLAS.dll');
          assert(hLib<>0, 'Cannot load [clBLAS] library!');
          getProc := getProcAddress;
        {$elseif defined(ANDROID) or defined(LINUX)}
          hLib := dlopen('libclBLAS.so',RTLD_NOW);
          assert(assigned(hLib), 'Cannot load [libclBLAS] library!');
          getProc := dlsym;
        {$endif}
        clBlasSgemm:= getproc(hlib, 'clblasSgemm');
      end;
      clBlasSGemm(
        clblasRowMajor, TCLblasTranspose(transA), TCLblasTranspose(transB),
        M, N, K, alpha, A, aOffset, lda, B, bOffset, ldb, beta, C, cOffset, ldc, 1, @ActiveQueue, length(events), pointer(events), event
      );
      goto done
  end;

  if useBLAS = CL_LIB_BLAST then begin
      if not assigned(clBlastSGemm) then begin
        {$if defined(MSWINDOWS)}
          hLib := LoadLibrary('clblast.dll');
          assert(hLib<>0, 'Cannot load [clBLAST] library!');
          getProc := getProcAddress;
        {$elseif defined(ANDROID) or defined(LINUX)}
          hLib := dlopen('libclblast.so', RTLD_NOW);
          assert(assigned(hLib), 'Cannot load [libclBLAST] library!');
          getProc := dlsym;
        {$endif}
        CLBlastSgemm:= getproc(hlib, 'CLBlastSgemm');
      end;
      clBlastSGemm(
        CLBlastLayoutRowMajor, TCLBlastTranspose(111+ord(transA)), TCLBlastTranspose(111+ord(transB)),
        M, N, K, alpha, A, aOffset, lda, B, bOffset, ldb, beta, C, cOffset, ldc, @ActiveQueue, pointer(events)
      );
      goto done
  end;

  if (not transA) and (not transB)then
    if N > M then begin
      SetGlobalWorkGroupSizes(N, M);
      //SetLocalWorkGroupSizes(NN, MM);
      kernelId :=KERNEL_sgemm2_nn;
    end else begin
      SetGlobalWorkGroupSizes(M, N);
    //  SetLocalWorkGroupSizes(MM, NN);
      kernelId :=KERNEL_sgemm1_nn;
    end

  else if (not transA) and transB then begin
    SetGlobalWorkGroupSizes(M, N);
    kernelId :=KERNEL_sgemm1_nt;
  end else if transA and (not transB) then begin
    SetGlobalWorkGroupSizes(M, N);
    kernelId := KERNEL_sgemm1_tn ;
  end;

  FErr:=clSetKernelArg(Kernels[kernelId], 0, SizeOf(K)       , @K);CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 1, SizeOf(ALPHA)   , @ALPHA);CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 2, SizeOf(cl_mem)  , @A); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 3, SizeOf(aOffset) , @aOffset); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 4, SizeOf(lda)     , @lda); CheckError();

  FErr:=clSetKernelArg(Kernels[kernelId], 5, SizeOf(cl_mem)  , @B); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 6, SizeOf(bOffset) , @bOffset); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 7, SizeOf(ldb)     , @ldb); CheckError();

  FErr:=clSetKernelArg(Kernels[kernelId], 8, SizeOf(BETA)     , @BETA); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 9, SizeOf(cl_mem)   , @C); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 10, SizeOf(cOffset) , @cOffset); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 11, SizeOf(ldc)     , @ldc); CheckError();

  FErr:=clEnqueueNDRangeKernel(FQueue, Kernels[kernelId] ,WorkItemDimensions ,@GlobalOffsets[0]
    , @GlobalWorkGroupSizes[0] ,nil{@FLocalWorkGroupSizes[0]}
    , length(events), pointer(events), event); CheckError();
  //inc(eventsCount);

done:
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opGemm);
  {$endif}
end;

procedure TNNOpenCL.gemmStridedBatched(const transA, transB: boolean; const M,
  N, K: SizeInt; const ALPHA: single; const A: cl_mem; aOffset: SizeInt;
  const lda, strideA: SizeInt; const B: cl_mem; bOffset: SizeInt; const ldb,
  strideB: SizeInt; const BETA: single; const C: cl_mem; cOffset: SizeInt;
  const ldc, strideC: SizeInt; const batchCount: SizeInt;
  const events: TCLEvents; event: PCLEvent);
var MM, KK, NN :SizeInt; kernelId, i:integer;
label done;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opGemm);
  {$endif}
  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]

  //MM := LSize(M);
  //NN := LSize(N);


  if useBLAS = CL_LIB_BLAS then begin
      if not assigned(clBlasSGemm) then begin
        {$if defined(MSWINDOWS)}
          hLib := LoadLibrary('clBLAS.dll');
          assert(hLib<>0, 'Cannot load [clBLAS] library!');
          getProc := getProcAddress;
        {$elseif defined(ANDROID) or defined(LINUX)}
          hLib := dlopen('libclBLAS.so',RTLD_NOW);
          assert(assigned(hLib), 'Cannot load [libclBLAS] library!');
          getProc := dlsym;
        {$endif}
        clBlasSgemm:= getproc(hlib, 'clblasSgemm');
      end;
      // todo implement clblasSGemmStridedBatched
      for i:= 0 to batchCount-1 do begin;
        clBlasSGemm(
          clblasRowMajor, TCLblasTranspose(transA), TCLblasTranspose(transB),
          M, N, K, alpha, A, aOffset, lda, B, bOffset, ldb, beta, C, cOffset, ldc, 1, @ActiveQueue, length(events), pointer(events), event
        );
        inc(aOffset, strideA);
        inc(bOffset, strideB);
        inc(cOffset, strideC);
      end;
      goto done
  end;

  if useBLAS = CL_LIB_BLAST then begin
      if not assigned(clBlastSGemm) then begin
        {$if defined(MSWINDOWS)}
          hLib := LoadLibrary('clblast.dll');
          assert(hLib<>0, 'Cannot load [clBLAST] library!');
          getProc := getProcAddress;
        {$elseif defined(ANDROID) or defined(LINUX)}
          hLib := dlopen('libclblast.so', RTLD_NOW);
          assert(assigned(hLib), 'Cannot load [libclBLAST] library!');
          getProc := dlsym;
        {$endif}
        CLBlastSgemmStridedBatched:= getproc(hlib, 'CLBlastSgemmStridedBatched');
      end;
      CLBlastSgemmStridedBatched(
        CLBlastLayoutRowMajor, TCLBlastTranspose(111+ord(transA)), TCLBlastTranspose(111+ord(transB)),
        M, N, K, alpha, A, aOffset, lda, strideA, B, bOffset, ldb, strideB, beta, C, cOffset, ldc, strideC, batchCount, @ActiveQueue, pointer(events)
      );
    goto done
  end;

  if (not transA) and (not transB)then
    if N > M then begin
      SetGlobalWorkGroupSizes(N, M);
      //SetLocalWorkGroupSizes(NN, MM);
      kernelId :=KERNEL_sgemm2_nn;
    end else begin
      SetGlobalWorkGroupSizes(M, N);
    //  SetLocalWorkGroupSizes(MM, NN);
      kernelId :=KERNEL_sgemm1_nn;
    end

  else if (not transA) and transB then begin
    SetGlobalWorkGroupSizes(M, N);
    kernelId := KERNEL_sgemm1_nt;
  end else if transA and (not transB) then begin
    SetGlobalWorkGroupSizes(M, N);
    kernelId := KERNEL_sgemm1_tn ;
  end;

  FErr:=clSetKernelArg(Kernels[kernelId], 0, SizeOf(K)       , @K);CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 1, SizeOf(ALPHA)   , @ALPHA);CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 2, SizeOf(cl_mem)  , @A); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 4, SizeOf(lda)     , @lda); CheckError();

  FErr:=clSetKernelArg(Kernels[kernelId], 5, SizeOf(cl_mem)  , @B); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 7, SizeOf(ldb)     , @ldb); CheckError();

  FErr:=clSetKernelArg(Kernels[kernelId], 8, SizeOf(BETA)     , @BETA); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 9, SizeOf(cl_mem)   , @C); CheckError();
  FErr:=clSetKernelArg(Kernels[kernelId], 11, SizeOf(ldc)     , @ldc); CheckError();
  for i:=0 to batchCount-1 do begin
    FErr:=clSetKernelArg(Kernels[kernelId], 3, SizeOf(aOffset) , @aOffset); CheckError();
    FErr:=clSetKernelArg(Kernels[kernelId], 6, SizeOf(bOffset) , @bOffset); CheckError();
    FErr:=clSetKernelArg(Kernels[kernelId], 10, SizeOf(cOffset) , @cOffset); CheckError();

    FErr:=clEnqueueNDRangeKernel(FQueue, Kernels[kernelId] ,WorkItemDimensions ,@GlobalOffsets[0]
      , @GlobalWorkGroupSizes[0] ,nil{@FLocalWorkGroupSizes[0]}
      , length(events), pointer(events), event); CheckError();
    inc(aOffset, strideA);
    inc(bOffset, strideB);
    inc(cOffset, strideC);
  end;
  //inc(eventsCount);

done:
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opGemm);
  {$endif}
end;

procedure TNNOpenCL.addvv(const N: SizeInt; const src1: cl_mem;
  const src1Offset, inca: SizeInt; const src2: cl_mem; const src2Offset,
  incb: SizeInt; dst: cl_mem; const dstOffset, incc: SizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_addv;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opAddvv);
  {$endif}
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(src1)       , @src1           );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(src1Offset) , @src1Offset     );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(inca)       , @inca           );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(src2)       , @src2           );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(src2Offset) , @src2Offset     );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb)       , @incb           );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(dst)        , @dst            );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7, SizeOf(dstOffset)  , @dstOffset      );   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8, SizeOf(incc)       , @incc           );   CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opAddvv);
{$endif}
end;

procedure TNNOpenCL.subvv(const N: SizeInt; const src1: cl_mem;
  const src1Offset, inca: SizeInt; const src2: cl_mem; const src2Offset,
  incb: SizeInt; dst: cl_mem; const dstOffset, incc: SizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_subv;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opSubvv);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(src1) , @src1);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(src1Offset) , @src1Offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(inca) , @inca);              CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(src2) , @src2);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(src2Offset) , @src2Offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb) , @incb);              CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(dst)  , @dst);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7, SizeOf(dstOffset) , @dstOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8, SizeOf(incc) , @incc);              CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
 {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opSubvv);
 {$endif}

end;

procedure TNNOpenCL.mulvv(const N: SizeInt; const src1: cl_mem;
  const src1Offset, inca: SizeInt; const src2: cl_mem; const src2Offset,
  incb: SizeInt; dst: cl_mem; const dstOffset, incc: SizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_mulv;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMulvv);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(src1) , @src1);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(src1Offset) , @src1Offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(inca) , @inca);              CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(src2) , @src2);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(src2Offset) , @src2Offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb) , @incb);              CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(dst)  , @dst);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7, SizeOf(dstOffset) , @dstOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8, SizeOf(incc) , @incc);              CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
 {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMulvv);
 {$endif}
end;

procedure TNNOpenCL.fmavv(const N: SizeInt;
  const src1: cl_mem; const src1Offset, inca: SizeInt;
  const src2: cl_mem; const src2Offset, incb: SizeInt;
  const src3: cl_mem; const src3Offset, incc: SizeInt;
  dst: cl_mem; const dstOffset, incd: SizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_fmav;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opFmavv);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(src1)       , @src1);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(src1Offset) , @src1Offset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(inca)       , @inca);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3 , SizeOf(src2)       , @src2);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4 , SizeOf(src2Offset) , @src2Offset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5 , SizeOf(incb)       , @incb);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6 , SizeOf(src3)       , @src3);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7 , SizeOf(src3Offset) , @src3Offset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8 , SizeOf(incc)       , @incc);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 9 , SizeOf(dst)        , @dst);        CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 10, SizeOf(dstOffset)  , @dstOffset);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 11, SizeOf(incd)       , @incd);       CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
 {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opFmavv);
 {$endif}
end;

procedure TNNOpenCL.axpy(const N: SizeInt; const a: single; const x: cl_mem;
  const xOffset: SizeInt; const incx: SizeInt; const y: cl_mem;
  const yOffset: SizeInt; const incy: sizeInt; const events: TCLEvents;
  event: PCLEvent);
const kernelId = KERNEL_axpy;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opAxpy);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)    , @a);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(x)    , @x);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(xOffset)    , @xOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(incx) , @incx); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(y)    , @y);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(yOffset)    , @yOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(incy) , @incy); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opAxpy);
  {$endif}
end;

procedure TNNOpenCL.power(const N: SizeInt; const x: cl_mem;
  const xOffset: SizeInt; const incx: SizeInt; const a: single;
  const y: cl_mem; const yOffset: SizeInt; const incy: sizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_power;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opPow);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)       , @x);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(xOffset) , @xOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(incx)    , @incx);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(a)       , @a); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(y)       , @y);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(yOffset) , @yOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(incy)    , @incy); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opPow);
  {$endif}
end;

procedure TNNOpenCL.scale(const N: SizeInt; const a: Single; const x: cl_mem;
  const stride: SizeInt; const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_scale;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMulvs);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)    , @a);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(x)    , @x);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(stride) , @stride); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMulvs);
  {$endif}

end;

procedure TNNOpenCL.crossEntropyLogistic(const N: SizeInt; const pred,
  truth: cl_mem; delta, error: cl_mem; const events: TCLEvents; event: PCLEvent
  );
const kernelId = KERNEL_crossEntropyLogistics;
var NN:SizeInt;
begin

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(pred)    , @pred);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(truth)   , @truth); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(delta)   , @delta); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(error)   , @error); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
end;

procedure TNNOpenCL.fill(const N: SizeInt; const x: cl_mem; const offset:SizeInt; const val: single;
  const stride: SizeInt; const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_fill;
var NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opFill);
  {$endif}

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)     , @x);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(offset), @offset);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(val)   , @val); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(stride), @stride); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opFill);
  {$endif}

end;

procedure TNNOpenCL.copy(const N: SizeInt; const src: cl_mem; const srcOffset,
  inca: SizeInt; const dst: cl_mem; const dstOffset, incb: SizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_copy;
var NN:SizeInt; sz : SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opCopy);
  {$endif}
  //sz := sizeof(single);
  //clEnqueueCopyBuffer(ActiveQueue, src, dst, srcOffset*sz, dstOffset*sz, N*sz
  //, length(events), pointer(events), event); CheckError();
  //{$ifdef USE_TELEMETRY}
  //finish();
  //tensorMetrics.finish(opCopy);
  //{$endif}
  //exit;

  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(src)       , @src);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(srcOffset) , @srcOffset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(inca)      , @inca);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(dst)       , @dst);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(dstOffset) , @dstOffset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb)      , @incb);      CheckError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opCopy);
  {$endif}
end;

procedure TNNOpenCL.softmaxBatch(const N: SizeInt; const input: cl_mem;
  const iOffset: SizeInt; const batch, batch_size, groups, group_size,
  stride: SizeInt; const temp: single; const output: cl_mem;
  const oOffset: SizeInt; const events: TCLEvents; event: PCLEvent);
const kernelId = KERNEL_softmaxBatch;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(batch, groups);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(input)       , @input);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(iOffset)     , @iOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(N)           , @N);          CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(batch_size)  , @batch_size); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(group_size)  , @group_size); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(stride)      , @stride);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(temp)        , @temp);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7, SizeOf(output)      , @output);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8, SizeOf(oOffset)     , @oOffset);    CheckError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
end;

procedure TNNOpenCL.crossEntropySoftmax(const N: SizeInt; const pred,
  truth: cl_mem; delta, error: cl_mem; const events: TCLEvents; event: PCLEvent
  );
const kernelId =  KERNEL_crossEntropySoftmax;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(pred)    , @pred);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(truth)   , @truth); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(delta)   , @delta); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(error)   , @error); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
end;

procedure TNNOpenCL.forwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  const input: cl_mem; const c, h, w: SizeInt; const stride_x, stride_y,
  padding, kernelSize: SizeInt; indexes, output: cl_mem;
  const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_forward_maxpool;
var NN, MM:SizeInt;
begin
  SetGlobalWorkGroupSizes(ABatch * outC, outH, outW);
  SetGlobalOffsets(0);
  //NN := LSize(ABatch * outC);
  //MM := LSize(outH* outW);
  //SetLocalWorkGroupSizes(NN, MM);

  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(input)      , @input);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(c)          , @c);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(h)          , @h);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3 , SizeOf(w)          , @w);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4 , SizeOf(stride_x)   , @stride_x);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5 , SizeOf(stride_y)   , @stride_y);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6 , SizeOf(padding)    , @padding);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7 , SizeOf(kernelSize) , @kernelSize); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8 , SizeOf(indexes)    , @indexes);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 9 , SizeOf(output)     , @output);     CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId], WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}, length(events), pointer(events), event); CheckError();
end;

procedure TNNOpenCL.backwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  output: cl_mem; const indexes, delta: cl_mem; const events: TCLEvents;
  event: PCLEvent);
const kernelId =  KERNEL_backward_maxpool;
var
    NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(ABatch*outC, outH*outW);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(output)  , @output);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(indexes) , @indexes); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(delta)   , @delta);   CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
end;

//procedure TNNOpenCL.im2col(const aChannels, aHeight, aWidth, kernelHeight,
//  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
//  dilationX: SizeInt; const im: cl_mem; const imOffset: SizeInt;
//  const col: cl_mem; const colOffset: SizeInt; const batch: SizeInt;
//  const events: TCLEvents; event: PCLEvent);
//const kernelId = 20;
//var
//    NN:SizeInt;
//begin
//  SetGlobalWorkGroupSizes(aChannels, kernelHeight* kernelWidth);
//  SetGlobalOffsets(0);
//  //NN:=LSize(N);
//  //SetLocalWorkGroupSizes(NN);
//  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(aHeight)         , @aHeight);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(aWidth)          , @aWidth); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(kernelHeight)    , @kernelHeight);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 3 , SizeOf(kernelWidth)     , @kernelWidth);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 4 , SizeOf(padHeight)       , @padHeight); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 5 , SizeOf(padWidth)        , @padWidth);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 6 , SizeOf(strideY)         , @strideY);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 7 , SizeOf(strideX)         , @strideX); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 8 , SizeOf(dilationY)       , @dilationY);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 9 , SizeOf(dilationX)       , @dilationX);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 10, SizeOf(im)              , @im); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 11, SizeOf(imOffset)        , @imOffset);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 12, SizeOf(col)             , @col);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 13, SizeOf(colOffset)       , @colOffset); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 14, SizeOf(batch)           , @batch);   CheckError();
//  FErr := clEnqueueNDRangeKernel(
//     ActiveQueue, Kernels[kernelId],
//     WorkItemDimensions, @GlobalOffsets[0],
//     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
//     , length(events), pointer(events), event); CheckError();
//end;

function ceil(const a, b: SizeInt):SizeInt;  overload;
begin
  result := (1 + (a-1) div b)*b
end;

const COPY_DIMX = 8;
      COPY_DIMY = 8;

procedure TNNOpenCL.im2col(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const im: cl_mem; const imOffset: SizeInt;
  const col: cl_mem; const colOffset: SizeInt; const events: TCLEvents;
  event: PCLEvent);
const kernelId =  KERNEL_Xim2colKernelNormal;
var
    NN, size_h, padding_h, col_h, padding_w, size_w, col_w, ceiled_w,
      ceiled_h:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opIm2col);
  {$endif}


  // Sets the height and width of the 'col' result
  size_h      := aHeight + 2 * padHeight;
  padding_h   := dilationY * (kernelHeight - 1) + 1;
  if size_h >= padding_h then
    col_h       := (size_h - padding_h) div strideY + 1
  else
    col_h       := 1;
  size_w      := aWidth + 2 * padWidth;
  padding_w   := dilationX * (kernelWidth - 1) + 1;
  if size_w >= padding_w then
    col_w       := (size_w - padding_w) div strideX + 1
  else
    col_w       := 1;
  ceiled_w    := Ceil(col_w, COPY_DIMX);
  ceiled_h    := Ceil(col_h, COPY_DIMY);

  SetGlobalWorkGroupSizes(ceiled_w, ceiled_h * aChannels);
  SetGlobalOffsets(0);

  // Sets the kernel arguments
  FErr := clSetKernelArg(Kernels[kernelId], 0,  sizeOf(aHeight       ), @aHeight       ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 1,  sizeOf(aWidth        ), @aWidth        ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 2,  sizeOf(aChannels     ), @aChannels     ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 3,  sizeOf(col_h         ), @col_h         ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 4,  sizeOf(col_w         ), @col_w         ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 5,  sizeOf(kernelHeight  ), @kernelHeight  ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 6,  sizeOf(kernelWidth   ), @kernelWidth   ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 7,  sizeOf(padHeight     ), @padHeight     ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 8,  sizeOf(padWidth      ), @padWidth      ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 9,  sizeOf(strideY       ), @strideY       ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 10, sizeOf(strideX       ), @strideX       ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 11, sizeOf(dilationY     ), @dilationY     ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 12, sizeOf(dilationX     ), @dilationX     ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 13, sizeOf(im            ), @im            ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 14, sizeOf(imOffset      ), @imOffset      ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 15, sizeOf(col           ), @col           ); checkError();
  FErr := clSetKernelArg(Kernels[kernelId], 16, sizeOf(colOffset     ), @colOffset     ); checkError();

  // Launches the kernel
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opIm2col);
  {$endif}

end;

procedure TNNOpenCL.col2im(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const col: cl_mem; const colOffset: SizeInt;
  const im: cl_mem; const imOffset: SizeInt; const events: TCLEvents;
  event: PCLEvent);
const kernelId =  KERNEL_Xcol2imKernelNormal;
var
    NN, size_h, padding_h, col_h, size_w, padding_w, col_w,
      dilation_bez_h, dilation_bez_w, gcd_h, gcd_w, stride_bez_h,
      stride_bez_w, w_ceiled, h_ceiled:SizeInt;

begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opCol2im);
  {$endif}

  //SetGlobalWorkGroupSizes(aChannels{, kernelHeight, kernelWidth});

  size_h          := AHeight + 2 * padHeight;
  padding_h       := dilationY * (kernelHeight - 1) + 1;
  if size_h >= padding_h then
    col_h           := (size_h - padding_h) div strideY + 1
  else
    col_h           := 1;
  size_w          := AWidth + 2 * padWidth;
  padding_w       := dilationX * (kernelWidth - 1) + 1;
  if size_w >= padding_w then
    col_w           := (size_w - padding_w) div strideX + 1
  else
    col_w           := 1;
  stride_bez_h    := 0;
  stride_bez_w    := 0;
  dilation_bez_h  := 0;
  dilation_bez_w  := 0;
  gcd_h           := 0;
  gcd_w           := 0;
  EuclidGCD(strideY, dilationY, stride_bez_h, dilation_bez_h, gcd_h);
  EuclidGCD(strideX, dilationX, stride_bez_w, dilation_bez_w, gcd_w);

  w_ceiled := Ceil((aWidth - 1) div gcd_w + 1, COPY_DIMX);
  h_ceiled := Ceil((aHeight - 1) div gcd_h + 1, COPY_DIMY);
  //const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
  SetGlobalWorkGroupSizes(w_ceiled, h_ceiled * aChannels);

  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(kernels[kernelId], 0,  sizeof(aHeight       ), @aHeight         ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 1,  sizeof(aWidth        ), @aWidth          ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 2,  sizeof(aChannels     ), @aChannels       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 3,  sizeof(col_h         ), @col_h           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 4,  sizeof(col_w         ), @col_w           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 5,  sizeof(kernelHeight  ), @kernelHeight    ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 6,  sizeof(kernelWidth   ), @kernelWidth     ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 7,  sizeof(padHeight     ), @padHeight       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 8,  sizeof(padWidth      ), @padWidth        ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 9,  sizeof(strideY       ), @strideY         ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 10, sizeof(strideX       ), @strideX         ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 11, sizeof(dilationY     ), @dilationY       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 12, sizeof(dilationX     ), @dilationX       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 13, sizeof(stride_bez_h  ), @stride_bez_h    ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 14, sizeof(stride_bez_w  ), @stride_bez_w    ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 15, sizeof(dilation_bez_h), @dilation_bez_h  ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 16, sizeof(dilation_bez_w), @dilation_bez_w  ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 17, sizeof(gcd_h         ), @gcd_h           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 18, sizeof(gcd_w         ), @gcd_w           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 19, sizeof(col           ), @col             ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 20, sizeof(colOffset     ), @colOffset       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 21, sizeof(im            ), @im              ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 22, sizeof(imOffset      ), @imOffset        ); checkError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opCol2im);
  {$endif}

end;

procedure TNNOpenCL.upSample(const aBatch, aChannels, outHeight,
  outWidth: SizeInt; const &in: cl_mem; const stride: SizeInt;
  const isForward: longint; const scale: single; const &out: cl_mem;
  const zeroIn: boolean; const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_upsample;
var
    NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(aBatch*aChannels, outHeight*stride, outWidth*stride);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(&in)      , @&in);        CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(stride)   , @stride);     CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(isForward), @isForward);  CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(scale)    , @scale);      CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(&out)     , @&out);       CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 5, sizeOf(Integer)  , @zeroIn);     CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
end;

procedure TNNOpenCL.fmavss(const N: SizeInt; const src: cl_mem;
  const offset: SizeInt; const scalar, bias: single; dst: cl_mem;
  const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_fmavss;
var
    NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opFmavss);
  {$endif}
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(src)    , @src);       CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(offset) , @offset);    CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(scalar) , @scalar);  CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(bias)   , @bias);    CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(dst)    , @dst);     CheckError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opFmavss);
  {$endif}
end;

procedure TNNOpenCL.meansAndVars(const srcSize, dstSize, groups: sizeInt;
  const src: cl_mem; const offset: sizeInt; means, vars: cl_mem;
  const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_means_vars;
var
  blockSize,  NN:SizeInt;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMeansVars);
  {$endif}

  blockSize := srcSize div (dstSize*groups);
  SetGlobalWorkGroupSizes(dstSize*OCL_BLOCK);
  SetLocalWorkGroupSizes(OCL_BLOCK);

  SetGlobalOffsets(0);
  //NN:=LSize(N);
  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(blockSize)   , @blockSize); CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(groups)      , @groups);    CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(src)         , @src);       CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(offset)      , @offset);    CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(means)       , @means);     CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 5, sizeOf(vars)        , @vars);      CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], @LocalWorkGroupSizes[0]
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMeansVars);
  {$endif}

end;

procedure TNNOpenCL.normalize(const srcSize, dstSize, groups: SizeInt;
  means: cl_mem; const meansStride: sizeInt; vars: cl_mem;
  const varsStride: SizeInt; dst: cl_mem; const dstOffset: sizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_normblkvv;
var
  blockSize,  NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opNormalize);
  {$endif}

  blockSize := dstSize div (srcSize*groups);
  SetGlobalWorkGroupSizes(srcSize, blocksize, groups);

  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(means)       , @means);       CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(meansStride) , @meansStride); CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(vars)        , @vars);        CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(varsStride)  , @varsStride);  CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(dst)         , @dst);         CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 5, sizeOf(dstOffset)   , @dstOffset);   CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opNormalize);
  {$endif}

end;

procedure TNNOpenCL.meansAndVarsDelta(const srcSize, dstSize, groups: SizeInt;
  delta, x: cl_mem; const offset: SizeInt; mean, variance, mean_delta,
  variance_delta: cl_mem; const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_means_vars_delta;
var
    blockSize, NN:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMeansVarsDelta);
  {$endif}


  blockSize := srcSize div (dstSize * groups);
  SetGlobalWorkGroupSizes(dstSize*OCL_BLOCK);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  SetLocalWorkGroupSizes(OCL_BLOCK);
  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(groups)         , @groups);          CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(blockSize)      , @blockSize);       CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(delta)          , @delta);           CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(x)              , @x);               CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(offset)         , @offset);          CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 5, sizeOf(mean)           , @mean);            CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 6, sizeOf(variance)       , @variance);        CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 7, sizeOf(mean_delta)     , @mean_delta);      CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 8, sizeOf(variance_delta) , @variance_delta);  CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], @LocalWorkGroupSizes[0]
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMeansVarsDelta);
  {$endif}

end;

procedure TNNOpenCL.normalizeDelta(const deltaSize, meanSize, groups: SizeInt;
  const delta, x: cl_mem; const offset: SizeInt; mean, variance, mean_delta,
  variance_delta: cl_mem; const events: TCLEvents; event: PCLEvent);
const kernelId =  KERNEL_norm_delta;
var
    blockSize, NN:SizeInt;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opNormalizeDelta);
  {$endif}

  blockSize := deltaSize div (meanSize * groups);
  SetGlobalWorkGroupSizes(meanSize, blockSize, groups);
  SetGlobalOffsets(0);
  //NN:=LSize(deltaSize);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(x)              , @x);               CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(offset)         , @offset);          CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(mean)           , @mean);            CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(variance)       , @variance);        CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(mean_delta)     , @mean_delta);      CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 5, sizeOf(variance_delta) , @variance_delta);  CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 6, sizeOf(delta)          , @delta);           CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opNormalizeDelta);
  {$endif}

end;

procedure TNNOpenCL.addDots(const N, nDst, groups: SizeInt; const src1,
  src2: cl_mem; const srcOffset: SizeInt; dst: cl_mem; const events: TCLEvents;
  event: PCLEvent);
const kernelId =  KERNEL_add_dots;
var
    blockSize, NN:SizeInt;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opAddDots);
  {$endif}

  blockSize := N div (nDst * groups);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  SetGlobalWorkGroupSizes(nDst*OCL_BLOCK);
  SetLocalWorkGroupSizes(OCL_BLOCK);

  FErr := clSetKernelarg(kernels[kernelId], 0, sizeOf(groups)     , @groups);    CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 1, sizeOf(blocksize)  , @blocksize); CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 2, sizeOf(src1)       , @src1);      CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 3, sizeOf(src2)       , @src2);      CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 4, sizeOf(srcOffset)  , @srcOffset); CheckError();
  FErr := clSetKernelarg(kernels[kernelId], 5, sizeOf(dst)        , @dst);       CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], @LocalWorkGroupSizes[0]
     , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opAddDots);
  {$endif}

end;

procedure TNNOpenCL.forwardScale(const dstSize: SizeInt; const dst: cl_mem;
  const offset: SizeInt; const scaleSize: SizeInt; const scale: cl_mem;
  const incb: SizeInt; const batch: SizeInt; const events: TCLEvents;
  event: PCLEvent);
const kernelId = KERNEL_forward_scale;
var
    blockSize, NN, MM , i, k, bOffset:SizeInt;
    reshape:integer;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opForwardScale);
  {$endif}

  blockSize := dstSize div (scaleSize*batch);
  //NN:=LSize(N);
  //MM:=LSize(blockSize);
  //writeln(N, ' ',batch, ' ', blocksize);
  reshape := integer(blockSize > scaleSize);
  if reshape>0 then
    SetGlobalWorkGroupSizes(blockSize, scaleSize, batch)
  else
    SetGlobalWorkGroupSizes(scaleSize, blockSize, batch);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(MM, NN);
  //SetGlobalWorkGroupSizes(N, blockSize);
  //SetLocalWorkGroupSizes(NN, MM);
  bOffset :=0;
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(reshape)  , @reshape);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(dst)      , @dst);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(Offset)   , @Offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(scale)    , @scale);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(bOffset)  , @bOffset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb)     , @incb);      CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opForwardScale);
  {$endif}

end;

procedure TNNOpenCL.forwardScaleAdd(const dstSize: SizeInt; const dst: cl_mem;
  const offset: SizeInt; const scaleSize: SizeInt; const scales,
  biases: cl_mem; const incb: SizeInt; const batch: SizeInt;
  const events: TCLEvents; event: PCLEvent);
const kernelId= KERNEL_forward_scale_add;
var
    NN, MM , i, k, bOffset, blockSize:SizeInt;
    reshape:integer;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opForwardScaleAdd);
  {$endif}

  //NN:=LSize(N);
  //MM:=LSize(blockSize);
  //writeln(N, ' ',batch, ' ', blocksize);
  blockSize := dstSize div (scaleSize*batch);
  //NN:=LSize(N);
  //MM:=LSize(blockSize);
  //writeln(N, ' ',batch, ' ', blocksize);
  reshape := integer(blockSize > scaleSize);
  if reshape>0 then
    SetGlobalWorkGroupSizes(blockSize, scaleSize, batch)
  else
    SetGlobalWorkGroupSizes(scaleSize, blockSize, batch);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(MM, NN);
  //SetGlobalWorkGroupSizes(dstSize, scaleSize);
  //SetLocalWorkGroupSizes(NN, MM);

  bOffset :=0;
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(reshape)  , @reshape);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(dst)      , @dst);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(offset)   , @offset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(scales)   , @scales);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(biases)   , @biases);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(bOffset)  , @bOffset);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(incb)     , @incb);     CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opForwardScaleAdd);
  {$endif}

end;

procedure TNNOpenCL.forwardDropout(const N: SizeInt; const src: cl_mem;
  const probability, scale: single; rnd: cl_mem; dst: cl_mem;
  const events: TCLEvents; event: PCLEvent);

const kernelId= KERNEL_forward_dropout;
var
    NN, MM :SizeInt;
    seed : longword;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opFwDropout);
  {$endif}
  //NN:=LSize(N);
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(NN);

  //randomize;
  seed := random($100000); // next randSeed

  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(Seed)        , @Seed);        CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(probability) , @probability); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(scale)       , @scale);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(src)         , @src);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(rnd)         , @rnd);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(dst)         , @dst);         CheckError();

  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opFwDropout);
  {$endif}

end;

procedure TNNOpenCL.backwardDropout(const N: SizeInt; const src: cl_mem;
  const probability, scale: single; const rnd: cl_mem; dst: cl_mem;
  const events: TCLEvents; event: PCLEvent);
const kernelId= KERNEL_backward_dropout;
var
    NN, MM :SizeInt;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opBwDropout);
  {$endif}
  //NN:=LSize(N);
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(NN);

  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(probability) , @probability); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(scale)       , @scale);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(src)         , @src);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(rnd)         , @rnd);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(dst)         , @dst);         CheckError();

  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opBwDropout);
  {$endif}

end;

procedure TNNOpenCL.costL2(const N: SizeInt; const pred, truth, delta,
  error: cl_mem; const events: TCLEvents; event: PCLEvent);
const kernelId= KERNEL_cost_l2;
var
    NN, MM :SizeInt;
begin

  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opL2);
  {$endif}
  //NN:=LSize(N);
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(pred)  , @pred);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(truth) , @truth);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(delta) , @delta);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(error) , @error);  CheckError();

    FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , length(events), pointer(events), event); CheckError();

  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opL2);
  {$endif}

end;


//procedure TNNOpenCL.halfTest(const N: SizeInt; a: cl_mem; b: cl_mem; c: cl_mem;
//  const events: TCLEvents; event: PCLEvent);
//const kernelId=41;
//var
//    NN, MM :SizeInt;
//begin
//
//  //NN:=LSize(N);
//  SetGlobalWorkGroupSizes(N);
//  SetGlobalOffsets(0);
//  //SetLocalWorkGroupSizes(NN);
//  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a) , @a);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(b) , @b);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(c) , @c);  CheckError();
//
//    FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
//  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
//  , length(events), pointer(events), event); CheckError();
//  finish()
//
//
//end;

end.
