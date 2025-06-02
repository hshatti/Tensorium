unit nnCuda;

{$ifdef FPC}
{$mode Delphi}
{$PackRecords C}
{$ModeSwitch advancedrecords}
{$ModeSwitch typehelpers}
{$endif}

interface

uses
  SysUtils, cudart, cudarttypes, cublas_api, nvrtc, nOpMetrics;

type
  TCUMem = pointer;

  TCudaParams = array[0..15] of pointer;

  { TCudaParamsHelper }

  TCudaParamsHelper = record helper for TCudaParams
    constructor Create(const p0 : pointer ; p1 : pointer = nil; p2 : pointer = nil; p3 : pointer = nil; p4 : pointer = nil; p5 : pointer = nil; p6 : pointer = nil; p7 : pointer = nil; p8 : pointer = nil; p9 : pointer = nil; p10 : pointer = nil; p11 : pointer = nil; p12 : pointer = nil; p13 : pointer = nil; p14 : pointer = nil; p15 : pointer = nil);
  end;

  { TNNCuda }

  TNNCuda = class
  private
    FJitOptions : TArray<cudaJitOption>;
    FJitOptionsValues : TArray<pointer>;
    FLibOptions : TArray<cudaLibraryOption>;
    FLibOptionsValues : TArray<pointer>;
    FKernels : TArray<cudaKernel_t>;
    function GetjitOptions(jo: cudaJitOption): pointer;
    function GetlibraryOptions(lo: cudaLibraryOption): pointer;
    procedure SetjitOptions(jo: cudaJitOption; AValue: pointer);
    procedure SetlibraryOptions(lo: cudaLibraryOption; AValue: pointer);
  public
    stream : cudaStream_t;
    properties : cudaDeviceProp;
    prog : nvrtcProgram;
    nnLib : cudaLibrary_t;
    useBLAS : integer;
    cublas: cublasHandle_t;
    property jitOptions[jo:cudaJitOption] : pointer read GetjitOptions write SetjitOptions;
    property libraryOptions[jo:cudaLibraryOption] : pointer read GetlibraryOptions write SetlibraryOptions;
    class function deviceCount() : longint;
    constructor Create(deviceIndex: longint = 0);
    destructor Destroy();  override;
    function CompileLog:ansistring;
    function createDeviceBuffer(const N:SizeInt):TCUMem;
    procedure freeDeviceBuffer(var cudaMem:TCUMem);
    procedure readBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer:pointer);
    procedure writeBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
    procedure ActivateArray(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const activation: longint);
    procedure activateArraySWISH(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const output_sigmoid, output: TCUMem);
    procedure DeriveArray(const N: SizeInt; const x: TCUMem; const offset:SizeInt; const activation: longint; delta: TCUMem);
    procedure forwardBias(const dstSize: SizeInt; const dst: TCUMem; const offset:SizeInt; const srcSize: SizeInt; const src: TCUMem; const incb: SizeInt; const batch: SizeInt);
    procedure backwardBias(const dstSize: SizeInt; const dst: TCUMem; const srcSize: SizeInt; const src: TCUMem; const srcOffset:SizeInt; const incb: SizeInt ; const batch: SizeInt);
    procedure gemm(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:single; const A:PSingle; const aOffset:SizeInt; lda:SizeInt; const B:PSingle; const bOffset:SizeInt; ldb:SizeInt; const BETA: single; const C:PSingle; const cOffset:SizeInt; ldc:SizeInt; const workspace: TCUMem = nil; const workspaceSize:SizeInt = 0);
    procedure addvv(const N:SizeInt; const src1:TCUMem; const src1Offset, inca:SizeInt; const src2:TCUMem; const src2Offset, incb:SizeInt; dst:TCUMem; const dstOffset, incc:SizeInt);
    procedure subvv(const N:SizeInt; const src1:TCUMem; const src1Offset, inca:SizeInt; const src2:TCUMem; const src2Offset, incb:SizeInt; dst:TCUMem; const dstOffset, incc:SizeInt);
    procedure mulvv(const N:SizeInt; const src1:TCUMem; const src1Offset, inca:SizeInt; const src2:TCUMem; const src2Offset, incb:SizeInt; dst:TCUMem; const dstOffset, incc:SizeInt);
    procedure fmavv(const N: SizeInt; const src1: TCUMem; const src1Offset, inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt; const src3: TCUMem; const src3Offset, incc: SizeInt; dst: TCUMem; const dstOffset, incd: SizeInt);
    procedure axpy(const N:SizeInt; const a:single; const x:TCUMem; const xOffset:SizeInt; const incx:SizeInt; const y:TCUMem; const yOffset:SizeInt; const incy:sizeInt);
    procedure power(const N:SizeInt; const x:TCUMem; const xOffset:SizeInt; const incx:SizeInt; const a:single; const y:TCUMem; const yOffset:SizeInt; const incy:sizeInt);
    procedure scale(const N:SizeInt; const a:Single; const x:TCUMem; const stride:SizeInt);
    procedure crossEntropyLogistic(const N:SizeInt; const pred, truth: TCUMem; delta, error: TCUMem);
    procedure fill(const N:SizeInt; const x: TCUMem; const offset:SizeInt; const val:single; const stride :SizeInt);
    procedure copy(const N:SizeInt; const src:TCUMem; const srcOffset, inca:SizeInt; const dst:TCUMem; const dstOffset, incb:SizeInt);
    procedure softmaxBatch(const N: SizeInt; const input: TCUMem; const iOffset: SizeInt; const batch, batch_size, groups, group_size, stride: SizeInt; const temp: single; const output: TCUMem; const oOffset: SizeInt);
    procedure crossEntropySoftmax(const N:SizeInt; const pred, truth: TCUMem; delta, error: TCUMem);
    procedure forwardMaxPool(const aBatch, outC, outH, outW: SizeInt; const input: TCUMem; const c, h, w: SizeInt; const stride_x, stride_y, padding, kernelSize: SizeInt; indexes, output: TCUMem);
    procedure backwardMaxPool(const aBatch, outC, outH, outW : SizeInt; output:TCUMem; const indexes, delta : TCUMem);
    procedure im2col(const aChannels, aHeight, aWidth , kernelHeight, kernelWidth, padHeight, padWidth , strideY, strideX, dilationY, dilationX : SizeInt ; const im :TCUMem; const imOffset : SizeInt ; const col:TCUMem; const colOffset:SizeInt);
    procedure col2im(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY, dilationX: SizeInt; const col: TCUMem; const colOffset: SizeInt; const im: TCUMem; const imOffset: SizeInt);
    procedure upSample(const aBatch, aChannels, outHeight, outWidth: SizeInt; const &in: TCUMem;const stride: SizeInt; const isForward: longint; const scale: single; const &out: TCUMem; const zeroIn :boolean = false);
    procedure fmavss(const N: SizeInt; const src: TCUMem; const offset: SizeInt; const scalar, bias: single; dst : TCUMem);
    procedure meansAndVars(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means, vars:TCUMem);
    procedure means(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means:TCUMem);
    procedure variances(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means, vars:TCUMem);
    procedure normalize(const srcSize, dstSize, groups:SizeInt; means:TCUMem; const meansStride:sizeInt; vars:TCUMem; const varsStride:SizeInt; dst:TCUMem; const dstOffset :sizeInt);
    procedure meansAndVarsDelta(const srcSize, dstSize, groups:SizeInt; delta, x: TCUMem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: TCUMem);
    procedure normalizeDelta(const deltaSize, meanSize, groups: SizeInt; const delta, x: TCUMem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: TCUMem);
    procedure addDots(const N, dstSize, groups:SizeInt; const src1, src2:TCUMem; const srcOffset:SizeInt; dst:TCUMem);
    procedure forwardScale(const dstSize: SizeInt; const dst: TCUMem; const offset :SizeInt; const scaleSize: SizeInt; const scale: TCUMem; const incb: SizeInt ; const batch: SizeInt);
    procedure forwardScaleAdd(const dstSize: SizeInt; const dst: TCUMem; const offset :SizeInt; const scaleSize: SizeInt; const scales, biases: TCUMem; const incb: SizeInt ; const batch: SizeInt);
    procedure forwardDropout(const N: SizeInt; const src: TCUMem; const probability, scale: single; rnd: TCUMem; dst: TCUMem);
    procedure backwardDropout(const N: SizeInt; const src: TCUMem; const probability, scale: single; const rnd: TCUMem; dst: TCUMem);
    procedure costL2(const N:SizeInt; const pred ,truth, delta, error: TCUMem);

    procedure finish();
    function compileToCUBIN(const code, name: ansistring; const headers: TArray<PAnsiChar> = nil; const includeNames: TArray<PAnsiChar> = nil): RawByteString;
    procedure loadCUBIN(const cubin : RawByteString);
    function compileFile(const filename : string):RawByteString;
    // procedure halfTest(//  const N : SizeInt; a:TCUMem; b:TCUMem ; c:TCUMem);

  end;

procedure SAFE_CALL(const res : cudaError_t);inline;   overload;

//var
//  cuda : TNNCuda;
implementation

//procedure SAFE_CALL(const res : CUresult);inline;     overload;
//var str:PAnsiChar;
//begin
//  cuGetErrorString(res, @str);
//  assert(res=CUDA_SUCCESS, str)
//end;

procedure SAFE_CALL(const res : cudaError_t);inline;   overload;
var str:PAnsiChar;
begin
  str := cudaGetErrorString(res);
  assert(res=cudaSuccess, str)
end;

procedure SAFE_CALL(const res : cublasStatus_t); inline;  overload;
var str:PAnsiChar;
begin
  str := cublasGetStatusString(res);
  assert(res=CUBLAS_STATUS_SUCCESS, str)
end;

//procedure SAFE_CALL(const res : cublasStatus); inline;  overload;
//var str:PAnsiChar;
//begin
//  //str := cublasGetStatusString(res);
//  assert(res=cublasStatus.CUBLAS_STATUS_SUCCESS)
//end;


procedure SAFE_CALL_RTC(const res : nvrtcResult); inline;  overload;
var
  str:PAnsiChar;
begin
  str := nvrtcGetErrorString(res);
  assert(res=NVRTC_SUCCESS, str)
end;

{ TCudaParamsHelper }

constructor TCudaParamsHelper.Create(const p0: pointer; p1: pointer;
  p2: pointer; p3: pointer; p4: pointer; p5: pointer; p6: pointer; p7: pointer;
  p8: pointer; p9: pointer; p10: pointer; p11: pointer; p12: pointer;
  p13: pointer; p14: pointer; p15: pointer);
begin
  self := default(TCudaParams);
  self[0] := p0;
  self[1] := p1;
  self[2] := p2;
  self[3] := p3;
  self[4] := p4;
  self[5] := p5;
  self[6] := p6;
  self[7] := p7;
  self[8] := p8;
  self[9] := p9;
  self[10] := p10;
  self[11] := p11;
  self[12] := p12;
  self[13] := p13;
  self[14] := p14;
  self[15] := p15;
end;

{ TNNCuda }

function TNNCuda.GetjitOptions(jo: cudaJitOption): pointer;
var i:sizeInt;
begin
  for i:=0 to High(FJitOptions) do
    if FjitOptions[i] = jo then
      exit(FJitOptionsValues[i])
end;

function TNNCuda.GetlibraryOptions(lo: cudaLibraryOption): pointer;
var i:sizeInt;
begin
  for i:=0 to High(FJitOptions) do
    if FLibOptions[i] = lo then
      exit(FLibOptionsValues[i])
end;

procedure TNNCuda.SetjitOptions(jo: cudaJitOption; AValue: pointer);
var i:sizeInt;
begin
  for i:=0 to High(FJitOptions) do
    if FjitOptions[i] = jo then
      if AValue=nil then begin
        delete(FJitOptions,i, 1);
        delete(FJitOptionsValues,i, 1);
        exit()
      end else begin
        FJitOptionsValues[i] := AValue;
        exit;
      end;
  insert(jo, FJitOptions, 0);
  insert(AValue, FJitOptionsValues, 0);
end;

procedure TNNCuda.SetlibraryOptions(lo: cudaLibraryOption; AValue: pointer);
var i:sizeInt;
begin
  for i:=0 to High(FLibOptions) do
    if FLibOptions[i] = lo then
      if AValue=nil then begin
        delete(FLibOptions,i, 1);
        delete(FLibOptionsValues,i, 1);
        exit()
      end else begin
        FLibOptionsValues[i] := AValue;
        exit;
      end;
  insert(lo, FLibOptions, 0);
  insert(AValue, FLibOptionsValues, 0);
end;

class function TNNCuda.deviceCount(): longint;
begin
  SAFE_CALL(cudaGetDeviceCount(@result))
end;

constructor TNNCuda.Create(deviceIndex: longint);
begin
  SAFE_CALL(cudaGetDeviceProperties(@properties, deviceIndex));
  SAFE_CALL(cudaSetDevice(deviceIndex));
  SAFE_CALL(cudaStreamCreate(@stream));
  SAFE_CALL(cublasCreate(@cublas));
  SAFE_CALL(cublasSetStream(cublas, stream));
  //SAFE_CALL(cublasInit());
  //SAFE_CALL(cublasSetKernelStream(stream));
  nnLib := nil;
end;

destructor TNNCuda.Destroy();
begin
  if assigned(nnLib) then
    SAFE_CALL(cudaLibraryUnload(nnlib));
  SAFE_CALL(cudaStreamDestroy(stream));
  SAFE_CALL(cublasDestroy(cublas));
  //SAFE_CALL(cublasShutdown());

end;

function TNNCuda.CompileLog: ansistring;
var
  progSize: size_t;
begin
  SAFE_CALL_RTC(nvrtcGetProgramLogSize(prog, @progSize));
  setLength(result, progSize);
  SAFE_CALL_RTC(nvrtcGetProgramLog(prog, pointer(result)));
end;

function TNNCuda.createDeviceBuffer(const N: SizeInt): TCUMem;
begin
  SAFE_CALL(cudaMalloc(@result, N));
end;

procedure TNNCuda.freeDeviceBuffer(var cudaMem: TCUMem);
begin
  SAFE_CALL(cudaFree(cudaMem));
  cudaMem := nil
end;

procedure TNNCuda.readBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opDeviceToHost);
{$endif}
  SAFE_CALL(cudaMemcpy(buffer, cudaMem, bufferSize, cudaMemcpyDeviceToHost)) ;

{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opDeviceToHost)
{$endif}
end;

procedure TNNCuda.writeBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opHostToDevice);
{$endif}
  SAFE_CALL(cudaMemcpy(cudaMem, buffer, bufferSize, cudaMemcpyHostToDevice))   ;
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opHostToDevice)
{$endif}
end;

const NUM_THREADS = $200;
procedure TNNCuda.ActivateArray(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const activation: longint);
const kernelId = 5;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  if activation= 4 then exit;
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @offset, @activation];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.activateArraySWISH(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const output_sigmoid, output: TCUMem);
const kernelId = 6;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @offset, @output_sigmoid, @output];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.DeriveArray(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const activation: longint; delta: TCUMem);
const kernelId = 7;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  if activation= 4 then exit;   // keep as is if acLINEAR
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @offset, @activation, @delta];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.forwardBias(const dstSize: SizeInt; const dst: TCUMem; const offset: SizeInt; const srcSize: SizeInt; const src: TCUMem; const incb: SizeInt; const batch: SizeInt);
const kernelId=4;
var
  blockSize, bOffset, num_grids: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opForwardBias) ;
{$endif}
  blockSize := dstSize div (srcSize*batch);
  bOffset := 1;
  num_grids := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @srcSize, @blockSize, @dst, @offset, @src, @bOffset, @incb];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opForwardBias)
{$endif}
end;

procedure TNNCuda.backwardBias(const dstSize: SizeInt; const dst: TCUMem;
  const srcSize: SizeInt; const src: TCUMem; const srcOffset: SizeInt;
  const incb: SizeInt; const batch: SizeInt);
const kernelId = 8;
var
  num_grids, blockSize: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opBackwardBias);
{$endif}

  blockSize := srcSize div (dstSize*batch);
  num_grids := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @dst, @blockSize, @src, @srcOffset, @batch];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(dstSize{num_grids}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opBackwardBias)
{$endif}
end;

procedure TNNCuda.gemm(const transA, transB: boolean; const M, N, K: SizeInt;
  const ALPHA: single; const A: PSingle; const aOffset: SizeInt; lda: SizeInt;
  const B: PSingle; const bOffset: SizeInt; ldb: SizeInt; const BETA: single;
  const C: PSingle; const cOffset: SizeInt; ldc: SizeInt;
  const workspace: TCUMem; const workspaceSize: SizeInt);
const dimThr:dim3 =(x: 4; y:64; z:1);
var
  kernelId: Integer;
  MM, NN : SizeInt;
  dim : dim3;
  params : array of pointer;

  opA, opB : cublasOperation_t;
  //opA, opB : ansichar;
  label done;
begin
  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]

{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opGemm);
{$endif}

// from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp
  //cublasSgemm(handle,
  //                                  CUBLAS_OP_N,
  //                                  CUBLAS_OP_N,
  //                                  matrix_size.uiWB,
  //                                  matrix_size.uiHA,
  //                                  matrix_size.uiWA,
  //                                  &alpha,
  //                                  d_B,
  //                                  matrix_size.uiWB,
  //                                  d_A,
  //                                  matrix_size.uiWA,
  //                                  &beta,
  //                                  d_C,
  //                                  matrix_size.uiWB)
  if transA then begin
    opA := CUBLAS_OP_T;
  end else begin
    opA := CUBLAS_OP_N  ;
  end;

  if transb then begin
    opB := CUBLAS_OP_T;
  end else begin
    opB := CUBLAS_OP_N  ;
  end;
  //SAFE_CALL(cublasSetWorkspace(cublas, workspace, workspaceSize));
  SAFE_CALL(cublasSgemm(cublas, opB, opA, N, M, K, @ALPHA, B+bOffset, ldb, A+aOffset, lda, @BETA, C+cOffset, ldc{, cublasComputeType_t.CUBLAS_COMPUTE_32F, cublasGemmAlgo_t.CUBLAS_GEMM_ALGO0}));
  goto done;// yes goto! and there is nothing you can do about it!

  if (not transA) and (not transB)then
    if N > M then begin
      dim := dim3.create(N, M);
      kernelId :=1;
    end else begin
      dim := dim3.create(M, N);
      kernelId :=0;
    end

  else if (not transA) and transB then begin
    dim := dim3.create(M, N);
    kernelId := 2;
  end else if transA and (not transB) then begin
    dim := dim3.create(M, N);
    kernelId := 3 ;
  end;
  dim := (dim + dimThr-1) div dimThr;
  //MM := (dim.x + NUM_THREADS-1) div NUM_THREADS;
  //NN := (dim.y + NUM_THREADS-1) div NUM_THREADS;

  params := [
         @M, @N, @K, @ALPHA,
         @A, @aOffset, @lda,
         @B, @bOffset, @ldb,
         @BETA,
         @C, @cOffset, @ldc
       ];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, dimThr, ppointer(params), 0, stream));


done:
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opGemm)
{$endif}
end;

procedure TNNCuda.addvv(const N: SizeInt; const src1: TCUMem; const src1Offset,
  inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt;
  dst: TCUMem; const dstOffset, incc: SizeInt);
const kernelId = 9;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
   num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
   params := [@N, @src1, @src1Offset, @inca, @src2, @src2Offset, @incb, @dst, @dstOffset, @incc];
   //SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelId], nnLib, 'addv'));
   SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.subvv(const N: SizeInt; const src1: TCUMem; const src1Offset,
  inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt;
  dst: TCUMem; const dstOffset, incc: SizeInt);
const kernelId = 10;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
   num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
   params := [@N, @src1, @src1Offset, @inca, @src2, @src2Offset, @incb, @dst, @dstOffset, @incc];
   SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.mulvv(const N: SizeInt; const src1: TCUMem; const src1Offset,
  inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt;
  dst: TCUMem; const dstOffset, incc: SizeInt);
const kernelId = 39;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
   num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
   params := [@N, @src1, @src1Offset, @inca, @src2, @src2Offset, @incb, @dst, @dstOffset, @incc];
   SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.fmavv(const N: SizeInt; const src1: TCUMem; const src1Offset,
  inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt;
  const src3: TCUMem; const src3Offset, incc: SizeInt; dst: TCUMem;
  const dstOffset, incd: SizeInt);
begin

end;

procedure TNNCuda.axpy(const N: SizeInt; const a: single; const x: TCUMem;
  const xOffset: SizeInt; const incx: SizeInt; const y: TCUMem;
  const yOffset: SizeInt; const incy: sizeInt);
const kernelId = 11;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opAxpy);
{$endif}
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @a, @x, @xOffset, @incx, @y, @yOffset, @incy];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opAxpy);
{$endif}
end;

procedure TNNCuda.power(const N: SizeInt; const x: TCUMem;
  const xOffset: SizeInt; const incx: SizeInt; const a: single;
  const y: TCUMem; const yOffset: SizeInt; const incy: sizeInt);
begin

end;

procedure TNNCuda.scale(const N: SizeInt; const a: Single; const x: TCUMem;
  const stride: SizeInt);
const kernelId = 12;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMulvs);
{$endif}
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @a, @x, @stride];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMulvs);
{$endif}
end;

procedure TNNCuda.crossEntropyLogistic(const N: SizeInt; const pred,
  truth: TCUMem; delta, error: TCUMem);
const kernelId = 13;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @pred, @truth, @delta, @error];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.fill(const N: SizeInt; const x: TCUMem;
  const offset: SizeInt; const val: single; const stride: SizeInt);
const kernelId = 14;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opFill);
{$endif}
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @offset, @val, @stride];
//  SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelId], nnLib, 'fill'));
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opFill)
{$endif}
end;

procedure TNNCuda.copy(const N: SizeInt; const src: TCUMem; const srcOffset,
  inca: SizeInt; const dst: TCUMem; const dstOffset, incb: SizeInt);
const kernelId = 15;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @src, @srcOffset, @inca, @dst, @dstOffset, @incb];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.softmaxBatch(const N: SizeInt; const input: TCUMem;
  const iOffset: SizeInt; const batch, batch_size, groups, group_size,
  stride: SizeInt; const temp: single; const output: TCUMem;
  const oOffset: SizeInt);
const kernelId = 18;
      blockDim:dim3 = (x:8; y:64; z:1);
var
  MM, NN: SizeInt;
  params : array of pointer;
  dim : dim3;
begin
  //MM := (batch + NUM_THREADS-1) div NUM_THREADS;
  //NN := (groups + NUM_THREADS-1) div NUM_THREADS;
  dim := dim3.create(batch, groups);
  dim := (dim + blockDim-1) div blockDim;
  params := [@batch, @groups, @input, @iOffset, @N, @batch_size, @group_size, @stride, @temp, @output, @oOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
end;

procedure TNNCuda.crossEntropySoftmax(const N: SizeInt; const pred,
  truth: TCUMem; delta, error: TCUMem);
const kernelId = 19;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @pred, @truth, @delta, @error];
  SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelId], nnLib,'crossEntropySoftmax'));
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.forwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  const input: TCUMem; const c, h, w: SizeInt; const stride_x, stride_y,
  padding, kernelSize: SizeInt; indexes, output: TCUMem);
const kernelId = 16;
      blockDim : dim3 = (x:8; y:8; z:8);
var N, MM, NN, KK:SizeInt;
  params : array of pointer;
  dim : dim3;
begin
  N := ABatch * outC;
  dim := dim3.create(N, outH, outW);
  dim := (dim + blockDim -1) div blockDim;
  //MM := (N + NUM_THREADS-1) div NUM_THREADS;
  //NN := (outH + NUM_THREADS-1) div NUM_THREADS;
  //KK := (outW + NUM_THREADS-1) div NUM_THREADS;;
  params := [
    @N
  , @outH
  , @outW
  , @input
  , @c
  , @h
  , @w
  , @stride_x
  , @stride_y
  , @padding
  , @kernelSize
  , @indexes
  , @output
  ];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
end;

procedure TNNCuda.backwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  output: TCUMem; const indexes, delta: TCUMem);
const kernelId = 17;
      blockDim:dim3 = (x:8; y:64; z:1);
var
  M, N : SizeInt;
  params : array of pointer;
  dim : dim3;
begin
  M := ABatch*outC;
  N := outH*outW;
  dim := dim3.create(M, N);
  dim := (dim + blockDim-1) div blockDim;
  params := [@M, @N, @output, @indexes, @delta];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockdim, ppointer(params), 0, stream));

end;

function ceil(const a, b: SizeInt):SizeInt;  overload;
begin
  result := (1 + (a-1) div b)*b
end;

const COPY_DIMX = 8;
      COPY_DIMY = 8;

procedure TNNCuda.im2col(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const im: TCUMem; const imOffset: SizeInt;
  const col: TCUMem; const colOffset: SizeInt);
const kernelId = 22;
      blockDim:dim3 =(x:8; y:64; z:1);
var
    N, size_h, padding_h, col_h, padding_w, size_w, col_w, ceiled_w,
      ceiled_h:SizeInt;
    params : array of pointer;
    dim :dim3;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opIm2col);
{$endif}

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

  N := ceiled_h * aChannels;
  dim := dim3.create(ceiled_w, N);
  dim := (dim + blockDim -1) div blockDim;

  params := [@ceiled_w, @N, @aHeight, @aWidth, @aChannels, @col_h, @col_w, @kernelHeight, @kernelWidth, @padHeight, @padWidth, @strideY, @strideX, @dilationY, @dilationX, @im, @imOffset, @col, @colOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opIm2col)
{$endif}
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

procedure TNNCuda.col2im(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const col: TCUMem; const colOffset: SizeInt;
  const im: TCUMem; const imOffset: SizeInt);
const kernelId = 24;
      blockDim:dim3 =(x:8; y:64; z:1);
var
    MM, NN, N, size_h, padding_h, col_h, size_w, padding_w, col_w,
      dilation_bez_h, dilation_bez_w, gcd_h, gcd_w, stride_bez_h,
      stride_bez_w, w_ceiled, h_ceiled:SizeInt;
    params : array of pointer;
    dim : dim3;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opCol2im) ;
{$endif}

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

  N:=  h_ceiled * aChannels ;
  dim := dim3.create(w_ceiled, N);
  dim := (dim + blockDim-1) div blockDim;
  //MM := (w_ceiled + NUM_THREADS-1) div NUM_THREADS;
  //NN := (N        + NUM_THREADS-1) div NUM_THREADS;
  params := [@w_ceiled, @N, @aHeight, @aWidth, @aChannels, @col_h, @col_w, @kernelHeight, @kernelWidth, @padHeight, @padWidth, @strideY, @strideX, @dilationY, @dilationX, @stride_bez_h, @stride_bez_w, @dilation_bez_h, @dilation_bez_w, @gcd_h, @gcd_w, @col, @colOffset, @im, @imOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opCol2im)
{$endif}
end;

procedure TNNCuda.upSample(const aBatch, aChannels, outHeight,
  outWidth: SizeInt; const &in: TCUMem; const stride: SizeInt;
  const isForward: longint; const scale: single; const &out: TCUMem;
  const zeroIn: boolean);

const kernelId = 25;
      blockDim : dim3 =(x:16; y:8; z:8);
var
  M, N, K, MM, NN, KK :SizeInt;
  params : array of pointer;
  dim : dim3;
begin
  M := aBatch*aChannels;
  N := outHeight*stride;
  K := outWidth*stride;
  dim := dim3.create(M, N, K);
  dim := (dim + blockDim-1) div blockDim;
  params := [@M, @N, @K, @&in, @stride, @isForward, @scale, @&out, @zeroIn];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
end;

procedure TNNCuda.fmavss(const N: SizeInt; const src: TCUMem;
  const offset: SizeInt; const scalar, bias: single; dst: TCUMem);

const kernelId = 26;
var
  NN : SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opFmavss);
{$endif}
  NN := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @src, @offset, @scalar, @bias, @dst];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(NN), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opFmavss);
{$endif}
end;

procedure TNNCuda.meansAndVars(const srcSize, dstSize, groups: sizeInt;
  const src: TCUMem; const offset: sizeInt; means, vars: TCUMem);
const kernelId = 27;
var
  MM, blockSize:SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMeansVars);
{$endif}
  MM := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  blockSize := srcSize div (dstSize*groups);
  params := [@dstSize, @blockSize, @groups, @src, @offset, @means, @vars];
  //SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
  SAFE_CALL(cudaLaunchKernel(FKernels[42], dim3.Create(dstSize{MM}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
  SAFE_CALL(cudaLaunchKernel(FKernels[43], dim3.Create(dstSize{MM}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMeansVars);
{$endif}
end;

procedure TNNCuda.means(const srcSize, dstSize, groups: sizeInt;
  const src: TCUMem; const offset: sizeInt; means: TCUMem);
const kernelId = 42;
var
  MM, blockSize:SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMeans);
{$endif}
  MM := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  blockSize := srcSize div (dstSize*groups);
  params := [@dstSize, @blockSize, @groups, @src, @offset, @means];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(dstSize{MM}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMeans);
{$endif}
end;

procedure TNNCuda.variances(const srcSize, dstSize, groups: sizeInt;
  const src: TCUMem; const offset: sizeInt; means, vars: TCUMem);
const kernelId = 43;
var
  MM, blockSize:SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opVariances);
{$endif}
  MM := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  blockSize := srcSize div (dstSize*groups);
  params := [@dstSize, @blockSize, @groups, @src, @offset, @means, @vars];
  //SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(dstSize{MM}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opVariances);
{$endif}
end;

procedure TNNCuda.normalize(const srcSize, dstSize, groups: SizeInt;
  means: TCUMem; const meansStride: sizeInt; vars: TCUMem;
  const varsStride: SizeInt; dst: TCUMem; const dstOffset: sizeInt);
const kernelId = 30;
      blockDim: dim3=(x:16; y:8; z:8);
var
  M, N, K, MM, NN, KK, blockSize:SizeInt;
  params : array of pointer;
  dim: dim3;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opNormalize);
{$endif}
  blockSize := dstSize div (srcSize*groups);
  M := srcSize;
  N := blocksize;
  K := groups;
  dim := dim3.create(M, N, K);
  dim := (dim + blockDim-1) div blockDim;
  params := [@M, @N, @K, @means, @meansStride, @vars, @varsStride, @dst, @dstOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opNormalize);
{$endif}
end;

procedure TNNCuda.meansAndVarsDelta(const srcSize, dstSize, groups: SizeInt;
  delta, x: TCUMem; const offset: SizeInt; mean, variance, mean_delta,
  variance_delta: TCUMem);
const
      //kernelId = 31;
      kernelId = 44;
var
  NN , blockSize: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opMeansVarsDelta);
{$endif}
  blockSize := srcSize div (dstSize * groups);
  NN := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @groups, @blockSize, @delta, @x, @offset, @mean, @variance, @mean_delta, @variance_delta];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(dstSize{NN}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opMeansVarsDelta);
{$endif}
end;

procedure TNNCuda.normalizeDelta(const deltaSize, meanSize, groups: SizeInt;
  const delta, x: TCUMem; const offset: SizeInt; mean, variance, mean_delta,
  variance_delta: TCUMem);
const kernelId = 32;
      blockDim: dim3 =(x:16; y:8; z:8);
var
  M, N, K, blockSize :SizeInt;
  params : array of pointer;
  dim: dim3;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opNormalizeDelta);
{$endif}
  blockSize := deltaSize div (meanSize * groups);
  M := meanSize;
  N := blocksize;
  K := groups;
  dim := dim3.create(M, N, K);
  dim := (dim + blockDim -1) div blockDim;
  params := [@M, @N, @K, @x, @offset, @mean, @variance, @mean_delta, @variance_delta, @delta];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, blockDim, ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opNormalizeDelta);
{$endif}
end;

procedure TNNCuda.addDots(const N, dstSize, groups: SizeInt; const src1,
  src2: TCUMem; const srcOffset: SizeInt; dst: TCUMem);
const kernelId = 33;
var
  blockSize, NN: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opAddDots);
{$endif}
  blockSize := N div (dstSize * groups);
  NN := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @groups, @blocksize, @src1, @src2, @srcOffset, @dst];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(dstSize{NN}), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opAddDots);
{$endif}
end;

procedure TNNCuda.forwardScale(const dstSize: SizeInt; const dst: TCUMem;
  const offset: SizeInt; const scaleSize: SizeInt; const scale: TCUMem;
  const incb: SizeInt; const batch: SizeInt);
const kernelId = 34;
var
  blockSize, bOffset , num_grids:SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opForwardScale);
{$endif}
  blockSize := dstSize div (scaleSize*batch);
  bOffset := 1;
  num_grids := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @scaleSize, @blockSize, @dst, @offset, @scale, @bOffset, @incb];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opForwardScale);
{$endif}
end;

procedure TNNCuda.forwardScaleAdd(const dstSize: SizeInt; const dst: TCUMem;
  const offset: SizeInt; const scaleSize: SizeInt; const scales,
  biases: TCUMem; const incb: SizeInt; const batch: SizeInt);
const kernelId = 35;
var
  blockSize, bOffset , num_grids:SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opForwardScaleAdd);
{$endif}
  blockSize := dstSize div (scaleSize*batch);
  bOffset := 1;
  num_grids := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @scaleSize, @blockSize, @dst, @offset, @scales, @biases, @bOffset, @incb];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opForwardScaleAdd);
{$endif}
end;

procedure TNNCuda.forwardDropout(const N: SizeInt; const src: TCUMem;
  const probability, scale: single; rnd: TCUMem; dst: TCUMem);
const kernelId=36;
var
  num_grids : SizeInt;
  params : array of pointer;
begin
  //randomize;
  random(1000); // next randSeed
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @RandSeed, @probability, @scale, @src, @rnd, @dst];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.backwardDropout(const N: SizeInt; const src: TCUMem;
  const probability, scale: single; const rnd: TCUMem; dst: TCUMem);
const kernelId=37;
var
  num_grids : SizeInt;
  params : array of pointer;
begin
  //randomize;
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @probability, @scale, @src, @rnd, @dst];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.costL2(const N: SizeInt; const pred, truth, delta, error: TCUMem);
const kernelId = 38;
var
  num_grids:SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @pred, @truth, @delta, @error];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.finish();
begin
  //SAFE_CALL(cudaDeviceSynchronize());
  SAFE_CALL(cudaStreamSynchronize(stream));
end;

function TNNCuda.compileToCUBIN(const code, name: ansistring; const headers: TArray<PAnsiChar>; const includeNames: TArray<PAnsiChar>): RawByteString;
var
  progSize : size_t;
  err: nvrtcResult;
  params : array of pansichar;
  log : ansistring;
begin
  SAFE_CALL_RTC(nvrtcCreateProgram(@prog, PAnsiChar(code), pAnsiChar(name), length(headers), Pointer(headers), Pointer(includeNames)));

  //params := [PAnsiChar('--gpu-architecture=sm_'+intToStr(devCapMajor)+intTostr(devCapMinor))];
  params := [PAnsiChar('-DBLOCK='+intToStr(NUM_THREADS)), PAnsiChar('-arch=sm_' + intToStr(properties.major)+intTostr(properties.minor)), '--use_fast_math'];
  err :=nvrtcCompileProgram(prog, length(params), pointer(params));
  log := CompileLog;
  if (log<>#0) and IsConsole then
    writeln(log);
  if err <>NVRTC_SUCCESS then begin
    assert(false, nvrtcGetErrorString(err) +sLineBreak+ log);
    //readln
  end;
  err := nvrtcGetCUBINSize(prog, @progSize);
  if progSize>0 then begin
    setLength(result, progSize);
    SAFE_CALL_RTC(nvrtcGetCUBIN(prog, pointer(result)));
  end;
  SAFE_CALL_RTC(nvrtcDestroyProgram(@prog));
end;

procedure TNNCuda.loadCUBIN(const cubin: RawByteString);
var kernelCount : longint;
  i : SizeInt;
  ker : cudaKernel_t;
begin
  SAFE_CALL(cudaLibraryLoadData(@nnLib, pointer(cubin), pointer(FJitOptions), pointer(FJitOptionsValues), length(FJitOptions),  pointer(FLibOptions), pointer(FLibOptionsValues), length(FLibOptions)));
  SAFE_CALL(cudaLibraryGetKernelCount(@kernelCount, nnLib));
  setLength(FKernels, kernelCount);
  SAFE_CALL(cudaLibraryEnumerateKernels(pointer(FKernels), kernelCount, nnlib));
  // cuda enumerates kernels reversed !#@%
  for i:=0 to kernelCount div 2 do begin
    ker := FKernels[i];
    FKernels[i] := Fkernels[kernelCount-i-1];
    Fkernels[kernelCount-i-1] := ker
  end;
end;

function TNNCuda.compileFile(const filename: string): RawByteString;
var f : TextFile;
  line, code:AnsiString;
  cuda_path:ansistring;
begin
  //cuda_path := GetEnvironmentVariable('CUDA_PATH')+'\include';
  assert(FileExists(filename), 'Cannot find file '+filename);
  AssignFile(F, filename);
  reset(F);
  code :='';
  while not EOF(F) do begin
    readln(F, line);
    code := code +line + sLineBreak;
  end;
  CloseFile(F);
  result := compileToCUBIN(code, filename)

end;

initialization
  //cuda := TNNCuda.create();
finalization
  //freeAndNil(cuda)
end.

