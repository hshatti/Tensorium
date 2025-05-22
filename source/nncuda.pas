unit nnCuda;

{$ifdef FPC}
{$mode Delphi}
{$PackRecords C}
{$ModeSwitch advancedrecords}
{$ModeSwitch typehelpers}
{$endif}

interface

uses
  SysUtils, cudart, cudarttypes, cublas_api, nvrtc;

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
    procedure gemm(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:single; const A:TCUMem; const aOffset:SizeInt; const lda:SizeInt; const B:TCUMem; const bOffset:SizeInt; const ldb:SizeInt; const BETA: single; const C:TCUMem; const cOffset:SizeInt; const ldc:SizeInt);
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
    procedure meanAndVars(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means, vars:TCUMem);
    procedure normalize(const srcSize, dstSize, groups:SizeInt; means:TCUMem; const meansStride:sizeInt; vars:TCUMem; const varsStride:SizeInt; dst:TCUMem; const dstOffset :sizeInt);
    procedure meansAndVarsDelta(const srcSize, dstSize, groups:SizeInt; delta, x: TCUMem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: TCUMem);
    procedure normalizeDelta(const deltaSize, meanSize, groups: SizeInt; const delta, x: TCUMem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: TCUMem);
    procedure addDots(const N, nDst, groups:SizeInt; const src1, src2:TCUMem; const srcOffset:SizeInt; dst:TCUMem);
    procedure forwardScale(const dstSize: SizeInt; const dst: TCUMem; const offset :SizeInt; const scaleSize: SizeInt; const scale: TCUMem; const incb: SizeInt ; const batch: SizeInt);
    procedure forwardScaleAdd(const dstSize: SizeInt; const dst: TCUMem; const offset :SizeInt; const scaleSize: SizeInt; const scales, biases: TCUMem; const incb: SizeInt ; const batch: SizeInt);
    procedure forwardDropout(const N: SizeInt; const src: TCUMem; const probability, scale: single; rnd: TCUMem; dst: TCUMem);
    procedure backwardDropout(const N: SizeInt; const src: TCUMem; const probability, scale: single; const rnd: TCUMem; dst: TCUMem);
    procedure costL2(const N:SizeInt; const pred ,truth, delta, error: TCUMem);

    procedure finish();
    function compileToCUBIN(const code, name: ansistring; const headers: TArray<PAnsiChar> = nil; const includeNames: TArray<PAnsiChar> = nil): RawByteString;
    procedure loadCUBIN(const cubin : RawByteString);
    function compileFile(const filename : string):RawByteString;
    //procedure halfTest(
    //  const N : SizeInt; a:TCUMem; b:TCUMem ; c:TCUMem);

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
//
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
  nnLib := nil;
end;

destructor TNNCuda.Destroy();
begin
  if assigned(nnLib) then
    SAFE_CALL(cudaLibraryUnload(nnlib));
  SAFE_CALL(cudaStreamDestroy(stream));
  SAFE_CALL(cublasDestroy(cublas));
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
  SAFE_CALL(cudaMemcpy(buffer, cudaMem, bufferSize, cudaMemcpyDeviceToHost))
end;

procedure TNNCuda.writeBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
begin
  SAFE_CALL(cudaMemcpy(cudaMem, buffer, bufferSize, cudaMemcpyHostToDevice))
end;

const NUM_THREADS = $20;
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
  if activation= 4 then exit;
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
  blockSize := dstSize div (srcSize*batch);
  bOffset := 1;
  num_grids := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @srcSize, @blockSize, @dst, @offset, @src, @bOffset, @incb];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));

end;

procedure TNNCuda.backwardBias(const dstSize: SizeInt; const dst: TCUMem;
  const srcSize: SizeInt; const src: TCUMem; const srcOffset: SizeInt;
  const incb: SizeInt; const batch: SizeInt);
const kernelId = 8;
var
  num_grids, blockSize: SizeInt;
  params : array of pointer;
begin
  blockSize := srcSize div (dstSize*batch);
  num_grids := (dstSize + NUM_THREADS-1) div NUM_THREADS;
  params := [@dstSize, @dst, @blockSize, @src, @srcOffset, @batch];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.gemm(const transA, transB: boolean; const M, N, K: SizeInt;
  const ALPHA: single; const A: TCUMem; const aOffset: SizeInt;
  const lda: SizeInt; const B: TCUMem; const bOffset: SizeInt;
  const ldb: SizeInt; const BETA: single; const C: TCUMem;
  const cOffset: SizeInt; const ldc: SizeInt);
var
  kernelId: Integer;
  MM, NN : SizeInt;
  dim : dim3;
  params : array of pointer;

  opA, opB : cublasOperation_t;
  dimGrid, dimThr :dim3;
begin
  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]
  if transA then
    opA := CUBLAS_OP_T else
    opA := CUBLAS_OP_N  ;
  if transb then
    opB := CUBLAS_OP_T else
    opB := CUBLAS_OP_N  ;

  //SAFE_CALL(cublasSgemm(cublas, opA, opB, M, N, K, @ALPHA, A, lda, B, ldb, @BETA, C, ldc));
  //exit;

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
  MM := (dim.x + NUM_THREADS-1) div NUM_THREADS;
  NN := (dim.y + NUM_THREADS-1) div NUM_THREADS;
  params := [
         @M, @N, @K, @ALPHA,
         @A, @aOffset, @lda,
         @B, @bOffset, @ldb,
         @BETA,
         @C, @cOffset, @ldc
       ];
  dimGrid := dim3.Create(MM, NN);
  dimThr := dim3.create(NUM_THREADS, NUM_THREADS);
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dimGrid, dimThr, ppointer(params), 0, stream));

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
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @a, @x, @xOffset, @incx, @y, @yOffset, @incy];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
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
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @a, @x, @stride];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
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
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @offset, @val, @stride];
  SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelId], nnLib, 'fill'));
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
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
var
  MM, NN: SizeInt;
  params : array of pointer;
begin
  MM := (batch + NUM_THREADS-1) div NUM_THREADS;
  NN := (groups + NUM_THREADS-1) div NUM_THREADS;

  params := [@batch, @groups, @input, @iOffset, @N, @batch_size, @group_size, @stride, @temp, @output, @oOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM, NN), dim3.create(NUM_THREADS, NUM_THREADS), ppointer(params), 0, stream));
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
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.forwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  const input: TCUMem; const c, h, w: SizeInt; const stride_x, stride_y,
  padding, kernelSize: SizeInt; indexes, output: TCUMem);
const kernelId = 16;
var N, MM, NN, KK:SizeInt;
  params : array of pointer;
begin
  N := ABatch * outC;
  MM := (N + NUM_THREADS-1) div NUM_THREADS;
  NN := (outH + NUM_THREADS-1) div NUM_THREADS;
  KK := (outW + NUM_THREADS-1) div NUM_THREADS;;
  params := [@N, @outH, @outW, @input, @c, @h, @w, @stride_x, @stride_y, @padding, @kernelSize, @indexes, @output];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM, NN, KK), dim3.create(NUM_THREADS, NUM_THREADS, NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.backwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  output: TCUMem; const indexes, delta: TCUMem);
const kernelId = 17;
var N, M, MM, NN:SizeInt;
  params : array of pointer;
begin
  M := ABatch*outC;
  N := outH*outW;
  MM := (M + NUM_THREADS-1) div NUM_THREADS;
  NN := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@M, @N, @output, @indexes, @delta];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM, NN), dim3.create(NUM_THREADS, NUM_THREADS), ppointer(params), 0, stream));

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
var
    MM, NN, N, size_h, padding_h, col_h, padding_w, size_w, col_w, ceiled_w,
      ceiled_h:SizeInt;
    params : array of pointer;
begin
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
  MM := (ceiled_w + NUM_THREADS-1) div NUM_THREADS;
  NN := (N        + NUM_THREADS-1) div NUM_THREADS;


  params := [@ceiled_w, @N, @aHeight, @aWidth, @aChannels, @col_h, @col_w, @kernelHeight, @kernelWidth, @padHeight, @padWidth, @strideY, @strideX, @dilationY, @dilationX, @im, @imOffset, @col, @colOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM, NN), dim3.create(NUM_THREADS, NUM_THREADS), ppointer(params), 0, stream));

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
var
    MM, NN, N, size_h, padding_h, col_h, size_w, padding_w, col_w,
      dilation_bez_h, dilation_bez_w, gcd_h, gcd_w, stride_bez_h,
      stride_bez_w, w_ceiled, h_ceiled:SizeInt;
    params : array of pointer;
begin
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

  MM := (w_ceiled + NUM_THREADS-1) div NUM_THREADS;
  NN := (N        + NUM_THREADS-1) div NUM_THREADS;
  params := [@w_ceiled, @N, @aHeight, @aWidth, @aChannels, @col_h, @col_w, @kernelHeight, @kernelWidth, @padHeight, @padWidth, @strideY, @strideX, @dilationY, @dilationX, @stride_bez_h, @stride_bez_w, @dilation_bez_h, @dilation_bez_w, @gcd_h, @gcd_w, @col, @colOffset, @im, @imOffset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(MM, NN), dim3.create(NUM_THREADS, NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda.upSample(const aBatch, aChannels, outHeight,
  outWidth: SizeInt; const &in: TCUMem; const stride: SizeInt;
  const isForward: longint; const scale: single; const &out: TCUMem;
  const zeroIn: boolean);
begin

end;

procedure TNNCuda.fmavss(const N: SizeInt; const src: TCUMem;
  const offset: SizeInt; const scalar, bias: single; dst: TCUMem);
begin

end;

procedure TNNCuda.meanAndVars(const srcSize, dstSize, groups: sizeInt;
  const src: TCUMem; const offset: sizeInt; means, vars: TCUMem);
begin

end;

procedure TNNCuda.normalize(const srcSize, dstSize, groups: SizeInt;
  means: TCUMem; const meansStride: sizeInt; vars: TCUMem;
  const varsStride: SizeInt; dst: TCUMem; const dstOffset: sizeInt);
begin

end;

procedure TNNCuda.meansAndVarsDelta(const srcSize, dstSize, groups: SizeInt;
  delta, x: TCUMem; const offset: SizeInt; mean, variance, mean_delta,
  variance_delta: TCUMem);
begin

end;

procedure TNNCuda.normalizeDelta(const deltaSize, meanSize, groups: SizeInt;
  const delta, x: TCUMem; const offset: SizeInt; mean, variance, mean_delta,
  variance_delta: TCUMem);
begin

end;

procedure TNNCuda.addDots(const N, nDst, groups: SizeInt; const src1,
  src2: TCUMem; const srcOffset: SizeInt; dst: TCUMem);
begin

end;

procedure TNNCuda.forwardScale(const dstSize: SizeInt; const dst: TCUMem;
  const offset: SizeInt; const scaleSize: SizeInt; const scale: TCUMem;
  const incb: SizeInt; const batch: SizeInt);
begin

end;

procedure TNNCuda.forwardScaleAdd(const dstSize: SizeInt; const dst: TCUMem;
  const offset: SizeInt; const scaleSize: SizeInt; const scales,
  biases: TCUMem; const incb: SizeInt; const batch: SizeInt);
begin

end;

procedure TNNCuda.forwardDropout(const N: SizeInt; const src: TCUMem;
  const probability, scale: single; rnd: TCUMem; dst: TCUMem);
begin

end;

procedure TNNCuda.backwardDropout(const N: SizeInt; const src: TCUMem;
  const probability, scale: single; const rnd: TCUMem; dst: TCUMem);
begin

end;

procedure TNNCuda.costL2(const N: SizeInt; const pred, truth, delta,
  error: TCUMem);
begin

end;

procedure TNNCuda.finish();
begin
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
  params := [PAnsiChar('-arch=sm_' + intToStr(properties.major)+intTostr(properties.minor)), '--use_fast_math'];
  err :=nvrtcCompileProgram(prog, length(params), pointer(params));
  log := CompileLog;
  if (log<>#0) and IsConsole then
    writeln(log);
  if err <>NVRTC_SUCCESS then
    assert(false, nvrtcGetErrorString(err) +sLineBreak+ log);
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
begin
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

