unit nnCuda;

{$ifdef FPC}
{$mode Delphi}
{$PackRecords C}
{$ModeSwitch advancedrecords}
{$ModeSwitch typehelpers}
{$endif}
{$PointerMath ON}
{$T-}
interface

uses
  SysUtils, TypInfo,
  cudart, cudarttypes,
  cudaTypes, cuda,
  cublas_api, nvrtc, nOpMetrics;


type
  //PCUMem = ^TCUMem;
  //PCUMem = pointer;
  //TCUMem = pointer;
  TDataType = (dtUnknowen, dtHalf, dtBF16, dtSingle, dtDouble, dtComplexHalf, dtComplexSingle, dtComplexBf16, dtComplexDouble, dtINT16, dtINT8, dtINT4);
  TCudaParams = array[0..15] of pointer;

  { TCudaParamsHelper }

//  TCudaParamsHelper = record helper for TCudaParams
//    constructor Create(const p0 : pointer ; p1 : pointer = nil; p2 : pointer = nil; p3 : pointer = nil; p4 : pointer = nil; p5 : pointer = nil; p6 : pointer = nil; p7 : pointer = nil; p8 : pointer = nil; p9 : pointer = nil; p10 : pointer = nil; p11 : pointer = nil; p12 : pointer = nil; p13 : pointer = nil; p14 : pointer = nil; p15 : pointer = nil);
//  end;

  { TNNCuda }

  TNNCuda<T> = class
  const
    NUM_THREADS = $200;
    COPY_DIMX = 8;
    COPY_DIMY = 8;

  type
    PCUMem = ^TCUMem;
    TCUMem = ^T;
  private
    FJitOptionsValues : TArray<pointer>;
    FLibOptionsValues : TArray<pointer>;
    //{$if defined(USE_CUDA)}
    FKernels : TArray<CUKernel>;
    FJitOptions : TArray<CUjit_option>;
    FLibOptions : TArray<CULibraryOption>;
    function GetjitOptions(jo: CUjit_option): pointer;
    function GetlibraryOptions(lo: CUlibraryOption): pointer;
    procedure SetjitOptions(jo: CUjit_option; AValue: pointer);
    procedure SetlibraryOptions(lo: CUlibraryOption; AValue: pointer);
    //{$else}
    //FKernels : TArray<cudaKernel_t>;
    //FJitOptions : TArray<cudaJitOption>;
    //FLibOptions : TArray<cudaLibraryOption>;
    //function GetjitOptions(jo: cudaJitOption): pointer;
    //function GetlibraryOptions(lo: cudaLibraryOption): pointer;
    //procedure SetjitOptions(jo: cudaJitOption; AValue: pointer);
    //procedure SetlibraryOptions(lo: cudaLibraryOption; AValue: pointer);
    //{$endif}
    //{$if defined(USE_CUDA)}
    class function cudaLibraryLoadData(&library : PCUlibrary ; const code : pointer ; jitOptions : PCUjit_option ; jitOptionsValues : PPointer ; numJitOptions : longword ; libraryOptions : PCUlibraryOption ; libraryOptionValues : PPointer; numLibraryOptions : longword):CUresult; static;
    class function cudaLibraryLoadFromFile(&library : PCUlibrary; const fileName:pansichar; jitOptions : PCUjit_option; jitOptionsValues : PPointer; numJitOptions: longword ; libraryOptions : PCUlibraryOption ; libraryOptionValues : PPointer; numLibraryOptions : longword):CUresult; static;
    class function cudaLibraryGetKernel(pKernel : PCUkernel; const &library : CUlibrary; const name : pansichar):CUresult; static;

    class function cudaLaunchKernel(const func:pointer; const gridDim:dim3; const blockDim:dim3; const args:Ppointer; const sharedMem:size_t; const stream:CUStream):CUresult;   static;
    class function cudaStreamCreate(stream: PCUStream):CUresult;   static;
    class function cudaLibraryUnload(const &library : CUlibrary):CUResult; static;
    class function cudaStreamDestroy(const stream: CUStream):CUresult; static;
    //{$endif}
    class function getTypeStr():shortstring; static;
    class function getType():TDataType;static;
  public
    //{$if defined(USE_CUDA)}
    stream : CUStream;
    ctx : CUContext;
    //{$else}
    //stream : cudaStream_t;
    //{$endif}
    properties : cudaDeviceProp;
    rtVersionMajor, rtVersionMinor : longint;
    prog : nvrtcProgram;
    //{$if defined(USE_CUDA)}
    nnLib : CUlibrary;
    //{$else}
    //nnLib : cudaLibrary_t;
    //{$endif}
    useBLAS : integer;
    cublas: cublasHandle_t;
    //{$if defined(USE_CUDA)}
    property jitOptions[jo:CUjit_option] : pointer read GetjitOptions write SetjitOptions;
    property libraryOptions[jo:CUlibraryOption] : pointer read GetlibraryOptions write SetlibraryOptions;
    //{$else}
    //property jitOptions[jo:cudaJitOption] : pointer read GetjitOptions write SetjitOptions;
    //property libraryOptions[jo:cudaLibraryOption] : pointer read GetlibraryOptions write SetlibraryOptions;
    //{$endif}
    class function deviceCount() : longint;
    constructor Create(deviceIndex: longint = 0);
    destructor Destroy();  override;
    function CompileLog:ansistring;
    function createDeviceBuffer(const N:SizeInt):TCUMem;
    procedure freeDeviceBuffer(cudaMem:TCUMem);
    procedure readBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer:pointer);
    procedure writeBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
    procedure ActivateArray(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const activation: longint);
    procedure activateArraySWISH(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const output_sigmoid, output: TCUMem);
    procedure DeriveArray(const N: SizeInt; const x: TCUMem; const offset:SizeInt; const activation: longint; delta: TCUMem);
    procedure forwardBias(const dstSize: SizeInt; const dst: TCUMem; const offset:SizeInt; const srcSize: SizeInt; const src: TCUMem; const incb: SizeInt; const batch: SizeInt);
    procedure backwardBias(const dstSize: SizeInt; const dst: TCUMem; const srcSize: SizeInt; const src: TCUMem; const srcOffset:SizeInt; const incb: SizeInt ; const batch: SizeInt);
    procedure gemm(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:T; const A:TCUMem; const aOffset:SizeInt; const lda:SizeInt; const B:TCUMem; const bOffset:SizeInt; const ldb:SizeInt; const BETA: T; const C:TCUMem; const cOffset:SizeInt; const ldc:SizeInt);
    procedure gemmBatched(const transA, transB: boolean; const M, N, K: SizeInt;
      const ALPHA: T; const A: PCUmem; const aOffset: SizeInt;
      const lda: SizeInt; const B: PCUMem; const bOffset: SizeInt;
      const ldb: SizeInt; const BETA: T; const C: PCUMem;
      const cOffset: SizeInt; const ldc: SizeInt; const batchCount: SizeInt);
    procedure gemmStridedBatched(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:T; A:TCUMem; const aOffset:SizeInt; const lda:SizeInt; const strideA:SizeInt; B:TCUMem; const bOffset:SizeInt; const ldb:SizeInt; const strideB:SizeInt; const BETA: T; C:TCUMem; const cOffset:SizeInt; const ldc:SizeInt; const strideC:SizeInt; const batchCount : SizeInt);
    procedure addvv(const N:SizeInt; const src1:TCUMem; const src1Offset, inca:SizeInt; const src2:TCUMem; const src2Offset, incb:SizeInt; dst:TCUMem; const dstOffset, incc:SizeInt);
    procedure subvv(const N:SizeInt; const src1:TCUMem; const src1Offset, inca:SizeInt; const src2:TCUMem; const src2Offset, incb:SizeInt; dst:TCUMem; const dstOffset, incc:SizeInt);
    procedure mulvv(const N:SizeInt; const src1:TCUMem; const src1Offset, inca:SizeInt; const src2:TCUMem; const src2Offset, incb:SizeInt; dst:TCUMem; const dstOffset, incc:SizeInt);
    procedure fmavv(const N: SizeInt; const src1: TCUMem; const src1Offset, inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt; const src3: TCUMem; const src3Offset, incc: SizeInt; dst: TCUMem; const dstOffset, incd: SizeInt);
    procedure axpy(const N:SizeInt; const a:T; const x:TCUMem; const xOffset:SizeInt; const incx:SizeInt; const y:TCUMem; const yOffset:SizeInt; const incy:sizeInt);
    procedure power(const N:SizeInt; const x:TCUMem; const xOffset:SizeInt; const incx:SizeInt; const a:T; const y:TCUMem; const yOffset:SizeInt; const incy:sizeInt);
    procedure scale(const N:SizeInt; const a:T; const x:TCUMem; const stride:SizeInt);
    procedure crossEntropyLogistic(const N:SizeInt; const pred, truth: TCUMem; delta, error: TCUMem);
    procedure fill(const N:SizeInt; const x: TCUMem; const offset:SizeInt; const val:T; const stride :SizeInt);
    procedure copy(const N:SizeInt; const src:TCUMem; const srcOffset, inca:SizeInt; const dst:TCUMem; const dstOffset, incb:SizeInt);
    procedure softmaxBatch(const N: SizeInt; const input: TCUMem; const iOffset: SizeInt; const batch, batch_size, groups, group_size, stride: SizeInt; const temp: T; const output: TCUMem; const oOffset: SizeInt);
    procedure crossEntropySoftmax(const N:SizeInt; const pred, truth: TCUMem; delta, error: TCUMem);
    procedure forwardMaxPool(const aBatch, outC, outH, outW: SizeInt; const input: TCUMem; const c, h, w: SizeInt; const stride_x, stride_y, padding, kernelSize: SizeInt; indexes, output: TCUMem);
    procedure backwardMaxPool(const aBatch, outC, outH, outW : SizeInt; output:TCUMem; const indexes, delta : TCUMem);
    procedure im2col(const aChannels, aHeight, aWidth , kernelHeight, kernelWidth, padHeight, padWidth , strideY, strideX, dilationY, dilationX : SizeInt ; const im :TCUMem; const imOffset : SizeInt ; const col:TCUMem; const colOffset:SizeInt);
    procedure col2im(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideY, strideX, dilationY, dilationX: SizeInt; const col: TCUMem; const colOffset: SizeInt; const im: TCUMem; const imOffset: SizeInt);
    procedure upSample(const aBatch, aChannels, outHeight, outWidth: SizeInt; const &in: TCUMem;const stride: SizeInt; const isForward: longint; const scale: T; const &out: TCUMem; const zeroIn :integer = 0);
    procedure fmavss(const N: SizeInt; const src: TCUMem; const offset: SizeInt; const scalar, bias: T; dst : TCUMem);
    procedure meansAndVars(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means, vars:TCUMem);
    procedure means(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means:TCUMem);
    procedure variances(const srcSize, dstSize, groups:sizeInt; const src:TCUMem; const offset:sizeInt; means, vars:TCUMem);
    procedure normalize(const srcSize, dstSize, groups:SizeInt; means:TCUMem; const meansStride:sizeInt; vars:TCUMem; const varsStride:SizeInt; dst:TCUMem; const dstOffset :sizeInt);
    procedure meansAndVarsDelta(const srcSize, dstSize, groups:SizeInt; delta, x: TCUMem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: TCUMem);
    procedure normalizeDelta(const deltaSize, meanSize, groups: SizeInt; const delta, x: TCUMem; const offset:SizeInt; mean, variance, mean_delta, variance_delta: TCUMem);
    procedure addDots(const N, dstSize, groups:SizeInt; const src1, src2:TCUMem; const srcOffset:SizeInt; dst:TCUMem);
    procedure forwardScale(const dstSize: SizeInt; const dst: TCUMem; const offset :SizeInt; const scaleSize: SizeInt; const scale: TCUMem; const incb: SizeInt ; const batch: SizeInt);
    procedure forwardScaleAdd(const dstSize: SizeInt; const dst: TCUMem; const offset :SizeInt; const scaleSize: SizeInt; const scales, biases: TCUMem; const incb: SizeInt ; const batch: SizeInt);
    procedure forwardDropout(const N: SizeInt; const src: TCUMem; const probability, scale: T; rnd: TCUMem; dst: TCUMem);
    procedure backwardDropout(const N: SizeInt; const src: TCUMem; const probability, scale: T; const rnd: TCUMem; dst: TCUMem);
    procedure costL2(const N:SizeInt; const pred ,truth, delta, error: TCUMem);
    procedure clamp(const N:SizeInt; const alpha :T; const src, dst: TCUMem; const stride: SizeInt =1; offset : SizeInt =0);
    procedure inverseSqrt(const N:SizeInt; const alpha :T; const src, dst: TCUMem; const stride: SizeInt =1; offset : SizeInt =0);

    procedure finish();
    function compileToCUBIN(const code, name: ansistring; const headers: TArray<PAnsiChar> = nil; const includeNames: TArray<PAnsiChar> = nil): RawByteString;
    procedure loadCUBIN(const cubin : RawByteString);
    function compileFile(const filename : ansistring):RawByteString;
    procedure loadCUBinFile(const filename: ansistring);
    // procedure halfTest (//  const N : SizeInt; a:TCUMem; b:TCUMem ; c:TCUMem);

  const
    kernelNames : array[0..46] of PAnsiChar =(
      'clamp',
      'inverse_sqrt',
      'means_vars_delta_fast',
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
      'array_activate_swish',
      'activate_array',
      'forward_bias',
      'sgemm1_tn',
      'sgemm1_nt',
      'sgemm2_nn',
      'sgemm1_nn'
    );

  end;
const
  TDataTypeNames : array[0.. ord(high(TDataType))] of shortstring =
    ('Unknowen', 'Half', 'BF16', 'Single', 'Double', 'ComplexHalf', 'ComplexSingle', 'ComplexBf16', 'ComplexDouble', 'INT16', 'INT8', 'INT4');

procedure SAFE_CALL(const res : cudaError_t);inline;   overload;
procedure SAFE_CALL(const res : cublasStatus_t);  inline;overload;
//{$if defined(USE_CUDA)}
procedure SAFE_CALL(const res : CUresult);inline ; overload;
//{$endif}
procedure SAFE_CALL_RTC(const res : nvrtcResult); inline;  overload;

function ceil(const a, b: SizeInt):SizeInt;  overload;
procedure EuclidGCD( a, b: SizeInt; var p, q, r:SizeInt);inline;

//var
//  cuda : TNNCuda;
implementation

//procedure SAFE_CALL(const res : CUresult);inline;     overload;
//var str:PAnsiChar;
//begin
//  cuGetErrorString(res, @str);
//  assert(res=CUDA_SUCCESS, str)
//end;

procedure SAFE_CALL(const res : cudaError_t);
var str:PAnsiChar;
begin
  str := cudaGetErrorString(res);
  assert(res=cudaSuccess, string(str))
end;

procedure SAFE_CALL(const res : cublasStatus_t);
var str:PAnsiChar;
begin
  str := cublasGetStatusString(res);
  assert(res=CUBLAS_STATUS_SUCCESS, string(str))
end;

//{$if defined(USE_CUDA)}
procedure SAFE_CALL(const res : CUresult);
var str:PAnsiChar;
begin
  cuGetErrorString(res, @str);
  assert(res=CUDA_SUCCESS, string(str))
end;
//{$endif}

//procedure SAFE_CALL(const res : cublasStatus); inline;  overload;
//var str:PAnsiChar;
//begin
//  //str := cublasGetStatusString(res);
//  assert(res=cublasStatus.CUBLAS_STATUS_SUCCESS)
//end;


procedure SAFE_CALL_RTC(const res : nvrtcResult);
var
  str:PAnsiChar;
begin
  str := nvrtcGetErrorString(res);
  assert(res=NVRTC_SUCCESS, string(str))
end;

{ TCudaParamsHelper }

//constructor TCudaParamsHelper.Create(const p0: pointer; p1: pointer;
//  p2: pointer; p3: pointer; p4: pointer; p5: pointer; p6: pointer; p7: pointer;
//  p8: pointer; p9: pointer; p10: pointer; p11: pointer; p12: pointer;
//  p13: pointer; p14: pointer; p15: pointer);
//begin
//  self := default(TCudaParams);
//  self[0] := p0;
//  self[1] := p1;
//  self[2] := p2;
//  self[3] := p3;
//  self[4] := p4;
//  self[5] := p5;
//  self[6] := p6;
//  self[7] := p7;
//  self[8] := p8;
//  self[9] := p9;
//  self[10] := p10;
//  self[11] := p11;
//  self[12] := p12;
//  self[13] := p13;
//  self[14] := p14;
//  self[15] := p15;
//
//end;

{ TNNCuda }

//{$if defined(USE_CUDA)}
function TNNCuda<T>.GetjitOptions(jo: CUjit_option): pointer;
//{$else}
//function TNNCuda<T>.GetjitOptions(jo: cudaJitOption): pointer;
//{$endif}
var i:sizeInt;
begin
  for i:=0 to High(FJitOptions) do
    if FjitOptions[i] = jo then
      exit(FJitOptionsValues[i])
end;

//{$if defined(USE_CUDA)}
function TNNCuda<T>.GetlibraryOptions(lo: CUlibraryOption): pointer;
//{$else}
//function TNNCuda<T>.GetlibraryOptions(lo: cudaLibraryOption): pointer;
//{$endif}
var i:sizeInt;
begin
  for i:=0 to High(FJitOptions) do
    if FLibOptions[i] = lo then
      exit(FLibOptionsValues[i])
end;

//{$if defined(USE_CUDA)}
procedure TNNCuda<T>.SetjitOptions(jo: CUjit_option; AValue: pointer);
//{$else}
//procedure TNNCuda<T>.SetjitOptions(jo: cudaJitOption; AValue: pointer);
//{$endif}
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

//{$if defined(USE_CUDA)}
procedure TNNCuda<T>.SetlibraryOptions(lo: CUlibraryOption; AValue: pointer);
//{$else}
//procedure TNNCuda<T>.SetlibraryOptions(lo: cudaLibraryOption; AValue: pointer);
//{$endif}
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

//{$if defined(USE_CUDA)}
class function TNNCuda<T>.cudaLibraryLoadData(&library: PCUlibrary;
  const code: pointer; jitOptions: PCUjit_option; jitOptionsValues: PPointer;
  numJitOptions: longword; libraryOptions: PCUlibraryOption;
  libraryOptionValues: PPointer; numLibraryOptions: longword): CUresult;
begin
  result := cuLibraryLoadData(&library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
end;

class function TNNCuda<T>.cudaLibraryLoadFromFile(&library: PCUlibrary;
  const fileName: pansichar; jitOptions: PCUjit_option; jitOptionsValues: PPointer;
  numJitOptions: longword; libraryOptions: PCUlibraryOption;
  libraryOptionValues: PPointer; numLibraryOptions: longword): CUresult;
begin
  result := cuLibraryLoadFromFile(&library, filename, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
end;

class function TNNCuda<T>.cudaLibraryGetKernel(pKernel: PCUkernel;
  const &library: CUlibrary; const name: pansichar): CUresult;
begin
  result := cuLibraryGetKernel(pKernel, &library, name);
end;

class function TNNCuda<T>.cudaLaunchKernel(const func: pointer;
  const gridDim: dim3; const blockDim: dim3; const args: Ppointer;
  const sharedMem: size_t; const stream: CUStream): CUresult;
var cuFunc : CUFunction;
begin
  SAFE_CALL(cuKernelGetFunction(@cuFunc, func));
  result := cuLaunchKernel(cuFunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, nil)
end;

class function TNNCuda<T>.cudaStreamCreate(stream: PCUStream): CUresult;
begin
  result := cuStreamCreate(stream, CU_STREAM_DEFAULT);
end;

class function TNNCuda<T>.cudaLibraryUnload(const &library: CUlibrary): CUResult;
begin
  result := cuLibraryUnload(&library);
end;

class function TNNCuda<T>.cudaStreamDestroy(const stream: CUStream): CUresult;
begin
  result := cuStreamDestroy(stream);
end;
//{$endif}

class function TNNCuda<T>.getTypeStr(): shortstring;
begin
  result := lowerCase(PTypeInfo(TypeInfo(T)).Name);
  if result = 'single' then
    result:='float'
end;

class function TNNCuda<T>.getType(): TDataType;
var aname : shortstring;
begin
  aname := LowerCase(PTypeInfo(TypeInfo(T)).name);
  if aname='bf16' then exit(dtBF16);
  if aname='half' then exit(dtHalf);
  if aname='single' then exit(dtSingle);
  if aname='double' then exit(dtDouble);
  if aname='complexhalf' then exit(dtComplexHalf);
  if aname='complexbf16' then exit(dtComplexBf16);
  if aname='complexsingle' then exit(dtComplexSingle);
  if aname='complexdouble' then exit(dtComplexDouble);
  if aname='int16' then exit(dtINT16);
  if aname='int8' then exit(dtINT8);
  if aname='int4' then exit(dtINT4)
end;

class function TNNCuda<T>.deviceCount(): longint;
begin
  SAFE_CALL(cudaGetDeviceCount(@result))
end;

constructor TNNCuda<T>.Create(deviceIndex: longint);
begin
  SAFE_CALL(cudaRuntimeGetVersion(@rtVersionMajor));
  rtVersionMinor:=rtVersionMajor mod 1000;
  rtVersionMajor:=rtVersionMajor div 1000;
  SAFE_CALL(cudaGetDeviceProperties(@properties, deviceIndex));
  SAFE_CALL(cudaSetDevice(deviceIndex));
  SAFE_CALL(cudaStreamCreate(@stream));
  //stream := nil;
  SAFE_CALL(cublasCreate(@cublas));
  SAFE_CALL(cublasSetStream(cublas, pointer(stream)));
  //SAFE_CALL(cublasInit());
  //SAFE_CALL(cublasSetKernelStream(stream));
  FJitOptions:=nil;
  FJitOptionsValues:= nil;
  FLibOptions:=nil;
  FLibOptionsValues:=nil;
  useBLAS := 0;
  nnLib := nil;
end;

destructor TNNCuda<T>.Destroy();
begin
  if assigned(nnLib) then
    SAFE_CALL(cudaLibraryUnload(nnlib));
  SAFE_CALL(cudaStreamDestroy(stream));
  SAFE_CALL(cublasDestroy(cublas));
  //SAFE_CALL(cublasShutdown());

end;

function TNNCuda<T>.CompileLog: ansistring;
var
  progSize: size_t;
begin
  SAFE_CALL_RTC(nvrtcGetProgramLogSize(prog, @progSize));
  setLength(result, progSize);
  SAFE_CALL_RTC(nvrtcGetProgramLog(prog, pointer(result)));
end;

function TNNCuda<T>.createDeviceBuffer(const N: SizeInt): TCUMem;
begin
  SAFE_CALL(cudaMalloc(@result, N));
end;

procedure TNNCuda<T>.freeDeviceBuffer(cudaMem: TCUMem);
begin
  SAFE_CALL(cudaFree(cudaMem));
  cudaMem := nil
end;

procedure TNNCuda<T>.readBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
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

procedure TNNCuda<T>.writeBuffer(const cudaMem: TCUMem; const bufferSize: size_t; const buffer: pointer);
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

procedure TNNCuda<T>.ActivateArray(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const activation: longint);
const kernelId = 5;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opActivate);
  {$endif}
  if activation <> 4 then begin
    num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
    params := [@N, @x, @offset, @activation];
    SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
  end;
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opActivate);
  {$endif}
end;

procedure TNNCuda<T>.activateArraySWISH(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const output_sigmoid, output: TCUMem);
const kernelId = 6;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opActivate);
  {$endif}
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @offset, @output_sigmoid, @output];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opActivate);
  {$endif}
end;

procedure TNNCuda<T>.DeriveArray(const N: SizeInt; const x: TCUMem; const offset: SizeInt; const activation: longint; delta: TCUMem);
const kernelId = 7;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  {$ifdef USE_TELEMETRY}
  tensorMetrics.start(opDerive);
  {$endif}
  if activation <> 4 then begin   // keep as is if acLINEAR
    num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
    params := [@N, @x, @offset, @activation, @delta];
    SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
  end;
  {$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opDerive);
  {$endif}
end;

procedure TNNCuda<T>.forwardBias(const dstSize: SizeInt; const dst: TCUMem; const offset: SizeInt; const srcSize: SizeInt; const src: TCUMem; const incb: SizeInt; const batch: SizeInt);
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

procedure TNNCuda<T>.backwardBias(const dstSize: SizeInt; const dst: TCUMem;
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

procedure TNNCuda<T>.gemm(const transA, transB: boolean; const M, N,
  K: SizeInt; const ALPHA: T; const A: TCUMem; const aOffset: SizeInt;
  const lda: SizeInt; const B: TCUMem; const bOffset: SizeInt;
  const ldb: SizeInt; const BETA: T; const C: TCUMem; const cOffset: SizeInt;
  const ldc: SizeInt);
const dimThr:dim3 =(x: 8; y:16; z:1);
var
  kernelId: Integer;
  dim : dim3;
  params : array of pointer;

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
  //if transA then begin
  //  opA := CUBLAS_OP_T;
  //end else begin
  //  opA := CUBLAS_OP_N  ;
  //end;
  //
  //if transb then begin
  //  opB := CUBLAS_OP_T;
  //end else begin
  //  opB := CUBLAS_OP_N  ;
  //end;
  //SAFE_CALL(cublasSetWorkspace(cublas, workspace, workspaceSize));

  if useBLAS = 1 then begin
    case getType() of
      dtHalf   :
        SAFE_CALL(cublasHgemm(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PHalf(B)+bOffset, ldb, PHalf(A)+aOffset, lda, @BETA, PHalf(C)+cOffset, ldc{, cublasComputeType_t.CUBLAS_COMPUTE_32F, cublasGemmAlgo_t.CUBLAS_GEMM_ALGO0}));
      dtSingle :
        SAFE_CALL(cublasSgemm(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PSingle(B)+bOffset, ldb, PSingle(A)+aOffset, lda, @BETA, PSingle(C)+cOffset, ldc{, cublasComputeType_t.CUBLAS_COMPUTE_32F, cublasGemmAlgo_t.CUBLAS_GEMM_ALGO0}));
        //SAFE_CALL(cublasGemmEx(cublas, cublasOperation_t(transB), cublasOperation_t(transA)
        //, N, M, K, @ALPHA, PSingle(B)+bOffset, CUDA_R_32F, ldb, PSingle(A)+aOffset, CUDA_R_32F, lda
        //, @BETA, PSingle(C)+cOffset, CUDA_R_32F, ldc
        //, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
      dtDouble :
        SAFE_CALL(cublasDgemm(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PDouble(B)+bOffset, ldb, PDouble(A)+aOffset, lda, @BETA, PDouble(C)+cOffset, ldc{, cublasComputeType_t.CUBLAS_COMPUTE_32F, cublasGemmAlgo_t.CUBLAS_GEMM_ALGO0}));
    end;
    goto done;// yes goto! and there is nothing you can do about it!
  end;

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

procedure TNNCuda<T>.gemmBatched(const transA, transB: boolean; const M, N,
  K: SizeInt; const ALPHA: T; const A: PCUmem; const aOffset: SizeInt;
  const lda: SizeInt; const B: PCUMem; const bOffset: SizeInt;
  const ldb: SizeInt; const BETA: T; const C: PCUMem; const cOffset: SizeInt;
  const ldc: SizeInt; const batchCount: SizeInt);

const dimThr:dim3 =(x: 8; y:32; z:1);
var
  kernelId , i: integer;
  dim : dim3;
  params : array of pointer;
  aa, bb, cc: array of pointer;
label done;
begin
  {$ifdef USE_TELEMETRY}
    tensorMetrics.start(opGemm);
  {$endif}

    //SAFE_CALL(cublasSetWorkspace(cublas, workspace, workspaceSize));

    if useBLAS = 1 then begin
      case getType() of
        dtHalf :
          SAFE_CALL(cublasHgemmBatched(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PPHalf(B), ldb, PPHalf(A), lda, @BETA, PPHalf(C), ldc, batchCount));
        dtSingle :
          SAFE_CALL(cublasSgemmBatched_64(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PPSingle(B), ldb, PPSingle(A), lda, @BETA, PPSingle(C), ldc, batchCount));
          //SAFE_CALL(cublasGemmBatchedEx(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PPointer(B), CUDA_R_32F, ldb, PPointer(A), CUDA_R_32F, lda
          //, @BETA, PPointer(C), CUDA_R_32F, ldc, batchCount, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        dtDouble :
          SAFE_CALL(cublasDgemmBatched(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PPDouble(B), ldb, PPDouble(A), lda, @BETA, PPDouble(C), ldc, batchCount));
      end;
      goto done;// yes goto! and there is nothing you can do about it!
    end;

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

    setLength(aa, batchCount);
    setLength(bb, batchCount);
    setLength(cc, batchCount);
    readBuffer(pointer(a), batchCount*sizeOf(pointer), pointer(aa));
    readBuffer(pointer(b), batchCount*sizeOf(pointer), pointer(bb));
    readBuffer(pointer(c), batchCount*sizeOf(pointer), pointer(cc));
    params := [
           @M, @N, @K, @ALPHA,
           @AA[0], @aOffset, @lda,
           @BB[0], @bOffset, @ldb,
           @BETA,
           @CC[0], @cOffset, @ldc
         ];
    SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, dimThr, ppointer(params), 0, stream));
    for i:=1 to batchCount-1 do begin
      params[4] := @AA[i];
      params[7] := @BB[i];
      params[11] := @CC[i];
      SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, dimThr, ppointer(params), 0, stream));
    end;

  done:
  {$ifdef USE_TELEMETRY}
    finish();
    tensorMetrics.finish(opGemm)
  {$endif}
end;

procedure TNNCuda<T>.gemmStridedBatched(const transA, transB: boolean; const M,
  N, K: SizeInt; const ALPHA: T; A: TCUMem; const aOffset: SizeInt;
  const lda: SizeInt; const strideA: SizeInt; B: TCUMem;
  const bOffset: SizeInt; const ldb: SizeInt; const strideB: SizeInt;
  const BETA: T; C: TCUMem; const cOffset: SizeInt; const ldc: SizeInt;
  const strideC: SizeInt; const batchCount: SizeInt);

const dimThr:dim3 =(x: 8; y:32; z:1);
var
  kernelId , i: integer;
  dim : dim3;
  params : array of pointer;
label done;
begin
  {$ifdef USE_TELEMETRY}
    tensorMetrics.start(opGemm);
  {$endif}

    //SAFE_CALL(cublasSetWorkspace(cublas, workspace, workspaceSize));

    if useBLAS = 1 then begin
      //case getType() of
      //  dtHalf:
      //    for i:=0 to batchCount-1 do
      //      cublasHgemm(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PHalf(B)+bOffset+i*strideB, ldb, PHalf(A)+aOffset+i*strideA, lda, @BETA, PHalf(C)+cOffset + i*strideC, ldc);
      //  dtSingle:
      //    for i:=0 to batchCount-1 do
      //      cublasSgemm(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PSingle(B)+bOffset+i*strideB, ldb, PSingle(A)+aOffset+i*strideA, lda, @BETA, PSingle(C)+cOffset + i*strideC, ldc);
      //  dtDouble:
      //     for i:=0 to batchCount-1 do
      //      cublasDgemm(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PDouble(B)+bOffset+i*strideB, ldb, PDouble(A)+aOffset+i*strideA, lda, @BETA, PDouble(C)+cOffset + i*strideC, ldc);
      //end;

      case getType() of
        dtHalf:
          SAFE_CALL(cublasHgemmStridedBatched(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PHalf(B)+bOffset, ldb, strideB, PHalf(A)+aOffset, lda, strideA, @BETA, PHalf(C)+cOffset, ldc, strideC, batchCount));
        dtSingle:
          SAFE_CALL(cublasSgemmStridedBatched(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PSingle(B)+bOffset, ldb, strideB, PSingle(A)+aOffset, lda, strideA, @BETA, PSingle(C)+cOffset, ldc, strideC, batchCount));
          //SAFE_CALL(cublasGemmStridedBatchedEx(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PSingle(B)+bOffset, CUDA_R_32F, ldb, strideB, PSingle(A)+aOffset, CUDA_R_32F, lda, strideA
          //, @BETA, PSingle(C)+cOffset, CUDA_R_32F, ldc, strideC, batchCount, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        dtDouble:
          SAFE_CALL(cublasDgemmStridedBatched(cublas, cublasOperation_t(transB), cublasOperation_t(transA), N, M, K, @ALPHA, PDouble(B)+bOffset, ldb, strideB, PDouble(A)+aOffset, lda, strideA, @BETA, PDouble(C)+cOffset, ldc, strideC, batchCount));
      end;
      goto done;// yes goto! and there is nothing you can do about it!
    end;

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
    ////MM := (dim.x + NUM_THREADS-1) div NUM_THREADS;
    ////NN := (dim.y + NUM_THREADS-1) div NUM_THREADS;
    //
    params := [
           @M, @N, @K, @ALPHA,
           @A, @aOffset, @lda,
           @B, @bOffset, @ldb,
           @BETA,
           @C, @cOffset, @ldc
         ];
    for i:=0 to batchCount-1 do begin
      SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim, dimThr, ppointer(params), 0, stream));
      inc(A, strideA);
      inc(B, strideB);
      inc(C, strideC);
    end;


  done:
  {$ifdef USE_TELEMETRY}
    finish();
    tensorMetrics.finish(opGemm)
  {$endif}
end;

procedure TNNCuda<T>.addvv(const N: SizeInt; const src1: TCUMem; const src1Offset,
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

procedure TNNCuda<T>.subvv(const N: SizeInt; const src1: TCUMem; const src1Offset,
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

procedure TNNCuda<T>.mulvv(const N: SizeInt; const src1: TCUMem; const src1Offset,
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

procedure TNNCuda<T>.fmavv(const N: SizeInt; const src1: TCUMem; const src1Offset,
  inca: SizeInt; const src2: TCUMem; const src2Offset, incb: SizeInt;
  const src3: TCUMem; const src3Offset, incc: SizeInt; dst: TCUMem;
  const dstOffset, incd: SizeInt);
begin

end;

procedure TNNCuda<T>.axpy(const N: SizeInt; const a: T; const x: TCUMem;
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
  //cublasSaxpy(cublas, N, @a, x, incx, y, incy);
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @a, @x, @xOffset, @incx, @y, @yOffset, @incy];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opAxpy);
{$endif}
end;

procedure TNNCuda<T>.power(const N: SizeInt; const x: TCUMem;
  const xOffset: SizeInt; const incx: SizeInt; const a: T; const y: TCUMem;
  const yOffset: SizeInt; const incy: sizeInt);
  const kernelId = 41;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opPow);
{$endif}
  //cublasSaxpy(cublas, N, @a, x, incx, y, incy);
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @x, @xOffset, @incx, @a, @y, @yOffset, @incy];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opPow);
{$endif}
end;
procedure TNNCuda<T>.scale(const N: SizeInt; const a: T; const x: TCUMem;
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

procedure TNNCuda<T>.crossEntropyLogistic(const N: SizeInt; const pred,
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

procedure TNNCuda<T>.fill(const N: SizeInt; const x: TCUMem;
  const offset: SizeInt; const val: T; const stride: SizeInt);
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

procedure TNNCuda<T>.copy(const N: SizeInt; const src: TCUMem; const srcOffset,
  inca: SizeInt; const dst: TCUMem; const dstOffset, incb: SizeInt);
const kernelId = 15;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
{$ifdef USE_TELEMETRY}
  tensorMetrics.start(opCopy);
{$endif}
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @src, @srcOffset, @inca, @dst, @dstOffset, @incb];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
{$ifdef USE_TELEMETRY}
  finish();
  tensorMetrics.finish(opCopy)
{$endif}
end;

procedure TNNCuda<T>.softmaxBatch(const N: SizeInt; const input: TCUMem;
  const iOffset: SizeInt; const batch, batch_size, groups, group_size,
  stride: SizeInt; const temp: T; const output: TCUMem; const oOffset: SizeInt);
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

procedure TNNCuda<T>.crossEntropySoftmax(const N: SizeInt; const pred,
  truth: TCUMem; delta, error: TCUMem);
const kernelId = 19;
var
  num_grids: SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @pred, @truth, @delta, @error];
  //SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelId], nnLib, 'crossEntropySoftmax'));
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.Create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda<T>.forwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
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

procedure TNNCuda<T>.backwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
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

function ceil(const a, b: SizeInt):SizeInt;
begin
  result := (1 + (a-1) div b)*b
end;

procedure TNNCuda<T>.im2col(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const im: TCUMem; const imOffset: SizeInt;
  const col: TCUMem; const colOffset: SizeInt);
const kernelId = 22;
      blockDim:dim3 =(x:4; y:64; z:1);
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
procedure EuclidGCD( a, b: SizeInt; var p, q, r:SizeInt);
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

procedure TNNCuda<T>.col2im(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const col: TCUMem; const colOffset: SizeInt;
  const im: TCUMem; const imOffset: SizeInt);
const kernelId = 24;
      blockDim:dim3 =(x:4; y:64; z:1);
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

procedure TNNCuda<T>.upSample(const aBatch, aChannels, outHeight,
  outWidth: SizeInt; const &in: TCUMem; const stride: SizeInt;
  const isForward: longint; const scale: T; const &out: TCUMem;
  const zeroIn: integer);

const kernelId = 25;
      blockDim : dim3 =(x:4; y:16; z:16);
var
  M, N, K :SizeInt;
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

procedure TNNCuda<T>.fmavss(const N: SizeInt; const src: TCUMem;
  const offset: SizeInt; const scalar, bias: T; dst: TCUMem);

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

procedure TNNCuda<T>.meansAndVars(const srcSize, dstSize, groups: sizeInt;
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

procedure TNNCuda<T>.means(const srcSize, dstSize, groups: sizeInt;
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

procedure TNNCuda<T>.variances(const srcSize, dstSize, groups: sizeInt;
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

procedure TNNCuda<T>.normalize(const srcSize, dstSize, groups: SizeInt;
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

procedure TNNCuda<T>.meansAndVarsDelta(const srcSize, dstSize, groups: SizeInt;
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

procedure TNNCuda<T>.normalizeDelta(const deltaSize, meanSize, groups: SizeInt;
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

procedure TNNCuda<T>.addDots(const N, dstSize, groups: SizeInt; const src1,
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

procedure TNNCuda<T>.forwardScale(const dstSize: SizeInt; const dst: TCUMem;
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

procedure TNNCuda<T>.forwardScaleAdd(const dstSize: SizeInt; const dst: TCUMem;
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

procedure TNNCuda<T>.forwardDropout(const N: SizeInt; const src: TCUMem;
  const probability, scale: T; rnd: TCUMem; dst: TCUMem);
const kernelId=36;
var
  num_grids : SizeInt;
  params : array of pointer;
  seed : uint64;
begin
  //randomize;
  seed := random($10000000); // next randSeed
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @seed{@RandSeed}, @probability, @scale, @src, @rnd, @dst];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda<T>.backwardDropout(const N: SizeInt; const src: TCUMem;
  const probability, scale: T; const rnd: TCUMem; dst: TCUMem);
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

procedure TNNCuda<T>.costL2(const N: SizeInt; const pred, truth, delta, error: TCUMem);
const kernelId = 38;
var
  num_grids:SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @pred, @truth, @delta, @error];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda<T>.clamp(const N: SizeInt; const alpha: T; const src,
  dst: TCUMem; const stride: SizeInt; offset: SizeInt);
const kernelId = 46;
var
  num_grids:SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @alpha, @src, @dst, @stride, @offset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda<T>.inverseSqrt(const N: SizeInt; const alpha: T; const src,
  dst: TCUMem; const stride: SizeInt; offset: SizeInt);
const kernelId = 45;
var
  num_grids:SizeInt;
  params : array of pointer;
begin
  num_grids := (N + NUM_THREADS-1) div NUM_THREADS;
  params := [@N, @src, @dst, @stride, @offset];
  SAFE_CALL(cudaLaunchKernel(FKernels[kernelId], dim3.create(num_grids), dim3.create(NUM_THREADS), ppointer(params), 0, stream));
end;

procedure TNNCuda<T>.finish();
begin
  //SAFE_CALL(cudaDeviceSynchronize());
  //{$if defined(USE_CUDA)}
  SAFE_CALL(cuStreamSynchronize(stream));
  //SAFE_CALL(cuCtxSynchronize());
  //{$else}
  //SAFE_CALL(cudaStreamSynchronize(stream));
  //{$endif}
end;

function TNNCuda<T>.compileToCUBIN(const code, name: ansistring; const headers: TArray<PAnsiChar>; const includeNames: TArray<PAnsiChar>): RawByteString;
var
  progSize : size_t;
  err: nvrtcResult;
  params : array of pansichar;
  paramsPtr : PPAnsiChar;
  log : ansistring;
begin
  result :='';
  //writeln('NVRTC compile start');
  SAFE_CALL_RTC(nvrtcCreateProgram(@prog, PAnsiChar(code), pAnsiChar(name), length(headers), Pointer(headers), Pointer(includeNames)));

  //params := [PAnsiChar('--gpu-architecture=sm_'+intToStr(devCapMajor)+intTostr(devCapMinor))];
  // Note : Delphi requires a string well formed before casting to a pansichar, this is why we cat to ansistring 1st
  params := [
              PAnsiChar(Ansistring('-arch=sm_' + intToStr(properties.major)+intTostr(properties.minor))),
              '--use_fast_math',
              PAnsiChar(ansistring('-DBLOCK='+intToStr(NUM_THREADS))),
              PAnsichar('-Dnfloat='+getTypeStr())
              ]; // causes weird exception on linux
  paramsPtr := pointer(params);
  err :=nvrtcCompileProgram(prog, length(params), paramsPtr);
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
  //writeln('NVRTC compile end');
end;

procedure TNNCuda<T>.loadCUBIN(const cubin: RawByteString);
var kernelCount : longint;
  i : SizeInt;
  //{$if defined(USE_CUDA)}
  ker : CUkernel;
  //{$else}
  //ker : cudaKernel_t;
  //{$endif}
begin
  SAFE_CALL(cudaLibraryLoadData(@nnLib, pointer(cubin), nil{pointer(FJitOptions)}, nil{pointer(FJitOptionsValues)}, 0{length(FJitOptions)},  nil{pointer(FLibOptions)}, nil{pointer(FLibOptionsValues)}, 0{length(FLibOptions)}));
  //SAFE_CALL(cudaLibraryGetKernelCount(@kernelCount, pointer(nnLib)));
  kernelCount := length(kernelNames);
  setLength(FKernels, kernelCount);
  //SAFE_CALL(cudaLibraryEnumerateKernels(pointer(FKernels), kernelCount, pointer(nnlib)));
  // cuda enumerates kernels reversed !#@%
  for i:=0 to kernelCount -1 do begin
    SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelCount-1-i], nnLib, kernelNames[i]))
  end;
end;

function TNNCuda<T>.compileFile(const filename: ansistring): RawByteString;
var
  tf : TextFile;
  f  : file;
  line, code:AnsiString;
  cuda_bin:ansistring;
begin
  //cuda_path := GetEnvironmentVariable('CUDA_PATH')+'\include';
  assert(FileExists(filename), 'Cannot find file '+filename);
  AssignFile(tf, filename);
  reset(tf);
  code :='';
  while not EOF(tf) do begin
    readln(tf, line);
    code := code +line + sLineBreak;
  end;
  CloseFile(tf);
  result := compileToCUBIN(code, filename);
  if length(result)>0 then begin
    cuda_bin := ChangeFileExt(filename, '.cubin');
    assignFile(f, cuda_bin);
    rewrite(f, 1);
    blockWrite(f, result[1], length(result));
    closeFile(f)
  end;

end;

procedure TNNCuda<T>.loadCUBinFile(const filename: ansistring);
var f: file;
  fs : SizeInt;
  bin : RawByteString;
  kernelCount : longint;
  i : SizeInt;
  //{$if defined(USE_CUDA)}
  ker : CUkernel;
  //{$else}
  //ker : cudaKernel_t;
  //{$endif}
begin
  assert(fileExists(Filename),'File does not exists!');
  SAFE_CALL(cudaLibraryLoadFromFile(@nnLib, PAnsiChar(filename), nil, nil, 0, nil, nil, 0));


  //SAFE_CALL(cudaLibraryGetKernelCount(@kernelCount, pointer(nnLib)));
  kernelCount := length(kernelNames);
  setLength(FKernels, kernelCount);
  //SAFE_CALL(cudaLibraryEnumerateKernels(pointer(FKernels), kernelCount, pointer(nnlib)));

  // cuda enumerates kernels reversed !#@%
  for i:=0 to kernelCount -1 do begin
    SAFE_CALL(cudaLibraryGetKernel(@FKernels[kernelCount-1-i], nnLib, kernelNames[i]))
  end;

  //assignFile(f, filename);
  //reset(f, 1);
  //fs := FileSize(f);
  //setLength(bin, fs);
  //BlockRead(f, bin[1], fs);
  //closeFile(f);
  //loadCUBIN(bin);
end;

initialization

finalization
  //freeAndNil(cuda)
end.

