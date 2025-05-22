unit cudaTypes;
{$ifdef FPC}
  {$mode delphi}
  {$PackRecords C}
  {$ModeSwitch advancedrecords}
{$endif}
{$H+}

interface

type
  { For backward compatibility purposes  }
    cudaDataType_t = (

        CUDA_R_32F  =  0, ///* real as a float */
        CUDA_R_64F  =  1, ///* real as a double */
        CUDA_R_16F  =  2, ///* real as a half */
        CUDA_R_8I   =  3, ///* real as a signed 8-bit int */
        CUDA_C_32F  =  4, ///* complex as a pair of float numbers */
        CUDA_C_64F  =  5, ///* complex as a pair of double numbers */
        CUDA_C_16F  =  6, ///* complex as a pair of half numbers */
        CUDA_C_8I   =  7, ///* complex as a pair of signed 8-bit int numbers */
        CUDA_R_8U   =  8, ///* real as a unsigned 8-bit int */
        CUDA_C_8U   =  9, ///* complex as a pair of unsigned 8-bit int numbers */
        CUDA_R_32I  = 10, ///* real as a signed 32-bit int */
        CUDA_C_32I  = 11, ///* complex as a pair of signed 32-bit int numbers */
        CUDA_R_32U  = 12, ///* real as a unsigned 32-bit int */
        CUDA_C_32U  = 13, ///* complex as a pair of unsigned 32-bit int numbers */
        CUDA_R_16BF = 14, ///* real as a nv_bfloat16 */
        CUDA_C_16BF = 15, ///* complex as a pair of nv_bfloat16 numbers */
        CUDA_R_4I   = 16, ///* real as a signed 4-bit int */
        CUDA_C_4I   = 17, ///* complex as a pair of signed 4-bit int numbers */
        CUDA_R_4U   = 18, ///* real as a unsigned 4-bit int */
        CUDA_C_4U   = 19, ///* complex as a pair of unsigned 4-bit int numbers */
        CUDA_R_16I  = 20, ///* real as a signed 16-bit int */
        CUDA_C_16I  = 21, ///* complex as a pair of signed 16-bit int numbers */
        CUDA_R_16U  = 22, ///* real as a unsigned 16-bit int */
        CUDA_C_16U  = 23, ///* complex as a pair of unsigned 16-bit int numbers */
        CUDA_R_64I  = 24, ///* real as a signed 64-bit int */
        CUDA_C_64I  = 25, ///* complex as a pair of signed 64-bit int numbers */
        CUDA_R_64U  = 26, ///* real as a unsigned 64-bit int */
        CUDA_C_64U  = 27, ///* complex as a pair of unsigned 64-bit int numbers */
        CUDA_R_8F_E4M3 = 28, ///* real as a nv_fp8_e4m3 */
        CUDA_R_8F_E5M2 = 29 ///* real as a nv_fp8_e5m2 */
     );
     cudaDataType = cudaDataType_t;

     libraryPropertyType_t = (
        MAJOR_VERSION,
        MINOR_VERSION,
        PATCH_LEVEL
     );
    libraryPropertyType = libraryPropertyType_t;
    CUStream_t = pointer;
    PcudaStream_t = ^cudaStream_t;
    cudaStream_t = ^CUStream_t;

    PPhalf = ^Phalf;
    Phalf = ^half;
    half = packed record
      __x : word;
    end;

    PPBfloat16 = ^Pbfloat16;
    PBfloat16 = ^Bfloat16;
    Bfloat16 = packed record
      __x : word;
    end;

    PPcuBFloat16Complex = ^PcuBFloat16Complex;
    PcuBFloat16Complex = ^cuBFloat16Complex;
    cuBFloat16Complex = packed record
      case byte of
        0 :(x: BFloat16 ; y : BFloat16);
        1 :(real: BFloat16 ; imaginary : BFloat16);
        2 :(a : array[0..1] of BFloat16);
    end;

    PPcuHalfComplex = ^PcuFloatComplex;
    PcuHalfComplex = ^cuFloatComplex;
    cuHalfComplex = packed record
      case byte of
        0 :(x: Half ; y : Half);
        1 :(real: Half ; imaginary : Half);
        2 :(a : array[0..1] of Half);
    end;

    PPcuFloatComplex = ^PcuFloatComplex;
    PcuFloatComplex = ^cuFloatComplex;
    cuFloatComplex = packed record
      case byte of
        0 :(x: Single ; y : Single);
        1 :(real: Single ; imaginary : Single);
        2 :(a : array[0..1] of Single);
    end;

    PPcuDoubleComplex = ^PcuFloatComplex;
    PcuDoubleComplex = ^cuFloatComplex;
    cuDoubleComplex = packed record
      case byte of
        0 :(x: Double ; y : Double);
        1 :(real: Double ; imaginary : Double);
        2 :(a : array[0..1] of Double)
    end;

    PPcuComplex = ^PcuComplex;
    PcuComplex = ^cuComplex;
    cuComplex = cuFloatComplex;


implementation

end.

