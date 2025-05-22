  {
   26/04/2025   converted to pascal by Haitham Shatti <haitham.shatti@gmail.com>

   * Copyright 1993-2022 NVIDIA Corporation. All rights reserved.
   *
   * NOTICE TO LICENSEE:
   *
   * This source code and/or documentation ("Licensed Deliverables") are
   * subject to NVIDIA intellectual property rights under U.S. and
   * international Copyright laws.
   *
   * These Licensed Deliverables contained herein is PROPRIETARY and
   * CONFIDENTIAL to NVIDIA and is being provided under the terms and
   * conditions of a form of NVIDIA software license agreement by and
   * between NVIDIA and Licensee ("License Agreement") or electronically
   * accepted by Licensee.  Notwithstanding any terms or conditions to
   * the contrary in the License Agreement, reproduction or disclosure
   * of the Licensed Deliverables to any third party without the express
   * written consent of NVIDIA is prohibited.
   *
   * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
   * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
   * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
   * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
   * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
   * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
   * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
   * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
   * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
   * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
   * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
   * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
   * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
   * OF THESE LICENSED DELIVERABLES.
   *
   * U.S. Government End Users.  These Licensed Deliverables are a
   * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
   * 1995), consisting of "commercial computer software" and "commercial
   * computer software documentation" as such terms are used in 48
   * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
   * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
   * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
   * U.S. Government End Users acquire the Licensed Deliverables with
   * only those rights set forth herein.
   *
   * Any use of the Licensed Deliverables in individual and commercial
   * software must include, in the user documentation and internal
   * comments to the code, the above Disclaimer and U.S. Government End
   * Users Notice.
    }
  {
   * This is the public header file for the CUBLAS library, defining the API
   *
   * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines)
   * on top of the CUDA runtime.
    }

{$ifdef FPC}
{$PACKRECORDS C}
{$mode Delphi}
{$endif}
{$define CUBLAS_V2}
unit cublas_api;
interface
uses cudaTypes;
{$i cublas.inc}

const
    CUBLAS_VER_MAJOR = 12;
    CUBLAS_VER_MINOR = 1;
    CUBLAS_VER_PATCH = 3;
    CUBLAS_VER_BUILD = 1;
    CUBLAS_VERSION   = (CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH);

  type
{$if not defined(PPSingles)}
  PPSingle = ^PSingle;
{$endif}

{$if not defined(PPSingles)}
  PPDouble = ^PDouble;
{$endif}

  { CUBLAS status type returns  }
    PcublasStatus_t = ^cublasStatus_t;
    cublasStatus_t = (CUBLAS_STATUS_SUCCESS = 0,CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16);
    //cublasStatus = cublasStatus_t;

    PcublasFillMode_t = ^cublasFillMode_t;
    cublasFillMode_t = (CUBLAS_FILL_MODE_LOWER = 0,CUBLAS_FILL_MODE_UPPER = 1,
      CUBLAS_FILL_MODE_FULL = 2);

    PcublasDiagType_t = ^cublasDiagType_t;
    cublasDiagType_t = (CUBLAS_DIAG_NON_UNIT = 0,CUBLAS_DIAG_UNIT = 1
      );

    PcublasSideMode_t = ^cublasSideMode_t;
    cublasSideMode_t = (CUBLAS_SIDE_LEFT = 0,CUBLAS_SIDE_RIGHT = 1
      );

    PcublasOperation_t = ^cublasOperation_t;
    cublasOperation_t = (CUBLAS_OP_N = 0,CUBLAS_OP_T = 1,CUBLAS_OP_C = 2,
  { conjugate, placeholder - not supported in the current release  }
      CUBLAS_OP_CONJG = 3
      );

    PcublasPointerMode_t = ^cublasPointerMode_t;
    cublasPointerMode_t = (CUBLAS_POINTER_MODE_HOST = 0,CUBLAS_POINTER_MODE_DEVICE = 1
      );

    PcublasAtomicsMode_t = ^cublasAtomicsMode_t;
    cublasAtomicsMode_t = (CUBLAS_ATOMICS_NOT_ALLOWED = 0,CUBLAS_ATOMICS_ALLOWED = 1
      );
  {For different GEMM algorithm  }
  { sliced 32x32 }
  { sliced 64x32 }
  { sliced 128x32 }
  { sliced 32x32  -splitK }
  { sliced 64x32  -splitK }
  { sliced 128x32 -splitK }

    PcublasGemmAlgo_t = ^cublasGemmAlgo_t;
    cublasGemmAlgo_t = (CUBLAS_GEMM_DEFAULT = -(1),
      CUBLAS_GEMM_ALGO0 = 0,CUBLAS_GEMM_ALGO1 = 1,
      CUBLAS_GEMM_ALGO2 = 2,CUBLAS_GEMM_ALGO3 = 3,
      CUBLAS_GEMM_ALGO4 = 4,CUBLAS_GEMM_ALGO5 = 5,
      CUBLAS_GEMM_ALGO6 = 6,CUBLAS_GEMM_ALGO7 = 7,
      CUBLAS_GEMM_ALGO8 = 8,CUBLAS_GEMM_ALGO9 = 9,
      CUBLAS_GEMM_ALGO10 = 10,CUBLAS_GEMM_ALGO11 = 11,
      CUBLAS_GEMM_ALGO12 = 12,CUBLAS_GEMM_ALGO13 = 13,
      CUBLAS_GEMM_ALGO14 = 14,CUBLAS_GEMM_ALGO15 = 15,
      CUBLAS_GEMM_ALGO16 = 16,CUBLAS_GEMM_ALGO17 = 17,
      CUBLAS_GEMM_ALGO18 = 18,CUBLAS_GEMM_ALGO19 = 19,
      CUBLAS_GEMM_ALGO20 = 20,CUBLAS_GEMM_ALGO21 = 21,
      CUBLAS_GEMM_ALGO22 = 22,CUBLAS_GEMM_ALGO23 = 23,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
      //CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
      CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
      CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
      CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
      CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
      CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
      CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
      CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
      CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
      CUBLAS_GEMM_ALGO15_TENSOR_OP = 115);
const
      CUBLAS_GEMM_DFALT = CUBLAS_GEMM_DEFAULT;
  { synonym if CUBLAS_OP_C  }
      CUBLAS_OP_HERMITAN = CUBLAS_OP_C;
      CUBLAS_GEMM_DFALT_TENSOR_OP = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
type

  {Enum for default math mode/tensor operation }
  { deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release  }
  { same as using matching _PEDANTIC compute type when using cublas<T>routine calls or cublasEx() calls with
       cudaDataType as compute type  }
  { allow accelerating single precision routines using TF32 tensor cores  }
  { flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines
       with lower size output type  }

    PcublasMath_t = ^cublasMath_t;
    cublasMath_t = (CUBLAS_DEFAULT_MATH = 0,CUBLAS_TENSOR_OP_MATH = 1,
      CUBLAS_PEDANTIC_MATH = 2,CUBLAS_TF32_TENSOR_OP_MATH = 3,
      CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
      );

    PcublasDataType_t = ^cublasDataType_t;
    cublasDataType_t = cudaDataType;
  { Enum for compute type
   *
   * - default types provide best available performance using all available hardware features
   *   and guarantee internal storage precision with at least the same precision and range;
   * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
   * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
    }
  { half - default  }
  { half - pedantic  }
  { float - default  }
  { float - pedantic  }
  { float - fast, allows down-converting inputs to half or TF32  }
  { float - fast, allows down-converting inputs to bfloat16 or TF32  }
  { float - fast, allows down-converting inputs to TF32  }
  { double - default  }
  { double - pedantic  }
  { signed 32-bit int - default  }
  { signed 32-bit int - pedantic  }

    PcublasComputeType_t = ^cublasComputeType_t;
    cublasComputeType_t = (CUBLAS_COMPUTE_16F = 64,CUBLAS_COMPUTE_16F_PEDANTIC = 65,
      CUBLAS_COMPUTE_32F = 68,CUBLAS_COMPUTE_32F_PEDANTIC = 69,
      CUBLAS_COMPUTE_64F = 70,
      CUBLAS_COMPUTE_64F_PEDANTIC = 71,CUBLAS_COMPUTE_32I = 72,
      CUBLAS_COMPUTE_32I_PEDANTIC = 73,
      CUBLAS_COMPUTE_32F_FAST_16F = 74,CUBLAS_COMPUTE_32F_FAST_16BF = 75,
      CUBLAS_COMPUTE_32F_FAST_TF32 = 77
      );
  { Opaque structure holding CUBLAS library context  }
    PcublasContext = ^cublasContext;
    cublasContext = record
        {undefined structure}
      end;


    PcublasHandle_t = ^cublasHandle_t;
    cublasHandle_t = PcublasContext;
  { Cublas logging  }

    PcublasLogCallback = ^cublasLogCallback;
    cublasLogCallback = procedure (msg:Pchar); WINAPI;
  { cuBLAS Exported API   }
  { --------------- CUBLAS Helper Functions  ----------------  }

  var
    //cublasInit : function():cublasStatus;  WINAPI;
    //cublasShutdown : function():cublasStatus; WINAPI;
    //cublasGetError : function():cublasStatus; WINAPI;

    //cublasGetVersion : function(version:Plongint):cublasStatus; WINAPI;
    //cublasAlloc : function(n, elemSize:longint; devicePtr:PPointer):cublasStatus; WINAPI;
    //
    //cublasFree : function(devicePtr:pointer):cublasStatus; WINAPI;
    //
    //cublasSetKernelStream : function(stream:cudaStream_t):cublasStatus; WINAPI;

    cublasCreate{$ifdef CUBLAS_V2}, cublasCreate_v2 {$endif} : function(handle:PcublasHandle_t):cublasStatus_t; WINAPI;
    cublasDestroy{$ifdef CUBLAS_V2}, cublasDestroy_v2 {$endif} : function(handle:cublasHandle_t):cublasStatus_t; WINAPI;
    cublasGetVersion{$ifdef CUBLAS_V2}, cublasGetVersion_v2 {$endif} : function(handle:cublasHandle_t; version:Plongint):cublasStatus_t; WINAPI;
    cublasGetProperty : function(_type:libraryPropertyType; value:Plongint):cublasStatus_t; WINAPI;
    cublasGetCudartVersion : function:size_t; WINAPI;
    cublasSetWorkspace{$ifdef CUBLAS_V2}, cublasSetWorkspace_v2 {$endif} : function(handle:cublasHandle_t; workspace:pointer; workspaceSizeInBytes:size_t):cublasStatus_t; WINAPI;
    cublasSetStream{$ifdef CUBLAS_V2}, cublasSetStream_v2 {$endif} : function(handle:cublasHandle_t; streamId:cudaStream_t):cublasStatus_t; WINAPI;
    cublasGetStream{$ifdef CUBLAS_V2}, cublasGetStream_v2 {$endif} : function(handle:cublasHandle_t; streamId:PcudaStream_t):cublasStatus_t; WINAPI;
    cublasGetPointerMode{$ifdef CUBLAS_V2}, cublasGetPointerMode_v2 {$endif} : function(handle:cublasHandle_t; mode:PcublasPointerMode_t):cublasStatus_t; WINAPI;
    cublasSetPointerMode{$ifdef CUBLAS_V2}, cublasSetPointerMode_v2 {$endif} : function(handle:cublasHandle_t; mode:cublasPointerMode_t):cublasStatus_t; WINAPI;
    cublasGetAtomicsMode : function(handle:cublasHandle_t; mode:PcublasAtomicsMode_t):cublasStatus_t; WINAPI;
    cublasSetAtomicsMode : function(handle:cublasHandle_t; mode:cublasAtomicsMode_t):cublasStatus_t; WINAPI;
    cublasGetMathMode : function(handle:cublasHandle_t; mode:PcublasMath_t):cublasStatus_t; WINAPI;
    cublasSetMathMode : function(handle:cublasHandle_t; mode:cublasMath_t):cublasStatus_t; WINAPI;
    cublasGetSmCountTarget : function(handle:cublasHandle_t; smCountTarget:Plongint):cublasStatus_t; WINAPI;
    cublasSetSmCountTarget : function(handle:cublasHandle_t; smCountTarget:longint):cublasStatus_t; WINAPI;

    cublasGetStatusName : function(status:cublasStatus_t):Pchar; WINAPI;

    cublasGetStatusString : function(status:cublasStatus_t):Pchar; WINAPI;

    cublasLoggerConfigure : function(logIsOn:longint; logToStdOut:longint; logToStdErr:longint; logFileName:Pchar):cublasStatus_t; WINAPI;
    cublasSetLoggerCallback : function(userCallback:cublasLogCallback):cublasStatus_t; WINAPI;
    cublasGetLoggerCallback : function(userCallback:PcublasLogCallback):cublasStatus_t; WINAPI;

    cublasSetVector : function(n:longint; elemSize:longint; x:pointer; incx:longint; devicePtr:pointer; 
      incy:longint):cublasStatus_t; WINAPI;

    cublasSetVector_64 : function(n:int64; elemSize:int64; x:pointer; incx:int64; devicePtr:pointer; 
      incy:int64):cublasStatus_t; WINAPI;

    cublasGetVector : function(n:longint; elemSize:longint; x:pointer; incx:longint; y:pointer; 
      incy:longint):cublasStatus_t; WINAPI;

    cublasGetVector_64 : function(n:int64; elemSize:int64; x:pointer; incx:int64; y:pointer; 
      incy:int64):cublasStatus_t; WINAPI;

    cublasSetMatrix : function(rows:longint; cols:longint; elemSize:longint; A:pointer; lda:longint; 
      B:pointer; ldb:longint):cublasStatus_t; WINAPI;

    cublasSetMatrix_64 : function(rows:int64; cols:int64; elemSize:int64; A:pointer; lda:int64; 
      B:pointer; ldb:int64):cublasStatus_t; WINAPI;

    cublasGetMatrix : function(rows:longint; cols:longint; elemSize:longint; A:pointer; lda:longint; 
      B:pointer; ldb:longint):cublasStatus_t; WINAPI;

    cublasGetMatrix_64 : function(rows:int64; cols:int64; elemSize:int64; A:pointer; lda:int64; 
      B:pointer; ldb:int64):cublasStatus_t; WINAPI;

    cublasSetVectorAsync : function(n:longint; elemSize:longint; hostPtr:pointer; incx:longint; devicePtr:pointer; 
      incy:longint; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasSetVectorAsync_64 : function(n:int64; elemSize:int64; hostPtr:pointer; incx:int64; devicePtr:pointer; 
      incy:int64; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasGetVectorAsync : function(n:longint; elemSize:longint; devicePtr:pointer; incx:longint; hostPtr:pointer; 
      incy:longint; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasGetVectorAsync_64 : function(n:int64; elemSize:int64; devicePtr:pointer; incx:int64; hostPtr:pointer; 
      incy:int64; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasSetMatrixAsync : function(rows:longint; cols:longint; elemSize:longint; A:pointer; lda:longint; 
      B:pointer; ldb:longint; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasSetMatrixAsync_64 : function(rows:int64; cols:int64; elemSize:int64; A:pointer; lda:int64; 
      B:pointer; ldb:int64; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasGetMatrixAsync : function(rows:longint; cols:longint; elemSize:longint; A:pointer; lda:longint; 
      B:pointer; ldb:longint; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasGetMatrixAsync_64 : function(rows:int64; cols:int64; elemSize:int64; A:pointer; lda:int64; 
      B:pointer; ldb:int64; stream:cudaStream_t):cublasStatus_t; WINAPI;

    cublasXerbla : procedure(srName:Pchar; info:longint); WINAPI;
  { --------------- CUBLAS BLAS1 Functions  ----------------  }

    cublasNrm2Ex : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      result:pointer; resultType:cudaDataType; executionType:cudaDataType):cublasStatus_t; WINAPI;

    cublasNrm2Ex_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      result:pointer; resultType:cudaDataType; executionType:cudaDataType):cublasStatus_t; WINAPI;

    cublasSnrm2{$ifdef CUBLAS_V2}, cublasSnrm2_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; result:Psingle):cublasStatus_t; WINAPI;

    cublasSnrm2_64{$ifdef CUBLAS_V2}, cublasSnrm2_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; result:Psingle):cublasStatus_t; WINAPI;

    cublasDnrm2{$ifdef CUBLAS_V2}, cublasDnrm2_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; result:Pdouble):cublasStatus_t; WINAPI;

    cublasDnrm2_64{$ifdef CUBLAS_V2}, cublasDnrm2_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; result:Pdouble):cublasStatus_t; WINAPI;

    cublasScnrm2{$ifdef CUBLAS_V2}, cublasScnrm2_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; result:Psingle):cublasStatus_t; WINAPI;

    cublasScnrm2_64{$ifdef CUBLAS_V2}, cublasScnrm2_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; result:Psingle):cublasStatus_t; WINAPI;

    cublasDznrm2{$ifdef CUBLAS_V2}, cublasDznrm2_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; result:Pdouble):cublasStatus_t; WINAPI;

    cublasDznrm2_64{$ifdef CUBLAS_V2}, cublasDznrm2_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; result:Pdouble):cublasStatus_t; WINAPI;


    cublasDotEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      y:pointer; yType:cudaDataType; incy:longint; result:pointer; resultType:cudaDataType; 
      executionType:cudaDataType):cublasStatus_t; WINAPI;


    cublasDotEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      y:pointer; yType:cudaDataType; incy:int64; result:pointer; resultType:cudaDataType; 
      executionType:cudaDataType):cublasStatus_t; WINAPI;


    cublasDotcEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      y:pointer; yType:cudaDataType; incy:longint; result:pointer; resultType:cudaDataType; 
      executionType:cudaDataType):cublasStatus_t; WINAPI;


    cublasDotcEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      y:pointer; yType:cudaDataType; incy:int64; result:pointer; resultType:cudaDataType; 
      executionType:cudaDataType):cublasStatus_t; WINAPI;


    cublasSdot{$ifdef CUBLAS_V2}, cublasSdot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; y:Psingle; 
      incy:longint; result:Psingle):cublasStatus_t; WINAPI;


    cublasSdot_64{$ifdef CUBLAS_V2}, cublasSdot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; y:Psingle; 
      incy:int64; result:Psingle):cublasStatus_t; WINAPI;


    cublasDdot{$ifdef CUBLAS_V2}, cublasDdot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; y:Pdouble; 
      incy:longint; result:Pdouble):cublasStatus_t; WINAPI;


    cublasDdot_64{$ifdef CUBLAS_V2}, cublasDdot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; y:Pdouble; 
      incy:int64; result:Pdouble):cublasStatus_t; WINAPI;


    cublasCdotu{$ifdef CUBLAS_V2}, cublasCdotu_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; y:PcuComplex; 
      incy:longint; result:PcuComplex):cublasStatus_t; WINAPI;


    cublasCdotu_64{$ifdef CUBLAS_V2}, cublasCdotu_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; y:PcuComplex; 
      incy:int64; result:PcuComplex):cublasStatus_t; WINAPI;


    cublasCdotc{$ifdef CUBLAS_V2}, cublasCdotc_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; y:PcuComplex; 
      incy:longint; result:PcuComplex):cublasStatus_t; WINAPI;


    cublasCdotc_64{$ifdef CUBLAS_V2}, cublasCdotc_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; y:PcuComplex; 
      incy:int64; result:PcuComplex):cublasStatus_t; WINAPI;


    cublasZdotu{$ifdef CUBLAS_V2}, cublasZdotu_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; y:PcuDoubleComplex; 
      incy:longint; result:PcuDoubleComplex):cublasStatus_t; WINAPI;


    cublasZdotu_64{$ifdef CUBLAS_V2}, cublasZdotu_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; y:PcuDoubleComplex; 
      incy:int64; result:PcuDoubleComplex):cublasStatus_t; WINAPI;


    cublasZdotc{$ifdef CUBLAS_V2}, cublasZdotc_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; y:PcuDoubleComplex; 
      incy:longint; result:PcuDoubleComplex):cublasStatus_t; WINAPI;


    cublasZdotc_64{$ifdef CUBLAS_V2}, cublasZdotc_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; y:PcuDoubleComplex; 
      incy:int64; result:PcuDoubleComplex):cublasStatus_t; WINAPI;

    cublasScalEx : function(handle:cublasHandle_t; n:longint; alpha:pointer; alphaType:cudaDataType; x:pointer; 
      xType:cudaDataType; incx:longint; executionType:cudaDataType):cublasStatus_t; WINAPI;

    cublasScalEx_64 : function(handle:cublasHandle_t; n:int64; alpha:pointer; alphaType:cudaDataType; x:pointer; 
      xType:cudaDataType; incx:int64; executionType:cudaDataType):cublasStatus_t; WINAPI;

    cublasSscal{$ifdef CUBLAS_V2}, cublasSscal_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:Psingle; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasSscal_64{$ifdef CUBLAS_V2}, cublasSscal_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:Psingle; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDscal{$ifdef CUBLAS_V2}, cublasDscal_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:Pdouble; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDscal_64{$ifdef CUBLAS_V2}, cublasDscal_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:Pdouble; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCscal{$ifdef CUBLAS_V2}, cublasCscal_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:PcuComplex; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCscal_64{$ifdef CUBLAS_V2}, cublasCscal_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:PcuComplex; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasCsscal{$ifdef CUBLAS_V2}, cublasCsscal_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:Psingle; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCsscal_64{$ifdef CUBLAS_V2}, cublasCsscal_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:Psingle; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZscal{$ifdef CUBLAS_V2}, cublasZscal_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZscal_64{$ifdef CUBLAS_V2}, cublasZscal_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZdscal{$ifdef CUBLAS_V2}, cublasZdscal_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:Pdouble; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZdscal_64{$ifdef CUBLAS_V2}, cublasZdscal_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:Pdouble; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;


    cublasAxpyEx : function(handle:cublasHandle_t; n:longint; alpha:pointer; alphaType:cudaDataType; x:pointer; 
      xType:cudaDataType; incx:longint; y:pointer; yType:cudaDataType; incy:longint; 
      executiontype:cudaDataType):cublasStatus_t; WINAPI;


    cublasAxpyEx_64 : function(handle:cublasHandle_t; n:int64; alpha:pointer; alphaType:cudaDataType; x:pointer; 
      xType:cudaDataType; incx:int64; y:pointer; yType:cudaDataType; incy:int64; 
      executiontype:cudaDataType):cublasStatus_t; WINAPI;


    cublasSaxpy{$ifdef CUBLAS_V2}, cublasSaxpy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:Psingle; x:Psingle; incx:longint; 
      y:Psingle; incy:longint):cublasStatus_t; WINAPI;


    cublasSaxpy_64{$ifdef CUBLAS_V2}, cublasSaxpy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:Psingle; x:Psingle; incx:int64; 
      y:Psingle; incy:int64):cublasStatus_t; WINAPI;


    cublasDaxpy{$ifdef CUBLAS_V2}, cublasDaxpy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:Pdouble; x:Pdouble; incx:longint; 
      y:Pdouble; incy:longint):cublasStatus_t; WINAPI;


    cublasDaxpy_64{$ifdef CUBLAS_V2}, cublasDaxpy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:Pdouble; x:Pdouble; incx:int64; 
      y:Pdouble; incy:int64):cublasStatus_t; WINAPI;


    cublasCaxpy{$ifdef CUBLAS_V2}, cublasCaxpy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:PcuComplex; x:PcuComplex; incx:longint; 
      y:PcuComplex; incy:longint):cublasStatus_t; WINAPI;


    cublasCaxpy_64{$ifdef CUBLAS_V2}, cublasCaxpy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:PcuComplex; x:PcuComplex; incx:int64; 
      y:PcuComplex; incy:int64):cublasStatus_t; WINAPI;


    cublasZaxpy{$ifdef CUBLAS_V2}, cublasZaxpy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; incx:longint; 
      y:PcuDoubleComplex; incy:longint):cublasStatus_t; WINAPI;


    cublasZaxpy_64{$ifdef CUBLAS_V2}, cublasZaxpy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; incx:int64; 
      y:PcuDoubleComplex; incy:int64):cublasStatus_t; WINAPI;

    cublasCopyEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      y:pointer; yType:cudaDataType; incy:longint):cublasStatus_t; WINAPI;

    cublasCopyEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      y:pointer; yType:cudaDataType; incy:int64):cublasStatus_t; WINAPI;

    cublasScopy{$ifdef CUBLAS_V2}, cublasScopy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; y:Psingle; 
      incy:longint):cublasStatus_t; WINAPI;

    cublasScopy_64{$ifdef CUBLAS_V2}, cublasScopy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; y:Psingle; 
      incy:int64):cublasStatus_t; WINAPI;

    cublasDcopy{$ifdef CUBLAS_V2}, cublasDcopy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; y:Pdouble; 
      incy:longint):cublasStatus_t; WINAPI;

    cublasDcopy_64{$ifdef CUBLAS_V2}, cublasDcopy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; y:Pdouble; 
      incy:int64):cublasStatus_t; WINAPI;

    cublasCcopy{$ifdef CUBLAS_V2}, cublasCcopy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; y:PcuComplex; 
      incy:longint):cublasStatus_t; WINAPI;

    cublasCcopy_64{$ifdef CUBLAS_V2}, cublasCcopy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; y:PcuComplex; 
      incy:int64):cublasStatus_t; WINAPI;

    cublasZcopy{$ifdef CUBLAS_V2}, cublasZcopy_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; y:PcuDoubleComplex; 
      incy:longint):cublasStatus_t; WINAPI;

    cublasZcopy_64{$ifdef CUBLAS_V2}, cublasZcopy_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; y:PcuDoubleComplex; 
      incy:int64):cublasStatus_t; WINAPI;
    cublasSswap{$ifdef CUBLAS_V2}, cublasSswap_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; y:Psingle; 
      incy:longint):cublasStatus_t; WINAPI;
    cublasSswap_64{$ifdef CUBLAS_V2}, cublasSswap_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; y:Psingle; 
      incy:int64):cublasStatus_t; WINAPI;
    cublasDswap{$ifdef CUBLAS_V2}, cublasDswap_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; y:Pdouble; 
      incy:longint):cublasStatus_t; WINAPI;
    cublasDswap_64{$ifdef CUBLAS_V2}, cublasDswap_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; y:Pdouble; 
      incy:int64):cublasStatus_t; WINAPI;
    cublasCswap{$ifdef CUBLAS_V2}, cublasCswap_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; y:PcuComplex; 
      incy:longint):cublasStatus_t; WINAPI;
    cublasCswap_64{$ifdef CUBLAS_V2}, cublasCswap_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; y:PcuComplex; 
      incy:int64):cublasStatus_t; WINAPI;
    cublasZswap{$ifdef CUBLAS_V2}, cublasZswap_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; y:PcuDoubleComplex; 
      incy:longint):cublasStatus_t; WINAPI;
    cublasZswap_64{$ifdef CUBLAS_V2}, cublasZswap_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; y:PcuDoubleComplex; 
      incy:int64):cublasStatus_t; WINAPI;
    cublasSwapEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      y:pointer; yType:cudaDataType; incy:longint):cublasStatus_t; WINAPI;
    cublasSwapEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      y:pointer; yType:cudaDataType; incy:int64):cublasStatus_t; WINAPI;

    cublasIsamax{$ifdef CUBLAS_V2}, cublasIsamax_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIsamax_64{$ifdef CUBLAS_V2}, cublasIsamax_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIdamax{$ifdef CUBLAS_V2}, cublasIdamax_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIdamax_64{$ifdef CUBLAS_V2}, cublasIdamax_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIcamax{$ifdef CUBLAS_V2}, cublasIcamax_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIcamax_64{$ifdef CUBLAS_V2}, cublasIcamax_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIzamax{$ifdef CUBLAS_V2}, cublasIzamax_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIzamax_64{$ifdef CUBLAS_V2}, cublasIzamax_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIamaxEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      result:Plongint):cublasStatus_t; WINAPI;

    cublasIamaxEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      result:PInt64):cublasStatus_t; WINAPI;

    cublasIsamin{$ifdef CUBLAS_V2}, cublasIsamin_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIsamin_64{$ifdef CUBLAS_V2}, cublasIsamin_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIdamin{$ifdef CUBLAS_V2}, cublasIdamin_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIdamin_64{$ifdef CUBLAS_V2}, cublasIdamin_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIcamin{$ifdef CUBLAS_V2}, cublasIcamin_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIcamin_64{$ifdef CUBLAS_V2}, cublasIcamin_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIzamin{$ifdef CUBLAS_V2}, cublasIzamin_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; result:Plongint):cublasStatus_t; WINAPI;

    cublasIzamin_64{$ifdef CUBLAS_V2}, cublasIzamin_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; result:PInt64):cublasStatus_t; WINAPI;

    cublasIaminEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      result:Plongint):cublasStatus_t; WINAPI;

    cublasIaminEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      result:PInt64):cublasStatus_t; WINAPI;

    cublasAsumEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      result:pointer; resultType:cudaDataType; executiontype:cudaDataType):cublasStatus_t; WINAPI;

    cublasAsumEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      result:pointer; resultType:cudaDataType; executiontype:cudaDataType):cublasStatus_t; WINAPI;

    cublasSasum{$ifdef CUBLAS_V2}, cublasSasum_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; result:Psingle):cublasStatus_t; WINAPI;

    cublasSasum_64{$ifdef CUBLAS_V2}, cublasSasum_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; result:Psingle):cublasStatus_t; WINAPI;

    cublasDasum{$ifdef CUBLAS_V2}, cublasDasum_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; result:Pdouble):cublasStatus_t; WINAPI;

    cublasDasum_64{$ifdef CUBLAS_V2}, cublasDasum_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; result:Pdouble):cublasStatus_t; WINAPI;

    cublasScasum{$ifdef CUBLAS_V2}, cublasScasum_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; result:Psingle):cublasStatus_t; WINAPI;

    cublasScasum_64{$ifdef CUBLAS_V2}, cublasScasum_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; result:Psingle):cublasStatus_t; WINAPI;

    cublasDzasum{$ifdef CUBLAS_V2}, cublasDzasum_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; result:Pdouble):cublasStatus_t; WINAPI;

    cublasDzasum_64{$ifdef CUBLAS_V2}, cublasDzasum_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; result:Pdouble):cublasStatus_t; WINAPI;


    cublasSrot{$ifdef CUBLAS_V2}, cublasSrot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; y:Psingle; 
      incy:longint; c:Psingle; s:Psingle):cublasStatus_t; WINAPI;


    cublasSrot_64{$ifdef CUBLAS_V2}, cublasSrot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; y:Psingle; 
      incy:int64; c:Psingle; s:Psingle):cublasStatus_t; WINAPI;


    cublasDrot{$ifdef CUBLAS_V2}, cublasDrot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; y:Pdouble; 
      incy:longint; c:Pdouble; s:Pdouble):cublasStatus_t; WINAPI;


    cublasDrot_64{$ifdef CUBLAS_V2}, cublasDrot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; y:Pdouble; 
      incy:int64; c:Pdouble; s:Pdouble):cublasStatus_t; WINAPI;


    cublasCrot{$ifdef CUBLAS_V2}, cublasCrot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; y:PcuComplex; 
      incy:longint; c:Psingle; s:PcuComplex):cublasStatus_t; WINAPI;


    cublasCrot_64{$ifdef CUBLAS_V2}, cublasCrot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; y:PcuComplex; 
      incy:int64; c:Psingle; s:PcuComplex):cublasStatus_t; WINAPI;


    cublasCsrot{$ifdef CUBLAS_V2}, cublasCsrot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuComplex; incx:longint; y:PcuComplex; 
      incy:longint; c:Psingle; s:Psingle):cublasStatus_t; WINAPI;


    cublasCsrot_64{$ifdef CUBLAS_V2}, cublasCsrot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuComplex; incx:int64; y:PcuComplex; 
      incy:int64; c:Psingle; s:Psingle):cublasStatus_t; WINAPI;


    cublasZrot{$ifdef CUBLAS_V2}, cublasZrot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; y:PcuDoubleComplex; 
      incy:longint; c:Pdouble; s:PcuDoubleComplex):cublasStatus_t; WINAPI;


    cublasZrot_64{$ifdef CUBLAS_V2}, cublasZrot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; y:PcuDoubleComplex; 
      incy:int64; c:Pdouble; s:PcuDoubleComplex):cublasStatus_t; WINAPI;


    cublasZdrot{$ifdef CUBLAS_V2}, cublasZdrot_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:PcuDoubleComplex; incx:longint; y:PcuDoubleComplex; 
      incy:longint; c:Pdouble; s:Pdouble):cublasStatus_t; WINAPI;


    cublasZdrot_64{$ifdef CUBLAS_V2}, cublasZdrot_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:PcuDoubleComplex; incx:int64; y:PcuDoubleComplex; 
      incy:int64; c:Pdouble; s:Pdouble):cublasStatus_t; WINAPI;


    cublasRotEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      y:pointer; yType:cudaDataType; incy:longint; c:pointer; s:pointer; 
      csType:cudaDataType; executiontype:cudaDataType):cublasStatus_t; WINAPI;


    cublasRotEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      y:pointer; yType:cudaDataType; incy:int64; c:pointer; s:pointer; 
      csType:cudaDataType; executiontype:cudaDataType):cublasStatus_t; WINAPI;
    cublasSrotg{$ifdef CUBLAS_V2}, cublasSrotg_v2 {$endif} : function(handle:cublasHandle_t; a:Psingle; b:Psingle; c:Psingle; s:Psingle):cublasStatus_t; WINAPI;
    cublasDrotg{$ifdef CUBLAS_V2}, cublasDrotg_v2 {$endif} : function(handle:cublasHandle_t; a:Pdouble; b:Pdouble; c:Pdouble; s:Pdouble):cublasStatus_t; WINAPI;
    cublasCrotg{$ifdef CUBLAS_V2}, cublasCrotg_v2 {$endif} : function(handle:cublasHandle_t; a:PcuComplex; b:PcuComplex; c:Psingle; s:PcuComplex):cublasStatus_t; WINAPI;
    cublasZrotg{$ifdef CUBLAS_V2}, cublasZrotg_v2 {$endif} : function(handle:cublasHandle_t; a:PcuDoubleComplex; b:PcuDoubleComplex; c:Pdouble; s:PcuDoubleComplex):cublasStatus_t; WINAPI;
    cublasRotgEx : function(handle:cublasHandle_t; a:pointer; b:pointer; abType:cudaDataType; c:pointer; 
      s:pointer; csType:cudaDataType; executiontype:cudaDataType):cublasStatus_t; WINAPI;

    cublasSrotm{$ifdef CUBLAS_V2}, cublasSrotm_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Psingle; incx:longint; y:Psingle; 
      incy:longint; param:Psingle):cublasStatus_t; WINAPI;

    cublasSrotm_64{$ifdef CUBLAS_V2}, cublasSrotm_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Psingle; incx:int64; y:Psingle; 
      incy:int64; param:Psingle):cublasStatus_t; WINAPI;

    cublasDrotm{$ifdef CUBLAS_V2}, cublasDrotm_v2 {$endif} : function(handle:cublasHandle_t; n:longint; x:Pdouble; incx:longint; y:Pdouble; 
      incy:longint; param:Pdouble):cublasStatus_t; WINAPI;

    cublasDrotm_64{$ifdef CUBLAS_V2}, cublasDrotm_v2_64 {$endif} : function(handle:cublasHandle_t; n:int64; x:Pdouble; incx:int64; y:Pdouble; 
      incy:int64; param:Pdouble):cublasStatus_t; WINAPI;

    cublasRotmEx : function(handle:cublasHandle_t; n:longint; x:pointer; xType:cudaDataType; incx:longint; 
      y:pointer; yType:cudaDataType; incy:longint; param:pointer; paramType:cudaDataType; 
      executiontype:cudaDataType):cublasStatus_t; WINAPI;

    cublasRotmEx_64 : function(handle:cublasHandle_t; n:int64; x:pointer; xType:cudaDataType; incx:int64; 
      y:pointer; yType:cudaDataType; incy:int64; param:pointer; paramType:cudaDataType; 
      executiontype:cudaDataType):cublasStatus_t; WINAPI;

    cublasSrotmg{$ifdef CUBLAS_V2}, cublasSrotmg_v2 {$endif} : function(handle:cublasHandle_t; d1:Psingle; d2:Psingle; x1:Psingle; y1:Psingle; 
      param:Psingle):cublasStatus_t; WINAPI;

    cublasDrotmg{$ifdef CUBLAS_V2}, cublasDrotmg_v2 {$endif} : function(handle:cublasHandle_t; d1:Pdouble; d2:Pdouble; x1:Pdouble; y1:Pdouble; 
      param:Pdouble):cublasStatus_t; WINAPI;

    cublasRotmgEx : function(handle:cublasHandle_t; d1:pointer; d1Type:cudaDataType; d2:pointer; d2Type:cudaDataType; 
      x1:pointer; x1Type:cudaDataType; y1:pointer; y1Type:cudaDataType; param:pointer; 
      paramType:cudaDataType; executiontype:cudaDataType):cublasStatus_t; WINAPI;
  { --------------- CUBLAS BLAS2 Functions  ----------------  }
  { GEMV  }




    cublasSgemv{$ifdef CUBLAS_V2}, cublasSgemv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      A:Psingle; lda:longint; x:Psingle; incx:longint; beta:Psingle; 
      y:Psingle; incy:longint):cublasStatus_t; WINAPI;




    cublasSgemv_64{$ifdef CUBLAS_V2}, cublasSgemv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      A:Psingle; lda:int64; x:Psingle; incx:int64; beta:Psingle; 
      y:Psingle; incy:int64):cublasStatus_t; WINAPI;




    cublasDgemv{$ifdef CUBLAS_V2}, cublasDgemv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Pdouble; 
      A:Pdouble; lda:longint; x:Pdouble; incx:longint; beta:Pdouble; 
      y:Pdouble; incy:longint):cublasStatus_t; WINAPI;




    cublasDgemv_64{$ifdef CUBLAS_V2}, cublasDgemv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Pdouble; 
      A:Pdouble; lda:int64; x:Pdouble; incx:int64; beta:Pdouble; 
      y:Pdouble; incy:int64):cublasStatus_t; WINAPI;




    cublasCgemv{$ifdef CUBLAS_V2}, cublasCgemv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:PcuComplex; 
      A:PcuComplex; lda:longint; x:PcuComplex; incx:longint; beta:PcuComplex; 
      y:PcuComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasCgemv_64{$ifdef CUBLAS_V2}, cublasCgemv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:PcuComplex; 
      A:PcuComplex; lda:int64; x:PcuComplex; incx:int64; beta:PcuComplex; 
      y:PcuComplex; incy:int64):cublasStatus_t; WINAPI;




    cublasZgemv{$ifdef CUBLAS_V2}, cublasZgemv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:PcuDoubleComplex; 
      A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; incx:longint; beta:PcuDoubleComplex; 
      y:PcuDoubleComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasZgemv_64{$ifdef CUBLAS_V2}, cublasZgemv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:PcuDoubleComplex; 
      A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; incx:int64; beta:PcuDoubleComplex; 
      y:PcuDoubleComplex; incy:int64):cublasStatus_t; WINAPI;
  { GBMV  }




    cublasSgbmv{$ifdef CUBLAS_V2}, cublasSgbmv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; kl:longint; 
      ku:longint; alpha:Psingle; A:Psingle; lda:longint; x:Psingle; 
      incx:longint; beta:Psingle; y:Psingle; incy:longint):cublasStatus_t; WINAPI;




    cublasSgbmv_64{$ifdef CUBLAS_V2}, cublasSgbmv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; kl:int64; 
      ku:int64; alpha:Psingle; A:Psingle; lda:int64; x:Psingle; 
      incx:int64; beta:Psingle; y:Psingle; incy:int64):cublasStatus_t; WINAPI;




    cublasDgbmv{$ifdef CUBLAS_V2}, cublasDgbmv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; kl:longint; 
      ku:longint; alpha:Pdouble; A:Pdouble; lda:longint; x:Pdouble; 
      incx:longint; beta:Pdouble; y:Pdouble; incy:longint):cublasStatus_t; WINAPI;




    cublasDgbmv_64{$ifdef CUBLAS_V2}, cublasDgbmv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; kl:int64; 
      ku:int64; alpha:Pdouble; A:Pdouble; lda:int64; x:Pdouble; 
      incx:int64; beta:Pdouble; y:Pdouble; incy:int64):cublasStatus_t; WINAPI;




    cublasCgbmv{$ifdef CUBLAS_V2}, cublasCgbmv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; kl:longint; 
      ku:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; x:PcuComplex; 
      incx:longint; beta:PcuComplex; y:PcuComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasCgbmv_64{$ifdef CUBLAS_V2}, cublasCgbmv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; kl:int64; 
      ku:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; x:PcuComplex; 
      incx:int64; beta:PcuComplex; y:PcuComplex; incy:int64):cublasStatus_t; WINAPI;




    cublasZgbmv{$ifdef CUBLAS_V2}, cublasZgbmv_v2 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; kl:longint; 
      ku:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; 
      incx:longint; beta:PcuDoubleComplex; y:PcuDoubleComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasZgbmv_64{$ifdef CUBLAS_V2}, cublasZgbmv_v2_64 {$endif} : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; kl:int64; 
      ku:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; 
      incx:int64; beta:PcuDoubleComplex; y:PcuDoubleComplex; incy:int64):cublasStatus_t; WINAPI;
  { TRMV  }

    cublasStrmv{$ifdef CUBLAS_V2}, cublasStrmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:Psingle; lda:longint; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasStrmv_64{$ifdef CUBLAS_V2}, cublasStrmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:Psingle; lda:int64; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDtrmv{$ifdef CUBLAS_V2}, cublasDtrmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:Pdouble; lda:longint; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDtrmv_64{$ifdef CUBLAS_V2}, cublasDtrmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:Pdouble; lda:int64; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCtrmv{$ifdef CUBLAS_V2}, cublasCtrmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:PcuComplex; lda:longint; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCtrmv_64{$ifdef CUBLAS_V2}, cublasCtrmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:PcuComplex; lda:int64; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZtrmv{$ifdef CUBLAS_V2}, cublasZtrmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZtrmv_64{$ifdef CUBLAS_V2}, cublasZtrmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;
  { TBMV  }

    cublasStbmv{$ifdef CUBLAS_V2}, cublasStbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:Psingle; lda:longint; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasStbmv_64{$ifdef CUBLAS_V2}, cublasStbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:Psingle; lda:int64; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDtbmv{$ifdef CUBLAS_V2}, cublasDtbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:Pdouble; lda:longint; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDtbmv_64{$ifdef CUBLAS_V2}, cublasDtbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:Pdouble; lda:int64; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCtbmv{$ifdef CUBLAS_V2}, cublasCtbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:PcuComplex; lda:longint; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCtbmv_64{$ifdef CUBLAS_V2}, cublasCtbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:PcuComplex; lda:int64; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZtbmv{$ifdef CUBLAS_V2}, cublasZtbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZtbmv_64{$ifdef CUBLAS_V2}, cublasZtbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;
  { TPMV  }

    cublasStpmv{$ifdef CUBLAS_V2}, cublasStpmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:Psingle; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasStpmv_64{$ifdef CUBLAS_V2}, cublasStpmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:Psingle; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDtpmv{$ifdef CUBLAS_V2}, cublasDtpmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:Pdouble; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDtpmv_64{$ifdef CUBLAS_V2}, cublasDtpmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:Pdouble; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCtpmv{$ifdef CUBLAS_V2}, cublasCtpmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:PcuComplex; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCtpmv_64{$ifdef CUBLAS_V2}, cublasCtpmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:PcuComplex; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZtpmv{$ifdef CUBLAS_V2}, cublasZtpmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:PcuDoubleComplex; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZtpmv_64{$ifdef CUBLAS_V2}, cublasZtpmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:PcuDoubleComplex; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;
  { TRSV  }

    cublasStrsv{$ifdef CUBLAS_V2}, cublasStrsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:Psingle; lda:longint; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasStrsv_64{$ifdef CUBLAS_V2}, cublasStrsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:Psingle; lda:int64; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDtrsv{$ifdef CUBLAS_V2}, cublasDtrsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:Pdouble; lda:longint; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDtrsv_64{$ifdef CUBLAS_V2}, cublasDtrsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:Pdouble; lda:int64; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCtrsv{$ifdef CUBLAS_V2}, cublasCtrsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:PcuComplex; lda:longint; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCtrsv_64{$ifdef CUBLAS_V2}, cublasCtrsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:PcuComplex; lda:int64; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZtrsv{$ifdef CUBLAS_V2}, cublasZtrsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZtrsv_64{$ifdef CUBLAS_V2}, cublasZtrsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;
  { TPSV  }

    cublasStpsv{$ifdef CUBLAS_V2}, cublasStpsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:Psingle; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasStpsv_64{$ifdef CUBLAS_V2}, cublasStpsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:Psingle; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDtpsv{$ifdef CUBLAS_V2}, cublasDtpsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:Pdouble; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDtpsv_64{$ifdef CUBLAS_V2}, cublasDtpsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:Pdouble; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCtpsv{$ifdef CUBLAS_V2}, cublasCtpsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:PcuComplex; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCtpsv_64{$ifdef CUBLAS_V2}, cublasCtpsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:PcuComplex; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZtpsv{$ifdef CUBLAS_V2}, cublasZtpsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      AP:PcuDoubleComplex; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZtpsv_64{$ifdef CUBLAS_V2}, cublasZtpsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      AP:PcuDoubleComplex; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;
  { TBSV  }

    cublasStbsv{$ifdef CUBLAS_V2}, cublasStbsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:Psingle; lda:longint; x:Psingle; incx:longint):cublasStatus_t; WINAPI;

    cublasStbsv_64{$ifdef CUBLAS_V2}, cublasStbsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:Psingle; lda:int64; x:Psingle; incx:int64):cublasStatus_t; WINAPI;

    cublasDtbsv{$ifdef CUBLAS_V2}, cublasDtbsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:Pdouble; lda:longint; x:Pdouble; incx:longint):cublasStatus_t; WINAPI;

    cublasDtbsv_64{$ifdef CUBLAS_V2}, cublasDtbsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:Pdouble; lda:int64; x:Pdouble; incx:int64):cublasStatus_t; WINAPI;

    cublasCtbsv{$ifdef CUBLAS_V2}, cublasCtbsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:PcuComplex; lda:longint; x:PcuComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasCtbsv_64{$ifdef CUBLAS_V2}, cublasCtbsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:PcuComplex; lda:int64; x:PcuComplex; incx:int64):cublasStatus_t; WINAPI;

    cublasZtbsv{$ifdef CUBLAS_V2}, cublasZtbsv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:longint; 
      k:longint; A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; incx:longint):cublasStatus_t; WINAPI;

    cublasZtbsv_64{$ifdef CUBLAS_V2}, cublasZtbsv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; n:int64; 
      k:int64; A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; incx:int64):cublasStatus_t; WINAPI;
  { SYMV/HEMV  }




    cublasSsymv{$ifdef CUBLAS_V2}, cublasSsymv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; A:Psingle; 
      lda:longint; x:Psingle; incx:longint; beta:Psingle; y:Psingle; 
      incy:longint):cublasStatus_t; WINAPI;




    cublasSsymv_64{$ifdef CUBLAS_V2}, cublasSsymv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; A:Psingle; 
      lda:int64; x:Psingle; incx:int64; beta:Psingle; y:Psingle; 
      incy:int64):cublasStatus_t; WINAPI;




    cublasDsymv{$ifdef CUBLAS_V2}, cublasDsymv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; A:Pdouble; 
      lda:longint; x:Pdouble; incx:longint; beta:Pdouble; y:Pdouble; 
      incy:longint):cublasStatus_t; WINAPI;




    cublasDsymv_64{$ifdef CUBLAS_V2}, cublasDsymv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; A:Pdouble; 
      lda:int64; x:Pdouble; incx:int64; beta:Pdouble; y:Pdouble; 
      incy:int64):cublasStatus_t; WINAPI;




    cublasCsymv{$ifdef CUBLAS_V2}, cublasCsymv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; A:PcuComplex; 
      lda:longint; x:PcuComplex; incx:longint; beta:PcuComplex; y:PcuComplex; 
      incy:longint):cublasStatus_t; WINAPI;




    cublasCsymv_64{$ifdef CUBLAS_V2}, cublasCsymv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; A:PcuComplex; 
      lda:int64; x:PcuComplex; incx:int64; beta:PcuComplex; y:PcuComplex; 
      incy:int64):cublasStatus_t; WINAPI;




    cublasZsymv{$ifdef CUBLAS_V2}, cublasZsymv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; 
      lda:longint; x:PcuDoubleComplex; incx:longint; beta:PcuDoubleComplex; y:PcuDoubleComplex; 
      incy:longint):cublasStatus_t; WINAPI;




    cublasZsymv_64{$ifdef CUBLAS_V2}, cublasZsymv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; 
      lda:int64; x:PcuDoubleComplex; incx:int64; beta:PcuDoubleComplex; y:PcuDoubleComplex; 
      incy:int64):cublasStatus_t; WINAPI;




    cublasChemv{$ifdef CUBLAS_V2}, cublasChemv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; A:PcuComplex; 
      lda:longint; x:PcuComplex; incx:longint; beta:PcuComplex; y:PcuComplex; 
      incy:longint):cublasStatus_t; WINAPI;




    cublasChemv_64{$ifdef CUBLAS_V2}, cublasChemv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; A:PcuComplex; 
      lda:int64; x:PcuComplex; incx:int64; beta:PcuComplex; y:PcuComplex; 
      incy:int64):cublasStatus_t; WINAPI;




    cublasZhemv{$ifdef CUBLAS_V2}, cublasZhemv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; 
      lda:longint; x:PcuDoubleComplex; incx:longint; beta:PcuDoubleComplex; y:PcuDoubleComplex; 
      incy:longint):cublasStatus_t; WINAPI;




    cublasZhemv_64{$ifdef CUBLAS_V2}, cublasZhemv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; 
      lda:int64; x:PcuDoubleComplex; incx:int64; beta:PcuDoubleComplex; y:PcuDoubleComplex; 
      incy:int64):cublasStatus_t; WINAPI;
  { SBMV/HBMV  }




    cublasSsbmv{$ifdef CUBLAS_V2}, cublasSsbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; k:longint; alpha:Psingle; 
      A:Psingle; lda:longint; x:Psingle; incx:longint; beta:Psingle; 
      y:Psingle; incy:longint):cublasStatus_t; WINAPI;




    cublasSsbmv_64{$ifdef CUBLAS_V2}, cublasSsbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; k:int64; alpha:Psingle; 
      A:Psingle; lda:int64; x:Psingle; incx:int64; beta:Psingle; 
      y:Psingle; incy:int64):cublasStatus_t; WINAPI;




    cublasDsbmv{$ifdef CUBLAS_V2}, cublasDsbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; k:longint; alpha:Pdouble; 
      A:Pdouble; lda:longint; x:Pdouble; incx:longint; beta:Pdouble; 
      y:Pdouble; incy:longint):cublasStatus_t; WINAPI;




    cublasDsbmv_64{$ifdef CUBLAS_V2}, cublasDsbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; k:int64; alpha:Pdouble; 
      A:Pdouble; lda:int64; x:Pdouble; incx:int64; beta:Pdouble; 
      y:Pdouble; incy:int64):cublasStatus_t; WINAPI;




    cublasChbmv{$ifdef CUBLAS_V2}, cublasChbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; k:longint; alpha:PcuComplex; 
      A:PcuComplex; lda:longint; x:PcuComplex; incx:longint; beta:PcuComplex; 
      y:PcuComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasChbmv_64{$ifdef CUBLAS_V2}, cublasChbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; k:int64; alpha:PcuComplex; 
      A:PcuComplex; lda:int64; x:PcuComplex; incx:int64; beta:PcuComplex; 
      y:PcuComplex; incy:int64):cublasStatus_t; WINAPI;




    cublasZhbmv{$ifdef CUBLAS_V2}, cublasZhbmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; k:longint; alpha:PcuDoubleComplex; 
      A:PcuDoubleComplex; lda:longint; x:PcuDoubleComplex; incx:longint; beta:PcuDoubleComplex; 
      y:PcuDoubleComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasZhbmv_64{$ifdef CUBLAS_V2}, cublasZhbmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; k:int64; alpha:PcuDoubleComplex; 
      A:PcuDoubleComplex; lda:int64; x:PcuDoubleComplex; incx:int64; beta:PcuDoubleComplex; 
      y:PcuDoubleComplex; incy:int64):cublasStatus_t; WINAPI;
  { SPMV/HPMV  }




    cublasSspmv{$ifdef CUBLAS_V2}, cublasSspmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; AP:Psingle; 
      x:Psingle; incx:longint; beta:Psingle; y:Psingle; incy:longint):cublasStatus_t; WINAPI;




    cublasSspmv_64{$ifdef CUBLAS_V2}, cublasSspmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; AP:Psingle; 
      x:Psingle; incx:int64; beta:Psingle; y:Psingle; incy:int64):cublasStatus_t; WINAPI;




    cublasDspmv{$ifdef CUBLAS_V2}, cublasDspmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; AP:Pdouble; 
      x:Pdouble; incx:longint; beta:Pdouble; y:Pdouble; incy:longint):cublasStatus_t; WINAPI;




    cublasDspmv_64{$ifdef CUBLAS_V2}, cublasDspmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; AP:Pdouble; 
      x:Pdouble; incx:int64; beta:Pdouble; y:Pdouble; incy:int64):cublasStatus_t; WINAPI;




    cublasChpmv{$ifdef CUBLAS_V2}, cublasChpmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; AP:PcuComplex; 
      x:PcuComplex; incx:longint; beta:PcuComplex; y:PcuComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasChpmv_64{$ifdef CUBLAS_V2}, cublasChpmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; AP:PcuComplex; 
      x:PcuComplex; incx:int64; beta:PcuComplex; y:PcuComplex; incy:int64):cublasStatus_t; WINAPI;




    cublasZhpmv{$ifdef CUBLAS_V2}, cublasZhpmv_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; AP:PcuDoubleComplex; 
      x:PcuDoubleComplex; incx:longint; beta:PcuDoubleComplex; y:PcuDoubleComplex; incy:longint):cublasStatus_t; WINAPI;




    cublasZhpmv_64{$ifdef CUBLAS_V2}, cublasZhpmv_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; AP:PcuDoubleComplex; 
      x:PcuDoubleComplex; incx:int64; beta:PcuDoubleComplex; y:PcuDoubleComplex; incy:int64):cublasStatus_t; WINAPI;
  { GER  }



    cublasSger{$ifdef CUBLAS_V2}, cublasSger_v2 {$endif} : function(handle:cublasHandle_t; m:longint; n:longint; alpha:Psingle; x:Psingle; 
      incx:longint; y:Psingle; incy:longint; A:Psingle; lda:longint):cublasStatus_t; WINAPI;



    cublasSger_64{$ifdef CUBLAS_V2}, cublasSger_v2_64 {$endif} : function(handle:cublasHandle_t; m:int64; n:int64; alpha:Psingle; x:Psingle; 
      incx:int64; y:Psingle; incy:int64; A:Psingle; lda:int64):cublasStatus_t; WINAPI;



    cublasDger{$ifdef CUBLAS_V2}, cublasDger_v2 {$endif} : function(handle:cublasHandle_t; m:longint; n:longint; alpha:Pdouble; x:Pdouble; 
      incx:longint; y:Pdouble; incy:longint; A:Pdouble; lda:longint):cublasStatus_t; WINAPI;



    cublasDger_64{$ifdef CUBLAS_V2}, cublasDger_v2_64 {$endif} : function(handle:cublasHandle_t; m:int64; n:int64; alpha:Pdouble; x:Pdouble; 
      incx:int64; y:Pdouble; incy:int64; A:Pdouble; lda:int64):cublasStatus_t; WINAPI;



    cublasCgeru{$ifdef CUBLAS_V2}, cublasCgeru_v2 {$endif} : function(handle:cublasHandle_t; m:longint; n:longint; alpha:PcuComplex; x:PcuComplex; 
      incx:longint; y:PcuComplex; incy:longint; A:PcuComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasCgeru_64{$ifdef CUBLAS_V2}, cublasCgeru_v2_64 {$endif} : function(handle:cublasHandle_t; m:int64; n:int64; alpha:PcuComplex; x:PcuComplex; 
      incx:int64; y:PcuComplex; incy:int64; A:PcuComplex; lda:int64):cublasStatus_t; WINAPI;



    cublasCgerc{$ifdef CUBLAS_V2}, cublasCgerc_v2 {$endif} : function(handle:cublasHandle_t; m:longint; n:longint; alpha:PcuComplex; x:PcuComplex; 
      incx:longint; y:PcuComplex; incy:longint; A:PcuComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasCgerc_64{$ifdef CUBLAS_V2}, cublasCgerc_v2_64 {$endif} : function(handle:cublasHandle_t; m:int64; n:int64; alpha:PcuComplex; x:PcuComplex; 
      incx:int64; y:PcuComplex; incy:int64; A:PcuComplex; lda:int64):cublasStatus_t; WINAPI;



    cublasZgeru{$ifdef CUBLAS_V2}, cublasZgeru_v2 {$endif} : function(handle:cublasHandle_t; m:longint; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:longint; y:PcuDoubleComplex; incy:longint; A:PcuDoubleComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasZgeru_64{$ifdef CUBLAS_V2}, cublasZgeru_v2_64 {$endif} : function(handle:cublasHandle_t; m:int64; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:int64; y:PcuDoubleComplex; incy:int64; A:PcuDoubleComplex; lda:int64):cublasStatus_t; WINAPI;



    cublasZgerc{$ifdef CUBLAS_V2}, cublasZgerc_v2 {$endif} : function(handle:cublasHandle_t; m:longint; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:longint; y:PcuDoubleComplex; incy:longint; A:PcuDoubleComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasZgerc_64{$ifdef CUBLAS_V2}, cublasZgerc_v2_64 {$endif} : function(handle:cublasHandle_t; m:int64; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:int64; y:PcuDoubleComplex; incy:int64; A:PcuDoubleComplex; lda:int64):cublasStatus_t; WINAPI;
  { SYR/HER  }


    cublasSsyr{$ifdef CUBLAS_V2}, cublasSsyr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; x:Psingle; 
      incx:longint; A:Psingle; lda:longint):cublasStatus_t; WINAPI;


    cublasSsyr_64{$ifdef CUBLAS_V2}, cublasSsyr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; x:Psingle; 
      incx:int64; A:Psingle; lda:int64):cublasStatus_t; WINAPI;


    cublasDsyr{$ifdef CUBLAS_V2}, cublasDsyr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; x:Pdouble; 
      incx:longint; A:Pdouble; lda:longint):cublasStatus_t; WINAPI;


    cublasDsyr_64{$ifdef CUBLAS_V2}, cublasDsyr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; x:Pdouble; 
      incx:int64; A:Pdouble; lda:int64):cublasStatus_t; WINAPI;


    cublasCsyr{$ifdef CUBLAS_V2}, cublasCsyr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; x:PcuComplex; 
      incx:longint; A:PcuComplex; lda:longint):cublasStatus_t; WINAPI;


    cublasCsyr_64{$ifdef CUBLAS_V2}, cublasCsyr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; x:PcuComplex; 
      incx:int64; A:PcuComplex; lda:int64):cublasStatus_t; WINAPI;


    cublasZsyr{$ifdef CUBLAS_V2}, cublasZsyr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:longint; A:PcuDoubleComplex; lda:longint):cublasStatus_t; WINAPI;


    cublasZsyr_64{$ifdef CUBLAS_V2}, cublasZsyr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:int64; A:PcuDoubleComplex; lda:int64):cublasStatus_t; WINAPI;


    cublasCher{$ifdef CUBLAS_V2}, cublasCher_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; x:PcuComplex; 
      incx:longint; A:PcuComplex; lda:longint):cublasStatus_t; WINAPI;


    cublasCher_64{$ifdef CUBLAS_V2}, cublasCher_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; x:PcuComplex; 
      incx:int64; A:PcuComplex; lda:int64):cublasStatus_t; WINAPI;


    cublasZher{$ifdef CUBLAS_V2}, cublasZher_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; x:PcuDoubleComplex; 
      incx:longint; A:PcuDoubleComplex; lda:longint):cublasStatus_t; WINAPI;


    cublasZher_64{$ifdef CUBLAS_V2}, cublasZher_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; x:PcuDoubleComplex; 
      incx:int64; A:PcuDoubleComplex; lda:int64):cublasStatus_t; WINAPI;
  { SPR/HPR  }


    cublasSspr{$ifdef CUBLAS_V2}, cublasSspr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; x:Psingle; 
      incx:longint; AP:Psingle):cublasStatus_t; WINAPI;


    cublasSspr_64{$ifdef CUBLAS_V2}, cublasSspr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; x:Psingle; 
      incx:int64; AP:Psingle):cublasStatus_t; WINAPI;


    cublasDspr{$ifdef CUBLAS_V2}, cublasDspr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; x:Pdouble; 
      incx:longint; AP:Pdouble):cublasStatus_t; WINAPI;


    cublasDspr_64{$ifdef CUBLAS_V2}, cublasDspr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; x:Pdouble; 
      incx:int64; AP:Pdouble):cublasStatus_t; WINAPI;


    cublasChpr{$ifdef CUBLAS_V2}, cublasChpr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; x:PcuComplex; 
      incx:longint; AP:PcuComplex):cublasStatus_t; WINAPI;


    cublasChpr_64{$ifdef CUBLAS_V2}, cublasChpr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; x:PcuComplex; 
      incx:int64; AP:PcuComplex):cublasStatus_t; WINAPI;


    cublasZhpr{$ifdef CUBLAS_V2}, cublasZhpr_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; x:PcuDoubleComplex; 
      incx:longint; AP:PcuDoubleComplex):cublasStatus_t; WINAPI;


    cublasZhpr_64{$ifdef CUBLAS_V2}, cublasZhpr_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; x:PcuDoubleComplex; 
      incx:int64; AP:PcuDoubleComplex):cublasStatus_t; WINAPI;
  { SYR2/HER2  }



    cublasSsyr2{$ifdef CUBLAS_V2}, cublasSsyr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; x:Psingle; 
      incx:longint; y:Psingle; incy:longint; A:Psingle; lda:longint):cublasStatus_t; WINAPI;



    cublasSsyr2_64{$ifdef CUBLAS_V2}, cublasSsyr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; x:Psingle; 
      incx:int64; y:Psingle; incy:int64; A:Psingle; lda:int64):cublasStatus_t; WINAPI;



    cublasDsyr2{$ifdef CUBLAS_V2}, cublasDsyr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; x:Pdouble; 
      incx:longint; y:Pdouble; incy:longint; A:Pdouble; lda:longint):cublasStatus_t; WINAPI;



    cublasDsyr2_64{$ifdef CUBLAS_V2}, cublasDsyr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; x:Pdouble; 
      incx:int64; y:Pdouble; incy:int64; A:Pdouble; lda:int64):cublasStatus_t; WINAPI;



    cublasCsyr2{$ifdef CUBLAS_V2}, cublasCsyr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; x:PcuComplex; 
      incx:longint; y:PcuComplex; incy:longint; A:PcuComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasCsyr2_64{$ifdef CUBLAS_V2}, cublasCsyr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; x:PcuComplex; 
      incx:int64; y:PcuComplex; incy:int64; A:PcuComplex; lda:int64):cublasStatus_t; WINAPI;



    cublasZsyr2{$ifdef CUBLAS_V2}, cublasZsyr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:longint; y:PcuDoubleComplex; incy:longint; A:PcuDoubleComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasZsyr2_64{$ifdef CUBLAS_V2}, cublasZsyr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:int64; y:PcuDoubleComplex; incy:int64; A:PcuDoubleComplex; lda:int64):cublasStatus_t; WINAPI;



    cublasCher2{$ifdef CUBLAS_V2}, cublasCher2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; x:PcuComplex; 
      incx:longint; y:PcuComplex; incy:longint; A:PcuComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasCher2_64{$ifdef CUBLAS_V2}, cublasCher2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; x:PcuComplex; 
      incx:int64; y:PcuComplex; incy:int64; A:PcuComplex; lda:int64):cublasStatus_t; WINAPI;



    cublasZher2{$ifdef CUBLAS_V2}, cublasZher2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:longint; y:PcuDoubleComplex; incy:longint; A:PcuDoubleComplex; lda:longint):cublasStatus_t; WINAPI;



    cublasZher2_64{$ifdef CUBLAS_V2}, cublasZher2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:int64; y:PcuDoubleComplex; incy:int64; A:PcuDoubleComplex; lda:int64):cublasStatus_t; WINAPI;
  { SPR2/HPR2  }



    cublasSspr2{$ifdef CUBLAS_V2}, cublasSspr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Psingle; x:Psingle; 
      incx:longint; y:Psingle; incy:longint; AP:Psingle):cublasStatus_t; WINAPI;



    cublasSspr2_64{$ifdef CUBLAS_V2}, cublasSspr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Psingle; x:Psingle; 
      incx:int64; y:Psingle; incy:int64; AP:Psingle):cublasStatus_t; WINAPI;



    cublasDspr2{$ifdef CUBLAS_V2}, cublasDspr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:Pdouble; x:Pdouble; 
      incx:longint; y:Pdouble; incy:longint; AP:Pdouble):cublasStatus_t; WINAPI;



    cublasDspr2_64{$ifdef CUBLAS_V2}, cublasDspr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:Pdouble; x:Pdouble; 
      incx:int64; y:Pdouble; incy:int64; AP:Pdouble):cublasStatus_t; WINAPI;



    cublasChpr2{$ifdef CUBLAS_V2}, cublasChpr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuComplex; x:PcuComplex; 
      incx:longint; y:PcuComplex; incy:longint; AP:PcuComplex):cublasStatus_t; WINAPI;



    cublasChpr2_64{$ifdef CUBLAS_V2}, cublasChpr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuComplex; x:PcuComplex; 
      incx:int64; y:PcuComplex; incy:int64; AP:PcuComplex):cublasStatus_t; WINAPI;



    cublasZhpr2{$ifdef CUBLAS_V2}, cublasZhpr2_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:longint; y:PcuDoubleComplex; incy:longint; AP:PcuDoubleComplex):cublasStatus_t; WINAPI;



    cublasZhpr2_64{$ifdef CUBLAS_V2}, cublasZhpr2_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:int64; alpha:PcuDoubleComplex; x:PcuDoubleComplex; 
      incx:int64; y:PcuDoubleComplex; incy:int64; AP:PcuDoubleComplex):cublasStatus_t; WINAPI;
  { BATCH GEMV  }







    cublasSgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      Aarray:PPsingle; lda:longint; xarray:PPsingle; incx:longint; beta:Psingle; 
      yarray:PPsingle; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasSgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      Aarray:PPsingle; lda:int64; xarray:PPsingle; incx:int64; beta:Psingle; 
      yarray:PPsingle; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasDgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Pdouble; 
      Aarray:PPdouble; lda:longint; xarray:PPdouble; incx:longint; beta:Pdouble; 
      yarray:PPdouble; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasDgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Pdouble; 
      Aarray:PPdouble; lda:int64; xarray:PPdouble; incx:int64; beta:Pdouble; 
      yarray:PPdouble; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasCgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:PcuComplex; 
      Aarray:PPcuComplex; lda:longint; xarray:PPcuComplex; incx:longint; beta:PcuComplex; 
      yarray:PPcuComplex; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasCgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:PcuComplex; 
      Aarray:PPcuComplex; lda:int64; xarray:PPcuComplex; incx:int64; beta:PcuComplex; 
      yarray:PPcuComplex; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasZgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:PcuDoubleComplex; 
      Aarray:PPcuDoubleComplex; lda:longint; xarray:PPcuDoubleComplex; incx:longint; beta:PcuDoubleComplex; 
      yarray:PPcuDoubleComplex; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasZgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:PcuDoubleComplex; 
      Aarray:PPcuDoubleComplex; lda:int64; xarray:PPcuDoubleComplex; incx:int64; beta:PcuDoubleComplex; 
      yarray:PPcuDoubleComplex; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasHSHgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      Aarray:PPhalf; lda:longint; xarray:PPhalf; incx:longint; beta:Psingle;
      yarray:PPhalf; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasHSHgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      Aarray:PPhalf; lda:int64; xarray:PPhalf; incx:int64; beta:Psingle;
      yarray:PPhalf; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasHSSgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      Aarray:PPhalf; lda:longint; xarray:PPhalf; incx:longint; beta:Psingle;
      yarray:PPsingle; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasHSSgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      Aarray:PPhalf; lda:int64; xarray:PPhalf; incx:int64; beta:Psingle;
      yarray:PPsingle; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasTSTgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      Aarray:PPbfloat16; lda:longint; xarray:PPbfloat16; incx:longint; beta:Psingle;
      yarray:PPbfloat16; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasTSTgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      Aarray:PPbfloat16; lda:int64; xarray:PPbfloat16; incx:int64; beta:Psingle;
      yarray:PPbfloat16; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasTSSgemvBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      Aarray:PPbfloat16; lda:longint; xarray:PPbfloat16; incx:longint; beta:Psingle;
      yarray:PPsingle; incy:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasTSSgemvBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      Aarray:PPbfloat16; lda:int64; xarray:PPbfloat16; incx:int64; beta:Psingle;
      yarray:PPsingle; incy:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasSgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      A:Psingle; lda:longint; strideA:int64; x:Psingle; incx:longint; 
      stridex:int64; beta:Psingle; y:Psingle; incy:longint; stridey:int64; 
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasSgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      A:Psingle; lda:int64; strideA:int64; x:Psingle; incx:int64; 
      stridex:int64; beta:Psingle; y:Psingle; incy:int64; stridey:int64; 
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasDgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Pdouble; 
      A:Pdouble; lda:longint; strideA:int64; x:Pdouble; incx:longint; 
      stridex:int64; beta:Pdouble; y:Pdouble; incy:longint; stridey:int64; 
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasDgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Pdouble; 
      A:Pdouble; lda:int64; strideA:int64; x:Pdouble; incx:int64; 
      stridex:int64; beta:Pdouble; y:Pdouble; incy:int64; stridey:int64; 
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasCgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:PcuComplex; 
      A:PcuComplex; lda:longint; strideA:int64; x:PcuComplex; incx:longint; 
      stridex:int64; beta:PcuComplex; y:PcuComplex; incy:longint; stridey:int64; 
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasCgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:PcuComplex; 
      A:PcuComplex; lda:int64; strideA:int64; x:PcuComplex; incx:int64; 
      stridex:int64; beta:PcuComplex; y:PcuComplex; incy:int64; stridey:int64; 
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasZgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:PcuDoubleComplex; 
      A:PcuDoubleComplex; lda:longint; strideA:int64; x:PcuDoubleComplex; incx:longint; 
      stridex:int64; beta:PcuDoubleComplex; y:PcuDoubleComplex; incy:longint; stridey:int64; 
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasZgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:PcuDoubleComplex; 
      A:PcuDoubleComplex; lda:int64; strideA:int64; x:PcuDoubleComplex; incx:int64; 
      stridex:int64; beta:PcuDoubleComplex; y:PcuDoubleComplex; incy:int64; stridey:int64; 
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasHSHgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      A:Phalf; lda:longint; strideA:int64; x:Phalf; incx:longint;
      stridex:int64; beta:Psingle; y:Phalf; incy:longint; stridey:int64;
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasHSHgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      A:Phalf; lda:int64; strideA:int64; x:Phalf; incx:int64;
      stridex:int64; beta:Psingle; y:Phalf; incy:int64; stridey:int64;
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasHSSgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      A:Phalf; lda:longint; strideA:int64; x:Phalf; incx:longint;
      stridex:int64; beta:Psingle; y:Psingle; incy:longint; stridey:int64; 
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasHSSgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      A:Phalf; lda:int64; strideA:int64; x:Phalf; incx:int64;
      stridex:int64; beta:Psingle; y:Psingle; incy:int64; stridey:int64; 
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasTSTgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      A:Pbfloat16; lda:longint; strideA:int64; x:Pbfloat16; incx:longint;
      stridex:int64; beta:Psingle; y:Pbfloat16; incy:longint; stridey:int64;
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasTSTgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      A:Pbfloat16; lda:int64; strideA:int64; x:Pbfloat16; incx:int64;
      stridex:int64; beta:Psingle; y:Pbfloat16; incy:int64; stridey:int64;
      batchCount:int64):cublasStatus_t; WINAPI;




    cublasTSSgemvStridedBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; alpha:Psingle; 
      A:Pbfloat16; lda:longint; strideA:int64; x:Pbfloat16; incx:longint;
      stridex:int64; beta:Psingle; y:Psingle; incy:longint; stridey:int64; 
      batchCount:longint):cublasStatus_t; WINAPI;




    cublasTSSgemvStridedBatched_64 : function(handle:cublasHandle_t; trans:cublasOperation_t; m:int64; n:int64; alpha:Psingle; 
      A:Pbfloat16; lda:int64; strideA:int64; x:Pbfloat16; incx:int64;
      stridex:int64; beta:Psingle; y:Psingle; incy:int64; stridey:int64; 
      batchCount:int64):cublasStatus_t; WINAPI;
  { ---------------- CUBLAS BLAS3 Functions ----------------  }
  { GEMM  }




    cublasSgemm{$ifdef CUBLAS_V2}, cublasSgemm_v2 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Psingle; A:Psingle; lda:longint; B:Psingle; 
      ldb:longint; beta:Psingle; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;




    cublasSgemm_64{$ifdef CUBLAS_V2}, cublasSgemm_v2_64 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Psingle; A:Psingle; lda:int64; B:Psingle; 
      ldb:int64; beta:Psingle; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;




    cublasDgemm{$ifdef CUBLAS_V2}, cublasDgemm_v2 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Pdouble; A:Pdouble; lda:longint; B:Pdouble; 
      ldb:longint; beta:Pdouble; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;




    cublasDgemm_64{$ifdef CUBLAS_V2}, cublasDgemm_v2_64 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Pdouble; A:Pdouble; lda:int64; B:Pdouble; 
      ldb:int64; beta:Pdouble; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;




    cublasCgemm{$ifdef CUBLAS_V2}, cublasCgemm_v2 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; 
      ldb:longint; beta:PcuComplex; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCgemm_64{$ifdef CUBLAS_V2}, cublasCgemm_v2_64 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; 
      ldb:int64; beta:PcuComplex; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasCgemm3m : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; 
      ldb:longint; beta:PcuComplex; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCgemm3m_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; 
      ldb:int64; beta:PcuComplex; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasCgemm3mEx : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:longint; 
      B:pointer; Btype:cudaDataType; ldb:longint; beta:PcuComplex; C:pointer; 
      Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;




    cublasCgemm3mEx_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:int64; 
      B:pointer; Btype:cudaDataType; ldb:int64; beta:PcuComplex; C:pointer; 
      Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;




    cublasZgemm{$ifdef CUBLAS_V2}, cublasZgemm_v2 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; 
      ldb:longint; beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZgemm_64{$ifdef CUBLAS_V2}, cublasZgemm_v2_64 {$endif} : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; 
      ldb:int64; beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZgemm3m : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; 
      ldb:longint; beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZgemm3m_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; 
      ldb:int64; beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasHgemm : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Phalf; A:Phalf; lda:longint; B:Phalf;
      ldb:longint; beta:Phalf; C:Phalf; ldc:longint):cublasStatus_t; WINAPI;




    cublasHgemm_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Phalf; A:Phalf; lda:int64; B:Phalf;
      ldb:int64; beta:Phalf; C:Phalf; ldc:int64):cublasStatus_t; WINAPI;




    cublasSgemmEx : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Psingle; A:pointer; Atype:cudaDataType; lda:longint; 
      B:pointer; Btype:cudaDataType; ldb:longint; beta:Psingle; C:pointer; 
      Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;




    cublasSgemmEx_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Psingle; A:pointer; Atype:cudaDataType; lda:int64; 
      B:pointer; Btype:cudaDataType; ldb:int64; beta:Psingle; C:pointer; 
      Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;




    cublasGemmEx : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:pointer; A:pointer; Atype:cudaDataType; lda:longint; 
      B:pointer; Btype:cudaDataType; ldb:longint; beta:pointer; C:pointer; 
      Ctype:cudaDataType; ldc:longint; computeType:cublasComputeType_t; algo:cublasGemmAlgo_t):cublasStatus_t; WINAPI;




    cublasGemmEx_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:pointer; A:pointer; Atype:cudaDataType; lda:int64; 
      B:pointer; Btype:cudaDataType; ldb:int64; beta:pointer; C:pointer; 
      Ctype:cudaDataType; ldc:int64; computeType:cublasComputeType_t; algo:cublasGemmAlgo_t):cublasStatus_t; WINAPI;




    cublasCgemmEx : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:longint; 
      B:pointer; Btype:cudaDataType; ldb:longint; beta:PcuComplex; C:pointer; 
      Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;




    cublasCgemmEx_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:int64; 
      B:pointer; Btype:cudaDataType; ldb:int64; beta:PcuComplex; C:pointer; 
      Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;
  { SYRK  }



    cublasSsyrk{$ifdef CUBLAS_V2}, cublasSsyrk_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Psingle; A:Psingle; lda:longint; beta:Psingle; C:Psingle; 
      ldc:longint):cublasStatus_t; WINAPI;



    cublasSsyrk_64{$ifdef CUBLAS_V2}, cublasSsyrk_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Psingle; A:Psingle; lda:int64; beta:Psingle; C:Psingle; 
      ldc:int64):cublasStatus_t; WINAPI;



    cublasDsyrk{$ifdef CUBLAS_V2}, cublasDsyrk_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Pdouble; A:Pdouble; lda:longint; beta:Pdouble; C:Pdouble; 
      ldc:longint):cublasStatus_t; WINAPI;



    cublasDsyrk_64{$ifdef CUBLAS_V2}, cublasDsyrk_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Pdouble; A:Pdouble; lda:int64; beta:Pdouble; C:Pdouble; 
      ldc:int64):cublasStatus_t; WINAPI;



    cublasCsyrk{$ifdef CUBLAS_V2}, cublasCsyrk_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; beta:PcuComplex; C:PcuComplex; 
      ldc:longint):cublasStatus_t; WINAPI;



    cublasCsyrk_64{$ifdef CUBLAS_V2}, cublasCsyrk_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; beta:PcuComplex; C:PcuComplex; 
      ldc:int64):cublasStatus_t; WINAPI;



    cublasZsyrk{$ifdef CUBLAS_V2}, cublasZsyrk_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; beta:PcuDoubleComplex; C:PcuDoubleComplex; 
      ldc:longint):cublasStatus_t; WINAPI;



    cublasZsyrk_64{$ifdef CUBLAS_V2}, cublasZsyrk_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; beta:PcuDoubleComplex; C:PcuDoubleComplex; 
      ldc:int64):cublasStatus_t; WINAPI;



    cublasCsyrkEx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:longint; beta:PcuComplex; 
      C:pointer; Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;



    cublasCsyrkEx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:int64; beta:PcuComplex; 
      C:pointer; Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;



    cublasCsyrk3mEx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:longint; beta:PcuComplex; 
      C:pointer; Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;



    cublasCsyrk3mEx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:pointer; Atype:cudaDataType; lda:int64; beta:PcuComplex; 
      C:pointer; Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;
  { HERK  }



    cublasCherk{$ifdef CUBLAS_V2}, cublasCherk_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Psingle; A:PcuComplex; lda:longint; beta:Psingle; C:PcuComplex; 
      ldc:longint):cublasStatus_t; WINAPI;



    cublasCherk_64{$ifdef CUBLAS_V2}, cublasCherk_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Psingle; A:PcuComplex; lda:int64; beta:Psingle; C:PcuComplex; 
      ldc:int64):cublasStatus_t; WINAPI;



    cublasZherk{$ifdef CUBLAS_V2}, cublasZherk_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Pdouble; A:PcuDoubleComplex; lda:longint; beta:Pdouble; C:PcuDoubleComplex; 
      ldc:longint):cublasStatus_t; WINAPI;



    cublasZherk_64{$ifdef CUBLAS_V2}, cublasZherk_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Pdouble; A:PcuDoubleComplex; lda:int64; beta:Pdouble; C:PcuDoubleComplex; 
      ldc:int64):cublasStatus_t; WINAPI;



    cublasCherkEx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Psingle; A:pointer; Atype:cudaDataType; lda:longint; beta:Psingle; 
      C:pointer; Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;



    cublasCherkEx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Psingle; A:pointer; Atype:cudaDataType; lda:int64; beta:Psingle; 
      C:pointer; Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;



    cublasCherk3mEx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Psingle; A:pointer; Atype:cudaDataType; lda:longint; beta:Psingle; 
      C:pointer; Ctype:cudaDataType; ldc:longint):cublasStatus_t; WINAPI;



    cublasCherk3mEx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Psingle; A:pointer; Atype:cudaDataType; lda:int64; beta:Psingle; 
      C:pointer; Ctype:cudaDataType; ldc:int64):cublasStatus_t; WINAPI;
  { SYR2K / HER2K  }




    cublasSsyr2k{$ifdef CUBLAS_V2}, cublasSsyr2k_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Psingle; A:Psingle; lda:longint; B:Psingle; ldb:longint; 
      beta:Psingle; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;




    cublasSsyr2k_64{$ifdef CUBLAS_V2}, cublasSsyr2k_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Psingle; A:Psingle; lda:int64; B:Psingle; ldb:int64; 
      beta:Psingle; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;




    cublasDsyr2k{$ifdef CUBLAS_V2}, cublasDsyr2k_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Pdouble; A:Pdouble; lda:longint; B:Pdouble; ldb:longint; 
      beta:Pdouble; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;




    cublasDsyr2k_64{$ifdef CUBLAS_V2}, cublasDsyr2k_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Pdouble; A:Pdouble; lda:int64; B:Pdouble; ldb:int64; 
      beta:Pdouble; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;




    cublasCsyr2k{$ifdef CUBLAS_V2}, cublasCsyr2k_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; ldb:longint; 
      beta:PcuComplex; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCsyr2k_64{$ifdef CUBLAS_V2}, cublasCsyr2k_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; ldb:int64; 
      beta:PcuComplex; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZsyr2k{$ifdef CUBLAS_V2}, cublasZsyr2k_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; ldb:longint; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZsyr2k_64{$ifdef CUBLAS_V2}, cublasZsyr2k_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; ldb:int64; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasCher2k{$ifdef CUBLAS_V2}, cublasCher2k_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; ldb:longint; 
      beta:Psingle; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCher2k_64{$ifdef CUBLAS_V2}, cublasCher2k_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; ldb:int64; 
      beta:Psingle; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZher2k{$ifdef CUBLAS_V2}, cublasZher2k_v2 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; ldb:longint; 
      beta:Pdouble; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZher2k_64{$ifdef CUBLAS_V2}, cublasZher2k_v2_64 {$endif} : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; ldb:int64; 
      beta:Pdouble; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { SYRKX / HERKX  }




    cublasSsyrkx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Psingle; A:Psingle; lda:longint; B:Psingle; ldb:longint; 
      beta:Psingle; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;




    cublasSsyrkx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Psingle; A:Psingle; lda:int64; B:Psingle; ldb:int64; 
      beta:Psingle; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;




    cublasDsyrkx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:Pdouble; A:Pdouble; lda:longint; B:Pdouble; ldb:longint; 
      beta:Pdouble; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;




    cublasDsyrkx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:Pdouble; A:Pdouble; lda:int64; B:Pdouble; ldb:int64; 
      beta:Pdouble; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;




    cublasCsyrkx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; ldb:longint; 
      beta:PcuComplex; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCsyrkx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; ldb:int64; 
      beta:PcuComplex; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZsyrkx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; ldb:longint; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZsyrkx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; ldb:int64; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasCherkx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; ldb:longint; 
      beta:Psingle; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCherkx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; ldb:int64; 
      beta:Psingle; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZherkx : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:longint; k:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; ldb:longint; 
      beta:Pdouble; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZherkx_64 : function(handle:cublasHandle_t; uplo:cublasFillMode_t; trans:cublasOperation_t; n:int64; k:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; ldb:int64; 
      beta:Pdouble; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { SYMM  }




    cublasSsymm{$ifdef CUBLAS_V2}, cublasSsymm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:longint; n:longint; 
      alpha:Psingle; A:Psingle; lda:longint; B:Psingle; ldb:longint; 
      beta:Psingle; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;




    cublasSsymm_64{$ifdef CUBLAS_V2}, cublasSsymm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:int64; n:int64; 
      alpha:Psingle; A:Psingle; lda:int64; B:Psingle; ldb:int64; 
      beta:Psingle; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;




    cublasDsymm{$ifdef CUBLAS_V2}, cublasDsymm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:longint; n:longint; 
      alpha:Pdouble; A:Pdouble; lda:longint; B:Pdouble; ldb:longint; 
      beta:Pdouble; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;




    cublasDsymm_64{$ifdef CUBLAS_V2}, cublasDsymm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:int64; n:int64; 
      alpha:Pdouble; A:Pdouble; lda:int64; B:Pdouble; ldb:int64; 
      beta:Pdouble; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;




    cublasCsymm{$ifdef CUBLAS_V2}, cublasCsymm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:longint; n:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; ldb:longint; 
      beta:PcuComplex; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCsymm_64{$ifdef CUBLAS_V2}, cublasCsymm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:int64; n:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; ldb:int64; 
      beta:PcuComplex; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZsymm{$ifdef CUBLAS_V2}, cublasZsymm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:longint; n:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; ldb:longint; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZsymm_64{$ifdef CUBLAS_V2}, cublasZsymm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:int64; n:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; ldb:int64; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { HEMM  }




    cublasChemm{$ifdef CUBLAS_V2}, cublasChemm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:longint; n:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; B:PcuComplex; ldb:longint; 
      beta:PcuComplex; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasChemm_64{$ifdef CUBLAS_V2}, cublasChemm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:int64; n:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; B:PcuComplex; ldb:int64; 
      beta:PcuComplex; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZhemm{$ifdef CUBLAS_V2}, cublasZhemm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:longint; n:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; B:PcuDoubleComplex; ldb:longint; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZhemm_64{$ifdef CUBLAS_V2}, cublasZhemm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; m:int64; n:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; B:PcuDoubleComplex; ldb:int64; 
      beta:PcuDoubleComplex; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { TRSM  }


    cublasStrsm{$ifdef CUBLAS_V2}, cublasStrsm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:Psingle; A:Psingle; lda:longint; 
      B:Psingle; ldb:longint):cublasStatus_t; WINAPI;


    cublasStrsm_64{$ifdef CUBLAS_V2}, cublasStrsm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:Psingle; A:Psingle; lda:int64; 
      B:Psingle; ldb:int64):cublasStatus_t; WINAPI;


    cublasDtrsm{$ifdef CUBLAS_V2}, cublasDtrsm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:Pdouble; A:Pdouble; lda:longint; 
      B:Pdouble; ldb:longint):cublasStatus_t; WINAPI;


    cublasDtrsm_64{$ifdef CUBLAS_V2}, cublasDtrsm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:Pdouble; A:Pdouble; lda:int64; 
      B:Pdouble; ldb:int64):cublasStatus_t; WINAPI;


    cublasCtrsm{$ifdef CUBLAS_V2}, cublasCtrsm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; 
      B:PcuComplex; ldb:longint):cublasStatus_t; WINAPI;


    cublasCtrsm_64{$ifdef CUBLAS_V2}, cublasCtrsm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; 
      B:PcuComplex; ldb:int64):cublasStatus_t; WINAPI;


    cublasZtrsm{$ifdef CUBLAS_V2}, cublasZtrsm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; 
      B:PcuDoubleComplex; ldb:longint):cublasStatus_t; WINAPI;


    cublasZtrsm_64{$ifdef CUBLAS_V2}, cublasZtrsm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; 
      B:PcuDoubleComplex; ldb:int64):cublasStatus_t; WINAPI;
  { TRMM  }



    cublasStrmm{$ifdef CUBLAS_V2}, cublasStrmm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:Psingle; A:Psingle; lda:longint; 
      B:Psingle; ldb:longint; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;



    cublasStrmm_64{$ifdef CUBLAS_V2}, cublasStrmm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:Psingle; A:Psingle; lda:int64; 
      B:Psingle; ldb:int64; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;



    cublasDtrmm{$ifdef CUBLAS_V2}, cublasDtrmm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:Pdouble; A:Pdouble; lda:longint; 
      B:Pdouble; ldb:longint; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;



    cublasDtrmm_64{$ifdef CUBLAS_V2}, cublasDtrmm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:Pdouble; A:Pdouble; lda:int64; 
      B:Pdouble; ldb:int64; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;



    cublasCtrmm{$ifdef CUBLAS_V2}, cublasCtrmm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; 
      B:PcuComplex; ldb:longint; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;



    cublasCtrmm_64{$ifdef CUBLAS_V2}, cublasCtrmm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; 
      B:PcuComplex; ldb:int64; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;



    cublasZtrmm{$ifdef CUBLAS_V2}, cublasZtrmm_v2 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; 
      B:PcuDoubleComplex; ldb:longint; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;



    cublasZtrmm_64{$ifdef CUBLAS_V2}, cublasZtrmm_v2_64 {$endif} : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; 
      B:PcuDoubleComplex; ldb:int64; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { BATCH GEMM  }


    cublasHgemmBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Phalf; Aarray:PPhalf; lda:longint; Barray:PPhalf;
      ldb:longint; beta:Phalf; Carray:PPhalf; ldc:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasHgemmBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Phalf; Aarray:PPhalf; lda:int64; Barray:PPhalf;
      ldb:int64; beta:Phalf; Carray:PPhalf; ldc:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasSgemmBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Psingle; Aarray:PPsingle; lda:longint; Barray:PPsingle; 
      ldb:longint; beta:Psingle; Carray:PPsingle; ldc:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasSgemmBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Psingle; Aarray:PPsingle; lda:int64; Barray:PPsingle; 
      ldb:int64; beta:Psingle; Carray:PPsingle; ldc:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasDgemmBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Pdouble; Aarray:PPdouble; lda:longint; Barray:PPdouble; 
      ldb:longint; beta:Pdouble; Carray:PPdouble; ldc:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasDgemmBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Pdouble; Aarray:PPdouble; lda:int64; Barray:PPdouble; 
      ldb:int64; beta:Pdouble; Carray:PPdouble; ldc:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasCgemmBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; Aarray:PPcuComplex; lda:longint; Barray:PPcuComplex; 
      ldb:longint; beta:PcuComplex; Carray:PPcuComplex; ldc:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasCgemmBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; Aarray:PPcuComplex; lda:int64; Barray:PPcuComplex; 
      ldb:int64; beta:PcuComplex; Carray:PPcuComplex; ldc:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasCgemm3mBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; Aarray:PPcuComplex; lda:longint; Barray:PPcuComplex; 
      ldb:longint; beta:PcuComplex; Carray:PPcuComplex; ldc:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasCgemm3mBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; Aarray:PPcuComplex; lda:int64; Barray:PPcuComplex; 
      ldb:int64; beta:PcuComplex; Carray:PPcuComplex; ldc:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasZgemmBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuDoubleComplex; Aarray:PPcuDoubleComplex; lda:longint; Barray:PPcuDoubleComplex; 
      ldb:longint; beta:PcuDoubleComplex; Carray:PPcuDoubleComplex; ldc:longint; batchCount:longint):cublasStatus_t; WINAPI;







    cublasZgemmBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuDoubleComplex; Aarray:PPcuDoubleComplex; lda:int64; Barray:PPcuDoubleComplex; 
      ldb:int64; beta:PcuDoubleComplex; Carray:PPcuDoubleComplex; ldc:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasHgemmStridedBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Phalf; A:Phalf; lda:longint; strideA:int64;
      B:Phalf; ldb:longint; strideB:int64; beta:Phalf; C:Phalf;
      ldc:longint; strideC:int64; batchCount:longint):cublasStatus_t; WINAPI;




    cublasHgemmStridedBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Phalf; A:Phalf; lda:int64; strideA:int64;
      B:Phalf; ldb:int64; strideB:int64; beta:Phalf; C:Phalf;
      ldc:int64; strideC:int64; batchCount:int64):cublasStatus_t; WINAPI;






  var
    cublasSgemmStridedBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Psingle; A:Psingle; lda:longint; strideA:int64; 
      B:Psingle; ldb:longint; strideB:int64; beta:Psingle; C:Psingle; 
      ldc:longint; strideC:int64; batchCount:longint):cublasStatus_t; WINAPI;




    cublasSgemmStridedBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Psingle; A:Psingle; lda:int64; strideA:int64; 
      B:Psingle; ldb:int64; strideB:int64; beta:Psingle; C:Psingle; 
      ldc:int64; strideC:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasDgemmStridedBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:Pdouble; A:Pdouble; lda:longint; strideA:int64; 
      B:Pdouble; ldb:longint; strideB:int64; beta:Pdouble; C:Pdouble; 
      ldc:longint; strideC:int64; batchCount:longint):cublasStatus_t; WINAPI;




    cublasDgemmStridedBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:Pdouble; A:Pdouble; lda:int64; strideA:int64; 
      B:Pdouble; ldb:int64; strideB:int64; beta:Pdouble; C:Pdouble; 
      ldc:int64; strideC:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasCgemmStridedBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; strideA:int64; 
      B:PcuComplex; ldb:longint; strideB:int64; beta:PcuComplex; C:PcuComplex; 
      ldc:longint; strideC:int64; batchCount:longint):cublasStatus_t; WINAPI;




    cublasCgemmStridedBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; strideA:int64; 
      B:PcuComplex; ldb:int64; strideB:int64; beta:PcuComplex; C:PcuComplex; 
      ldc:int64; strideC:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasCgemm3mStridedBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuComplex; A:PcuComplex; lda:longint; strideA:int64; 
      B:PcuComplex; ldb:longint; strideB:int64; beta:PcuComplex; C:PcuComplex; 
      ldc:longint; strideC:int64; batchCount:longint):cublasStatus_t; WINAPI;




    cublasCgemm3mStridedBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuComplex; A:PcuComplex; lda:int64; strideA:int64; 
      B:PcuComplex; ldb:int64; strideB:int64; beta:PcuComplex; C:PcuComplex; 
      ldc:int64; strideC:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasZgemmStridedBatched : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; strideA:int64; 
      B:PcuDoubleComplex; ldb:longint; strideB:int64; beta:PcuDoubleComplex; C:PcuDoubleComplex; 
      ldc:longint; strideC:int64; batchCount:longint):cublasStatus_t; WINAPI;




    cublasZgemmStridedBatched_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; strideA:int64; 
      B:PcuDoubleComplex; ldb:int64; strideB:int64; beta:PcuDoubleComplex; C:PcuDoubleComplex; 
      ldc:int64; strideC:int64; batchCount:int64):cublasStatus_t; WINAPI;







    cublasGemmBatchedEx : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:pointer; Aarray:Ppointer; Atype:cudaDataType; lda:longint; 
      Barray:Ppointer; Btype:cudaDataType; ldb:longint; beta:pointer; Carray:Ppointer; 
      Ctype:cudaDataType; ldc:longint; batchCount:longint; computeType:cublasComputeType_t; algo:cublasGemmAlgo_t):cublasStatus_t; WINAPI;







    cublasGemmBatchedEx_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:pointer; Aarray:Ppointer; Atype:cudaDataType; lda:int64; 
      Barray:Ppointer; Btype:cudaDataType; ldb:int64; beta:pointer; Carray:Ppointer; 
      Ctype:cudaDataType; ldc:int64; batchCount:int64; computeType:cublasComputeType_t; algo:cublasGemmAlgo_t):cublasStatus_t; WINAPI;




    cublasGemmStridedBatchedEx : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      k:longint; alpha:pointer; A:pointer; Atype:cudaDataType; lda:longint; 
      strideA:int64; B:pointer; Btype:cudaDataType; ldb:longint; strideB:int64; 
      beta:pointer; C:pointer; Ctype:cudaDataType; ldc:longint; strideC:int64; 
      batchCount:longint; computeType:cublasComputeType_t; algo:cublasGemmAlgo_t):cublasStatus_t; WINAPI;




    cublasGemmStridedBatchedEx_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      k:int64; alpha:pointer; A:pointer; Atype:cudaDataType; lda:int64; 
      strideA:int64; B:pointer; Btype:cudaDataType; ldb:int64; strideB:int64; 
      beta:pointer; C:pointer; Ctype:cudaDataType; ldc:int64; strideC:int64; 
      batchCount:int64; computeType:cublasComputeType_t; algo:cublasGemmAlgo_t):cublasStatus_t; WINAPI;
  { ---------------- CUBLAS BLAS-like Extension ----------------  }
  { GEAM  }




    cublasSgeam : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      alpha:Psingle; A:Psingle; lda:longint; beta:Psingle; B:Psingle; 
      ldb:longint; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;




    cublasSgeam_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      alpha:Psingle; A:Psingle; lda:int64; beta:Psingle; B:Psingle; 
      ldb:int64; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;




    cublasDgeam : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      alpha:Pdouble; A:Pdouble; lda:longint; beta:Pdouble; B:Pdouble; 
      ldb:longint; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;




    cublasDgeam_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      alpha:Pdouble; A:Pdouble; lda:int64; beta:Pdouble; B:Pdouble; 
      ldb:int64; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;




    cublasCgeam : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      alpha:PcuComplex; A:PcuComplex; lda:longint; beta:PcuComplex; B:PcuComplex; 
      ldb:longint; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasCgeam_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      alpha:PcuComplex; A:PcuComplex; lda:int64; beta:PcuComplex; B:PcuComplex; 
      ldb:int64; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;




    cublasZgeam : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:longint; n:longint; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:longint; beta:PcuDoubleComplex; B:PcuDoubleComplex; 
      ldb:longint; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;




    cublasZgeam_64 : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; m:int64; n:int64; 
      alpha:PcuDoubleComplex; A:PcuDoubleComplex; lda:int64; beta:PcuDoubleComplex; B:PcuDoubleComplex; 
      ldb:int64; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { TRSM - Batched Triangular Solver  }




    cublasStrsmBatched : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:Psingle; A:PPsingle; lda:longint; 
      B:PPsingle; ldb:longint; batchCount:longint):cublasStatus_t; WINAPI;




    cublasStrsmBatched_64 : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:Psingle; A:PPsingle; lda:int64; 
      B:PPsingle; ldb:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasDtrsmBatched : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:Pdouble; A:PPdouble; lda:longint; 
      B:PPdouble; ldb:longint; batchCount:longint):cublasStatus_t; WINAPI;




    cublasDtrsmBatched_64 : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:Pdouble; A:PPdouble; lda:int64; 
      B:PPdouble; ldb:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasCtrsmBatched : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:PcuComplex; A:PPcuComplex; lda:longint; 
      B:PPcuComplex; ldb:longint; batchCount:longint):cublasStatus_t; WINAPI;




    cublasCtrsmBatched_64 : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:PcuComplex; A:PPcuComplex; lda:int64; 
      B:PPcuComplex; ldb:int64; batchCount:int64):cublasStatus_t; WINAPI;




    cublasZtrsmBatched : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:longint; n:longint; alpha:PcuDoubleComplex; A:PPcuDoubleComplex; lda:longint; 
      B:PPcuDoubleComplex; ldb:longint; batchCount:longint):cublasStatus_t; WINAPI;




    cublasZtrsmBatched_64 : function(handle:cublasHandle_t; side:cublasSideMode_t; uplo:cublasFillMode_t; trans:cublasOperation_t; diag:cublasDiagType_t; 
      m:int64; n:int64; alpha:PcuDoubleComplex; A:PPcuDoubleComplex; lda:int64; 
      B:PPcuDoubleComplex; ldb:int64; batchCount:int64):cublasStatus_t; WINAPI;
  { DGMM  }


    cublasSdgmm : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:longint; n:longint; A:Psingle; 
      lda:longint; x:Psingle; incx:longint; C:Psingle; ldc:longint):cublasStatus_t; WINAPI;


    cublasSdgmm_64 : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:int64; n:int64; A:Psingle; 
      lda:int64; x:Psingle; incx:int64; C:Psingle; ldc:int64):cublasStatus_t; WINAPI;


    cublasDdgmm : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:longint; n:longint; A:Pdouble; 
      lda:longint; x:Pdouble; incx:longint; C:Pdouble; ldc:longint):cublasStatus_t; WINAPI;


    cublasDdgmm_64 : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:int64; n:int64; A:Pdouble; 
      lda:int64; x:Pdouble; incx:int64; C:Pdouble; ldc:int64):cublasStatus_t; WINAPI;


    cublasCdgmm : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:longint; n:longint; A:PcuComplex; 
      lda:longint; x:PcuComplex; incx:longint; C:PcuComplex; ldc:longint):cublasStatus_t; WINAPI;


    cublasCdgmm_64 : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:int64; n:int64; A:PcuComplex; 
      lda:int64; x:PcuComplex; incx:int64; C:PcuComplex; ldc:int64):cublasStatus_t; WINAPI;


    cublasZdgmm : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:longint; n:longint; A:PcuDoubleComplex; 
      lda:longint; x:PcuDoubleComplex; incx:longint; C:PcuDoubleComplex; ldc:longint):cublasStatus_t; WINAPI;


    cublasZdgmm_64 : function(handle:cublasHandle_t; mode:cublasSideMode_t; m:int64; n:int64; A:PcuDoubleComplex; 
      lda:int64; x:PcuDoubleComplex; incx:int64; C:PcuDoubleComplex; ldc:int64):cublasStatus_t; WINAPI;
  { Batched - MATINV }



    cublasSmatinvBatched : function(handle:cublasHandle_t; n:longint; A:PPsingle; lda:longint; Ainv:PPsingle; 
      lda_inv:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;



    cublasDmatinvBatched : function(handle:cublasHandle_t; n:longint; A:PPdouble; lda:longint; Ainv:PPdouble; 
      lda_inv:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;



    cublasCmatinvBatched : function(handle:cublasHandle_t; n:longint; A:PPcuComplex; lda:longint; Ainv:PPcuComplex; 
      lda_inv:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;



    cublasZmatinvBatched : function(handle:cublasHandle_t; n:longint; A:PPcuDoubleComplex; lda:longint; Ainv:PPcuDoubleComplex; 
      lda_inv:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;
  { Batch QR Factorization  }


    cublasSgeqrfBatched : function(handle:cublasHandle_t; m:longint; n:longint; Aarray:PPsingle; lda:longint; 
      TauArray:PPsingle; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;


    cublasDgeqrfBatched : function(handle:cublasHandle_t; m:longint; n:longint; Aarray:PPdouble; lda:longint; 
      TauArray:PPdouble; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;


    cublasCgeqrfBatched : function(handle:cublasHandle_t; m:longint; n:longint; Aarray:PPcuComplex; lda:longint; 
      TauArray:PPcuComplex; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;


    cublasZgeqrfBatched : function(handle:cublasHandle_t; m:longint; n:longint; Aarray:PPcuDoubleComplex; lda:longint; 
      TauArray:PPcuDoubleComplex; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;
  { Least Square Min only m >= n and Non-transpose supported  }


    cublasSgelsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; nrhs:longint; 
      Aarray:PPsingle; lda:longint; Carray:PPsingle; ldc:longint; info:Plongint; 
      devInfoArray:Plongint; batchSize:longint):cublasStatus_t; WINAPI;


    cublasDgelsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; nrhs:longint; 
      Aarray:PPdouble; lda:longint; Carray:PPdouble; ldc:longint; info:Plongint; 
      devInfoArray:Plongint; batchSize:longint):cublasStatus_t; WINAPI;


    cublasCgelsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; nrhs:longint; 
      Aarray:PPcuComplex; lda:longint; Carray:PPcuComplex; ldc:longint; info:Plongint; 
      devInfoArray:Plongint; batchSize:longint):cublasStatus_t; WINAPI;


    cublasZgelsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; m:longint; n:longint; nrhs:longint; 
      Aarray:PPcuDoubleComplex; lda:longint; Carray:PPcuDoubleComplex; ldc:longint; info:Plongint; 
      devInfoArray:Plongint; batchSize:longint):cublasStatus_t; WINAPI;
  { TPTTR : Triangular Pack format to Triangular format  }

    cublasStpttr : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; AP:Psingle; A:Psingle; 
      lda:longint):cublasStatus_t; WINAPI;

    cublasDtpttr : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; AP:Pdouble; A:Pdouble; 
      lda:longint):cublasStatus_t; WINAPI;

    cublasCtpttr : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; AP:PcuComplex; A:PcuComplex; 
      lda:longint):cublasStatus_t; WINAPI;

    cublasZtpttr : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; AP:PcuDoubleComplex; A:PcuDoubleComplex; 
      lda:longint):cublasStatus_t; WINAPI;
  { TRTTP : Triangular format to Triangular Pack format  }

    cublasStrttp : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; A:Psingle; lda:longint; 
      AP:Psingle):cublasStatus_t; WINAPI;

    cublasDtrttp : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; A:Pdouble; lda:longint; 
      AP:Pdouble):cublasStatus_t; WINAPI;

    cublasCtrttp : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; A:PcuComplex; lda:longint; 
      AP:PcuComplex):cublasStatus_t; WINAPI;

    cublasZtrttp : function(handle:cublasHandle_t; uplo:cublasFillMode_t; n:longint; A:PcuDoubleComplex; lda:longint; 
      AP:PcuDoubleComplex):cublasStatus_t; WINAPI;
  { Batched LU - GETRF }

    cublasSgetrfBatched : function(handle:cublasHandle_t; n:longint; A:PPsingle; lda:longint; P:Plongint; 
      info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;

    cublasDgetrfBatched : function(handle:cublasHandle_t; n:longint; A:PPdouble; lda:longint; P:Plongint; 
      info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;

    cublasCgetrfBatched : function(handle:cublasHandle_t; n:longint; A:PPcuComplex; lda:longint; P:Plongint; 
      info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;

    cublasZgetrfBatched : function(handle:cublasHandle_t; n:longint; A:PPcuDoubleComplex; lda:longint; P:Plongint; 
      info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;
  { Batched inversion based on LU factorization from getrf  }




    cublasSgetriBatched : function(handle:cublasHandle_t; n:longint; A:PPsingle; lda:longint; P:Plongint; 
      C:PPsingle; ldc:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;




    cublasDgetriBatched : function(handle:cublasHandle_t; n:longint; A:PPdouble; lda:longint; P:Plongint; 
      C:PPdouble; ldc:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;




    cublasCgetriBatched : function(handle:cublasHandle_t; n:longint; A:PPcuComplex; lda:longint; P:Plongint; 
      C:PPcuComplex; ldc:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;




    cublasZgetriBatched : function(handle:cublasHandle_t; n:longint; A:PPcuDoubleComplex; lda:longint; P:Plongint; 
      C:PPcuDoubleComplex; ldc:longint; info:Plongint; batchSize:longint):cublasStatus_t; WINAPI;
  { Batched solver based on LU factorization from getrf  }




    cublasSgetrsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; n:longint; nrhs:longint; Aarray:PPsingle;
      lda:longint; devIpiv:Plongint; Barray:PPsingle; ldb:longint; info:Plongint;
      batchSize:longint):cublasStatus_t; WINAPI;




    cublasDgetrsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; n:longint; nrhs:longint; Aarray:PPdouble;
      lda:longint; devIpiv:Plongint; Barray:PPdouble; ldb:longint; info:Plongint;
      batchSize:longint):cublasStatus_t; WINAPI;




    cublasCgetrsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; n:longint; nrhs:longint; Aarray:PPcuComplex;
      lda:longint; devIpiv:Plongint; Barray:PPcuComplex; ldb:longint; info:Plongint;
      batchSize:longint):cublasStatus_t; WINAPI;




    cublasZgetrsBatched : function(handle:cublasHandle_t; trans:cublasOperation_t; n:longint; nrhs:longint; Aarray:PPcuDoubleComplex;
      lda:longint; devIpiv:Plongint; Barray:PPcuDoubleComplex; ldb:longint; info:Plongint;
      batchSize:longint):cublasStatus_t; WINAPI;
  { Deprecated  }


    cublasUint8gemmBias : function(handle:cublasHandle_t; transa:cublasOperation_t; transb:cublasOperation_t; transc:cublasOperation_t; m:longint;
      n:longint; k:longint; A:Pbyte; A_bias:longint; lda:longint;
      B:Pbyte; B_bias:longint; ldb:longint; C:Pbyte; C_bias:longint;
      ldc:longint; C_mult:longint; C_shift:longint):cublasStatus_t; WINAPI;

implementation
  uses
    SysUtils, dynlibs;
  {  cuBLAS Exported API  }
(*
  static inline cublasStatus_t cublasMigrateComputeType(cublasHandle_t handle,
                                                        cudaDataType_t dataType,
                                                        cublasComputeType_t* computeType) {
    cublasMath_t mathMode = CUBLAS_DEFAULT_MATH;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    status = cublasGetMathMode(handle, &mathMode);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;


    bool isPedantic = ((mathMode & 0xf) == CUBLAS_PEDANTIC_MATH);

    switch (dataType)
      case CUDA_R_32F:
      case CUDA_C_32F:
        *computeType = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        return CUBLAS_STATUS_SUCCESS;
      case CUDA_R_64F:
      case CUDA_C_64F:
        *computeType = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        return CUBLAS_STATUS_SUCCESS;
      case CUDA_R_16F:
        *computeType = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        return CUBLAS_STATUS_SUCCESS;
      case CUDA_R_32I:
        *computeType = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        return CUBLAS_STATUS_SUCCESS;
      default:
        return CUBLAS_STATUS_NOT_SUPPORTED;
  }
*)
function cublasMigrateComputeType(handle: cublasHandle_t; dataType: cudaDataType_t; computeType: PcublasComputeType_t):cublasStatus_t; inline;
var
    mathMode: cublasMath_t;
    status: cublasStatus_t;
    isPedantic: boolean;
begin
    mathMode := CUBLAS_DEFAULT_MATH;
    status := CUBLAS_STATUS_SUCCESS;
    status := cublasGetMathMode(handle, @mathMode);
    if status <> CUBLAS_STATUS_SUCCESS then
        exit(status);
    isPedantic := ((longint(mathMode) and $f) = longint(CUBLAS_PEDANTIC_MATH));
    case dataType of
        CUDA_R_32F, CUDA_C_32F:
        begin
            if isPedantic then
                 computeType^ := CUBLAS_COMPUTE_32F_PEDANTIC
            else
                 computeType^ := CUBLAS_COMPUTE_32F;
            exit(CUBLAS_STATUS_SUCCESS)
        end;
        CUDA_R_64F, CUDA_C_64F:
        begin
            if isPedantic then
                 computeType^ := CUBLAS_COMPUTE_64F_PEDANTIC
            else
                 computeType^ := CUBLAS_COMPUTE_64F;
            exit(CUBLAS_STATUS_SUCCESS)
        end;
        CUDA_R_16F:
        begin
            if isPedantic then
                 computeType^ := CUBLAS_COMPUTE_16F_PEDANTIC
            else
                 computeType^ := CUBLAS_COMPUTE_16F;
            exit(CUBLAS_STATUS_SUCCESS)
        end;
        CUDA_R_32I:
        begin
            if isPedantic then
                 computeType^ := CUBLAS_COMPUTE_32I_PEDANTIC
            else
                 computeType^ := CUBLAS_COMPUTE_32I;
            exit(CUBLAS_STATUS_SUCCESS)
        end;
        else
        exit(CUBLAS_STATUS_NOT_SUPPORTED)

    end;
end;

(*
  // wrappers to accept old code with cudaDataType computeType when referenced from c++ code

  /*
  static inline cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                                            cublasOperation_t transa,
                                            cublasOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            const void* alpha, // host or device pointer
                                            const void* A,
                                            cudaDataType Atype,
                                            int lda,
                                            const void* B,
                                            cudaDataType Btype,
                                            int ldb,
                                            const void* beta, // host or device pointer
                                            void* C,
                                            cudaDataType Ctype,
                                            int ldc,
                                            cudaDataType computeType,
                                            cublasGemmAlgo_t algo) {
    cublasComputeType_t migratedComputeType = CUBLAS_COMPUTE_32F;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;


    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        Atype,
                        lda,
                        B,
                        Btype,
                        ldb,
                        beta,
                        C,
                        Ctype,
                        ldc,
                        migratedComputeType,
                        algo);

  }


function cublasGemmEx(handle: cublasHandle_t; transa: cublasOperation_t; transb: cublasOperation_t; m: longint; n: longint; k: longint; const alpha: Pointer; const A: Pointer; Atype: cudaDataType; lda: longint; const B: Pointer; Btype: cudaDataType; ldb: longint; const beta: Pointer; C: Pointer; Ctype: cudaDataType; ldc: longint; computeType: cudaDataType; algo: cublasGemmAlgo_t):cublasStatus_t;inline;
var
    migratedComputeType: cublasComputeType_t;
    status: cublasStatus_t;
begin
    migratedComputeType := CUBLAS_COMPUTE_32F;
    status := CUBLAS_STATUS_SUCCESS;
    status := cublasMigrateComputeType(handle, computeType, @migratedComputeType);
    if status <> CUBLAS_STATUS_SUCCESS then
        exit(status);
    exit(cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, migratedComputeType, algo))
end;

  static inline cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                                   cublasOperation_t transa,
                                                   cublasOperation_t transb,
                                                   int m,
                                                   int n,
                                                   int k,
                                                   const void* alpha, // host or device pointer
                                                   const void* const Aarray[],
                                                   cudaDataType Atype,
                                                   int lda,
                                                   const void* const Barray[],
                                                   cudaDataType Btype,
                                                   int ldb,
                                                   const void* beta, // host or device pointer
                                                   void* const Carray[],
                                                   cudaDataType Ctype,
                                                   int ldc,
                                                   int batchCount,
                                                   cudaDataType computeType,
                                                   cublasGemmAlgo_t algo) {
    cublasComputeType_t migratedComputeType;
    cublasStatus_t status;
    status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;


    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               alpha,
                               Aarray,
                               Atype,
                               lda,
                               Barray,
                               Btype,
                               ldb,
                               beta,
                               Carray,
                               Ctype,
                               ldc,
                               batchCount,
                               migratedComputeType,
                               algo);
  }

function cublasGemmBatchedEx(handle: cublasHandle_t; transa: cublasOperation_t; transb: cublasOperation_t; m: longint; n: longint; k: longint; const alpha: Pointer; const Aarray: array of Pointer; Atype: cudaDataType; lda: longint; const Barray: array of Pointer; Btype: cudaDataType; ldb: longint; const beta: Pointer; Carray: array of P; Ctype: cudaDataType; ldc: longint; batchCount: longint; computeType: cudaDataType; algo: cublasGemmAlgo_t):cublasStatus_t;inline;
var
    migratedComputeType: cublasComputeType_t;
    status: cublasStatus_t;
begin
    status := cublasMigrateComputeType(handle, computeType, @migratedComputeType);
    if status <> CUBLAS_STATUS_SUCCESS then
        exit(status);
    exit(cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, migratedComputeType, algo))
end;


  static inline cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const void* alpha, // host or device pointer
                                                          const void* A,
                                                          cudaDataType Atype,
                                                          int lda,
                                                          long long int strideA, // purposely signed
                                                          const void* B,
                                                          cudaDataType Btype,
                                                          int ldb,
                                                          long long int strideB,
                                                          const void* beta, // host or device pointer
                                                          void* C,
                                                          cudaDataType Ctype,
                                                          int ldc,
                                                          long long int strideC,
                                                          int batchCount,
                                                          cudaDataType computeType,
                                                          cublasGemmAlgo_t algo) {
    cublasComputeType_t migratedComputeType;
    cublasStatus_t status;
    status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;


    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      Atype,
                                      lda,
                                      strideA,
                                      B,
                                      Btype,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      Ctype,
                                      ldc,
                                      strideC,
                                      batchCount,
                                      migratedComputeType,
                                      algo);

   }

function cublasGemmStridedBatchedEx(handle: cublasHandle_t; transa: cublasOperation_t; transb: cublasOperation_t; m: longint; n: longint; k: longint; const alpha: Pointer; const A: Pointer; Atype: cudaDataType; lda: longint; strideA: int64; const B: Pointer; Btype: cudaDataType; ldb: longint; strideB: int64; const beta: Pointer; C: P; Ctype: cudaDataType; ldc: longint; strideC: int64; batchCount: longint; computeType: cudaDataType; algo: cublasGemmAlgo_t):cublasStatus_t;inline;
var
    migratedComputeType: cublasComputeType_t;
    status: cublasStatus_t;
begin
    status := cublasMigrateComputeType(handle, computeType,  @migratedComputeType);
    if status <> CUBLAS_STATUS_SUCCESS then
        exit(status);
    exit(cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, migratedComputeType, algo))
end;
*)


  var
    hlib : tlibhandle;


  procedure Freecublas_api;
    begin
      FreeLibrary(hlib);
      cublasGetProperty:=nil;
      cublasGetCudartVersion:=nil;
      cublasGetAtomicsMode:=nil;
      cublasSetAtomicsMode:=nil;
      cublasGetMathMode:=nil;
      cublasSetMathMode:=nil;
      cublasGetSmCountTarget:=nil;
      cublasSetSmCountTarget:=nil;
      cublasGetStatusName:=nil;
      cublasGetStatusString:=nil;
      cublasLoggerConfigure:=nil;
      cublasSetLoggerCallback:=nil;
      cublasGetLoggerCallback:=nil;
      cublasSetVector:=nil;
      cublasSetVector_64:=nil;
      cublasGetVector:=nil;
      cublasGetVector_64:=nil;
      cublasSetMatrix:=nil;
      cublasSetMatrix_64:=nil;
      cublasGetMatrix:=nil;
      cublasGetMatrix_64:=nil;
      cublasSetVectorAsync:=nil;
      cublasSetVectorAsync_64:=nil;
      cublasGetVectorAsync:=nil;
      cublasGetVectorAsync_64:=nil;
      cublasSetMatrixAsync:=nil;
      cublasSetMatrixAsync_64:=nil;
      cublasGetMatrixAsync:=nil;
      cublasGetMatrixAsync_64:=nil;
      cublasXerbla:=nil;
      cublasNrm2Ex:=nil;
      cublasNrm2Ex_64:=nil;
      cublasDotEx:=nil;
      cublasDotEx_64:=nil;
      cublasDotcEx:=nil;
      cublasDotcEx_64:=nil;
      cublasScalEx:=nil;
      cublasScalEx_64:=nil;
      cublasAxpyEx:=nil;
      cublasAxpyEx_64:=nil;
      cublasCopyEx:=nil;
      cublasCopyEx_64:=nil;
      cublasSwapEx:=nil;
      cublasSwapEx_64:=nil;
      cublasIamaxEx:=nil;
      cublasIamaxEx_64:=nil;
      cublasIaminEx:=nil;
      cublasIaminEx_64:=nil;
      cublasAsumEx:=nil;
      cublasAsumEx_64:=nil;
      cublasRotEx:=nil;
      cublasRotEx_64:=nil;
      cublasRotgEx:=nil;
      cublasRotmEx:=nil;
      cublasRotmEx_64:=nil;
      cublasRotmgEx:=nil;
      cublasSgemvBatched:=nil;
      cublasSgemvBatched_64:=nil;
      cublasDgemvBatched:=nil;
      cublasDgemvBatched_64:=nil;
      cublasCgemvBatched:=nil;
      cublasCgemvBatched_64:=nil;
      cublasZgemvBatched:=nil;
      cublasZgemvBatched_64:=nil;
      cublasHSHgemvBatched:=nil;
      cublasHSHgemvBatched_64:=nil;
      cublasHSSgemvBatched:=nil;
      cublasHSSgemvBatched_64:=nil;
      cublasTSTgemvBatched:=nil;
      cublasTSTgemvBatched_64:=nil;
      cublasTSSgemvBatched:=nil;
      cublasTSSgemvBatched_64:=nil;
      cublasSgemvStridedBatched:=nil;
      cublasSgemvStridedBatched_64:=nil;
      cublasDgemvStridedBatched:=nil;
      cublasDgemvStridedBatched_64:=nil;
      cublasCgemvStridedBatched:=nil;
      cublasCgemvStridedBatched_64:=nil;
      cublasZgemvStridedBatched:=nil;
      cublasZgemvStridedBatched_64:=nil;
      cublasHSHgemvStridedBatched:=nil;
      cublasHSHgemvStridedBatched_64:=nil;
      cublasHSSgemvStridedBatched:=nil;
      cublasHSSgemvStridedBatched_64:=nil;
      cublasTSTgemvStridedBatched:=nil;
      cublasTSTgemvStridedBatched_64:=nil;
      cublasTSSgemvStridedBatched:=nil;
      cublasTSSgemvStridedBatched_64:=nil;
      cublasCgemm3m:=nil;
      cublasCgemm3m_64:=nil;
      cublasCgemm3mEx:=nil;
      cublasCgemm3mEx_64:=nil;
      cublasZgemm3m:=nil;
      cublasZgemm3m_64:=nil;
      cublasHgemm:=nil;
      cublasHgemm_64:=nil;
      cublasSgemmEx:=nil;
      cublasSgemmEx_64:=nil;
      cublasGemmEx:=nil;
      cublasGemmEx_64:=nil;
      cublasCgemmEx:=nil;
      cublasCgemmEx_64:=nil;
      cublasCsyrkEx:=nil;
      cublasCsyrkEx_64:=nil;
      cublasCsyrk3mEx:=nil;
      cublasCsyrk3mEx_64:=nil;
      cublasCherkEx:=nil;
      cublasCherkEx_64:=nil;
      cublasCherk3mEx:=nil;
      cublasCherk3mEx_64:=nil;
      cublasSsyrkx:=nil;
      cublasSsyrkx_64:=nil;
      cublasDsyrkx:=nil;
      cublasDsyrkx_64:=nil;
      cublasCsyrkx:=nil;
      cublasCsyrkx_64:=nil;
      cublasZsyrkx:=nil;
      cublasZsyrkx_64:=nil;
      cublasCherkx:=nil;
      cublasCherkx_64:=nil;
      cublasZherkx:=nil;
      cublasZherkx_64:=nil;
      cublasHgemmBatched:=nil;
      cublasHgemmBatched_64:=nil;
      cublasSgemmBatched:=nil;
      cublasSgemmBatched_64:=nil;
      cublasDgemmBatched:=nil;
      cublasDgemmBatched_64:=nil;
      cublasCgemmBatched:=nil;
      cublasCgemmBatched_64:=nil;
      cublasCgemm3mBatched:=nil;
      cublasCgemm3mBatched_64:=nil;
      cublasZgemmBatched:=nil;
      cublasZgemmBatched_64:=nil;
      cublasHgemmStridedBatched:=nil;
      cublasHgemmStridedBatched_64:=nil;
      cublasSgemmStridedBatched:=nil;
      cublasSgemmStridedBatched_64:=nil;
      cublasDgemmStridedBatched:=nil;
      cublasDgemmStridedBatched_64:=nil;
      cublasCgemmStridedBatched:=nil;
      cublasCgemmStridedBatched_64:=nil;
      cublasCgemm3mStridedBatched:=nil;
      cublasCgemm3mStridedBatched_64:=nil;
      cublasZgemmStridedBatched:=nil;
      cublasZgemmStridedBatched_64:=nil;
      cublasGemmBatchedEx:=nil;
      cublasGemmBatchedEx_64:=nil;
      cublasGemmStridedBatchedEx:=nil;
      cublasGemmStridedBatchedEx_64:=nil;
      cublasSgeam:=nil;
      cublasSgeam_64:=nil;
      cublasDgeam:=nil;
      cublasDgeam_64:=nil;
      cublasCgeam:=nil;
      cublasCgeam_64:=nil;
      cublasZgeam:=nil;
      cublasZgeam_64:=nil;
      cublasStrsmBatched:=nil;
      cublasStrsmBatched_64:=nil;
      cublasDtrsmBatched:=nil;
      cublasDtrsmBatched_64:=nil;
      cublasCtrsmBatched:=nil;
      cublasCtrsmBatched_64:=nil;
      cublasZtrsmBatched:=nil;
      cublasZtrsmBatched_64:=nil;
      cublasSdgmm:=nil;
      cublasSdgmm_64:=nil;
      cublasDdgmm:=nil;
      cublasDdgmm_64:=nil;
      cublasCdgmm:=nil;
      cublasCdgmm_64:=nil;
      cublasZdgmm:=nil;
      cublasZdgmm_64:=nil;
      cublasSmatinvBatched:=nil;
      cublasDmatinvBatched:=nil;
      cublasCmatinvBatched:=nil;
      cublasZmatinvBatched:=nil;
      cublasSgeqrfBatched:=nil;
      cublasDgeqrfBatched:=nil;
      cublasCgeqrfBatched:=nil;
      cublasZgeqrfBatched:=nil;
      cublasSgelsBatched:=nil;
      cublasDgelsBatched:=nil;
      cublasCgelsBatched:=nil;
      cublasZgelsBatched:=nil;
      cublasStpttr:=nil;
      cublasDtpttr:=nil;
      cublasCtpttr:=nil;
      cublasZtpttr:=nil;
      cublasStrttp:=nil;
      cublasDtrttp:=nil;
      cublasCtrttp:=nil;
      cublasZtrttp:=nil;
      cublasSgetrfBatched:=nil;
      cublasDgetrfBatched:=nil;
      cublasCgetrfBatched:=nil;
      cublasZgetrfBatched:=nil;
      cublasSgetriBatched:=nil;
      cublasDgetriBatched:=nil;
      cublasCgetriBatched:=nil;
      cublasZgetriBatched:=nil;
      cublasSgetrsBatched:=nil;
      cublasDgetrsBatched:=nil;
      cublasCgetrsBatched:=nil;
      cublasZgetrsBatched:=nil;
      cublasUint8gemmBias:=nil;

      cublasCreate:=nil;
      cublasDestroy:=nil;
      cublasGetVersion:=nil;
      cublasSetWorkspace:=nil;
      cublasSetStream:=nil;
      cublasGetStream:=nil;
      cublasGetPointerMode:=nil;
      cublasSetPointerMode:=nil;
      cublasSnrm2:=nil;
      cublasSnrm2_64:=nil;
      cublasDnrm2:=nil;
      cublasDnrm2_64:=nil;
      cublasScnrm2:=nil;
      cublasScnrm2_64:=nil;
      cublasDznrm2:=nil;
      cublasDznrm2_64:=nil;
      cublasSdot:=nil;
      cublasSdot_64:=nil;
      cublasDdot:=nil;
      cublasDdot_64:=nil;
      cublasCdotu:=nil;
      cublasCdotu_64:=nil;
      cublasCdotc:=nil;
      cublasCdotc_64:=nil;
      cublasZdotu:=nil;
      cublasZdotu_64:=nil;
      cublasZdotc:=nil;
      cublasZdotc_64:=nil;
      cublasSscal:=nil;
      cublasSscal_64:=nil;
      cublasDscal:=nil;
      cublasDscal_64:=nil;
      cublasCscal:=nil;
      cublasCscal_64:=nil;
      cublasCsscal:=nil;
      cublasCsscal_64:=nil;
      cublasZscal:=nil;
      cublasZscal_64:=nil;
      cublasZdscal:=nil;
      cublasZdscal_64:=nil;
      cublasSaxpy:=nil;
      cublasSaxpy_64:=nil;
      cublasDaxpy:=nil;
      cublasDaxpy_64:=nil;
      cublasCaxpy:=nil;
      cublasCaxpy_64:=nil;
      cublasZaxpy:=nil;
      cublasZaxpy_64:=nil;
      cublasScopy:=nil;
      cublasScopy_64:=nil;
      cublasDcopy:=nil;
      cublasDcopy_64:=nil;
      cublasCcopy:=nil;
      cublasCcopy_64:=nil;
      cublasZcopy:=nil;
      cublasZcopy_64:=nil;
      cublasSswap:=nil;
      cublasSswap_64:=nil;
      cublasDswap:=nil;
      cublasDswap_64:=nil;
      cublasCswap:=nil;
      cublasCswap_64:=nil;
      cublasZswap:=nil;
      cublasZswap_64:=nil;
      cublasIsamax:=nil;
      cublasIsamax_64:=nil;
      cublasIdamax:=nil;
      cublasIdamax_64:=nil;
      cublasIcamax:=nil;
      cublasIcamax_64:=nil;
      cublasIzamax:=nil;
      cublasIzamax_64:=nil;
      cublasIsamin:=nil;
      cublasIsamin_64:=nil;
      cublasIdamin:=nil;
      cublasIdamin_64:=nil;
      cublasIcamin:=nil;
      cublasIcamin_64:=nil;
      cublasIzamin:=nil;
      cublasIzamin_64:=nil;
      cublasSasum:=nil;
      cublasSasum_64:=nil;
      cublasDasum:=nil;
      cublasDasum_64:=nil;
      cublasScasum:=nil;
      cublasScasum_64:=nil;
      cublasDzasum:=nil;
      cublasDzasum_64:=nil;
      cublasSrot:=nil;
      cublasSrot_64:=nil;
      cublasDrot:=nil;
      cublasDrot_64:=nil;
      cublasCrot:=nil;
      cublasCrot_64:=nil;
      cublasCsrot:=nil;
      cublasCsrot_64:=nil;
      cublasZrot:=nil;
      cublasZrot_64:=nil;
      cublasZdrot:=nil;
      cublasZdrot_64:=nil;
      cublasSrotg:=nil;
      cublasDrotg:=nil;
      cublasCrotg:=nil;
      cublasZrotg:=nil;
      cublasSrotm:=nil;
      cublasSrotm_64:=nil;
      cublasDrotm:=nil;
      cublasDrotm_64:=nil;
      cublasSrotmg:=nil;
      cublasDrotmg:=nil;
      cublasSgemv:=nil;
      cublasSgemv_64:=nil;
      cublasDgemv:=nil;
      cublasDgemv_64:=nil;
      cublasCgemv:=nil;
      cublasCgemv_64:=nil;
      cublasZgemv:=nil;
      cublasZgemv_64:=nil;
      cublasSgbmv:=nil;
      cublasSgbmv_64:=nil;
      cublasDgbmv:=nil;
      cublasDgbmv_64:=nil;
      cublasCgbmv:=nil;
      cublasCgbmv_64:=nil;
      cublasZgbmv:=nil;
      cublasZgbmv_64:=nil;
      cublasStrmv:=nil;
      cublasStrmv_64:=nil;
      cublasDtrmv:=nil;
      cublasDtrmv_64:=nil;
      cublasCtrmv:=nil;
      cublasCtrmv_64:=nil;
      cublasZtrmv:=nil;
      cublasZtrmv_64:=nil;
      cublasStbmv:=nil;
      cublasStbmv_64:=nil;
      cublasDtbmv:=nil;
      cublasDtbmv_64:=nil;
      cublasCtbmv:=nil;
      cublasCtbmv_64:=nil;
      cublasZtbmv:=nil;
      cublasZtbmv_64:=nil;
      cublasStpmv:=nil;
      cublasStpmv_64:=nil;
      cublasDtpmv:=nil;
      cublasDtpmv_64:=nil;
      cublasCtpmv:=nil;
      cublasCtpmv_64:=nil;
      cublasZtpmv:=nil;
      cublasZtpmv_64:=nil;
      cublasStrsv:=nil;
      cublasStrsv_64:=nil;
      cublasDtrsv:=nil;
      cublasDtrsv_64:=nil;
      cublasCtrsv:=nil;
      cublasCtrsv_64:=nil;
      cublasZtrsv:=nil;
      cublasZtrsv_64:=nil;
      cublasStpsv:=nil;
      cublasStpsv_64:=nil;
      cublasDtpsv:=nil;
      cublasDtpsv_64:=nil;
      cublasCtpsv:=nil;
      cublasCtpsv_64:=nil;
      cublasZtpsv:=nil;
      cublasZtpsv_64:=nil;
      cublasStbsv:=nil;
      cublasStbsv_64:=nil;
      cublasDtbsv:=nil;
      cublasDtbsv_64:=nil;
      cublasCtbsv:=nil;
      cublasCtbsv_64:=nil;
      cublasZtbsv:=nil;
      cublasZtbsv_64:=nil;
      cublasSsymv:=nil;
      cublasSsymv_64:=nil;
      cublasDsymv:=nil;
      cublasDsymv_64:=nil;
      cublasCsymv:=nil;
      cublasCsymv_64:=nil;
      cublasZsymv:=nil;
      cublasZsymv_64:=nil;
      cublasChemv:=nil;
      cublasChemv_64:=nil;
      cublasZhemv:=nil;
      cublasZhemv_64:=nil;
      cublasSsbmv:=nil;
      cublasSsbmv_64:=nil;
      cublasDsbmv:=nil;
      cublasDsbmv_64:=nil;
      cublasChbmv:=nil;
      cublasChbmv_64:=nil;
      cublasZhbmv:=nil;
      cublasZhbmv_64:=nil;
      cublasSspmv:=nil;
      cublasSspmv_64:=nil;
      cublasDspmv:=nil;
      cublasDspmv_64:=nil;
      cublasChpmv:=nil;
      cublasChpmv_64:=nil;
      cublasZhpmv:=nil;
      cublasZhpmv_64:=nil;
      cublasSger:=nil;
      cublasSger_64:=nil;
      cublasDger:=nil;
      cublasDger_64:=nil;
      cublasCgeru:=nil;
      cublasCgeru_64:=nil;
      cublasCgerc:=nil;
      cublasCgerc_64:=nil;
      cublasZgeru:=nil;
      cublasZgeru_64:=nil;
      cublasZgerc:=nil;
      cublasZgerc_64:=nil;
      cublasSsyr:=nil;
      cublasSsyr_64:=nil;
      cublasDsyr:=nil;
      cublasDsyr_64:=nil;
      cublasCsyr:=nil;
      cublasCsyr_64:=nil;
      cublasZsyr:=nil;
      cublasZsyr_64:=nil;
      cublasCher:=nil;
      cublasCher_64:=nil;
      cublasZher:=nil;
      cublasZher_64:=nil;
      cublasSspr:=nil;
      cublasSspr_64:=nil;
      cublasDspr:=nil;
      cublasDspr_64:=nil;
      cublasChpr:=nil;
      cublasChpr_64:=nil;
      cublasZhpr:=nil;
      cublasZhpr_64:=nil;
      cublasSsyr2:=nil;
      cublasSsyr2_64:=nil;
      cublasDsyr2:=nil;
      cublasDsyr2_64:=nil;
      cublasCsyr2:=nil;
      cublasCsyr2_64:=nil;
      cublasZsyr2:=nil;
      cublasZsyr2_64:=nil;
      cublasCher2:=nil;
      cublasCher2_64:=nil;
      cublasZher2:=nil;
      cublasZher2_64:=nil;
      cublasSspr2:=nil;
      cublasSspr2_64:=nil;
      cublasDspr2:=nil;
      cublasDspr2_64:=nil;
      cublasChpr2:=nil;
      cublasChpr2_64:=nil;
      cublasZhpr2:=nil;
      cublasZhpr2_64:=nil;
      cublasSgemm:=nil;
      cublasSgemm_64:=nil;
      cublasDgemm:=nil;
      cublasDgemm_64:=nil;
      cublasCgemm:=nil;
      cublasCgemm_64:=nil;
      cublasZgemm:=nil;
      cublasZgemm_64:=nil;
      cublasSsyrk:=nil;
      cublasSsyrk_64:=nil;
      cublasDsyrk:=nil;
      cublasDsyrk_64:=nil;
      cublasCsyrk:=nil;
      cublasCsyrk_64:=nil;
      cublasZsyrk:=nil;
      cublasZsyrk_64:=nil;
      cublasCherk:=nil;
      cublasCherk_64:=nil;
      cublasZherk:=nil;
      cublasZherk_64:=nil;
      cublasSsyr2k:=nil;
      cublasSsyr2k_64:=nil;
      cublasDsyr2k:=nil;
      cublasDsyr2k_64:=nil;
      cublasCsyr2k:=nil;
      cublasCsyr2k_64:=nil;
      cublasZsyr2k:=nil;
      cublasZsyr2k_64:=nil;
      cublasCher2k:=nil;
      cublasCher2k_64:=nil;
      cublasZher2k:=nil;
      cublasZher2k_64:=nil;
      cublasSsymm:=nil;
      cublasSsymm_64:=nil;
      cublasDsymm:=nil;
      cublasDsymm_64:=nil;
      cublasCsymm:=nil;
      cublasCsymm_64:=nil;
      cublasZsymm:=nil;
      cublasZsymm_64:=nil;
      cublasChemm:=nil;
      cublasChemm_64:=nil;
      cublasZhemm:=nil;
      cublasZhemm_64:=nil;
      cublasStrsm:=nil;
      cublasStrsm_64:=nil;
      cublasDtrsm:=nil;
      cublasDtrsm_64:=nil;
      cublasCtrsm:=nil;
      cublasCtrsm_64:=nil;
      cublasZtrsm:=nil;
      cublasZtrsm_64:=nil;
      cublasStrmm:=nil;
      cublasStrmm_64:=nil;
      cublasDtrmm:=nil;
      cublasDtrmm_64:=nil;
      cublasCtrmm:=nil;
      cublasCtrmm_64:=nil;
      cublasZtrmm:=nil;
      cublasZtrmm_64:=nil;

      {$ifdef CUBLAS_V2}
      cublasCreate_v2:=nil;
      cublasDestroy_v2:=nil;
      cublasGetVersion_v2:=nil;
      cublasSetWorkspace_v2:=nil;
      cublasSetStream_v2:=nil;
      cublasGetStream_v2:=nil;
      cublasGetPointerMode_v2:=nil;
      cublasSetPointerMode_v2:=nil;
      cublasSnrm2_v2:=nil;
      cublasSnrm2_v2_64:=nil;
      cublasDnrm2_v2:=nil;
      cublasDnrm2_v2_64:=nil;
      cublasScnrm2_v2:=nil;
      cublasScnrm2_v2_64:=nil;
      cublasDznrm2_v2:=nil;
      cublasDznrm2_v2_64:=nil;
      cublasSdot_v2:=nil;
      cublasSdot_v2_64:=nil;
      cublasDdot_v2:=nil;
      cublasDdot_v2_64:=nil;
      cublasCdotu_v2:=nil;
      cublasCdotu_v2_64:=nil;
      cublasCdotc_v2:=nil;
      cublasCdotc_v2_64:=nil;
      cublasZdotu_v2:=nil;
      cublasZdotu_v2_64:=nil;
      cublasZdotc_v2:=nil;
      cublasZdotc_v2_64:=nil;
      cublasSscal_v2:=nil;
      cublasSscal_v2_64:=nil;
      cublasDscal_v2:=nil;
      cublasDscal_v2_64:=nil;
      cublasCscal_v2:=nil;
      cublasCscal_v2_64:=nil;
      cublasCsscal_v2:=nil;
      cublasCsscal_v2_64:=nil;
      cublasZscal_v2:=nil;
      cublasZscal_v2_64:=nil;
      cublasZdscal_v2:=nil;
      cublasZdscal_v2_64:=nil;
      cublasSaxpy_v2:=nil;
      cublasSaxpy_v2_64:=nil;
      cublasDaxpy_v2:=nil;
      cublasDaxpy_v2_64:=nil;
      cublasCaxpy_v2:=nil;
      cublasCaxpy_v2_64:=nil;
      cublasZaxpy_v2:=nil;
      cublasZaxpy_v2_64:=nil;
      cublasScopy_v2:=nil;
      cublasScopy_v2_64:=nil;
      cublasDcopy_v2:=nil;
      cublasDcopy_v2_64:=nil;
      cublasCcopy_v2:=nil;
      cublasCcopy_v2_64:=nil;
      cublasZcopy_v2:=nil;
      cublasZcopy_v2_64:=nil;
      cublasSswap_v2:=nil;
      cublasSswap_v2_64:=nil;
      cublasDswap_v2:=nil;
      cublasDswap_v2_64:=nil;
      cublasCswap_v2:=nil;
      cublasCswap_v2_64:=nil;
      cublasZswap_v2:=nil;
      cublasZswap_v2_64:=nil;
      cublasIsamax_v2:=nil;
      cublasIsamax_v2_64:=nil;
      cublasIdamax_v2:=nil;
      cublasIdamax_v2_64:=nil;
      cublasIcamax_v2:=nil;
      cublasIcamax_v2_64:=nil;
      cublasIzamax_v2:=nil;
      cublasIzamax_v2_64:=nil;
      cublasIsamin_v2:=nil;
      cublasIsamin_v2_64:=nil;
      cublasIdamin_v2:=nil;
      cublasIdamin_v2_64:=nil;
      cublasIcamin_v2:=nil;
      cublasIcamin_v2_64:=nil;
      cublasIzamin_v2:=nil;
      cublasIzamin_v2_64:=nil;
      cublasSasum_v2:=nil;
      cublasSasum_v2_64:=nil;
      cublasDasum_v2:=nil;
      cublasDasum_v2_64:=nil;
      cublasScasum_v2:=nil;
      cublasScasum_v2_64:=nil;
      cublasDzasum_v2:=nil;
      cublasDzasum_v2_64:=nil;
      cublasSrot_v2:=nil;
      cublasSrot_v2_64:=nil;
      cublasDrot_v2:=nil;
      cublasDrot_v2_64:=nil;
      cublasCrot_v2:=nil;
      cublasCrot_v2_64:=nil;
      cublasCsrot_v2:=nil;
      cublasCsrot_v2_64:=nil;
      cublasZrot_v2:=nil;
      cublasZrot_v2_64:=nil;
      cublasZdrot_v2:=nil;
      cublasZdrot_v2_64:=nil;
      cublasSrotg_v2:=nil;
      cublasDrotg_v2:=nil;
      cublasCrotg_v2:=nil;
      cublasZrotg_v2:=nil;
      cublasSrotm_v2:=nil;
      cublasSrotm_v2_64:=nil;
      cublasDrotm_v2:=nil;
      cublasDrotm_v2_64:=nil;
      cublasSrotmg_v2:=nil;
      cublasDrotmg_v2:=nil;
      cublasSgemv_v2:=nil;
      cublasSgemv_v2_64:=nil;
      cublasDgemv_v2:=nil;
      cublasDgemv_v2_64:=nil;
      cublasCgemv_v2:=nil;
      cublasCgemv_v2_64:=nil;
      cublasZgemv_v2:=nil;
      cublasZgemv_v2_64:=nil;
      cublasSgbmv_v2:=nil;
      cublasSgbmv_v2_64:=nil;
      cublasDgbmv_v2:=nil;
      cublasDgbmv_v2_64:=nil;
      cublasCgbmv_v2:=nil;
      cublasCgbmv_v2_64:=nil;
      cublasZgbmv_v2:=nil;
      cublasZgbmv_v2_64:=nil;
      cublasStrmv_v2:=nil;
      cublasStrmv_v2_64:=nil;
      cublasDtrmv_v2:=nil;
      cublasDtrmv_v2_64:=nil;
      cublasCtrmv_v2:=nil;
      cublasCtrmv_v2_64:=nil;
      cublasZtrmv_v2:=nil;
      cublasZtrmv_v2_64:=nil;
      cublasStbmv_v2:=nil;
      cublasStbmv_v2_64:=nil;
      cublasDtbmv_v2:=nil;
      cublasDtbmv_v2_64:=nil;
      cublasCtbmv_v2:=nil;
      cublasCtbmv_v2_64:=nil;
      cublasZtbmv_v2:=nil;
      cublasZtbmv_v2_64:=nil;
      cublasStpmv_v2:=nil;
      cublasStpmv_v2_64:=nil;
      cublasDtpmv_v2:=nil;
      cublasDtpmv_v2_64:=nil;
      cublasCtpmv_v2:=nil;
      cublasCtpmv_v2_64:=nil;
      cublasZtpmv_v2:=nil;
      cublasZtpmv_v2_64:=nil;
      cublasStrsv_v2:=nil;
      cublasStrsv_v2_64:=nil;
      cublasDtrsv_v2:=nil;
      cublasDtrsv_v2_64:=nil;
      cublasCtrsv_v2:=nil;
      cublasCtrsv_v2_64:=nil;
      cublasZtrsv_v2:=nil;
      cublasZtrsv_v2_64:=nil;
      cublasStpsv_v2:=nil;
      cublasStpsv_v2_64:=nil;
      cublasDtpsv_v2:=nil;
      cublasDtpsv_v2_64:=nil;
      cublasCtpsv_v2:=nil;
      cublasCtpsv_v2_64:=nil;
      cublasZtpsv_v2:=nil;
      cublasZtpsv_v2_64:=nil;
      cublasStbsv_v2:=nil;
      cublasStbsv_v2_64:=nil;
      cublasDtbsv_v2:=nil;
      cublasDtbsv_v2_64:=nil;
      cublasCtbsv_v2:=nil;
      cublasCtbsv_v2_64:=nil;
      cublasZtbsv_v2:=nil;
      cublasZtbsv_v2_64:=nil;
      cublasSsymv_v2:=nil;
      cublasSsymv_v2_64:=nil;
      cublasDsymv_v2:=nil;
      cublasDsymv_v2_64:=nil;
      cublasCsymv_v2:=nil;
      cublasCsymv_v2_64:=nil;
      cublasZsymv_v2:=nil;
      cublasZsymv_v2_64:=nil;
      cublasChemv_v2:=nil;
      cublasChemv_v2_64:=nil;
      cublasZhemv_v2:=nil;
      cublasZhemv_v2_64:=nil;
      cublasSsbmv_v2:=nil;
      cublasSsbmv_v2_64:=nil;
      cublasDsbmv_v2:=nil;
      cublasDsbmv_v2_64:=nil;
      cublasChbmv_v2:=nil;
      cublasChbmv_v2_64:=nil;
      cublasZhbmv_v2:=nil;
      cublasZhbmv_v2_64:=nil;
      cublasSspmv_v2:=nil;
      cublasSspmv_v2_64:=nil;
      cublasDspmv_v2:=nil;
      cublasDspmv_v2_64:=nil;
      cublasChpmv_v2:=nil;
      cublasChpmv_v2_64:=nil;
      cublasZhpmv_v2:=nil;
      cublasZhpmv_v2_64:=nil;
      cublasSger_v2:=nil;
      cublasSger_v2_64:=nil;
      cublasDger_v2:=nil;
      cublasDger_v2_64:=nil;
      cublasCgeru_v2:=nil;
      cublasCgeru_v2_64:=nil;
      cublasCgerc_v2:=nil;
      cublasCgerc_v2_64:=nil;
      cublasZgeru_v2:=nil;
      cublasZgeru_v2_64:=nil;
      cublasZgerc_v2:=nil;
      cublasZgerc_v2_64:=nil;
      cublasSsyr_v2:=nil;
      cublasSsyr_v2_64:=nil;
      cublasDsyr_v2:=nil;
      cublasDsyr_v2_64:=nil;
      cublasCsyr_v2:=nil;
      cublasCsyr_v2_64:=nil;
      cublasZsyr_v2:=nil;
      cublasZsyr_v2_64:=nil;
      cublasCher_v2:=nil;
      cublasCher_v2_64:=nil;
      cublasZher_v2:=nil;
      cublasZher_v2_64:=nil;
      cublasSspr_v2:=nil;
      cublasSspr_v2_64:=nil;
      cublasDspr_v2:=nil;
      cublasDspr_v2_64:=nil;
      cublasChpr_v2:=nil;
      cublasChpr_v2_64:=nil;
      cublasZhpr_v2:=nil;
      cublasZhpr_v2_64:=nil;
      cublasSsyr2_v2:=nil;
      cublasSsyr2_v2_64:=nil;
      cublasDsyr2_v2:=nil;
      cublasDsyr2_v2_64:=nil;
      cublasCsyr2_v2:=nil;
      cublasCsyr2_v2_64:=nil;
      cublasZsyr2_v2:=nil;
      cublasZsyr2_v2_64:=nil;
      cublasCher2_v2:=nil;
      cublasCher2_v2_64:=nil;
      cublasZher2_v2:=nil;
      cublasZher2_v2_64:=nil;
      cublasSspr2_v2:=nil;
      cublasSspr2_v2_64:=nil;
      cublasDspr2_v2:=nil;
      cublasDspr2_v2_64:=nil;
      cublasChpr2_v2:=nil;
      cublasChpr2_v2_64:=nil;
      cublasZhpr2_v2:=nil;
      cublasZhpr2_v2_64:=nil;
      cublasSgemm_v2:=nil;
      cublasSgemm_v2_64:=nil;
      cublasDgemm_v2:=nil;
      cublasDgemm_v2_64:=nil;
      cublasCgemm_v2:=nil;
      cublasCgemm_v2_64:=nil;
      cublasZgemm_v2:=nil;
      cublasZgemm_v2_64:=nil;
      cublasSsyrk_v2:=nil;
      cublasSsyrk_v2_64:=nil;
      cublasDsyrk_v2:=nil;
      cublasDsyrk_v2_64:=nil;
      cublasCsyrk_v2:=nil;
      cublasCsyrk_v2_64:=nil;
      cublasZsyrk_v2:=nil;
      cublasZsyrk_v2_64:=nil;
      cublasCherk_v2:=nil;
      cublasCherk_v2_64:=nil;
      cublasZherk_v2:=nil;
      cublasZherk_v2_64:=nil;
      cublasSsyr2k_v2:=nil;
      cublasSsyr2k_v2_64:=nil;
      cublasDsyr2k_v2:=nil;
      cublasDsyr2k_v2_64:=nil;
      cublasCsyr2k_v2:=nil;
      cublasCsyr2k_v2_64:=nil;
      cublasZsyr2k_v2:=nil;
      cublasZsyr2k_v2_64:=nil;
      cublasCher2k_v2:=nil;
      cublasCher2k_v2_64:=nil;
      cublasZher2k_v2:=nil;
      cublasZher2k_v2_64:=nil;
      cublasSsymm_v2:=nil;
      cublasSsymm_v2_64:=nil;
      cublasDsymm_v2:=nil;
      cublasDsymm_v2_64:=nil;
      cublasCsymm_v2:=nil;
      cublasCsymm_v2_64:=nil;
      cublasZsymm_v2:=nil;
      cublasZsymm_v2_64:=nil;
      cublasChemm_v2:=nil;
      cublasChemm_v2_64:=nil;
      cublasZhemm_v2:=nil;
      cublasZhemm_v2_64:=nil;
      cublasStrsm_v2:=nil;
      cublasStrsm_v2_64:=nil;
      cublasDtrsm_v2:=nil;
      cublasDtrsm_v2_64:=nil;
      cublasCtrsm_v2:=nil;
      cublasCtrsm_v2_64:=nil;
      cublasZtrsm_v2:=nil;
      cublasZtrsm_v2_64:=nil;
      cublasStrmm_v2:=nil;
      cublasStrmm_v2_64:=nil;
      cublasDtrmm_v2:=nil;
      cublasDtrmm_v2_64:=nil;
      cublasCtrmm_v2:=nil;
      cublasCtrmm_v2_64:=nil;
      cublasZtrmm_v2:=nil;
      cublasZtrmm_v2_64:=nil;
      {$endif}
    end;


  procedure Loadcublas_api(lib : pchar);
    begin
      Freecublas_api;
      hlib:=LoadLibrary(lib);
      if hlib=0 then
        raise Exception.Create(format('Could not load library: %s',[lib]));

      //cublasInit := GetProcAddress(hlib,'cublasInit');
      //cublasShutdown := GetProcAddress(hlib,'cublasShutdown');
      //cublasGetError := GetProcAddress(hlib,'cublasGetError');
      //cublasGetError : GetProcAddress(hlib,'cublasGetError');
      //cublasAlloc := GetProcAddress(hlib,'cublasAlloc');
      //cublasFree := GetProcAddress(hlib,'cublasFree');
      //cublasSetKernelStream := GetProcAddress(hlib,'cublasSetKernelStream');

      cublasGetProperty:=GetProcAddress(hlib,'cublasGetProperty');
      cublasGetCudartVersion:=GetProcAddress(hlib,'cublasGetCudartVersion');
      cublasGetAtomicsMode:=GetProcAddress(hlib,'cublasGetAtomicsMode');
      cublasSetAtomicsMode:=GetProcAddress(hlib,'cublasSetAtomicsMode');
      cublasGetMathMode:=GetProcAddress(hlib,'cublasGetMathMode');
      cublasSetMathMode:=GetProcAddress(hlib,'cublasSetMathMode');
      cublasGetSmCountTarget:=GetProcAddress(hlib,'cublasGetSmCountTarget');
      cublasSetSmCountTarget:=GetProcAddress(hlib,'cublasSetSmCountTarget');
      cublasGetStatusName:=GetProcAddress(hlib,'cublasGetStatusName');
      cublasGetStatusString:=GetProcAddress(hlib,'cublasGetStatusString');
      cublasLoggerConfigure:=GetProcAddress(hlib,'cublasLoggerConfigure');
      cublasSetLoggerCallback:=GetProcAddress(hlib,'cublasSetLoggerCallback');
      cublasGetLoggerCallback:=GetProcAddress(hlib,'cublasGetLoggerCallback');
      cublasSetVector:=GetProcAddress(hlib,'cublasSetVector');
      cublasSetVector_64:=GetProcAddress(hlib,'cublasSetVector_64');
      cublasGetVector:=GetProcAddress(hlib,'cublasGetVector');
      cublasGetVector_64:=GetProcAddress(hlib,'cublasGetVector_64');
      cublasSetMatrix:=GetProcAddress(hlib,'cublasSetMatrix');
      cublasSetMatrix_64:=GetProcAddress(hlib,'cublasSetMatrix_64');
      cublasGetMatrix:=GetProcAddress(hlib,'cublasGetMatrix');
      cublasGetMatrix_64:=GetProcAddress(hlib,'cublasGetMatrix_64');
      cublasSetVectorAsync:=GetProcAddress(hlib,'cublasSetVectorAsync');
      cublasSetVectorAsync_64:=GetProcAddress(hlib,'cublasSetVectorAsync_64');
      cublasGetVectorAsync:=GetProcAddress(hlib,'cublasGetVectorAsync');
      cublasGetVectorAsync_64:=GetProcAddress(hlib,'cublasGetVectorAsync_64');
      cublasSetMatrixAsync:=GetProcAddress(hlib,'cublasSetMatrixAsync');
      cublasSetMatrixAsync_64:=GetProcAddress(hlib,'cublasSetMatrixAsync_64');
      cublasGetMatrixAsync:=GetProcAddress(hlib,'cublasGetMatrixAsync');
      cublasGetMatrixAsync_64:=GetProcAddress(hlib,'cublasGetMatrixAsync_64');
      cublasXerbla:=GetProcAddress(hlib,'cublasXerbla');
      cublasNrm2Ex:=GetProcAddress(hlib,'cublasNrm2Ex');
      cublasNrm2Ex_64:=GetProcAddress(hlib,'cublasNrm2Ex_64');
      cublasDotEx:=GetProcAddress(hlib,'cublasDotEx');
      cublasDotEx_64:=GetProcAddress(hlib,'cublasDotEx_64');
      cublasDotcEx:=GetProcAddress(hlib,'cublasDotcEx');
      cublasDotcEx_64:=GetProcAddress(hlib,'cublasDotcEx_64');
      cublasScalEx:=GetProcAddress(hlib,'cublasScalEx');
      cublasScalEx_64:=GetProcAddress(hlib,'cublasScalEx_64');
      cublasAxpyEx:=GetProcAddress(hlib,'cublasAxpyEx');
      cublasAxpyEx_64:=GetProcAddress(hlib,'cublasAxpyEx_64');
      cublasCopyEx:=GetProcAddress(hlib,'cublasCopyEx');
      cublasCopyEx_64:=GetProcAddress(hlib,'cublasCopyEx_64');
      cublasSwapEx:=GetProcAddress(hlib,'cublasSwapEx');
      cublasSwapEx_64:=GetProcAddress(hlib,'cublasSwapEx_64');
      cublasIamaxEx:=GetProcAddress(hlib,'cublasIamaxEx');
      cublasIamaxEx_64:=GetProcAddress(hlib,'cublasIamaxEx_64');
      cublasIaminEx:=GetProcAddress(hlib,'cublasIaminEx');
      cublasIaminEx_64:=GetProcAddress(hlib,'cublasIaminEx_64');
      cublasAsumEx:=GetProcAddress(hlib,'cublasAsumEx');
      cublasAsumEx_64:=GetProcAddress(hlib,'cublasAsumEx_64');
      cublasRotEx:=GetProcAddress(hlib,'cublasRotEx');
      cublasRotEx_64:=GetProcAddress(hlib,'cublasRotEx_64');
      cublasRotgEx:=GetProcAddress(hlib,'cublasRotgEx');
      cublasRotmEx:=GetProcAddress(hlib,'cublasRotmEx');
      cublasRotmEx_64:=GetProcAddress(hlib,'cublasRotmEx_64');
      cublasRotmgEx:=GetProcAddress(hlib,'cublasRotmgEx');
      cublasSgemvBatched:=GetProcAddress(hlib,'cublasSgemvBatched');
      cublasSgemvBatched_64:=GetProcAddress(hlib,'cublasSgemvBatched_64');
      cublasDgemvBatched:=GetProcAddress(hlib,'cublasDgemvBatched');
      cublasDgemvBatched_64:=GetProcAddress(hlib,'cublasDgemvBatched_64');
      cublasCgemvBatched:=GetProcAddress(hlib,'cublasCgemvBatched');
      cublasCgemvBatched_64:=GetProcAddress(hlib,'cublasCgemvBatched_64');
      cublasZgemvBatched:=GetProcAddress(hlib,'cublasZgemvBatched');
      cublasZgemvBatched_64:=GetProcAddress(hlib,'cublasZgemvBatched_64');
      cublasHSHgemvBatched:=GetProcAddress(hlib,'cublasHSHgemvBatched');
      cublasHSHgemvBatched_64:=GetProcAddress(hlib,'cublasHSHgemvBatched_64');
      cublasHSSgemvBatched:=GetProcAddress(hlib,'cublasHSSgemvBatched');
      cublasHSSgemvBatched_64:=GetProcAddress(hlib,'cublasHSSgemvBatched_64');
      cublasTSTgemvBatched:=GetProcAddress(hlib,'cublasTSTgemvBatched');
      cublasTSTgemvBatched_64:=GetProcAddress(hlib,'cublasTSTgemvBatched_64');
      cublasTSSgemvBatched:=GetProcAddress(hlib,'cublasTSSgemvBatched');
      cublasTSSgemvBatched_64:=GetProcAddress(hlib,'cublasTSSgemvBatched_64');
      cublasSgemvStridedBatched:=GetProcAddress(hlib,'cublasSgemvStridedBatched');
      cublasSgemvStridedBatched_64:=GetProcAddress(hlib,'cublasSgemvStridedBatched_64');
      cublasDgemvStridedBatched:=GetProcAddress(hlib,'cublasDgemvStridedBatched');
      cublasDgemvStridedBatched_64:=GetProcAddress(hlib,'cublasDgemvStridedBatched_64');
      cublasCgemvStridedBatched:=GetProcAddress(hlib,'cublasCgemvStridedBatched');
      cublasCgemvStridedBatched_64:=GetProcAddress(hlib,'cublasCgemvStridedBatched_64');
      cublasZgemvStridedBatched:=GetProcAddress(hlib,'cublasZgemvStridedBatched');
      cublasZgemvStridedBatched_64:=GetProcAddress(hlib,'cublasZgemvStridedBatched_64');
      cublasHSHgemvStridedBatched:=GetProcAddress(hlib,'cublasHSHgemvStridedBatched');
      cublasHSHgemvStridedBatched_64:=GetProcAddress(hlib,'cublasHSHgemvStridedBatched_64');
      cublasHSSgemvStridedBatched:=GetProcAddress(hlib,'cublasHSSgemvStridedBatched');
      cublasHSSgemvStridedBatched_64:=GetProcAddress(hlib,'cublasHSSgemvStridedBatched_64');
      cublasTSTgemvStridedBatched:=GetProcAddress(hlib,'cublasTSTgemvStridedBatched');
      cublasTSTgemvStridedBatched_64:=GetProcAddress(hlib,'cublasTSTgemvStridedBatched_64');
      cublasTSSgemvStridedBatched:=GetProcAddress(hlib,'cublasTSSgemvStridedBatched');
      cublasTSSgemvStridedBatched_64:=GetProcAddress(hlib,'cublasTSSgemvStridedBatched_64');
      cublasCgemm3m:=GetProcAddress(hlib,'cublasCgemm3m');
      cublasCgemm3m_64:=GetProcAddress(hlib,'cublasCgemm3m_64');
      cublasCgemm3mEx:=GetProcAddress(hlib,'cublasCgemm3mEx');
      cublasCgemm3mEx_64:=GetProcAddress(hlib,'cublasCgemm3mEx_64');
      cublasZgemm3m:=GetProcAddress(hlib,'cublasZgemm3m');
      cublasZgemm3m_64:=GetProcAddress(hlib,'cublasZgemm3m_64');
      cublasHgemm:=GetProcAddress(hlib,'cublasHgemm');
      cublasHgemm_64:=GetProcAddress(hlib,'cublasHgemm_64');
      cublasSgemmEx:=GetProcAddress(hlib,'cublasSgemmEx');
      cublasSgemmEx_64:=GetProcAddress(hlib,'cublasSgemmEx_64');
      cublasGemmEx:=GetProcAddress(hlib,'cublasGemmEx');
      cublasGemmEx_64:=GetProcAddress(hlib,'cublasGemmEx_64');
      cublasCgemmEx:=GetProcAddress(hlib,'cublasCgemmEx');
      cublasCgemmEx_64:=GetProcAddress(hlib,'cublasCgemmEx_64');
      cublasCsyrkEx:=GetProcAddress(hlib,'cublasCsyrkEx');
      cublasCsyrkEx_64:=GetProcAddress(hlib,'cublasCsyrkEx_64');
      cublasCsyrk3mEx:=GetProcAddress(hlib,'cublasCsyrk3mEx');
      cublasCsyrk3mEx_64:=GetProcAddress(hlib,'cublasCsyrk3mEx_64');
      cublasCherkEx:=GetProcAddress(hlib,'cublasCherkEx');
      cublasCherkEx_64:=GetProcAddress(hlib,'cublasCherkEx_64');
      cublasCherk3mEx:=GetProcAddress(hlib,'cublasCherk3mEx');
      cublasCherk3mEx_64:=GetProcAddress(hlib,'cublasCherk3mEx_64');
      cublasSsyrkx:=GetProcAddress(hlib,'cublasSsyrkx');
      cublasSsyrkx_64:=GetProcAddress(hlib,'cublasSsyrkx_64');
      cublasDsyrkx:=GetProcAddress(hlib,'cublasDsyrkx');
      cublasDsyrkx_64:=GetProcAddress(hlib,'cublasDsyrkx_64');
      cublasCsyrkx:=GetProcAddress(hlib,'cublasCsyrkx');
      cublasCsyrkx_64:=GetProcAddress(hlib,'cublasCsyrkx_64');
      cublasZsyrkx:=GetProcAddress(hlib,'cublasZsyrkx');
      cublasZsyrkx_64:=GetProcAddress(hlib,'cublasZsyrkx_64');
      cublasCherkx:=GetProcAddress(hlib,'cublasCherkx');
      cublasCherkx_64:=GetProcAddress(hlib,'cublasCherkx_64');
      cublasZherkx:=GetProcAddress(hlib,'cublasZherkx');
      cublasZherkx_64:=GetProcAddress(hlib,'cublasZherkx_64');
      cublasHgemmBatched:=GetProcAddress(hlib,'cublasHgemmBatched');
      cublasHgemmBatched_64:=GetProcAddress(hlib,'cublasHgemmBatched_64');
      cublasSgemmBatched:=GetProcAddress(hlib,'cublasSgemmBatched');
      cublasSgemmBatched_64:=GetProcAddress(hlib,'cublasSgemmBatched_64');
      cublasDgemmBatched:=GetProcAddress(hlib,'cublasDgemmBatched');
      cublasDgemmBatched_64:=GetProcAddress(hlib,'cublasDgemmBatched_64');
      cublasCgemmBatched:=GetProcAddress(hlib,'cublasCgemmBatched');
      cublasCgemmBatched_64:=GetProcAddress(hlib,'cublasCgemmBatched_64');
      cublasCgemm3mBatched:=GetProcAddress(hlib,'cublasCgemm3mBatched');
      cublasCgemm3mBatched_64:=GetProcAddress(hlib,'cublasCgemm3mBatched_64');
      cublasZgemmBatched:=GetProcAddress(hlib,'cublasZgemmBatched');
      cublasZgemmBatched_64:=GetProcAddress(hlib,'cublasZgemmBatched_64');
      cublasHgemmStridedBatched:=GetProcAddress(hlib,'cublasHgemmStridedBatched');
      cublasHgemmStridedBatched_64:=GetProcAddress(hlib,'cublasHgemmStridedBatched_64');
      cublasSgemmStridedBatched:=GetProcAddress(hlib,'cublasSgemmStridedBatched');
      cublasSgemmStridedBatched_64:=GetProcAddress(hlib,'cublasSgemmStridedBatched_64');
      cublasDgemmStridedBatched:=GetProcAddress(hlib,'cublasDgemmStridedBatched');
      cublasDgemmStridedBatched_64:=GetProcAddress(hlib,'cublasDgemmStridedBatched_64');
      cublasCgemmStridedBatched:=GetProcAddress(hlib,'cublasCgemmStridedBatched');
      cublasCgemmStridedBatched_64:=GetProcAddress(hlib,'cublasCgemmStridedBatched_64');
      cublasCgemm3mStridedBatched:=GetProcAddress(hlib,'cublasCgemm3mStridedBatched');
      cublasCgemm3mStridedBatched_64:=GetProcAddress(hlib,'cublasCgemm3mStridedBatched_64');
      cublasZgemmStridedBatched:=GetProcAddress(hlib,'cublasZgemmStridedBatched');
      cublasZgemmStridedBatched_64:=GetProcAddress(hlib,'cublasZgemmStridedBatched_64');
      cublasGemmBatchedEx:=GetProcAddress(hlib,'cublasGemmBatchedEx');
      cublasGemmBatchedEx_64:=GetProcAddress(hlib,'cublasGemmBatchedEx_64');
      cublasGemmStridedBatchedEx:=GetProcAddress(hlib,'cublasGemmStridedBatchedEx');
      cublasGemmStridedBatchedEx_64:=GetProcAddress(hlib,'cublasGemmStridedBatchedEx_64');
      cublasSgeam:=GetProcAddress(hlib,'cublasSgeam');
      cublasSgeam_64:=GetProcAddress(hlib,'cublasSgeam_64');
      cublasDgeam:=GetProcAddress(hlib,'cublasDgeam');
      cublasDgeam_64:=GetProcAddress(hlib,'cublasDgeam_64');
      cublasCgeam:=GetProcAddress(hlib,'cublasCgeam');
      cublasCgeam_64:=GetProcAddress(hlib,'cublasCgeam_64');
      cublasZgeam:=GetProcAddress(hlib,'cublasZgeam');
      cublasZgeam_64:=GetProcAddress(hlib,'cublasZgeam_64');
      cublasStrsmBatched:=GetProcAddress(hlib,'cublasStrsmBatched');
      cublasStrsmBatched_64:=GetProcAddress(hlib,'cublasStrsmBatched_64');
      cublasDtrsmBatched:=GetProcAddress(hlib,'cublasDtrsmBatched');
      cublasDtrsmBatched_64:=GetProcAddress(hlib,'cublasDtrsmBatched_64');
      cublasCtrsmBatched:=GetProcAddress(hlib,'cublasCtrsmBatched');
      cublasCtrsmBatched_64:=GetProcAddress(hlib,'cublasCtrsmBatched_64');
      cublasZtrsmBatched:=GetProcAddress(hlib,'cublasZtrsmBatched');
      cublasZtrsmBatched_64:=GetProcAddress(hlib,'cublasZtrsmBatched_64');
      cublasSdgmm:=GetProcAddress(hlib,'cublasSdgmm');
      cublasSdgmm_64:=GetProcAddress(hlib,'cublasSdgmm_64');
      cublasDdgmm:=GetProcAddress(hlib,'cublasDdgmm');
      cublasDdgmm_64:=GetProcAddress(hlib,'cublasDdgmm_64');
      cublasCdgmm:=GetProcAddress(hlib,'cublasCdgmm');
      cublasCdgmm_64:=GetProcAddress(hlib,'cublasCdgmm_64');
      cublasZdgmm:=GetProcAddress(hlib,'cublasZdgmm');
      cublasZdgmm_64:=GetProcAddress(hlib,'cublasZdgmm_64');
      cublasSmatinvBatched:=GetProcAddress(hlib,'cublasSmatinvBatched');
      cublasDmatinvBatched:=GetProcAddress(hlib,'cublasDmatinvBatched');
      cublasCmatinvBatched:=GetProcAddress(hlib,'cublasCmatinvBatched');
      cublasZmatinvBatched:=GetProcAddress(hlib,'cublasZmatinvBatched');
      cublasSgeqrfBatched:=GetProcAddress(hlib,'cublasSgeqrfBatched');
      cublasDgeqrfBatched:=GetProcAddress(hlib,'cublasDgeqrfBatched');
      cublasCgeqrfBatched:=GetProcAddress(hlib,'cublasCgeqrfBatched');
      cublasZgeqrfBatched:=GetProcAddress(hlib,'cublasZgeqrfBatched');
      cublasSgelsBatched:=GetProcAddress(hlib,'cublasSgelsBatched');
      cublasDgelsBatched:=GetProcAddress(hlib,'cublasDgelsBatched');
      cublasCgelsBatched:=GetProcAddress(hlib,'cublasCgelsBatched');
      cublasZgelsBatched:=GetProcAddress(hlib,'cublasZgelsBatched');
      cublasStpttr:=GetProcAddress(hlib,'cublasStpttr');
      cublasDtpttr:=GetProcAddress(hlib,'cublasDtpttr');
      cublasCtpttr:=GetProcAddress(hlib,'cublasCtpttr');
      cublasZtpttr:=GetProcAddress(hlib,'cublasZtpttr');
      cublasStrttp:=GetProcAddress(hlib,'cublasStrttp');
      cublasDtrttp:=GetProcAddress(hlib,'cublasDtrttp');
      cublasCtrttp:=GetProcAddress(hlib,'cublasCtrttp');
      cublasZtrttp:=GetProcAddress(hlib,'cublasZtrttp');
      cublasSgetrfBatched:=GetProcAddress(hlib,'cublasSgetrfBatched');
      cublasDgetrfBatched:=GetProcAddress(hlib,'cublasDgetrfBatched');
      cublasCgetrfBatched:=GetProcAddress(hlib,'cublasCgetrfBatched');
      cublasZgetrfBatched:=GetProcAddress(hlib,'cublasZgetrfBatched');
      cublasSgetriBatched:=GetProcAddress(hlib,'cublasSgetriBatched');
      cublasDgetriBatched:=GetProcAddress(hlib,'cublasDgetriBatched');
      cublasCgetriBatched:=GetProcAddress(hlib,'cublasCgetriBatched');
      cublasZgetriBatched:=GetProcAddress(hlib,'cublasZgetriBatched');
      cublasSgetrsBatched:=GetProcAddress(hlib,'cublasSgetrsBatched');
      cublasDgetrsBatched:=GetProcAddress(hlib,'cublasDgetrsBatched');
      cublasCgetrsBatched:=GetProcAddress(hlib,'cublasCgetrsBatched');
      cublasZgetrsBatched:=GetProcAddress(hlib,'cublasZgetrsBatched');
      cublasUint8gemmBias:=GetProcAddress(hlib,'cublasUint8gemmBias');
      {$ifdef CUBLAS_V2}
      cublasCreate_v2:=GetProcAddress(hlib,'cublasCreate_v2');
	cublasCreate := cublasCreate_v2;
      cublasDestroy_v2:=GetProcAddress(hlib,'cublasDestroy_v2');
	cublasDestroy := cublasDestroy_v2;
      cublasGetVersion_v2:=GetProcAddress(hlib,'cublasGetVersion_v2');
	cublasGetVersion := cublasGetVersion_v2;
      cublasSetWorkspace_v2:=GetProcAddress(hlib,'cublasSetWorkspace_v2');
	cublasSetWorkspace := cublasSetWorkspace_v2;
      cublasSetStream_v2:=GetProcAddress(hlib,'cublasSetStream_v2');
	cublasSetStream := cublasSetStream_v2;
      cublasGetStream_v2:=GetProcAddress(hlib,'cublasGetStream_v2');
	cublasGetStream := cublasGetStream_v2;
      cublasGetPointerMode_v2:=GetProcAddress(hlib,'cublasGetPointerMode_v2');
	cublasGetPointerMode := cublasGetPointerMode_v2;
      cublasSetPointerMode_v2:=GetProcAddress(hlib,'cublasSetPointerMode_v2');
	cublasSetPointerMode := cublasSetPointerMode_v2;
      cublasSnrm2_v2:=GetProcAddress(hlib,'cublasSnrm2_v2');
	cublasSnrm2 := cublasSnrm2_v2;
      cublasSnrm2_v2_64:=GetProcAddress(hlib,'cublasSnrm2_v2_64');
	cublasSnrm2_64 := cublasSnrm2_v2_64;
      cublasDnrm2_v2:=GetProcAddress(hlib,'cublasDnrm2_v2');
	cublasDnrm2 := cublasDnrm2_v2;
      cublasDnrm2_v2_64:=GetProcAddress(hlib,'cublasDnrm2_v2_64');
	cublasDnrm2_64 := cublasDnrm2_v2_64;
      cublasScnrm2_v2:=GetProcAddress(hlib,'cublasScnrm2_v2');
	cublasScnrm2 := cublasScnrm2_v2;
      cublasScnrm2_v2_64:=GetProcAddress(hlib,'cublasScnrm2_v2_64');
	cublasScnrm2_64 := cublasScnrm2_v2_64;
      cublasDznrm2_v2:=GetProcAddress(hlib,'cublasDznrm2_v2');
	cublasDznrm2 := cublasDznrm2_v2;
      cublasDznrm2_v2_64:=GetProcAddress(hlib,'cublasDznrm2_v2_64');
	cublasDznrm2_64 := cublasDznrm2_v2_64;
      cublasSdot_v2:=GetProcAddress(hlib,'cublasSdot_v2');
	cublasSdot := cublasSdot_v2;
      cublasSdot_v2_64:=GetProcAddress(hlib,'cublasSdot_v2_64');
	cublasSdot_64 := cublasSdot_v2_64;
      cublasDdot_v2:=GetProcAddress(hlib,'cublasDdot_v2');
	cublasDdot := cublasDdot_v2;
      cublasDdot_v2_64:=GetProcAddress(hlib,'cublasDdot_v2_64');
	cublasDdot_64 := cublasDdot_v2_64;
      cublasCdotu_v2:=GetProcAddress(hlib,'cublasCdotu_v2');
	cublasCdotu := cublasCdotu_v2;
      cublasCdotu_v2_64:=GetProcAddress(hlib,'cublasCdotu_v2_64');
	cublasCdotu_64 := cublasCdotu_v2_64;
      cublasCdotc_v2:=GetProcAddress(hlib,'cublasCdotc_v2');
	cublasCdotc := cublasCdotc_v2;
      cublasCdotc_v2_64:=GetProcAddress(hlib,'cublasCdotc_v2_64');
	cublasCdotc_64 := cublasCdotc_v2_64;
      cublasZdotu_v2:=GetProcAddress(hlib,'cublasZdotu_v2');
	cublasZdotu := cublasZdotu_v2;
      cublasZdotu_v2_64:=GetProcAddress(hlib,'cublasZdotu_v2_64');
	cublasZdotu_64 := cublasZdotu_v2_64;
      cublasZdotc_v2:=GetProcAddress(hlib,'cublasZdotc_v2');
	cublasZdotc := cublasZdotc_v2;
      cublasZdotc_v2_64:=GetProcAddress(hlib,'cublasZdotc_v2_64');
	cublasZdotc_64 := cublasZdotc_v2_64;
      cublasSscal_v2:=GetProcAddress(hlib,'cublasSscal_v2');
	cublasSscal := cublasSscal_v2;
      cublasSscal_v2_64:=GetProcAddress(hlib,'cublasSscal_v2_64');
	cublasSscal_64 := cublasSscal_v2_64;
      cublasDscal_v2:=GetProcAddress(hlib,'cublasDscal_v2');
	cublasDscal := cublasDscal_v2;
      cublasDscal_v2_64:=GetProcAddress(hlib,'cublasDscal_v2_64');
	cublasDscal_64 := cublasDscal_v2_64;
      cublasCscal_v2:=GetProcAddress(hlib,'cublasCscal_v2');
	cublasCscal := cublasCscal_v2;
      cublasCscal_v2_64:=GetProcAddress(hlib,'cublasCscal_v2_64');
	cublasCscal_64 := cublasCscal_v2_64;
      cublasCsscal_v2:=GetProcAddress(hlib,'cublasCsscal_v2');
	cublasCsscal := cublasCsscal_v2;
      cublasCsscal_v2_64:=GetProcAddress(hlib,'cublasCsscal_v2_64');
	cublasCsscal_64 := cublasCsscal_v2_64;
      cublasZscal_v2:=GetProcAddress(hlib,'cublasZscal_v2');
	cublasZscal := cublasZscal_v2;
      cublasZscal_v2_64:=GetProcAddress(hlib,'cublasZscal_v2_64');
	cublasZscal_64 := cublasZscal_v2_64;
      cublasZdscal_v2:=GetProcAddress(hlib,'cublasZdscal_v2');
	cublasZdscal := cublasZdscal_v2;
      cublasZdscal_v2_64:=GetProcAddress(hlib,'cublasZdscal_v2_64');
	cublasZdscal_64 := cublasZdscal_v2_64;
      cublasSaxpy_v2:=GetProcAddress(hlib,'cublasSaxpy_v2');
	cublasSaxpy := cublasSaxpy_v2;
      cublasSaxpy_v2_64:=GetProcAddress(hlib,'cublasSaxpy_v2_64');
	cublasSaxpy_64 := cublasSaxpy_v2_64;
      cublasDaxpy_v2:=GetProcAddress(hlib,'cublasDaxpy_v2');
	cublasDaxpy := cublasDaxpy_v2;
      cublasDaxpy_v2_64:=GetProcAddress(hlib,'cublasDaxpy_v2_64');
	cublasDaxpy_64 := cublasDaxpy_v2_64;
      cublasCaxpy_v2:=GetProcAddress(hlib,'cublasCaxpy_v2');
	cublasCaxpy := cublasCaxpy_v2;
      cublasCaxpy_v2_64:=GetProcAddress(hlib,'cublasCaxpy_v2_64');
	cublasCaxpy_64 := cublasCaxpy_v2_64;
      cublasZaxpy_v2:=GetProcAddress(hlib,'cublasZaxpy_v2');
	cublasZaxpy := cublasZaxpy_v2;
      cublasZaxpy_v2_64:=GetProcAddress(hlib,'cublasZaxpy_v2_64');
	cublasZaxpy_64 := cublasZaxpy_v2_64;
      cublasScopy_v2:=GetProcAddress(hlib,'cublasScopy_v2');
	cublasScopy := cublasScopy_v2;
      cublasScopy_v2_64:=GetProcAddress(hlib,'cublasScopy_v2_64');
	cublasScopy_64 := cublasScopy_v2_64;
      cublasDcopy_v2:=GetProcAddress(hlib,'cublasDcopy_v2');
	cublasDcopy := cublasDcopy_v2;
      cublasDcopy_v2_64:=GetProcAddress(hlib,'cublasDcopy_v2_64');
	cublasDcopy_64 := cublasDcopy_v2_64;
      cublasCcopy_v2:=GetProcAddress(hlib,'cublasCcopy_v2');
	cublasCcopy := cublasCcopy_v2;
      cublasCcopy_v2_64:=GetProcAddress(hlib,'cublasCcopy_v2_64');
	cublasCcopy_64 := cublasCcopy_v2_64;
      cublasZcopy_v2:=GetProcAddress(hlib,'cublasZcopy_v2');
	cublasZcopy := cublasZcopy_v2;
      cublasZcopy_v2_64:=GetProcAddress(hlib,'cublasZcopy_v2_64');
	cublasZcopy_64 := cublasZcopy_v2_64;
      cublasSswap_v2:=GetProcAddress(hlib,'cublasSswap_v2');
	cublasSswap := cublasSswap_v2;
      cublasSswap_v2_64:=GetProcAddress(hlib,'cublasSswap_v2_64');
	cublasSswap_64 := cublasSswap_v2_64;
      cublasDswap_v2:=GetProcAddress(hlib,'cublasDswap_v2');
	cublasDswap := cublasDswap_v2;
      cublasDswap_v2_64:=GetProcAddress(hlib,'cublasDswap_v2_64');
	cublasDswap_64 := cublasDswap_v2_64;
      cublasCswap_v2:=GetProcAddress(hlib,'cublasCswap_v2');
	cublasCswap := cublasCswap_v2;
      cublasCswap_v2_64:=GetProcAddress(hlib,'cublasCswap_v2_64');
	cublasCswap_64 := cublasCswap_v2_64;
      cublasZswap_v2:=GetProcAddress(hlib,'cublasZswap_v2');
	cublasZswap := cublasZswap_v2;
      cublasZswap_v2_64:=GetProcAddress(hlib,'cublasZswap_v2_64');
	cublasZswap_64 := cublasZswap_v2_64;
      cublasIsamax_v2:=GetProcAddress(hlib,'cublasIsamax_v2');
	cublasIsamax := cublasIsamax_v2;
      cublasIsamax_v2_64:=GetProcAddress(hlib,'cublasIsamax_v2_64');
	cublasIsamax_64 := cublasIsamax_v2_64;
      cublasIdamax_v2:=GetProcAddress(hlib,'cublasIdamax_v2');
	cublasIdamax := cublasIdamax_v2;
      cublasIdamax_v2_64:=GetProcAddress(hlib,'cublasIdamax_v2_64');
	cublasIdamax_64 := cublasIdamax_v2_64;
      cublasIcamax_v2:=GetProcAddress(hlib,'cublasIcamax_v2');
	cublasIcamax := cublasIcamax_v2;
      cublasIcamax_v2_64:=GetProcAddress(hlib,'cublasIcamax_v2_64');
	cublasIcamax_64 := cublasIcamax_v2_64;
      cublasIzamax_v2:=GetProcAddress(hlib,'cublasIzamax_v2');
	cublasIzamax := cublasIzamax_v2;
      cublasIzamax_v2_64:=GetProcAddress(hlib,'cublasIzamax_v2_64');
	cublasIzamax_64 := cublasIzamax_v2_64;
      cublasIsamin_v2:=GetProcAddress(hlib,'cublasIsamin_v2');
	cublasIsamin := cublasIsamin_v2;
      cublasIsamin_v2_64:=GetProcAddress(hlib,'cublasIsamin_v2_64');
	cublasIsamin_64 := cublasIsamin_v2_64;
      cublasIdamin_v2:=GetProcAddress(hlib,'cublasIdamin_v2');
	cublasIdamin := cublasIdamin_v2;
      cublasIdamin_v2_64:=GetProcAddress(hlib,'cublasIdamin_v2_64');
	cublasIdamin_64 := cublasIdamin_v2_64;
      cublasIcamin_v2:=GetProcAddress(hlib,'cublasIcamin_v2');
	cublasIcamin := cublasIcamin_v2;
      cublasIcamin_v2_64:=GetProcAddress(hlib,'cublasIcamin_v2_64');
	cublasIcamin_64 := cublasIcamin_v2_64;
      cublasIzamin_v2:=GetProcAddress(hlib,'cublasIzamin_v2');
	cublasIzamin := cublasIzamin_v2;
      cublasIzamin_v2_64:=GetProcAddress(hlib,'cublasIzamin_v2_64');
	cublasIzamin_64 := cublasIzamin_v2_64;
      cublasSasum_v2:=GetProcAddress(hlib,'cublasSasum_v2');
	cublasSasum := cublasSasum_v2;
      cublasSasum_v2_64:=GetProcAddress(hlib,'cublasSasum_v2_64');
	cublasSasum_64 := cublasSasum_v2_64;
      cublasDasum_v2:=GetProcAddress(hlib,'cublasDasum_v2');
	cublasDasum := cublasDasum_v2;
      cublasDasum_v2_64:=GetProcAddress(hlib,'cublasDasum_v2_64');
	cublasDasum_64 := cublasDasum_v2_64;
      cublasScasum_v2:=GetProcAddress(hlib,'cublasScasum_v2');
	cublasScasum := cublasScasum_v2;
      cublasScasum_v2_64:=GetProcAddress(hlib,'cublasScasum_v2_64');
	cublasScasum_64 := cublasScasum_v2_64;
      cublasDzasum_v2:=GetProcAddress(hlib,'cublasDzasum_v2');
	cublasDzasum := cublasDzasum_v2;
      cublasDzasum_v2_64:=GetProcAddress(hlib,'cublasDzasum_v2_64');
	cublasDzasum_64 := cublasDzasum_v2_64;
      cublasSrot_v2:=GetProcAddress(hlib,'cublasSrot_v2');
	cublasSrot := cublasSrot_v2;
      cublasSrot_v2_64:=GetProcAddress(hlib,'cublasSrot_v2_64');
	cublasSrot_64 := cublasSrot_v2_64;
      cublasDrot_v2:=GetProcAddress(hlib,'cublasDrot_v2');
	cublasDrot := cublasDrot_v2;
      cublasDrot_v2_64:=GetProcAddress(hlib,'cublasDrot_v2_64');
	cublasDrot_64 := cublasDrot_v2_64;
      cublasCrot_v2:=GetProcAddress(hlib,'cublasCrot_v2');
	cublasCrot := cublasCrot_v2;
      cublasCrot_v2_64:=GetProcAddress(hlib,'cublasCrot_v2_64');
	cublasCrot_64 := cublasCrot_v2_64;
      cublasCsrot_v2:=GetProcAddress(hlib,'cublasCsrot_v2');
	cublasCsrot := cublasCsrot_v2;
      cublasCsrot_v2_64:=GetProcAddress(hlib,'cublasCsrot_v2_64');
	cublasCsrot_64 := cublasCsrot_v2_64;
      cublasZrot_v2:=GetProcAddress(hlib,'cublasZrot_v2');
	cublasZrot := cublasZrot_v2;
      cublasZrot_v2_64:=GetProcAddress(hlib,'cublasZrot_v2_64');
	cublasZrot_64 := cublasZrot_v2_64;
      cublasZdrot_v2:=GetProcAddress(hlib,'cublasZdrot_v2');
	cublasZdrot := cublasZdrot_v2;
      cublasZdrot_v2_64:=GetProcAddress(hlib,'cublasZdrot_v2_64');
	cublasZdrot_64 := cublasZdrot_v2_64;
      cublasSrotg_v2:=GetProcAddress(hlib,'cublasSrotg_v2');
	cublasSrotg := cublasSrotg_v2;
      cublasDrotg_v2:=GetProcAddress(hlib,'cublasDrotg_v2');
	cublasDrotg := cublasDrotg_v2;
      cublasCrotg_v2:=GetProcAddress(hlib,'cublasCrotg_v2');
	cublasCrotg := cublasCrotg_v2;
      cublasZrotg_v2:=GetProcAddress(hlib,'cublasZrotg_v2');
	cublasZrotg := cublasZrotg_v2;
      cublasSrotm_v2:=GetProcAddress(hlib,'cublasSrotm_v2');
	cublasSrotm := cublasSrotm_v2;
      cublasSrotm_v2_64:=GetProcAddress(hlib,'cublasSrotm_v2_64');
	cublasSrotm_64 := cublasSrotm_v2_64;
      cublasDrotm_v2:=GetProcAddress(hlib,'cublasDrotm_v2');
	cublasDrotm := cublasDrotm_v2;
      cublasDrotm_v2_64:=GetProcAddress(hlib,'cublasDrotm_v2_64');
	cublasDrotm_64 := cublasDrotm_v2_64;
      cublasSrotmg_v2:=GetProcAddress(hlib,'cublasSrotmg_v2');
	cublasSrotmg := cublasSrotmg_v2;
      cublasDrotmg_v2:=GetProcAddress(hlib,'cublasDrotmg_v2');
	cublasDrotmg := cublasDrotmg_v2;
      cublasSgemv_v2:=GetProcAddress(hlib,'cublasSgemv_v2');
	cublasSgemv := cublasSgemv_v2;
      cublasSgemv_v2_64:=GetProcAddress(hlib,'cublasSgemv_v2_64');
	cublasSgemv_64 := cublasSgemv_v2_64;
      cublasDgemv_v2:=GetProcAddress(hlib,'cublasDgemv_v2');
	cublasDgemv := cublasDgemv_v2;
      cublasDgemv_v2_64:=GetProcAddress(hlib,'cublasDgemv_v2_64');
	cublasDgemv_64 := cublasDgemv_v2_64;
      cublasCgemv_v2:=GetProcAddress(hlib,'cublasCgemv_v2');
	cublasCgemv := cublasCgemv_v2;
      cublasCgemv_v2_64:=GetProcAddress(hlib,'cublasCgemv_v2_64');
	cublasCgemv_64 := cublasCgemv_v2_64;
      cublasZgemv_v2:=GetProcAddress(hlib,'cublasZgemv_v2');
	cublasZgemv := cublasZgemv_v2;
      cublasZgemv_v2_64:=GetProcAddress(hlib,'cublasZgemv_v2_64');
	cublasZgemv_64 := cublasZgemv_v2_64;
      cublasSgbmv_v2:=GetProcAddress(hlib,'cublasSgbmv_v2');
	cublasSgbmv := cublasSgbmv_v2;
      cublasSgbmv_v2_64:=GetProcAddress(hlib,'cublasSgbmv_v2_64');
	cublasSgbmv_64 := cublasSgbmv_v2_64;
      cublasDgbmv_v2:=GetProcAddress(hlib,'cublasDgbmv_v2');
	cublasDgbmv := cublasDgbmv_v2;
      cublasDgbmv_v2_64:=GetProcAddress(hlib,'cublasDgbmv_v2_64');
	cublasDgbmv_64 := cublasDgbmv_v2_64;
      cublasCgbmv_v2:=GetProcAddress(hlib,'cublasCgbmv_v2');
	cublasCgbmv := cublasCgbmv_v2;
      cublasCgbmv_v2_64:=GetProcAddress(hlib,'cublasCgbmv_v2_64');
	cublasCgbmv_64 := cublasCgbmv_v2_64;
      cublasZgbmv_v2:=GetProcAddress(hlib,'cublasZgbmv_v2');
	cublasZgbmv := cublasZgbmv_v2;
      cublasZgbmv_v2_64:=GetProcAddress(hlib,'cublasZgbmv_v2_64');
	cublasZgbmv_64 := cublasZgbmv_v2_64;
      cublasStrmv_v2:=GetProcAddress(hlib,'cublasStrmv_v2');
	cublasStrmv := cublasStrmv_v2;
      cublasStrmv_v2_64:=GetProcAddress(hlib,'cublasStrmv_v2_64');
	cublasStrmv_64 := cublasStrmv_v2_64;
      cublasDtrmv_v2:=GetProcAddress(hlib,'cublasDtrmv_v2');
	cublasDtrmv := cublasDtrmv_v2;
      cublasDtrmv_v2_64:=GetProcAddress(hlib,'cublasDtrmv_v2_64');
	cublasDtrmv_64 := cublasDtrmv_v2_64;
      cublasCtrmv_v2:=GetProcAddress(hlib,'cublasCtrmv_v2');
	cublasCtrmv := cublasCtrmv_v2;
      cublasCtrmv_v2_64:=GetProcAddress(hlib,'cublasCtrmv_v2_64');
	cublasCtrmv_64 := cublasCtrmv_v2_64;
      cublasZtrmv_v2:=GetProcAddress(hlib,'cublasZtrmv_v2');
	cublasZtrmv := cublasZtrmv_v2;
      cublasZtrmv_v2_64:=GetProcAddress(hlib,'cublasZtrmv_v2_64');
	cublasZtrmv_64 := cublasZtrmv_v2_64;
      cublasStbmv_v2:=GetProcAddress(hlib,'cublasStbmv_v2');
	cublasStbmv := cublasStbmv_v2;
      cublasStbmv_v2_64:=GetProcAddress(hlib,'cublasStbmv_v2_64');
	cublasStbmv_64 := cublasStbmv_v2_64;
      cublasDtbmv_v2:=GetProcAddress(hlib,'cublasDtbmv_v2');
	cublasDtbmv := cublasDtbmv_v2;
      cublasDtbmv_v2_64:=GetProcAddress(hlib,'cublasDtbmv_v2_64');
	cublasDtbmv_64 := cublasDtbmv_v2_64;
      cublasCtbmv_v2:=GetProcAddress(hlib,'cublasCtbmv_v2');
	cublasCtbmv := cublasCtbmv_v2;
      cublasCtbmv_v2_64:=GetProcAddress(hlib,'cublasCtbmv_v2_64');
	cublasCtbmv_64 := cublasCtbmv_v2_64;
      cublasZtbmv_v2:=GetProcAddress(hlib,'cublasZtbmv_v2');
	cublasZtbmv := cublasZtbmv_v2;
      cublasZtbmv_v2_64:=GetProcAddress(hlib,'cublasZtbmv_v2_64');
	cublasZtbmv_64 := cublasZtbmv_v2_64;
      cublasStpmv_v2:=GetProcAddress(hlib,'cublasStpmv_v2');
	cublasStpmv := cublasStpmv_v2;
      cublasStpmv_v2_64:=GetProcAddress(hlib,'cublasStpmv_v2_64');
	cublasStpmv_64 := cublasStpmv_v2_64;
      cublasDtpmv_v2:=GetProcAddress(hlib,'cublasDtpmv_v2');
	cublasDtpmv := cublasDtpmv_v2;
      cublasDtpmv_v2_64:=GetProcAddress(hlib,'cublasDtpmv_v2_64');
	cublasDtpmv_64 := cublasDtpmv_v2_64;
      cublasCtpmv_v2:=GetProcAddress(hlib,'cublasCtpmv_v2');
	cublasCtpmv := cublasCtpmv_v2;
      cublasCtpmv_v2_64:=GetProcAddress(hlib,'cublasCtpmv_v2_64');
	cublasCtpmv_64 := cublasCtpmv_v2_64;
      cublasZtpmv_v2:=GetProcAddress(hlib,'cublasZtpmv_v2');
	cublasZtpmv := cublasZtpmv_v2;
      cublasZtpmv_v2_64:=GetProcAddress(hlib,'cublasZtpmv_v2_64');
	cublasZtpmv_64 := cublasZtpmv_v2_64;
      cublasStrsv_v2:=GetProcAddress(hlib,'cublasStrsv_v2');
	cublasStrsv := cublasStrsv_v2;
      cublasStrsv_v2_64:=GetProcAddress(hlib,'cublasStrsv_v2_64');
	cublasStrsv_64 := cublasStrsv_v2_64;
      cublasDtrsv_v2:=GetProcAddress(hlib,'cublasDtrsv_v2');
	cublasDtrsv := cublasDtrsv_v2;
      cublasDtrsv_v2_64:=GetProcAddress(hlib,'cublasDtrsv_v2_64');
	cublasDtrsv_64 := cublasDtrsv_v2_64;
      cublasCtrsv_v2:=GetProcAddress(hlib,'cublasCtrsv_v2');
	cublasCtrsv := cublasCtrsv_v2;
      cublasCtrsv_v2_64:=GetProcAddress(hlib,'cublasCtrsv_v2_64');
	cublasCtrsv_64 := cublasCtrsv_v2_64;
      cublasZtrsv_v2:=GetProcAddress(hlib,'cublasZtrsv_v2');
	cublasZtrsv := cublasZtrsv_v2;
      cublasZtrsv_v2_64:=GetProcAddress(hlib,'cublasZtrsv_v2_64');
	cublasZtrsv_64 := cublasZtrsv_v2_64;
      cublasStpsv_v2:=GetProcAddress(hlib,'cublasStpsv_v2');
	cublasStpsv := cublasStpsv_v2;
      cublasStpsv_v2_64:=GetProcAddress(hlib,'cublasStpsv_v2_64');
	cublasStpsv_64 := cublasStpsv_v2_64;
      cublasDtpsv_v2:=GetProcAddress(hlib,'cublasDtpsv_v2');
	cublasDtpsv := cublasDtpsv_v2;
      cublasDtpsv_v2_64:=GetProcAddress(hlib,'cublasDtpsv_v2_64');
	cublasDtpsv_64 := cublasDtpsv_v2_64;
      cublasCtpsv_v2:=GetProcAddress(hlib,'cublasCtpsv_v2');
	cublasCtpsv := cublasCtpsv_v2;
      cublasCtpsv_v2_64:=GetProcAddress(hlib,'cublasCtpsv_v2_64');
	cublasCtpsv_64 := cublasCtpsv_v2_64;
      cublasZtpsv_v2:=GetProcAddress(hlib,'cublasZtpsv_v2');
	cublasZtpsv := cublasZtpsv_v2;
      cublasZtpsv_v2_64:=GetProcAddress(hlib,'cublasZtpsv_v2_64');
	cublasZtpsv_64 := cublasZtpsv_v2_64;
      cublasStbsv_v2:=GetProcAddress(hlib,'cublasStbsv_v2');
	cublasStbsv := cublasStbsv_v2;
      cublasStbsv_v2_64:=GetProcAddress(hlib,'cublasStbsv_v2_64');
	cublasStbsv_64 := cublasStbsv_v2_64;
      cublasDtbsv_v2:=GetProcAddress(hlib,'cublasDtbsv_v2');
	cublasDtbsv := cublasDtbsv_v2;
      cublasDtbsv_v2_64:=GetProcAddress(hlib,'cublasDtbsv_v2_64');
	cublasDtbsv_64 := cublasDtbsv_v2_64;
      cublasCtbsv_v2:=GetProcAddress(hlib,'cublasCtbsv_v2');
	cublasCtbsv := cublasCtbsv_v2;
      cublasCtbsv_v2_64:=GetProcAddress(hlib,'cublasCtbsv_v2_64');
	cublasCtbsv_64 := cublasCtbsv_v2_64;
      cublasZtbsv_v2:=GetProcAddress(hlib,'cublasZtbsv_v2');
	cublasZtbsv := cublasZtbsv_v2;
      cublasZtbsv_v2_64:=GetProcAddress(hlib,'cublasZtbsv_v2_64');
	cublasZtbsv_64 := cublasZtbsv_v2_64;
      cublasSsymv_v2:=GetProcAddress(hlib,'cublasSsymv_v2');
	cublasSsymv := cublasSsymv_v2;
      cublasSsymv_v2_64:=GetProcAddress(hlib,'cublasSsymv_v2_64');
	cublasSsymv_64 := cublasSsymv_v2_64;
      cublasDsymv_v2:=GetProcAddress(hlib,'cublasDsymv_v2');
	cublasDsymv := cublasDsymv_v2;
      cublasDsymv_v2_64:=GetProcAddress(hlib,'cublasDsymv_v2_64');
	cublasDsymv_64 := cublasDsymv_v2_64;
      cublasCsymv_v2:=GetProcAddress(hlib,'cublasCsymv_v2');
	cublasCsymv := cublasCsymv_v2;
      cublasCsymv_v2_64:=GetProcAddress(hlib,'cublasCsymv_v2_64');
	cublasCsymv_64 := cublasCsymv_v2_64;
      cublasZsymv_v2:=GetProcAddress(hlib,'cublasZsymv_v2');
	cublasZsymv := cublasZsymv_v2;
      cublasZsymv_v2_64:=GetProcAddress(hlib,'cublasZsymv_v2_64');
	cublasZsymv_64 := cublasZsymv_v2_64;
      cublasChemv_v2:=GetProcAddress(hlib,'cublasChemv_v2');
	cublasChemv := cublasChemv_v2;
      cublasChemv_v2_64:=GetProcAddress(hlib,'cublasChemv_v2_64');
	cublasChemv_64 := cublasChemv_v2_64;
      cublasZhemv_v2:=GetProcAddress(hlib,'cublasZhemv_v2');
	cublasZhemv := cublasZhemv_v2;
      cublasZhemv_v2_64:=GetProcAddress(hlib,'cublasZhemv_v2_64');
	cublasZhemv_64 := cublasZhemv_v2_64;
      cublasSsbmv_v2:=GetProcAddress(hlib,'cublasSsbmv_v2');
	cublasSsbmv := cublasSsbmv_v2;
      cublasSsbmv_v2_64:=GetProcAddress(hlib,'cublasSsbmv_v2_64');
	cublasSsbmv_64 := cublasSsbmv_v2_64;
      cublasDsbmv_v2:=GetProcAddress(hlib,'cublasDsbmv_v2');
	cublasDsbmv := cublasDsbmv_v2;
      cublasDsbmv_v2_64:=GetProcAddress(hlib,'cublasDsbmv_v2_64');
	cublasDsbmv_64 := cublasDsbmv_v2_64;
      cublasChbmv_v2:=GetProcAddress(hlib,'cublasChbmv_v2');
	cublasChbmv := cublasChbmv_v2;
      cublasChbmv_v2_64:=GetProcAddress(hlib,'cublasChbmv_v2_64');
	cublasChbmv_64 := cublasChbmv_v2_64;
      cublasZhbmv_v2:=GetProcAddress(hlib,'cublasZhbmv_v2');
	cublasZhbmv := cublasZhbmv_v2;
      cublasZhbmv_v2_64:=GetProcAddress(hlib,'cublasZhbmv_v2_64');
	cublasZhbmv_64 := cublasZhbmv_v2_64;
      cublasSspmv_v2:=GetProcAddress(hlib,'cublasSspmv_v2');
	cublasSspmv := cublasSspmv_v2;
      cublasSspmv_v2_64:=GetProcAddress(hlib,'cublasSspmv_v2_64');
	cublasSspmv_64 := cublasSspmv_v2_64;
      cublasDspmv_v2:=GetProcAddress(hlib,'cublasDspmv_v2');
	cublasDspmv := cublasDspmv_v2;
      cublasDspmv_v2_64:=GetProcAddress(hlib,'cublasDspmv_v2_64');
	cublasDspmv_64 := cublasDspmv_v2_64;
      cublasChpmv_v2:=GetProcAddress(hlib,'cublasChpmv_v2');
	cublasChpmv := cublasChpmv_v2;
      cublasChpmv_v2_64:=GetProcAddress(hlib,'cublasChpmv_v2_64');
	cublasChpmv_64 := cublasChpmv_v2_64;
      cublasZhpmv_v2:=GetProcAddress(hlib,'cublasZhpmv_v2');
	cublasZhpmv := cublasZhpmv_v2;
      cublasZhpmv_v2_64:=GetProcAddress(hlib,'cublasZhpmv_v2_64');
	cublasZhpmv_64 := cublasZhpmv_v2_64;
      cublasSger_v2:=GetProcAddress(hlib,'cublasSger_v2');
	cublasSger := cublasSger_v2;
      cublasSger_v2_64:=GetProcAddress(hlib,'cublasSger_v2_64');
	cublasSger_64 := cublasSger_v2_64;
      cublasDger_v2:=GetProcAddress(hlib,'cublasDger_v2');
	cublasDger := cublasDger_v2;
      cublasDger_v2_64:=GetProcAddress(hlib,'cublasDger_v2_64');
	cublasDger_64 := cublasDger_v2_64;
      cublasCgeru_v2:=GetProcAddress(hlib,'cublasCgeru_v2');
	cublasCgeru := cublasCgeru_v2;
      cublasCgeru_v2_64:=GetProcAddress(hlib,'cublasCgeru_v2_64');
	cublasCgeru_64 := cublasCgeru_v2_64;
      cublasCgerc_v2:=GetProcAddress(hlib,'cublasCgerc_v2');
	cublasCgerc := cublasCgerc_v2;
      cublasCgerc_v2_64:=GetProcAddress(hlib,'cublasCgerc_v2_64');
	cublasCgerc_64 := cublasCgerc_v2_64;
      cublasZgeru_v2:=GetProcAddress(hlib,'cublasZgeru_v2');
	cublasZgeru := cublasZgeru_v2;
      cublasZgeru_v2_64:=GetProcAddress(hlib,'cublasZgeru_v2_64');
	cublasZgeru_64 := cublasZgeru_v2_64;
      cublasZgerc_v2:=GetProcAddress(hlib,'cublasZgerc_v2');
	cublasZgerc := cublasZgerc_v2;
      cublasZgerc_v2_64:=GetProcAddress(hlib,'cublasZgerc_v2_64');
	cublasZgerc_64 := cublasZgerc_v2_64;
      cublasSsyr_v2:=GetProcAddress(hlib,'cublasSsyr_v2');
	cublasSsyr := cublasSsyr_v2;
      cublasSsyr_v2_64:=GetProcAddress(hlib,'cublasSsyr_v2_64');
	cublasSsyr_64 := cublasSsyr_v2_64;
      cublasDsyr_v2:=GetProcAddress(hlib,'cublasDsyr_v2');
	cublasDsyr := cublasDsyr_v2;
      cublasDsyr_v2_64:=GetProcAddress(hlib,'cublasDsyr_v2_64');
	cublasDsyr_64 := cublasDsyr_v2_64;
      cublasCsyr_v2:=GetProcAddress(hlib,'cublasCsyr_v2');
	cublasCsyr := cublasCsyr_v2;
      cublasCsyr_v2_64:=GetProcAddress(hlib,'cublasCsyr_v2_64');
	cublasCsyr_64 := cublasCsyr_v2_64;
      cublasZsyr_v2:=GetProcAddress(hlib,'cublasZsyr_v2');
	cublasZsyr := cublasZsyr_v2;
      cublasZsyr_v2_64:=GetProcAddress(hlib,'cublasZsyr_v2_64');
	cublasZsyr_64 := cublasZsyr_v2_64;
      cublasCher_v2:=GetProcAddress(hlib,'cublasCher_v2');
	cublasCher := cublasCher_v2;
      cublasCher_v2_64:=GetProcAddress(hlib,'cublasCher_v2_64');
	cublasCher_64 := cublasCher_v2_64;
      cublasZher_v2:=GetProcAddress(hlib,'cublasZher_v2');
	cublasZher := cublasZher_v2;
      cublasZher_v2_64:=GetProcAddress(hlib,'cublasZher_v2_64');
	cublasZher_64 := cublasZher_v2_64;
      cublasSspr_v2:=GetProcAddress(hlib,'cublasSspr_v2');
	cublasSspr := cublasSspr_v2;
      cublasSspr_v2_64:=GetProcAddress(hlib,'cublasSspr_v2_64');
	cublasSspr_64 := cublasSspr_v2_64;
      cublasDspr_v2:=GetProcAddress(hlib,'cublasDspr_v2');
	cublasDspr := cublasDspr_v2;
      cublasDspr_v2_64:=GetProcAddress(hlib,'cublasDspr_v2_64');
	cublasDspr_64 := cublasDspr_v2_64;
      cublasChpr_v2:=GetProcAddress(hlib,'cublasChpr_v2');
	cublasChpr := cublasChpr_v2;
      cublasChpr_v2_64:=GetProcAddress(hlib,'cublasChpr_v2_64');
	cublasChpr_64 := cublasChpr_v2_64;
      cublasZhpr_v2:=GetProcAddress(hlib,'cublasZhpr_v2');
	cublasZhpr := cublasZhpr_v2;
      cublasZhpr_v2_64:=GetProcAddress(hlib,'cublasZhpr_v2_64');
	cublasZhpr_64 := cublasZhpr_v2_64;
      cublasSsyr2_v2:=GetProcAddress(hlib,'cublasSsyr2_v2');
	cublasSsyr2 := cublasSsyr2_v2;
      cublasSsyr2_v2_64:=GetProcAddress(hlib,'cublasSsyr2_v2_64');
	cublasSsyr2_64 := cublasSsyr2_v2_64;
      cublasDsyr2_v2:=GetProcAddress(hlib,'cublasDsyr2_v2');
	cublasDsyr2 := cublasDsyr2_v2;
      cublasDsyr2_v2_64:=GetProcAddress(hlib,'cublasDsyr2_v2_64');
	cublasDsyr2_64 := cublasDsyr2_v2_64;
      cublasCsyr2_v2:=GetProcAddress(hlib,'cublasCsyr2_v2');
	cublasCsyr2 := cublasCsyr2_v2;
      cublasCsyr2_v2_64:=GetProcAddress(hlib,'cublasCsyr2_v2_64');
	cublasCsyr2_64 := cublasCsyr2_v2_64;
      cublasZsyr2_v2:=GetProcAddress(hlib,'cublasZsyr2_v2');
	cublasZsyr2 := cublasZsyr2_v2;
      cublasZsyr2_v2_64:=GetProcAddress(hlib,'cublasZsyr2_v2_64');
	cublasZsyr2_64 := cublasZsyr2_v2_64;
      cublasCher2_v2:=GetProcAddress(hlib,'cublasCher2_v2');
	cublasCher2 := cublasCher2_v2;
      cublasCher2_v2_64:=GetProcAddress(hlib,'cublasCher2_v2_64');
	cublasCher2_64 := cublasCher2_v2_64;
      cublasZher2_v2:=GetProcAddress(hlib,'cublasZher2_v2');
	cublasZher2 := cublasZher2_v2;
      cublasZher2_v2_64:=GetProcAddress(hlib,'cublasZher2_v2_64');
	cublasZher2_64 := cublasZher2_v2_64;
      cublasSspr2_v2:=GetProcAddress(hlib,'cublasSspr2_v2');
	cublasSspr2 := cublasSspr2_v2;
      cublasSspr2_v2_64:=GetProcAddress(hlib,'cublasSspr2_v2_64');
	cublasSspr2_64 := cublasSspr2_v2_64;
      cublasDspr2_v2:=GetProcAddress(hlib,'cublasDspr2_v2');
	cublasDspr2 := cublasDspr2_v2;
      cublasDspr2_v2_64:=GetProcAddress(hlib,'cublasDspr2_v2_64');
	cublasDspr2_64 := cublasDspr2_v2_64;
      cublasChpr2_v2:=GetProcAddress(hlib,'cublasChpr2_v2');
	cublasChpr2 := cublasChpr2_v2;
      cublasChpr2_v2_64:=GetProcAddress(hlib,'cublasChpr2_v2_64');
	cublasChpr2_64 := cublasChpr2_v2_64;
      cublasZhpr2_v2:=GetProcAddress(hlib,'cublasZhpr2_v2');
	cublasZhpr2 := cublasZhpr2_v2;
      cublasZhpr2_v2_64:=GetProcAddress(hlib,'cublasZhpr2_v2_64');
	cublasZhpr2_64 := cublasZhpr2_v2_64;
      cublasSgemm_v2:=GetProcAddress(hlib,'cublasSgemm_v2');
	cublasSgemm := cublasSgemm_v2;
      cublasSgemm_v2_64:=GetProcAddress(hlib,'cublasSgemm_v2_64');
	cublasSgemm_64 := cublasSgemm_v2_64;
      cublasDgemm_v2:=GetProcAddress(hlib,'cublasDgemm_v2');
	cublasDgemm := cublasDgemm_v2;
      cublasDgemm_v2_64:=GetProcAddress(hlib,'cublasDgemm_v2_64');
	cublasDgemm_64 := cublasDgemm_v2_64;
      cublasCgemm_v2:=GetProcAddress(hlib,'cublasCgemm_v2');
	cublasCgemm := cublasCgemm_v2;
      cublasCgemm_v2_64:=GetProcAddress(hlib,'cublasCgemm_v2_64');
	cublasCgemm_64 := cublasCgemm_v2_64;
      cublasZgemm_v2:=GetProcAddress(hlib,'cublasZgemm_v2');
	cublasZgemm := cublasZgemm_v2;
      cublasZgemm_v2_64:=GetProcAddress(hlib,'cublasZgemm_v2_64');
	cublasZgemm_64 := cublasZgemm_v2_64;
      cublasSsyrk_v2:=GetProcAddress(hlib,'cublasSsyrk_v2');
	cublasSsyrk := cublasSsyrk_v2;
      cublasSsyrk_v2_64:=GetProcAddress(hlib,'cublasSsyrk_v2_64');
	cublasSsyrk_64 := cublasSsyrk_v2_64;
      cublasDsyrk_v2:=GetProcAddress(hlib,'cublasDsyrk_v2');
	cublasDsyrk := cublasDsyrk_v2;
      cublasDsyrk_v2_64:=GetProcAddress(hlib,'cublasDsyrk_v2_64');
	cublasDsyrk_64 := cublasDsyrk_v2_64;
      cublasCsyrk_v2:=GetProcAddress(hlib,'cublasCsyrk_v2');
	cublasCsyrk := cublasCsyrk_v2;
      cublasCsyrk_v2_64:=GetProcAddress(hlib,'cublasCsyrk_v2_64');
	cublasCsyrk_64 := cublasCsyrk_v2_64;
      cublasZsyrk_v2:=GetProcAddress(hlib,'cublasZsyrk_v2');
	cublasZsyrk := cublasZsyrk_v2;
      cublasZsyrk_v2_64:=GetProcAddress(hlib,'cublasZsyrk_v2_64');
	cublasZsyrk_64 := cublasZsyrk_v2_64;
      cublasCherk_v2:=GetProcAddress(hlib,'cublasCherk_v2');
	cublasCherk := cublasCherk_v2;
      cublasCherk_v2_64:=GetProcAddress(hlib,'cublasCherk_v2_64');
	cublasCherk_64 := cublasCherk_v2_64;
      cublasZherk_v2:=GetProcAddress(hlib,'cublasZherk_v2');
	cublasZherk := cublasZherk_v2;
      cublasZherk_v2_64:=GetProcAddress(hlib,'cublasZherk_v2_64');
	cublasZherk_64 := cublasZherk_v2_64;
      cublasSsyr2k_v2:=GetProcAddress(hlib,'cublasSsyr2k_v2');
	cublasSsyr2k := cublasSsyr2k_v2;
      cublasSsyr2k_v2_64:=GetProcAddress(hlib,'cublasSsyr2k_v2_64');
	cublasSsyr2k_64 := cublasSsyr2k_v2_64;
      cublasDsyr2k_v2:=GetProcAddress(hlib,'cublasDsyr2k_v2');
	cublasDsyr2k := cublasDsyr2k_v2;
      cublasDsyr2k_v2_64:=GetProcAddress(hlib,'cublasDsyr2k_v2_64');
	cublasDsyr2k_64 := cublasDsyr2k_v2_64;
      cublasCsyr2k_v2:=GetProcAddress(hlib,'cublasCsyr2k_v2');
	cublasCsyr2k := cublasCsyr2k_v2;
      cublasCsyr2k_v2_64:=GetProcAddress(hlib,'cublasCsyr2k_v2_64');
	cublasCsyr2k_64 := cublasCsyr2k_v2_64;
      cublasZsyr2k_v2:=GetProcAddress(hlib,'cublasZsyr2k_v2');
	cublasZsyr2k := cublasZsyr2k_v2;
      cublasZsyr2k_v2_64:=GetProcAddress(hlib,'cublasZsyr2k_v2_64');
	cublasZsyr2k_64 := cublasZsyr2k_v2_64;
      cublasCher2k_v2:=GetProcAddress(hlib,'cublasCher2k_v2');
	cublasCher2k := cublasCher2k_v2;
      cublasCher2k_v2_64:=GetProcAddress(hlib,'cublasCher2k_v2_64');
	cublasCher2k_64 := cublasCher2k_v2_64;
      cublasZher2k_v2:=GetProcAddress(hlib,'cublasZher2k_v2');
	cublasZher2k := cublasZher2k_v2;
      cublasZher2k_v2_64:=GetProcAddress(hlib,'cublasZher2k_v2_64');
	cublasZher2k_64 := cublasZher2k_v2_64;
      cublasSsymm_v2:=GetProcAddress(hlib,'cublasSsymm_v2');
	cublasSsymm := cublasSsymm_v2;
      cublasSsymm_v2_64:=GetProcAddress(hlib,'cublasSsymm_v2_64');
	cublasSsymm_64 := cublasSsymm_v2_64;
      cublasDsymm_v2:=GetProcAddress(hlib,'cublasDsymm_v2');
	cublasDsymm := cublasDsymm_v2;
      cublasDsymm_v2_64:=GetProcAddress(hlib,'cublasDsymm_v2_64');
	cublasDsymm_64 := cublasDsymm_v2_64;
      cublasCsymm_v2:=GetProcAddress(hlib,'cublasCsymm_v2');
	cublasCsymm := cublasCsymm_v2;
      cublasCsymm_v2_64:=GetProcAddress(hlib,'cublasCsymm_v2_64');
	cublasCsymm_64 := cublasCsymm_v2_64;
      cublasZsymm_v2:=GetProcAddress(hlib,'cublasZsymm_v2');
	cublasZsymm := cublasZsymm_v2;
      cublasZsymm_v2_64:=GetProcAddress(hlib,'cublasZsymm_v2_64');
	cublasZsymm_64 := cublasZsymm_v2_64;
      cublasChemm_v2:=GetProcAddress(hlib,'cublasChemm_v2');
	cublasChemm := cublasChemm_v2;
      cublasChemm_v2_64:=GetProcAddress(hlib,'cublasChemm_v2_64');
	cublasChemm_64 := cublasChemm_v2_64;
      cublasZhemm_v2:=GetProcAddress(hlib,'cublasZhemm_v2');
	cublasZhemm := cublasZhemm_v2;
      cublasZhemm_v2_64:=GetProcAddress(hlib,'cublasZhemm_v2_64');
	cublasZhemm_64 := cublasZhemm_v2_64;
      cublasStrsm_v2:=GetProcAddress(hlib,'cublasStrsm_v2');
	cublasStrsm := cublasStrsm_v2;
      cublasStrsm_v2_64:=GetProcAddress(hlib,'cublasStrsm_v2_64');
	cublasStrsm_64 := cublasStrsm_v2_64;
      cublasDtrsm_v2:=GetProcAddress(hlib,'cublasDtrsm_v2');
	cublasDtrsm := cublasDtrsm_v2;
      cublasDtrsm_v2_64:=GetProcAddress(hlib,'cublasDtrsm_v2_64');
	cublasDtrsm_64 := cublasDtrsm_v2_64;
      cublasCtrsm_v2:=GetProcAddress(hlib,'cublasCtrsm_v2');
	cublasCtrsm := cublasCtrsm_v2;
      cublasCtrsm_v2_64:=GetProcAddress(hlib,'cublasCtrsm_v2_64');
	cublasCtrsm_64 := cublasCtrsm_v2_64;
      cublasZtrsm_v2:=GetProcAddress(hlib,'cublasZtrsm_v2');
	cublasZtrsm := cublasZtrsm_v2;
      cublasZtrsm_v2_64:=GetProcAddress(hlib,'cublasZtrsm_v2_64');
	cublasZtrsm_64 := cublasZtrsm_v2_64;
      cublasStrmm_v2:=GetProcAddress(hlib,'cublasStrmm_v2');
	cublasStrmm := cublasStrmm_v2;
      cublasStrmm_v2_64:=GetProcAddress(hlib,'cublasStrmm_v2_64');
	cublasStrmm_64 := cublasStrmm_v2_64;
      cublasDtrmm_v2:=GetProcAddress(hlib,'cublasDtrmm_v2');
	cublasDtrmm := cublasDtrmm_v2;
      cublasDtrmm_v2_64:=GetProcAddress(hlib,'cublasDtrmm_v2_64');
	cublasDtrmm_64 := cublasDtrmm_v2_64;
      cublasCtrmm_v2:=GetProcAddress(hlib,'cublasCtrmm_v2');
	cublasCtrmm := cublasCtrmm_v2;
      cublasCtrmm_v2_64:=GetProcAddress(hlib,'cublasCtrmm_v2_64');
	cublasCtrmm_64 := cublasCtrmm_v2_64;
      cublasZtrmm_v2:=GetProcAddress(hlib,'cublasZtrmm_v2');
	cublasZtrmm := cublasZtrmm_v2;
      cublasZtrmm_v2_64:=GetProcAddress(hlib,'cublasZtrmm_v2_64');
	cublasZtrmm_64 := cublasZtrmm_v2_64;
      {$else}
      cublasCreate:=GetProcAddress(hlib,'cublasCreate');
      cublasDestroy:=GetProcAddress(hlib,'cublasDestroy');
      cublasGetVersion:=GetProcAddress(hlib,'cublasGetVersion');
      cublasSetWorkspace:=GetProcAddress(hlib,'cublasSetWorkspace');
      cublasSetStream:=GetProcAddress(hlib,'cublasSetStream');
      cublasGetStream:=GetProcAddress(hlib,'cublasGetStream');
      cublasGetPointerMode:=GetProcAddress(hlib,'cublasGetPointerMode');
      cublasSetPointerMode:=GetProcAddress(hlib,'cublasSetPointerMode');
      cublasSnrm2:=GetProcAddress(hlib,'cublasSnrm2');
      cublasSnrm2_64:=GetProcAddress(hlib,'cublasSnrm2_64');
      cublasDnrm2:=GetProcAddress(hlib,'cublasDnrm2');
      cublasDnrm2_64:=GetProcAddress(hlib,'cublasDnrm2_64');
      cublasScnrm2:=GetProcAddress(hlib,'cublasScnrm2');
      cublasScnrm2_64:=GetProcAddress(hlib,'cublasScnrm2_64');
      cublasDznrm2:=GetProcAddress(hlib,'cublasDznrm2');
      cublasDznrm2_64:=GetProcAddress(hlib,'cublasDznrm2_64');
      cublasSdot:=GetProcAddress(hlib,'cublasSdot');
      cublasSdot_64:=GetProcAddress(hlib,'cublasSdot_64');
      cublasDdot:=GetProcAddress(hlib,'cublasDdot');
      cublasDdot_64:=GetProcAddress(hlib,'cublasDdot_64');
      cublasCdotu:=GetProcAddress(hlib,'cublasCdotu');
      cublasCdotu_64:=GetProcAddress(hlib,'cublasCdotu_64');
      cublasCdotc:=GetProcAddress(hlib,'cublasCdotc');
      cublasCdotc_64:=GetProcAddress(hlib,'cublasCdotc_64');
      cublasZdotu:=GetProcAddress(hlib,'cublasZdotu');
      cublasZdotu_64:=GetProcAddress(hlib,'cublasZdotu_64');
      cublasZdotc:=GetProcAddress(hlib,'cublasZdotc');
      cublasZdotc_64:=GetProcAddress(hlib,'cublasZdotc_64');
      cublasSscal:=GetProcAddress(hlib,'cublasSscal');
      cublasSscal_64:=GetProcAddress(hlib,'cublasSscal_64');
      cublasDscal:=GetProcAddress(hlib,'cublasDscal');
      cublasDscal_64:=GetProcAddress(hlib,'cublasDscal_64');
      cublasCscal:=GetProcAddress(hlib,'cublasCscal');
      cublasCscal_64:=GetProcAddress(hlib,'cublasCscal_64');
      cublasCsscal:=GetProcAddress(hlib,'cublasCsscal');
      cublasCsscal_64:=GetProcAddress(hlib,'cublasCsscal_64');
      cublasZscal:=GetProcAddress(hlib,'cublasZscal');
      cublasZscal_64:=GetProcAddress(hlib,'cublasZscal_64');
      cublasZdscal:=GetProcAddress(hlib,'cublasZdscal');
      cublasZdscal_64:=GetProcAddress(hlib,'cublasZdscal_64');
      cublasSaxpy:=GetProcAddress(hlib,'cublasSaxpy');
      cublasSaxpy_64:=GetProcAddress(hlib,'cublasSaxpy_64');
      cublasDaxpy:=GetProcAddress(hlib,'cublasDaxpy');
      cublasDaxpy_64:=GetProcAddress(hlib,'cublasDaxpy_64');
      cublasCaxpy:=GetProcAddress(hlib,'cublasCaxpy');
      cublasCaxpy_64:=GetProcAddress(hlib,'cublasCaxpy_64');
      cublasZaxpy:=GetProcAddress(hlib,'cublasZaxpy');
      cublasZaxpy_64:=GetProcAddress(hlib,'cublasZaxpy_64');
      cublasScopy:=GetProcAddress(hlib,'cublasScopy');
      cublasScopy_64:=GetProcAddress(hlib,'cublasScopy_64');
      cublasDcopy:=GetProcAddress(hlib,'cublasDcopy');
      cublasDcopy_64:=GetProcAddress(hlib,'cublasDcopy_64');
      cublasCcopy:=GetProcAddress(hlib,'cublasCcopy');
      cublasCcopy_64:=GetProcAddress(hlib,'cublasCcopy_64');
      cublasZcopy:=GetProcAddress(hlib,'cublasZcopy');
      cublasZcopy_64:=GetProcAddress(hlib,'cublasZcopy_64');
      cublasSswap:=GetProcAddress(hlib,'cublasSswap');
      cublasSswap_64:=GetProcAddress(hlib,'cublasSswap_64');
      cublasDswap:=GetProcAddress(hlib,'cublasDswap');
      cublasDswap_64:=GetProcAddress(hlib,'cublasDswap_64');
      cublasCswap:=GetProcAddress(hlib,'cublasCswap');
      cublasCswap_64:=GetProcAddress(hlib,'cublasCswap_64');
      cublasZswap:=GetProcAddress(hlib,'cublasZswap');
      cublasZswap_64:=GetProcAddress(hlib,'cublasZswap_64');
      cublasIsamax:=GetProcAddress(hlib,'cublasIsamax');
      cublasIsamax_64:=GetProcAddress(hlib,'cublasIsamax_64');
      cublasIdamax:=GetProcAddress(hlib,'cublasIdamax');
      cublasIdamax_64:=GetProcAddress(hlib,'cublasIdamax_64');
      cublasIcamax:=GetProcAddress(hlib,'cublasIcamax');
      cublasIcamax_64:=GetProcAddress(hlib,'cublasIcamax_64');
      cublasIzamax:=GetProcAddress(hlib,'cublasIzamax');
      cublasIzamax_64:=GetProcAddress(hlib,'cublasIzamax_64');
      cublasIsamin:=GetProcAddress(hlib,'cublasIsamin');
      cublasIsamin_64:=GetProcAddress(hlib,'cublasIsamin_64');
      cublasIdamin:=GetProcAddress(hlib,'cublasIdamin');
      cublasIdamin_64:=GetProcAddress(hlib,'cublasIdamin_64');
      cublasIcamin:=GetProcAddress(hlib,'cublasIcamin');
      cublasIcamin_64:=GetProcAddress(hlib,'cublasIcamin_64');
      cublasIzamin:=GetProcAddress(hlib,'cublasIzamin');
      cublasIzamin_64:=GetProcAddress(hlib,'cublasIzamin_64');
      cublasSasum:=GetProcAddress(hlib,'cublasSasum');
      cublasSasum_64:=GetProcAddress(hlib,'cublasSasum_64');
      cublasDasum:=GetProcAddress(hlib,'cublasDasum');
      cublasDasum_64:=GetProcAddress(hlib,'cublasDasum_64');
      cublasScasum:=GetProcAddress(hlib,'cublasScasum');
      cublasScasum_64:=GetProcAddress(hlib,'cublasScasum_64');
      cublasDzasum:=GetProcAddress(hlib,'cublasDzasum');
      cublasDzasum_64:=GetProcAddress(hlib,'cublasDzasum_64');
      cublasSrot:=GetProcAddress(hlib,'cublasSrot');
      cublasSrot_64:=GetProcAddress(hlib,'cublasSrot_64');
      cublasDrot:=GetProcAddress(hlib,'cublasDrot');
      cublasDrot_64:=GetProcAddress(hlib,'cublasDrot_64');
      cublasCrot:=GetProcAddress(hlib,'cublasCrot');
      cublasCrot_64:=GetProcAddress(hlib,'cublasCrot_64');
      cublasCsrot:=GetProcAddress(hlib,'cublasCsrot');
      cublasCsrot_64:=GetProcAddress(hlib,'cublasCsrot_64');
      cublasZrot:=GetProcAddress(hlib,'cublasZrot');
      cublasZrot_64:=GetProcAddress(hlib,'cublasZrot_64');
      cublasZdrot:=GetProcAddress(hlib,'cublasZdrot');
      cublasZdrot_64:=GetProcAddress(hlib,'cublasZdrot_64');
      cublasSrotg:=GetProcAddress(hlib,'cublasSrotg');
      cublasDrotg:=GetProcAddress(hlib,'cublasDrotg');
      cublasCrotg:=GetProcAddress(hlib,'cublasCrotg');
      cublasZrotg:=GetProcAddress(hlib,'cublasZrotg');
      cublasSrotm:=GetProcAddress(hlib,'cublasSrotm');
      cublasSrotm_64:=GetProcAddress(hlib,'cublasSrotm_64');
      cublasDrotm:=GetProcAddress(hlib,'cublasDrotm');
      cublasDrotm_64:=GetProcAddress(hlib,'cublasDrotm_64');
      cublasSrotmg:=GetProcAddress(hlib,'cublasSrotmg');
      cublasDrotmg:=GetProcAddress(hlib,'cublasDrotmg');
      cublasSgemv:=GetProcAddress(hlib,'cublasSgemv');
      cublasSgemv_64:=GetProcAddress(hlib,'cublasSgemv_64');
      cublasDgemv:=GetProcAddress(hlib,'cublasDgemv');
      cublasDgemv_64:=GetProcAddress(hlib,'cublasDgemv_64');
      cublasCgemv:=GetProcAddress(hlib,'cublasCgemv');
      cublasCgemv_64:=GetProcAddress(hlib,'cublasCgemv_64');
      cublasZgemv:=GetProcAddress(hlib,'cublasZgemv');
      cublasZgemv_64:=GetProcAddress(hlib,'cublasZgemv_64');
      cublasSgbmv:=GetProcAddress(hlib,'cublasSgbmv');
      cublasSgbmv_64:=GetProcAddress(hlib,'cublasSgbmv_64');
      cublasDgbmv:=GetProcAddress(hlib,'cublasDgbmv');
      cublasDgbmv_64:=GetProcAddress(hlib,'cublasDgbmv_64');
      cublasCgbmv:=GetProcAddress(hlib,'cublasCgbmv');
      cublasCgbmv_64:=GetProcAddress(hlib,'cublasCgbmv_64');
      cublasZgbmv:=GetProcAddress(hlib,'cublasZgbmv');
      cublasZgbmv_64:=GetProcAddress(hlib,'cublasZgbmv_64');
      cublasStrmv:=GetProcAddress(hlib,'cublasStrmv');
      cublasStrmv_64:=GetProcAddress(hlib,'cublasStrmv_64');
      cublasDtrmv:=GetProcAddress(hlib,'cublasDtrmv');
      cublasDtrmv_64:=GetProcAddress(hlib,'cublasDtrmv_64');
      cublasCtrmv:=GetProcAddress(hlib,'cublasCtrmv');
      cublasCtrmv_64:=GetProcAddress(hlib,'cublasCtrmv_64');
      cublasZtrmv:=GetProcAddress(hlib,'cublasZtrmv');
      cublasZtrmv_64:=GetProcAddress(hlib,'cublasZtrmv_64');
      cublasStbmv:=GetProcAddress(hlib,'cublasStbmv');
      cublasStbmv_64:=GetProcAddress(hlib,'cublasStbmv_64');
      cublasDtbmv:=GetProcAddress(hlib,'cublasDtbmv');
      cublasDtbmv_64:=GetProcAddress(hlib,'cublasDtbmv_64');
      cublasCtbmv:=GetProcAddress(hlib,'cublasCtbmv');
      cublasCtbmv_64:=GetProcAddress(hlib,'cublasCtbmv_64');
      cublasZtbmv:=GetProcAddress(hlib,'cublasZtbmv');
      cublasZtbmv_64:=GetProcAddress(hlib,'cublasZtbmv_64');
      cublasStpmv:=GetProcAddress(hlib,'cublasStpmv');
      cublasStpmv_64:=GetProcAddress(hlib,'cublasStpmv_64');
      cublasDtpmv:=GetProcAddress(hlib,'cublasDtpmv');
      cublasDtpmv_64:=GetProcAddress(hlib,'cublasDtpmv_64');
      cublasCtpmv:=GetProcAddress(hlib,'cublasCtpmv');
      cublasCtpmv_64:=GetProcAddress(hlib,'cublasCtpmv_64');
      cublasZtpmv:=GetProcAddress(hlib,'cublasZtpmv');
      cublasZtpmv_64:=GetProcAddress(hlib,'cublasZtpmv_64');
      cublasStrsv:=GetProcAddress(hlib,'cublasStrsv');
      cublasStrsv_64:=GetProcAddress(hlib,'cublasStrsv_64');
      cublasDtrsv:=GetProcAddress(hlib,'cublasDtrsv');
      cublasDtrsv_64:=GetProcAddress(hlib,'cublasDtrsv_64');
      cublasCtrsv:=GetProcAddress(hlib,'cublasCtrsv');
      cublasCtrsv_64:=GetProcAddress(hlib,'cublasCtrsv_64');
      cublasZtrsv:=GetProcAddress(hlib,'cublasZtrsv');
      cublasZtrsv_64:=GetProcAddress(hlib,'cublasZtrsv_64');
      cublasStpsv:=GetProcAddress(hlib,'cublasStpsv');
      cublasStpsv_64:=GetProcAddress(hlib,'cublasStpsv_64');
      cublasDtpsv:=GetProcAddress(hlib,'cublasDtpsv');
      cublasDtpsv_64:=GetProcAddress(hlib,'cublasDtpsv_64');
      cublasCtpsv:=GetProcAddress(hlib,'cublasCtpsv');
      cublasCtpsv_64:=GetProcAddress(hlib,'cublasCtpsv_64');
      cublasZtpsv:=GetProcAddress(hlib,'cublasZtpsv');
      cublasZtpsv_64:=GetProcAddress(hlib,'cublasZtpsv_64');
      cublasStbsv:=GetProcAddress(hlib,'cublasStbsv');
      cublasStbsv_64:=GetProcAddress(hlib,'cublasStbsv_64');
      cublasDtbsv:=GetProcAddress(hlib,'cublasDtbsv');
      cublasDtbsv_64:=GetProcAddress(hlib,'cublasDtbsv_64');
      cublasCtbsv:=GetProcAddress(hlib,'cublasCtbsv');
      cublasCtbsv_64:=GetProcAddress(hlib,'cublasCtbsv_64');
      cublasZtbsv:=GetProcAddress(hlib,'cublasZtbsv');
      cublasZtbsv_64:=GetProcAddress(hlib,'cublasZtbsv_64');
      cublasSsymv:=GetProcAddress(hlib,'cublasSsymv');
      cublasSsymv_64:=GetProcAddress(hlib,'cublasSsymv_64');
      cublasDsymv:=GetProcAddress(hlib,'cublasDsymv');
      cublasDsymv_64:=GetProcAddress(hlib,'cublasDsymv_64');
      cublasCsymv:=GetProcAddress(hlib,'cublasCsymv');
      cublasCsymv_64:=GetProcAddress(hlib,'cublasCsymv_64');
      cublasZsymv:=GetProcAddress(hlib,'cublasZsymv');
      cublasZsymv_64:=GetProcAddress(hlib,'cublasZsymv_64');
      cublasChemv:=GetProcAddress(hlib,'cublasChemv');
      cublasChemv_64:=GetProcAddress(hlib,'cublasChemv_64');
      cublasZhemv:=GetProcAddress(hlib,'cublasZhemv');
      cublasZhemv_64:=GetProcAddress(hlib,'cublasZhemv_64');
      cublasSsbmv:=GetProcAddress(hlib,'cublasSsbmv');
      cublasSsbmv_64:=GetProcAddress(hlib,'cublasSsbmv_64');
      cublasDsbmv:=GetProcAddress(hlib,'cublasDsbmv');
      cublasDsbmv_64:=GetProcAddress(hlib,'cublasDsbmv_64');
      cublasChbmv:=GetProcAddress(hlib,'cublasChbmv');
      cublasChbmv_64:=GetProcAddress(hlib,'cublasChbmv_64');
      cublasZhbmv:=GetProcAddress(hlib,'cublasZhbmv');
      cublasZhbmv_64:=GetProcAddress(hlib,'cublasZhbmv_64');
      cublasSspmv:=GetProcAddress(hlib,'cublasSspmv');
      cublasSspmv_64:=GetProcAddress(hlib,'cublasSspmv_64');
      cublasDspmv:=GetProcAddress(hlib,'cublasDspmv');
      cublasDspmv_64:=GetProcAddress(hlib,'cublasDspmv_64');
      cublasChpmv:=GetProcAddress(hlib,'cublasChpmv');
      cublasChpmv_64:=GetProcAddress(hlib,'cublasChpmv_64');
      cublasZhpmv:=GetProcAddress(hlib,'cublasZhpmv');
      cublasZhpmv_64:=GetProcAddress(hlib,'cublasZhpmv_64');
      cublasSger:=GetProcAddress(hlib,'cublasSger');
      cublasSger_64:=GetProcAddress(hlib,'cublasSger_64');
      cublasDger:=GetProcAddress(hlib,'cublasDger');
      cublasDger_64:=GetProcAddress(hlib,'cublasDger_64');
      cublasCgeru:=GetProcAddress(hlib,'cublasCgeru');
      cublasCgeru_64:=GetProcAddress(hlib,'cublasCgeru_64');
      cublasCgerc:=GetProcAddress(hlib,'cublasCgerc');
      cublasCgerc_64:=GetProcAddress(hlib,'cublasCgerc_64');
      cublasZgeru:=GetProcAddress(hlib,'cublasZgeru');
      cublasZgeru_64:=GetProcAddress(hlib,'cublasZgeru_64');
      cublasZgerc:=GetProcAddress(hlib,'cublasZgerc');
      cublasZgerc_64:=GetProcAddress(hlib,'cublasZgerc_64');
      cublasSsyr:=GetProcAddress(hlib,'cublasSsyr');
      cublasSsyr_64:=GetProcAddress(hlib,'cublasSsyr_64');
      cublasDsyr:=GetProcAddress(hlib,'cublasDsyr');
      cublasDsyr_64:=GetProcAddress(hlib,'cublasDsyr_64');
      cublasCsyr:=GetProcAddress(hlib,'cublasCsyr');
      cublasCsyr_64:=GetProcAddress(hlib,'cublasCsyr_64');
      cublasZsyr:=GetProcAddress(hlib,'cublasZsyr');
      cublasZsyr_64:=GetProcAddress(hlib,'cublasZsyr_64');
      cublasCher:=GetProcAddress(hlib,'cublasCher');
      cublasCher_64:=GetProcAddress(hlib,'cublasCher_64');
      cublasZher:=GetProcAddress(hlib,'cublasZher');
      cublasZher_64:=GetProcAddress(hlib,'cublasZher_64');
      cublasSspr:=GetProcAddress(hlib,'cublasSspr');
      cublasSspr_64:=GetProcAddress(hlib,'cublasSspr_64');
      cublasDspr:=GetProcAddress(hlib,'cublasDspr');
      cublasDspr_64:=GetProcAddress(hlib,'cublasDspr_64');
      cublasChpr:=GetProcAddress(hlib,'cublasChpr');
      cublasChpr_64:=GetProcAddress(hlib,'cublasChpr_64');
      cublasZhpr:=GetProcAddress(hlib,'cublasZhpr');
      cublasZhpr_64:=GetProcAddress(hlib,'cublasZhpr_64');
      cublasSsyr2:=GetProcAddress(hlib,'cublasSsyr2');
      cublasSsyr2_64:=GetProcAddress(hlib,'cublasSsyr2_64');
      cublasDsyr2:=GetProcAddress(hlib,'cublasDsyr2');
      cublasDsyr2_64:=GetProcAddress(hlib,'cublasDsyr2_64');
      cublasCsyr2:=GetProcAddress(hlib,'cublasCsyr2');
      cublasCsyr2_64:=GetProcAddress(hlib,'cublasCsyr2_64');
      cublasZsyr2:=GetProcAddress(hlib,'cublasZsyr2');
      cublasZsyr2_64:=GetProcAddress(hlib,'cublasZsyr2_64');
      cublasCher2:=GetProcAddress(hlib,'cublasCher2');
      cublasCher2_64:=GetProcAddress(hlib,'cublasCher2_64');
      cublasZher2:=GetProcAddress(hlib,'cublasZher2');
      cublasZher2_64:=GetProcAddress(hlib,'cublasZher2_64');
      cublasSspr2:=GetProcAddress(hlib,'cublasSspr2');
      cublasSspr2_64:=GetProcAddress(hlib,'cublasSspr2_64');
      cublasDspr2:=GetProcAddress(hlib,'cublasDspr2');
      cublasDspr2_64:=GetProcAddress(hlib,'cublasDspr2_64');
      cublasChpr2:=GetProcAddress(hlib,'cublasChpr2');
      cublasChpr2_64:=GetProcAddress(hlib,'cublasChpr2_64');
      cublasZhpr2:=GetProcAddress(hlib,'cublasZhpr2');
      cublasZhpr2_64:=GetProcAddress(hlib,'cublasZhpr2_64');
      cublasSgemm:=GetProcAddress(hlib,'cublasSgemm');
      cublasSgemm_64:=GetProcAddress(hlib,'cublasSgemm_64');
      cublasDgemm:=GetProcAddress(hlib,'cublasDgemm');
      cublasDgemm_64:=GetProcAddress(hlib,'cublasDgemm_64');
      cublasCgemm:=GetProcAddress(hlib,'cublasCgemm');
      cublasCgemm_64:=GetProcAddress(hlib,'cublasCgemm_64');
      cublasZgemm:=GetProcAddress(hlib,'cublasZgemm');
      cublasZgemm_64:=GetProcAddress(hlib,'cublasZgemm_64');
      cublasSsyrk:=GetProcAddress(hlib,'cublasSsyrk');
      cublasSsyrk_64:=GetProcAddress(hlib,'cublasSsyrk_64');
      cublasDsyrk:=GetProcAddress(hlib,'cublasDsyrk');
      cublasDsyrk_64:=GetProcAddress(hlib,'cublasDsyrk_64');
      cublasCsyrk:=GetProcAddress(hlib,'cublasCsyrk');
      cublasCsyrk_64:=GetProcAddress(hlib,'cublasCsyrk_64');
      cublasZsyrk:=GetProcAddress(hlib,'cublasZsyrk');
      cublasZsyrk_64:=GetProcAddress(hlib,'cublasZsyrk_64');
      cublasCherk:=GetProcAddress(hlib,'cublasCherk');
      cublasCherk_64:=GetProcAddress(hlib,'cublasCherk_64');
      cublasZherk:=GetProcAddress(hlib,'cublasZherk');
      cublasZherk_64:=GetProcAddress(hlib,'cublasZherk_64');
      cublasSsyr2k:=GetProcAddress(hlib,'cublasSsyr2k');
      cublasSsyr2k_64:=GetProcAddress(hlib,'cublasSsyr2k_64');
      cublasDsyr2k:=GetProcAddress(hlib,'cublasDsyr2k');
      cublasDsyr2k_64:=GetProcAddress(hlib,'cublasDsyr2k_64');
      cublasCsyr2k:=GetProcAddress(hlib,'cublasCsyr2k');
      cublasCsyr2k_64:=GetProcAddress(hlib,'cublasCsyr2k_64');
      cublasZsyr2k:=GetProcAddress(hlib,'cublasZsyr2k');
      cublasZsyr2k_64:=GetProcAddress(hlib,'cublasZsyr2k_64');
      cublasCher2k:=GetProcAddress(hlib,'cublasCher2k');
      cublasCher2k_64:=GetProcAddress(hlib,'cublasCher2k_64');
      cublasZher2k:=GetProcAddress(hlib,'cublasZher2k');
      cublasZher2k_64:=GetProcAddress(hlib,'cublasZher2k_64');
      cublasSsymm:=GetProcAddress(hlib,'cublasSsymm');
      cublasSsymm_64:=GetProcAddress(hlib,'cublasSsymm_64');
      cublasDsymm:=GetProcAddress(hlib,'cublasDsymm');
      cublasDsymm_64:=GetProcAddress(hlib,'cublasDsymm_64');
      cublasCsymm:=GetProcAddress(hlib,'cublasCsymm');
      cublasCsymm_64:=GetProcAddress(hlib,'cublasCsymm_64');
      cublasZsymm:=GetProcAddress(hlib,'cublasZsymm');
      cublasZsymm_64:=GetProcAddress(hlib,'cublasZsymm_64');
      cublasChemm:=GetProcAddress(hlib,'cublasChemm');
      cublasChemm_64:=GetProcAddress(hlib,'cublasChemm_64');
      cublasZhemm:=GetProcAddress(hlib,'cublasZhemm');
      cublasZhemm_64:=GetProcAddress(hlib,'cublasZhemm_64');
      cublasStrsm:=GetProcAddress(hlib,'cublasStrsm');
      cublasStrsm_64:=GetProcAddress(hlib,'cublasStrsm_64');
      cublasDtrsm:=GetProcAddress(hlib,'cublasDtrsm');
      cublasDtrsm_64:=GetProcAddress(hlib,'cublasDtrsm_64');
      cublasCtrsm:=GetProcAddress(hlib,'cublasCtrsm');
      cublasCtrsm_64:=GetProcAddress(hlib,'cublasCtrsm_64');
      cublasZtrsm:=GetProcAddress(hlib,'cublasZtrsm');
      cublasZtrsm_64:=GetProcAddress(hlib,'cublasZtrsm_64');
      cublasStrmm:=GetProcAddress(hlib,'cublasStrmm');
      cublasStrmm_64:=GetProcAddress(hlib,'cublasStrmm_64');
      cublasDtrmm:=GetProcAddress(hlib,'cublasDtrmm');
      cublasDtrmm_64:=GetProcAddress(hlib,'cublasDtrmm_64');
      cublasCtrmm:=GetProcAddress(hlib,'cublasCtrmm');
      cublasCtrmm_64:=GetProcAddress(hlib,'cublasCtrmm_64');
      cublasZtrmm:=GetProcAddress(hlib,'cublasZtrmm');
      cublasZtrmm_64:=GetProcAddress(hlib,'cublasZtrmm_64');
      {$endif}
    end;


initialization
  Loadcublas_api(libcublas);
finalization
  Freecublas_api;

end.
