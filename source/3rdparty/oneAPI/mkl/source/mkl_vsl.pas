{$IFDEF FPC}
{$mode delphi}
{$PACKRECORDS C}
{$ENDIF}
unit mkl_vsl;

interface

uses mkl_types;


{$i mkl.inc}


{ file: mkl_vsl_types.h  }
{******************************************************************************
* Copyright 2006-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
****************************************************************************** }
  const
    VSL_STATUS_OK = 0;    
    VSL_ERROR_OK = 0;    
  {
  // Common errors (-1..-999)
   }
    VSL_ERROR_FEATURE_NOT_IMPLEMENTED = -(1);    
    VSL_ERROR_UNKNOWN = -(2);    
    VSL_ERROR_BADARGS = -(3);    
    VSL_ERROR_MEM_FAILURE = -(4);    
    VSL_ERROR_NULL_PTR = -(5);    
    VSL_ERROR_CPU_NOT_SUPPORTED = -(6);    
  {
  // RNG errors (-1000..-1999)
   }
  { brng errors  }
    VSL_RNG_ERROR_INVALID_BRNG_INDEX = -(1000);    
    VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED = -(1002);    
    VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED = -(1003);    
    VSL_RNG_ERROR_SKIPAHEADEX_UNSUPPORTED = -(1004);    
    VSL_RNG_ERROR_BRNGS_INCOMPATIBLE = -(1005);    
    VSL_RNG_ERROR_BAD_STREAM = -(1006);    
    VSL_RNG_ERROR_BRNG_TABLE_FULL = -(1007);    
    VSL_RNG_ERROR_BAD_STREAM_STATE_SIZE = -(1008);    
    VSL_RNG_ERROR_BAD_WORD_SIZE = -(1009);    
    VSL_RNG_ERROR_BAD_NSEEDS = -(1010);    
    VSL_RNG_ERROR_BAD_NBITS = -(1011);    
    VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED = -(1012);    
    VSL_RNG_ERROR_LEAPFROG_NSTREAMS_TOO_BIG = -(1013);    
    VSL_RNG_ERROR_BRNG_NOT_SUPPORTED = -(1014);    
  { abstract stream related errors  }
    VSL_RNG_ERROR_BAD_UPDATE = -(1120);    
    VSL_RNG_ERROR_NO_NUMBERS = -(1121);    
    VSL_RNG_ERROR_INVALID_ABSTRACT_STREAM = -(1122);    
  { non deterministic stream related errors  }
    VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED = -(1130);    
    VSL_RNG_ERROR_NONDETERM_NRETRIES_EXCEEDED = -(1131);    
  { ARS5 stream related errors  }
    VSL_RNG_ERROR_ARS5_NOT_SUPPORTED = -(1140);    
  { Multinomial distribution probability array related errors  }
    VSL_DISTR_MULTINOMIAL_BAD_PROBABILITY_ARRAY = -(1150);    
  { read/write stream to file errors  }
    VSL_RNG_ERROR_FILE_CLOSE = -(1100);    
    VSL_RNG_ERROR_FILE_OPEN = -(1101);    
    VSL_RNG_ERROR_FILE_WRITE = -(1102);    
    VSL_RNG_ERROR_FILE_READ = -(1103);    
    VSL_RNG_ERROR_BAD_FILE_FORMAT = -(1110);    
    VSL_RNG_ERROR_UNSUPPORTED_FILE_VER = -(1111);    
    VSL_RNG_ERROR_BAD_MEM_FORMAT = -(1200);    
  { Convolution/correlation errors  }
    VSL_CC_ERROR_NOT_IMPLEMENTED = -(2000);    
    VSL_CC_ERROR_ALLOCATION_FAILURE = -(2001);    
    VSL_CC_ERROR_BAD_DESCRIPTOR = -(2200);    
    VSL_CC_ERROR_SERVICE_FAILURE = -(2210);    
    VSL_CC_ERROR_EDIT_FAILURE = -(2211);    
    VSL_CC_ERROR_EDIT_PROHIBITED = -(2212);    
    VSL_CC_ERROR_COMMIT_FAILURE = -(2220);    
    VSL_CC_ERROR_COPY_FAILURE = -(2230);    
    VSL_CC_ERROR_DELETE_FAILURE = -(2240);    
    VSL_CC_ERROR_BAD_ARGUMENT = -(2300);    
    VSL_CC_ERROR_DIMS = -(2301);    
    VSL_CC_ERROR_START = -(2302);    
    VSL_CC_ERROR_DECIMATION = -(2303);    
    VSL_CC_ERROR_XSHAPE = -(2311);    
    VSL_CC_ERROR_YSHAPE = -(2312);    
    VSL_CC_ERROR_ZSHAPE = -(2313);    
    VSL_CC_ERROR_XSTRIDE = -(2321);    
    VSL_CC_ERROR_YSTRIDE = -(2322);    
    VSL_CC_ERROR_ZSTRIDE = -(2323);    
    VSL_CC_ERROR_X = -(2331);    
    VSL_CC_ERROR_Y = -(2332);    
    VSL_CC_ERROR_Z = -(2333);    
    VSL_CC_ERROR_JOB = -(2100);    
    VSL_CC_ERROR_KIND = -(2110);    
    VSL_CC_ERROR_MODE = -(2120);    
    VSL_CC_ERROR_TYPE = -(2130);    
    VSL_CC_ERROR_PRECISION = -(2140);    
    VSL_CC_ERROR_EXTERNAL_PRECISION = -(2141);    
    VSL_CC_ERROR_INTERNAL_PRECISION = -(2142);    
    VSL_CC_ERROR_METHOD = -(2400);    
    VSL_CC_ERROR_OTHER = -(2800);    
  {
  //++
  // SUMMARY STATTISTICS ERROR/WARNING CODES
  //--
   }
  {
  // Warnings
   }
    VSL_SS_NOT_FULL_RANK_MATRIX = 4028;    
    VSL_SS_SEMIDEFINITE_COR = 4029;    
  {
  // Errors (-4000..-4999)
   }
    VSL_SS_ERROR_ALLOCATION_FAILURE = -(4000);    
    VSL_SS_ERROR_BAD_DIMEN = -(4001);    
    VSL_SS_ERROR_BAD_OBSERV_N = -(4002);    
    VSL_SS_ERROR_STORAGE_NOT_SUPPORTED = -(4003);    
    VSL_SS_ERROR_BAD_INDC_ADDR = -(4004);    
    VSL_SS_ERROR_BAD_WEIGHTS = -(4005);    
    VSL_SS_ERROR_BAD_MEAN_ADDR = -(4006);    
    VSL_SS_ERROR_BAD_2R_MOM_ADDR = -(4007);    
    VSL_SS_ERROR_BAD_3R_MOM_ADDR = -(4008);    
    VSL_SS_ERROR_BAD_4R_MOM_ADDR = -(4009);    
    VSL_SS_ERROR_BAD_2C_MOM_ADDR = -(4010);    
    VSL_SS_ERROR_BAD_3C_MOM_ADDR = -(4011);    
    VSL_SS_ERROR_BAD_4C_MOM_ADDR = -(4012);    
    VSL_SS_ERROR_BAD_KURTOSIS_ADDR = -(4013);    
    VSL_SS_ERROR_BAD_SKEWNESS_ADDR = -(4014);    
    VSL_SS_ERROR_BAD_MIN_ADDR = -(4015);    
    VSL_SS_ERROR_BAD_MAX_ADDR = -(4016);    
    VSL_SS_ERROR_BAD_VARIATION_ADDR = -(4017);    
    VSL_SS_ERROR_BAD_COV_ADDR = -(4018);    
    VSL_SS_ERROR_BAD_COR_ADDR = -(4019);    
    VSL_SS_ERROR_BAD_ACCUM_WEIGHT_ADDR = -(4020);    
    VSL_SS_ERROR_BAD_QUANT_ORDER_ADDR = -(4021);    
    VSL_SS_ERROR_BAD_QUANT_ORDER = -(4022);    
    VSL_SS_ERROR_BAD_QUANT_ADDR = -(4023);    
    VSL_SS_ERROR_BAD_ORDER_STATS_ADDR = -(4024);    
    VSL_SS_ERROR_MOMORDER_NOT_SUPPORTED = -(4025);    
    VSL_SS_ERROR_ALL_OBSERVS_OUTLIERS = -(4026);    
    VSL_SS_ERROR_BAD_ROBUST_COV_ADDR = -(4027);    
    VSL_SS_ERROR_BAD_ROBUST_MEAN_ADDR = -(4028);    
    VSL_SS_ERROR_METHOD_NOT_SUPPORTED = -(4029);    
    VSL_SS_ERROR_BAD_GROUP_INDC_ADDR = -(4030);    
    VSL_SS_ERROR_NULL_TASK_DESCRIPTOR = -(4031);    
    VSL_SS_ERROR_BAD_OBSERV_ADDR = -(4032);    
    VSL_SS_ERROR_SINGULAR_COV = -(4033);    
    VSL_SS_ERROR_BAD_POOLED_COV_ADDR = -(4034);    
    VSL_SS_ERROR_BAD_POOLED_MEAN_ADDR = -(4035);    
    VSL_SS_ERROR_BAD_GROUP_COV_ADDR = -(4036);    
    VSL_SS_ERROR_BAD_GROUP_MEAN_ADDR = -(4037);    
    VSL_SS_ERROR_BAD_GROUP_INDC = -(4038);    
    VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_ADDR = -(4039);    
    VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_N_ADDR = -(4040);    
    VSL_SS_ERROR_BAD_OUTLIERS_WEIGHTS_ADDR = -(4041);    
    VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_ADDR = -(4042);    
    VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_N_ADDR = -(4043);    
    VSL_SS_ERROR_BAD_STORAGE_ADDR = -(4044);    
    VSL_SS_ERROR_BAD_PARTIAL_COV_IDX_ADDR = -(4045);    
    VSL_SS_ERROR_BAD_PARTIAL_COV_ADDR = -(4046);    
    VSL_SS_ERROR_BAD_PARTIAL_COR_ADDR = -(4047);    
    VSL_SS_ERROR_BAD_MI_PARAMS_ADDR = -(4048);    
    VSL_SS_ERROR_BAD_MI_PARAMS_N_ADDR = -(4049);    
    VSL_SS_ERROR_BAD_MI_BAD_PARAMS_N = -(4050);    
    VSL_SS_ERROR_BAD_MI_PARAMS = -(4051);    
    VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_N_ADDR = -(4052);    
    VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_ADDR = -(4053);    
    VSL_SS_ERROR_BAD_MI_SIMUL_VALS_ADDR = -(4054);    
    VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N_ADDR = -(4055);    
    VSL_SS_ERROR_BAD_MI_ESTIMATES_N_ADDR = -(4056);    
    VSL_SS_ERROR_BAD_MI_ESTIMATES_ADDR = -(4057);    
    VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N = -(4058);    
    VSL_SS_ERROR_BAD_MI_ESTIMATES_N = -(4059);    
    VSL_SS_ERROR_BAD_MI_OUTPUT_PARAMS = -(4060);    
    VSL_SS_ERROR_BAD_MI_PRIOR_N_ADDR = -(4061);    
    VSL_SS_ERROR_BAD_MI_PRIOR_ADDR = -(4062);    
    VSL_SS_ERROR_BAD_MI_MISSING_VALS_N = -(4063);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N_ADDR = -(4064);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_ADDR = -(4065);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N = -(4066);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS = -(4067);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER_ADDR = -(4068);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER = -(4069);    
    VSL_SS_ERROR_BAD_STREAM_QUANT_ADDR = -(4070);    
    VSL_SS_ERROR_BAD_PARAMTR_COR_ADDR = -(4071);    
    VSL_SS_ERROR_BAD_COR = -(4072);    
    VSL_SS_ERROR_BAD_PARTIAL_COV_IDX = -(4073);    
    VSL_SS_ERROR_BAD_SUM_ADDR = -(4074);    
    VSL_SS_ERROR_BAD_2R_SUM_ADDR = -(4075);    
    VSL_SS_ERROR_BAD_3R_SUM_ADDR = -(4076);    
    VSL_SS_ERROR_BAD_4R_SUM_ADDR = -(4077);    
    VSL_SS_ERROR_BAD_2C_SUM_ADDR = -(4078);    
    VSL_SS_ERROR_BAD_3C_SUM_ADDR = -(4079);    
    VSL_SS_ERROR_BAD_4C_SUM_ADDR = -(4080);    
    VSL_SS_ERROR_BAD_CP_ADDR = -(4081);    
    VSL_SS_ERROR_BAD_MDAD_ADDR = -(4082);    
    VSL_SS_ERROR_BAD_MNAD_ADDR = -(4083);    
    VSL_SS_ERROR_BAD_SORTED_OBSERV_ADDR = -(4084);    
    VSL_SS_ERROR_INDICES_NOT_SUPPORTED = -(4085);    
  {
  // Internal errors caused by internal routines of the functions
   }
    VSL_SS_ERROR_ROBCOV_INTERN_C1 = -(5000);    
    VSL_SS_ERROR_PARTIALCOV_INTERN_C1 = -(5010);    
    VSL_SS_ERROR_PARTIALCOV_INTERN_C2 = -(5011);    
    VSL_SS_ERROR_MISSINGVALS_INTERN_C1 = -(5021);    
    VSL_SS_ERROR_MISSINGVALS_INTERN_C2 = -(5022);    
    VSL_SS_ERROR_MISSINGVALS_INTERN_C3 = -(5023);    
    VSL_SS_ERROR_MISSINGVALS_INTERN_C4 = -(5024);    
    VSL_SS_ERROR_MISSINGVALS_INTERN_C5 = -(5025);    
    VSL_SS_ERROR_PARAMTRCOR_INTERN_C1 = -(5030);    
    VSL_SS_ERROR_COVRANK_INTERNAL_ERROR_C1 = -(5040);    
    VSL_SS_ERROR_INVCOV_INTERNAL_ERROR_C1 = -(5041);    
    VSL_SS_ERROR_INVCOV_INTERNAL_ERROR_C2 = -(5042);    
  {
  // CONV/CORR RELATED MACRO DEFINITIONS
   }
    VSL_CONV_MODE_AUTO = 0;    
    VSL_CORR_MODE_AUTO = 0;    
    VSL_CONV_MODE_DIRECT = 1;    
    VSL_CORR_MODE_DIRECT = 1;    
    VSL_CONV_MODE_FFT = 2;    
    VSL_CORR_MODE_FFT = 2;    
    VSL_CONV_PRECISION_SINGLE = 1;    
    VSL_CORR_PRECISION_SINGLE = 1;    
    VSL_CONV_PRECISION_DOUBLE = 2;    
    VSL_CORR_PRECISION_DOUBLE = 2;    
  {
  //++
  //  BASIC RANDOM NUMBER GENERATOR (BRNG) RELATED MACRO DEFINITIONS
  //--
   }
  {
  //  MAX NUMBER OF BRNGS CAN BE REGISTERED IN VSL
  //  No more than VSL_MAX_REG_BRNGS basic generators can be registered in VSL
  //  (including predefined basic generators).
  //
  //  Change this number to increase/decrease number of BRNGs can be registered.
   }
    VSL_MAX_REG_BRNGS = 512;    
  {
  //  PREDEFINED BRNG NAMES
   }
    VSL_BRNG_SHIFT = 20;    
    VSL_BRNG_INC = 1 shl VSL_BRNG_SHIFT;    
    VSL_BRNG_MCG31 = VSL_BRNG_INC;    
    VSL_BRNG_R250 = VSL_BRNG_MCG31+VSL_BRNG_INC;    
    VSL_BRNG_MRG32K3A = VSL_BRNG_R250+VSL_BRNG_INC;    
    VSL_BRNG_MCG59 = VSL_BRNG_MRG32K3A+VSL_BRNG_INC;    
    VSL_BRNG_WH = VSL_BRNG_MCG59+VSL_BRNG_INC;    
    VSL_BRNG_SOBOL = VSL_BRNG_WH+VSL_BRNG_INC;    
    VSL_BRNG_NIEDERR = VSL_BRNG_SOBOL+VSL_BRNG_INC;    
    VSL_BRNG_MT19937 = VSL_BRNG_NIEDERR+VSL_BRNG_INC;    
    VSL_BRNG_MT2203 = VSL_BRNG_MT19937+VSL_BRNG_INC;    
    VSL_BRNG_IABSTRACT = VSL_BRNG_MT2203+VSL_BRNG_INC;    
    VSL_BRNG_DABSTRACT = VSL_BRNG_IABSTRACT+VSL_BRNG_INC;    
    VSL_BRNG_SABSTRACT = VSL_BRNG_DABSTRACT+VSL_BRNG_INC;    
    VSL_BRNG_SFMT19937 = VSL_BRNG_SABSTRACT+VSL_BRNG_INC;    
    VSL_BRNG_NONDETERM = VSL_BRNG_SFMT19937+VSL_BRNG_INC;    
    VSL_BRNG_ARS5 = VSL_BRNG_NONDETERM+VSL_BRNG_INC;    
    VSL_BRNG_PHILOX4X32X10 = VSL_BRNG_ARS5+VSL_BRNG_INC;    
  {
  // PREDEFINED PARAMETERS FOR NON-DETERMNINISTIC RANDOM NUMBER
  // GENERATOR
  // The library provides an abstraction to the source of non-deterministic
  // random numbers supported in HW. Current version of the library provides
  // interface to RDRAND-based only, available in latest Intel CPU.
   }
    VSL_BRNG_RDRAND = $0;    
    VSL_BRNG_NONDETERM_NRETRIES = 10;    
  {
  //  LEAPFROG METHOD FOR GRAY-CODE BASED QUASI-RANDOM NUMBER BASIC GENERATORS
  //  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random number
  //  basic generators. In contrast to pseudorandom number basic generators,
  //  quasi-random ones take the dimension as initialization parameter.
  //
  //  Suppose that quasi-random number generator (QRNG) dimension is S. QRNG
  //  sequence is a sequence of S-dimensional vectors:
  //
  //     x0=(x0[0],x0[1],...,x0[S-1]),x1=(x1[0],x1[1],...,x1[S-1]),...
  //
  //  VSL treats the output of any basic generator as 1-dimensional, however:
  //
  //     x0[0],x0[1],...,x0[S-1],x1[0],x1[1],...,x1[S-1],...
  //
  //  Because of nature of VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR QRNGs,
  //  the only S-stride Leapfrog method is supported for them. In other words,
  //  user can generate subsequences, which consist of fixed elements of
  //  vectors x0,x1,... For example, if 0 element is fixed, the following
  //  subsequence is generated:
  //
  //     x0[1],x1[1],x2[1],...
  //
  //  To use the s-stride Leapfrog method with given QRNG, user should call
  //  vslLeapfrogStream function with parameter k equal to element to be fixed
  //  (0<=k<S) and parameter nstreams equal to VSL_QRNG_LEAPFROG_COMPONENTS.
   }
    VSL_QRNG_LEAPFROG_COMPONENTS = $7fffffff;    
  {
  //  USER-DEFINED PARAMETERS FOR QUASI-RANDOM NUMBER BASIC GENERATORS
  //  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random
  //  number basic generators. Default parameters of the generators
  //  support generation of quasi-random number vectors of dimensions
  //  S<=40 for SOBOL and S<=318 for NIEDERRITER. The library provides
  //  opportunity to register user-defined initial values for the
  //  generators and generate quasi-random vectors of desirable dimension.
  //  There is also opportunity to register user-defined parameters for
  //  default dimensions and obtain another sequence of quasi-random vectors.
  //  Service function vslNewStreamEx is used to pass the parameters to
  //  the library. Data are packed into array params, parameter of the routine.
  //  First element of the array is used for dimension S, second element
  //  contains indicator, VSL_USER_QRNG_INITIAL_VALUES, of user-defined
  //  parameters for quasi-random number generators.
  //  Macros VSL_USER_PRIMITIVE_POLYMS and VSL_USER_INIT_DIRECTION_NUMBERS
  //  are used to describe which data are passed to SOBOL QRNG and
  //  VSL_USER_IRRED_POLYMS - which data are passed to NIEDERRITER QRNG.
  //  For example, to demonstrate that both primitive polynomials and initial
  //  direction numbers are passed in SOBOL one should set third element of the
  //  array params to  VSL_USER_PRIMITIVE_POLYMS | VSL_USER_DIRECTION_NUMBERS.
  //  Macro VSL_QRNG_OVERRIDE_1ST_DIM_INIT is used to override default
  //  initialization for the first dimension. Macro VSL_USER_DIRECTION_NUMBERS
  //  is used when direction numbers calculated on the user side are passed
  //  into the generators. More detailed description of interface for
  //  registration of user-defined QRNG initial parameters can be found
  //  in VslNotes.pdf.
   }
    VSL_USER_QRNG_INITIAL_VALUES = $1;    
    VSL_USER_PRIMITIVE_POLYMS = $1;    
    VSL_USER_INIT_DIRECTION_NUMBERS = $2;    
    VSL_USER_IRRED_POLYMS = $1;    
    VSL_USER_DIRECTION_NUMBERS = $4;    
    VSL_QRNG_OVERRIDE_1ST_DIM_INIT = $8;    
  {
  //  INITIALIZATION METHODS FOR USER-DESIGNED BASIC RANDOM NUMBER GENERATORS.
  //  Each BRNG must support at least VSL_INIT_METHOD_STANDARD initialization
  //  method. In addition, VSL_INIT_METHOD_LEAPFROG, VSL_INIT_METHOD_SKIPAHEAD and
  //  VSL_INIT_METHOD_SKIPAHEADEX initialization methods can be supported.
  //
  //  If VSL_INIT_METHOD_LEAPFROG is not supported then initialization routine
  //  must return VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED error code.
  //
  //  If VSL_INIT_METHOD_SKIPAHEAD is not supported then initialization routine
  //  must return VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED error code.
  //
  //  If VSL_INIT_METHOD_SKIPAHEADEX is not supported then initialization routine
  //  must return VSL_RNG_ERROR_SKIPAHEADEX_UNSUPPORTED error code.
  //
  //  If there is no error during initialization, the initialization routine must
  //  return VSL_ERROR_OK code.
   }
    VSL_INIT_METHOD_STANDARD = 0;    
    VSL_INIT_METHOD_LEAPFROG = 1;    
    VSL_INIT_METHOD_SKIPAHEAD = 2;    
    VSL_INIT_METHOD_SKIPAHEADEX = 3;    
  {
  //++
  //  ACCURACY FLAG FOR DISTRIBUTION GENERATORS
  //  This flag defines mode of random number generation.
  //  If accuracy mode is set distribution generators will produce
  //  numbers lying exactly within definitional domain for all values
  //  of distribution parameters. In this case slight performance
  //  degradation is expected. By default accuracy mode is switched off
  //  admitting random numbers to be out of the definitional domain for
  //  specific values of distribution parameters.
  //  This macro is used to form names for accuracy versions of
  //  distribution number generators
  //--
   }
    VSL_RNG_METHOD_ACCURACY_FLAG = 1 shl 30;    
  {
  //++
  //  TRANSFORMATION METHOD NAMES FOR DISTRIBUTION RANDOM NUMBER GENERATORS
  //  VSL interface allows more than one generation method in a distribution
  //  transformation subroutine. Following macro definitions are used to
  //  specify generation method for given distribution generator.
  //
  //  Method name macro is constructed as
  //
  //     VSL_RNG_METHOD_<Distribution>_<Method>
  //
  //  where
  //
  //     <Distribution> - probability distribution
  //     <Method> - method name
  //
  //  VSL_RNG_METHOD_<Distribution>_<Method> should be used with
  //  vsl<precision>Rng<Distribution> function only, where
  //
  //     <precision> - s (single) or d (double)
  //     <Distribution> - probability distribution
  //--
   }
  {
  // Uniform
  //
  // <Method>   <Short Description>
  // STD        standard method. Currently there is only one method for this
  //            distribution generator
   }
  { vsls,d,iRngUniform  }
    VSL_RNG_METHOD_UNIFORM_STD = 0;    
 
    VSL_RNG_METHOD_UNIFORM_STD_ACCURATE = VSL_RNG_METHOD_UNIFORM_STD or VSL_RNG_METHOD_ACCURACY_FLAG;

    { accurate mode of vsld,sRngUniform  }
    {
    // Uniform Bits
    //
    // <Method>   <Short Description>
    // STD        standard method. Currently there is only one method for this
    //            distribution generator
     }
    { vsliRngUniformBits  }
      VSL_RNG_METHOD_UNIFORMBITS_STD = 0;      
    {
    // Uniform Bits 32
    //
    // <Method>   <Short Description>                                                                  
    // STD        standard method. Currently there is only one method for this
    //            distribution generator
     }
    { vsliRngUniformBits32  }
      VSL_RNG_METHOD_UNIFORMBITS32_STD = 0;      
    {
    // Uniform Bits 64
    //
    // <Method>   <Short Description>
    // STD        standard method. Currently there is only one method for this
    //            distribution generator
     }
    { vsliRngUniformBits64  }
      VSL_RNG_METHOD_UNIFORMBITS64_STD = 0;      
    {
    // Gaussian
    //
    // <Method>   <Short Description>
    // BOXMULLER  generates normally distributed random number x thru the pair of
    //            uniformly distributed numbers u1 and u2 according to the formula:
    //
    //               x=sqrt(-ln(u1))*sin(2*Pi*u2)
    //
    // BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
    //            thru the pair of uniformly dustributed numbers u1 and u2
    //            according to the formula
    //
    //               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
    //               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
    //
    //            NOTE: implementation correctly works with odd vector lengths
    //
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngGaussian  }
      VSL_RNG_METHOD_GAUSSIAN_BOXMULLER = 0;      
    { vsld,sRngGaussian  }
      VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 = 1;      
    { vsld,sRngGaussian  }
      VSL_RNG_METHOD_GAUSSIAN_ICDF = 2;      
    {
    // GaussianMV - multivariate (correlated) normal
    // Multivariate (correlated) normal random number generator is based on
    // uncorrelated Gaussian random number generator (see vslsRngGaussian and
    // vsldRngGaussian functions):
    //
    // <Method>   <Short Description>
    // BOXMULLER  generates normally distributed random number x thru the pair of
    //            uniformly distributed numbers u1 and u2 according to the formula:
    //
    //               x=sqrt(-ln(u1))*sin(2*Pi*u2)
    //
    // BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
    //            thru the pair of uniformly dustributed numbers u1 and u2
    //            according to the formula
    //
    //               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
    //               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
    //
    //            NOTE: implementation correctly works with odd vector lengths
    //
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngGaussianMV  }
      VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER = 0;      
    { vsld,sRngGaussianMV  }
      VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2 = 1;      
    { vsld,sRngGaussianMV  }
      VSL_RNG_METHOD_GAUSSIANMV_ICDF = 2;      
    {
    // Exponential
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngExponential  }
      VSL_RNG_METHOD_EXPONENTIAL_ICDF = 0;      
      VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE = VSL_RNG_METHOD_EXPONENTIAL_ICDF or VSL_RNG_METHOD_ACCURACY_FLAG  ;

    { accurate mode of vsld,sRngExponential  }
    {
    // Laplace
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
    //
    // ICDF - inverse cumulative distribution function method:
    //
    //           x=+/-ln(u) with probability 1/2,
    //
    //        where
    //
    //           x - random number with Laplace distribution,
    //           u - uniformly distributed random number
     }
    { vsld,sRngLaplace  }
      VSL_RNG_METHOD_LAPLACE_ICDF = 0;      
    {
    // Weibull
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngWeibull  }
      VSL_RNG_METHOD_WEIBULL_ICDF = 0;      
      VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE = VSL_RNG_METHOD_WEIBULL_ICDF or VSL_RNG_METHOD_ACCURACY_FLAG ;
    { accurate mode of vsld,sRngWeibull  }
    {
    // Cauchy
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngCauchy  }
      VSL_RNG_METHOD_CAUCHY_ICDF = 0;      
    {
    // Rayleigh
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngRayleigh  }
      VSL_RNG_METHOD_RAYLEIGH_ICDF = 0;      
      VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE = VSL_RNG_METHOD_RAYLEIGH_ICDF or VSL_RNG_METHOD_ACCURACY_FLAG ;
    { accurate mode of vsld,sRngRayleigh  }
    {
    // Lognormal
    //
    // <Method>   <Short Description>
    // BOXMULLER2       Box-Muller 2 algorithm based method
     }
    { vsld,sRngLognormal  }
      VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 = 0;      
    { vsld,sRngLognormal  }
      VSL_RNG_METHOD_LOGNORMAL_ICDF = 1;      
      VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2_ACCURATE = VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 or VSL_RNG_METHOD_ACCURACY_FLAG ;
    { accurate mode of vsld,sRngLognormal  }
      VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE = VSL_RNG_METHOD_LOGNORMAL_ICDF or VSL_RNG_METHOD_ACCURACY_FLAG   ;
    { accurate mode of vsld,sRngLognormal  }
    {
    // Gumbel
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsld,sRngGumbel  }
      VSL_RNG_METHOD_GUMBEL_ICDF = 0;      
    {
    // Gamma
    //
    // Comments:
    // alpha>1             - algorithm of Marsaglia is used, nonlinear
    //                       transformation of gaussian numbers based on
    //                       acceptance/rejection method with squeezes;
    // alpha>=0.6, alpha<1 - rejection from the Weibull distribution is used;
    // alpha<0.6           - transformation of exponential power distribution
    //                       (EPD) is used, EPD random numbers are generated
    //                       by means of acceptance/rejection technique;
    // alpha=1             - gamma distribution reduces to exponential
    //                       distribution
     }
    { vsld,sRngGamma  }
      VSL_RNG_METHOD_GAMMA_GNORM = 0;      

      VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE = VSL_RNG_METHOD_GAMMA_GNORM or VSL_RNG_METHOD_ACCURACY_FLAG ;
    { accurate mode of vsld,sRngGamma  }
    {
    // Beta
    //
    // Comments:
    // CJA - stands for first letters of Cheng, Johnk, and Atkinson.
    // Cheng    - for min(p,q) > 1 method of Cheng,
    //            generation of beta random numbers of the second kind
    //            based on acceptance/rejection technique and its
    //            transformation to beta random numbers of the first kind;
    +// Johnk    - for max(p,q) < 1 methods of Johnk and Atkinson:
    //            if q + K*p^2+C<=0, K=0.852..., C=-0.956...
    //            algorithm of Johnk:
    //            beta distributed random number is generated as
    //            u1^(1/p) / (u1^(1/p)+u2^(1/q)), if u1^(1/p)+u2^(1/q)<=1;
    //            otherwise switching algorithm of Atkinson: interval (0,1)
    //            is divided into two domains (0,t) and (t,1), on each interval
    //            acceptance/rejection technique with convenient majorizing
    //            function is used;
    // Atkinson - for min(p,q)<1, max(p,q)>1 switching algorithm of Atkinson
    //            is used (with another point t, see short description above);
    // ICDF     - inverse cumulative distribution function method according
    //            to formulas x=1-u^(1/q) for p = 1, and x = u^(1/p) for q=1,
    //            where x is beta distributed random number,
    //            u - uniformly distributed random number.
    //            for p=q=1 beta distribution reduces to uniform distribution.
    //
     }
    { vsld,sRngBeta  }
      VSL_RNG_METHOD_BETA_CJA = 0;      
      VSL_RNG_METHOD_BETA_CJA_ACCURATE = VSL_RNG_METHOD_BETA_CJA or VSL_RNG_METHOD_ACCURACY_FLAG;
    { accurate mode of vsld,sRngBeta  }
    {
    // ChiSquare
    //
    // Comments:
    // v = 1, v = 3               - chi-square distributed random number is
    //                              generated as a sum of squares of v independent
    //                              normal random numbers;
    // v is even and v = 16       - chi-square distributed random number is
    //                              generated using the following formula:
    //                              x = -2*ln(u[0]*...*u[v/2-1]),
    //                              where u[i] - random numbers uniformly
    //                              distributed over the interval (0,1);
    // v > 16, v is odd and v > 3 - chi-square distribution reduces to gamma
    //                              distribution;
     }
    { vsld,sRngChiSquare  }
      VSL_RNG_METHOD_CHISQUARE_CHI2GAMMA = 0;      
    {
    // Bernoulli
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsliRngBernoulli  }
      VSL_RNG_METHOD_BERNOULLI_ICDF = 0;      
    {
    // Geometric
    //
    // <Method>   <Short Description>
    // ICDF       inverse cumulative distribution function method
     }
    { vsliRngGeometric  }
      VSL_RNG_METHOD_GEOMETRIC_ICDF = 0;      
    {
    // Binomial
    //
    // <Method>   <Short Description>
    // BTPE       for ntrial*min(p,1-p)>30 acceptance/rejection method with
    //            decomposition onto 4 regions:
    //
    //               * 2 parallelograms;
    //               * triangle;
    //               * left exponential tail;
    //               * right exponential tail.
    //
    //            otherwise table lookup method is used
     }
    { vsliRngBinomial  }
      VSL_RNG_METHOD_BINOMIAL_BTPE = 0;      
    {
    // Multinomial
    //
    // <Method>    <Short Description>
    // MULTPOISSON Poisson Approximation of Multinomial Distribution method
     }
    { vsliRngMultinomial  }
      VSL_RNG_METHOD_MULTINOMIAL_MULTPOISSON = 0;      
    {
    // Hypergeometric
    //
    // <Method>   <Short Description>
    // H2PE       if mode of distribution is large, acceptance/rejection method is
    //            used with decomposition onto 3 regions:
    //
    //               * rectangular;
    //               * left exponential tail;
    //               * right exponential tail.
    //
    //            otherwise table lookup method is used
     }
    { vsliRngHypergeometric  }
      VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE = 0;      
    {
    // Poisson
    //
    // <Method>   <Short Description>
    // PTPE       if lambda>=27, acceptance/rejection method is used with
    //            decomposition onto 4 regions:
    //
    //               * 2 parallelograms;
    //               * triangle;
    //               * left exponential tail;
    //               * right exponential tail.
    //
    //            otherwise table lookup method is used
    //
    // POISNORM   for lambda>=1 method is based on Poisson inverse CDF
    //            approximation by Gaussian inverse CDF; for lambda<1
    //            table lookup method is used.
     }
    { vsliRngPoisson  }
      VSL_RNG_METHOD_POISSON_PTPE = 0;      
    { vsliRngPoisson  }
      VSL_RNG_METHOD_POISSON_POISNORM = 1;      
    {
    // Poisson
    //
    // <Method>   <Short Description>
    // POISNORM   for lambda>=1 method is based on Poisson inverse CDF
    //            approximation by Gaussian inverse CDF; for lambda<1
    //            ICDF method is used.
     }
    { vsliRngPoissonV  }
      VSL_RNG_METHOD_POISSONV_POISNORM = 0;      
    {
    // Negbinomial
    //
    // <Method>   <Short Description>
    // NBAR       if (a-1)*(1-p)/p>=100, acceptance/rejection method is used with
    //            decomposition onto 5 regions:
    //
    //               * rectangular;
    //               * 2 trapezoid;
    //               * left exponential tail;
    //               * right exponential tail.
    //
    //            otherwise table lookup method is used.
     }
    { vsliRngNegbinomial  }
      VSL_RNG_METHOD_NEGBINOMIAL_NBAR = 0;      
    {
    //++
    //  MATRIX STORAGE SCHEMES
    //--
     }
    {
    // Some multivariate random number generators, e.g. GaussianMV, operate
    // with matrix parameters. To optimize matrix parameters usage VSL offers
    // following matrix storage schemes. (See VSL documentation for more details).
    //
    // FULL     - whole matrix is stored
    // PACKED   - lower/higher triangular matrix is packed in 1-dimensional array
    // DIAGONAL - diagonal elements are packed in 1-dimensional array
     }
      VSL_MATRIX_STORAGE_FULL = 0;      
      VSL_MATRIX_STORAGE_PACKED = 1;      
      VSL_MATRIX_STORAGE_DIAGONAL = 2;      
    {
    // SUMMARY STATISTICS (SS) RELATED MACRO DEFINITIONS
     }
    {
    //++
    //  MATRIX STORAGE SCHEMES
    //--
     }
    {
    // SS routines work with matrix parameters, e.g. matrix of observations,
    // variance-covariance matrix. To optimize work with matrices the library
    // provides the following storage matrix schemes.
     }
    {
    // Matrix of observations:
    // ROWS    - observations of the random vector are stored in raws, that
    //           is, i-th row of the matrix of observations contains values
    //           of i-th component of the random vector
    // COLS    - observations of the random vector are stored in columns that
    //           is, i-th column of the matrix of observations contains values
    //           of i-th component of the random vector
     }
      VSL_SS_MATRIX_STORAGE_ROWS = $00010000;      
      VSL_SS_MATRIX_STORAGE_COLS = $00020000;      
    {
    // Variance-covariance/correlation matrix:
    // FULL     - whole matrix is stored
    // L_PACKED - lower triangular matrix is stored as 1-dimensional array
    // U_PACKED - upper triangular matrix is stored as 1-dimensional array
     }
      VSL_SS_MATRIX_STORAGE_FULL = $00000000;      
      VSL_SS_MATRIX_STORAGE_L_PACKED = $00000001;      
      VSL_SS_MATRIX_STORAGE_U_PACKED = $00000002;      
    {
    //++
    //  Summary Statistics METHODS
    //--
     }
    {
    // SS routines provide computation of basic statistical estimates
    // (central/raw moments up to 4th order, variance-covariance,
    //  minimum, maximum, skewness/kurtosis) using the following methods
    //  - FAST  - estimates are computed for price of one or two passes over
    //            observations using highly optimized oneMKL routines
    //  - 1PASS - estimate is computed for price of one pass of the observations
    //  - FAST_USER_MEAN - estimates are computed for price of one or two passes
    //            over observations given user defined mean for central moments,
    //            covariance and correlation
    //  - CP_TO_COVCOR - convert cross-product matrix to variance-covariance/
    //            correlation matrix
    //  - SUM_TO_MOM - convert raw/central sums to raw/central moments
    //
     }
      VSL_SS_METHOD_FAST = $00000001;      
      VSL_SS_METHOD_1PASS = $00000002;      
      VSL_SS_METHOD_FAST_USER_MEAN = $00000100;      
      VSL_SS_METHOD_CP_TO_COVCOR = $00000200;      
      VSL_SS_METHOD_SUM_TO_MOM = $00000400;      
    {
    // SS provides routine for parametrization of correlation matrix using
    // SPECTRAL DECOMPOSITION (SD) method
     }
      VSL_SS_METHOD_SD = $00000004;      
    {
    // SS routine for robust estimation of variance-covariance matrix
    // and mean supports Rocke algorithm, TBS-estimator
     }
      VSL_SS_METHOD_TBS = $00000008;      
    {
    //  SS routine for estimation of missing values
    //  supports Multiple Imputation (MI) method
     }
      VSL_SS_METHOD_MI = $00000010;      
    {
    // SS provides routine for detection of outliers, BACON method
     }
      VSL_SS_METHOD_BACON = $00000020;      
    {
    // SS supports routine for estimation of quantiles for streaming data
    // using the following methods:
    // - ZW      - intermediate estimates of quantiles during processing
    //             the next block are computed
    // - ZW_FAST - intermediate estimates of quantiles during processing
    //             the next block are not computed
     }
      VSL_SS_METHOD_SQUANTS_ZW = $00000040;      
      VSL_SS_METHOD_SQUANTS_ZW_FAST = $00000080;      
    {
    // Input of BACON algorithm is set of 3 parameters:
    // - Initialization method of the algorithm
    // - Parameter alfa such that 1-alfa is percentile of Chi2 distribution
    // - Stopping criterion
     }
    {
    // Number of BACON algorithm parameters
     }
      VSL_SS_BACON_PARAMS_N = 3;      
    {
    // SS implementation of BACON algorithm supports two initialization methods:
    // - Mahalanobis distance based method
    // - Median based method
     }
      VSL_SS_METHOD_BACON_MAHALANOBIS_INIT = $00000001;      
      VSL_SS_METHOD_BACON_MEDIAN_INIT = $00000002;      
    {
    // SS routine for sorting data, RADIX method
     }
      VSL_SS_METHOD_RADIX = $00100000;      
    {
    // Input of TBS algorithm is set of 4 parameters:
    // - Breakdown point
    // - Asymptotic rejection probability
    // - Stopping criterion
    // - Maximum number of iterations
     }
    {
    // Number of TBS algorithm parameters
     }
      VSL_SS_TBS_PARAMS_N = 4;      
    {
    // Input of MI algorithm is set of 5 parameters:
    // - Maximal number of iterations for EM algorithm
    // - Maximal number of iterations for DA algorithm
    // - Stopping criterion
    // - Number of sets to impute
    // - Total number of missing values in dataset
     }
    {
    // Number of MI algorithm parameters
     }
      VSL_SS_MI_PARAMS_SIZE = 5;      
    {
    // SS MI algorithm expects that missing values are
    // marked with NANs
     }
      VSL_SS_DNAN = $FFF8000000000000;      
      VSL_SS_SNAN = $FFC00000;      
    {
    // Input of ZW algorithm is 1 parameter:
    // - accuracy of quantile estimation
     }
    {
    // Number of ZW algorithm parameters
     }
      VSL_SS_SQUANTS_ZW_PARAMS_N = 1;      
    {
    //++
    // MACROS USED SS EDIT AND COMPUTE ROUTINES
    //--
     }
    {
    // SS EditTask routine is way to edit input and output parameters of the task,
    // e.g., pointers to arrays which hold observations, weights of observations,
    // arrays of mean estimates or covariance estimates.
    // Macros below define parameters available for modification
     }
      VSL_SS_ED_DIMEN = 1;      
      VSL_SS_ED_OBSERV_N = 2;      
      VSL_SS_ED_OBSERV = 3;      
      VSL_SS_ED_OBSERV_STORAGE = 4;      
      VSL_SS_ED_INDC = 5;      
      VSL_SS_ED_WEIGHTS = 6;      
      VSL_SS_ED_MEAN = 7;      
      VSL_SS_ED_2R_MOM = 8;      
      VSL_SS_ED_3R_MOM = 9;      
      VSL_SS_ED_4R_MOM = 10;      
      VSL_SS_ED_2C_MOM = 11;      
      VSL_SS_ED_3C_MOM = 12;      
      VSL_SS_ED_4C_MOM = 13;      
      VSL_SS_ED_SUM = 67;      
      VSL_SS_ED_2R_SUM = 68;      
      VSL_SS_ED_3R_SUM = 69;      
      VSL_SS_ED_4R_SUM = 70;      
      VSL_SS_ED_2C_SUM = 71;      
      VSL_SS_ED_3C_SUM = 72;      
      VSL_SS_ED_4C_SUM = 73;      
      VSL_SS_ED_KURTOSIS = 14;      
      VSL_SS_ED_SKEWNESS = 15;      
      VSL_SS_ED_MIN = 16;      
      VSL_SS_ED_MAX = 17;      
      VSL_SS_ED_VARIATION = 18;      
      VSL_SS_ED_COV = 19;      
      VSL_SS_ED_COV_STORAGE = 20;      
      VSL_SS_ED_COR = 21;      
      VSL_SS_ED_COR_STORAGE = 22;      
      VSL_SS_ED_CP = 74;      
      VSL_SS_ED_CP_STORAGE = 75;      
      VSL_SS_ED_ACCUM_WEIGHT = 23;      
      VSL_SS_ED_QUANT_ORDER_N = 24;      
      VSL_SS_ED_QUANT_ORDER = 25;      
      VSL_SS_ED_QUANT_QUANTILES = 26;      
      VSL_SS_ED_ORDER_STATS = 27;      
      VSL_SS_ED_GROUP_INDC = 28;      
      VSL_SS_ED_POOLED_COV_STORAGE = 29;      
      VSL_SS_ED_POOLED_MEAN = 30;      
      VSL_SS_ED_POOLED_COV = 31;      
      VSL_SS_ED_GROUP_COV_INDC = 32;      
      VSL_SS_ED_REQ_GROUP_INDC = 32;      
      VSL_SS_ED_GROUP_MEAN = 33;      
      VSL_SS_ED_GROUP_COV_STORAGE = 34;      
      VSL_SS_ED_GROUP_COV = 35;      
      VSL_SS_ED_ROBUST_COV_STORAGE = 36;      
      VSL_SS_ED_ROBUST_COV_PARAMS_N = 37;      
      VSL_SS_ED_ROBUST_COV_PARAMS = 38;      
      VSL_SS_ED_ROBUST_MEAN = 39;      
      VSL_SS_ED_ROBUST_COV = 40;      
      VSL_SS_ED_OUTLIERS_PARAMS_N = 41;      
      VSL_SS_ED_OUTLIERS_PARAMS = 42;      
      VSL_SS_ED_OUTLIERS_WEIGHT = 43;      
      VSL_SS_ED_ORDER_STATS_STORAGE = 44;      
      VSL_SS_ED_PARTIAL_COV_IDX = 45;      
      VSL_SS_ED_PARTIAL_COV = 46;      
      VSL_SS_ED_PARTIAL_COV_STORAGE = 47;      
      VSL_SS_ED_PARTIAL_COR = 48;      
      VSL_SS_ED_PARTIAL_COR_STORAGE = 49;      
      VSL_SS_ED_MI_PARAMS_N = 50;      
      VSL_SS_ED_MI_PARAMS = 51;      
      VSL_SS_ED_MI_INIT_ESTIMATES_N = 52;      
      VSL_SS_ED_MI_INIT_ESTIMATES = 53;      
      VSL_SS_ED_MI_SIMUL_VALS_N = 54;      
      VSL_SS_ED_MI_SIMUL_VALS = 55;      
      VSL_SS_ED_MI_ESTIMATES_N = 56;      
      VSL_SS_ED_MI_ESTIMATES = 57;      
      VSL_SS_ED_MI_PRIOR_N = 58;      
      VSL_SS_ED_MI_PRIOR = 59;      
      VSL_SS_ED_PARAMTR_COR = 60;      
      VSL_SS_ED_PARAMTR_COR_STORAGE = 61;      
      VSL_SS_ED_STREAM_QUANT_PARAMS_N = 62;      
      VSL_SS_ED_STREAM_QUANT_PARAMS = 63;      
      VSL_SS_ED_STREAM_QUANT_ORDER_N = 64;      
      VSL_SS_ED_STREAM_QUANT_ORDER = 65;      
      VSL_SS_ED_STREAM_QUANT_QUANTILES = 66;      
      VSL_SS_ED_MDAD = 76;      
      VSL_SS_ED_MNAD = 77;      
      VSL_SS_ED_SORTED_OBSERV = 78;      
      VSL_SS_ED_SORTED_OBSERV_STORAGE = 79;      
    {
    // SS Compute routine calculates estimates supported by the library
    // Macros below define estimates to compute
     }
      VSL_SS_MEAN = $0000000000000001;      
      VSL_SS_2R_MOM = $0000000000000002;      
      VSL_SS_3R_MOM = $0000000000000004;      
      VSL_SS_4R_MOM = $0000000000000008;      
      VSL_SS_2C_MOM = $0000000000000010;      
      VSL_SS_3C_MOM = $0000000000000020;      
      VSL_SS_4C_MOM = $0000000000000040;      
      VSL_SS_SUM = $0000000002000000;      
      VSL_SS_2R_SUM = $0000000004000000;      
      VSL_SS_3R_SUM = $0000000008000000;      
      VSL_SS_4R_SUM = $0000000010000000;      
      VSL_SS_2C_SUM = $0000000020000000;      
      VSL_SS_3C_SUM = $0000000040000000;      
      VSL_SS_4C_SUM = $0000000080000000;      
      VSL_SS_KURTOSIS = $0000000000000080;      
      VSL_SS_SKEWNESS = $0000000000000100;      
      VSL_SS_VARIATION = $0000000000000200;      
      VSL_SS_MIN = $0000000000000400;      
      VSL_SS_MAX = $0000000000000800;      
      VSL_SS_COV = $0000000000001000;      
      VSL_SS_COR = $0000000000002000;      
      VSL_SS_CP = $0000000100000000;      
      VSL_SS_POOLED_COV = $0000000000004000;      
      VSL_SS_GROUP_COV = $0000000000008000;      
      VSL_SS_POOLED_MEAN = $0000000800000000;      
      VSL_SS_GROUP_MEAN = $0000001000000000;      
      VSL_SS_QUANTS = $0000000000010000;      
      VSL_SS_ORDER_STATS = $0000000000020000;      
      VSL_SS_SORTED_OBSERV = $0000008000000000;      
      VSL_SS_ROBUST_COV = $0000000000040000;      
      VSL_SS_OUTLIERS = $0000000000080000;      
      VSL_SS_PARTIAL_COV = $0000000000100000;      
      VSL_SS_PARTIAL_COR = $0000000000200000;      
      VSL_SS_MISSING_VALS = $0000000000400000;      
      VSL_SS_PARAMTR_COR = $0000000000800000;      
      VSL_SS_STREAM_QUANTS = $0000000001000000;      
      VSL_SS_MDAD = $0000000200000000;      
      VSL_SS_MNAD = $0000000400000000;      

type
{
//  POINTER TO STREAM STATE STRUCTURE
//  This is a void pointer to hide implementation details.
 }
  PVSLStreamStatePtr = ^VSLStreamStatePtr;
  VSLStreamStatePtr = pointer;

  PVSLConvTaskPtr = ^VSLConvTaskPtr;
  VSLConvTaskPtr = pointer;

  PVSLCorrTaskPtr = ^VSLCorrTaskPtr;
  VSLCorrTaskPtr = pointer;

  PVSLSSTaskPtr = ^VSLSSTaskPtr;
  VSLSSTaskPtr = pointer;

//  POINTERS TO BASIC RANDOM NUMBER GENERATOR FUNCTIONS
//  Each BRNG must have following implementations:
//
//  * Stream initialization (InitStreamPtr)
//  * Integer-value recurrence implementation (iBRngPtr)
//  * Single precision implementation (sBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
//  * Double precision implementation (dBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
  InitStreamPtr = function (method:longint; stream:VSLStreamStatePtr; n:longint; const params:Pdword):longint;cdecl;

  sBRngPtr = function (stream:VSLStreamStatePtr; n:longint; r:Psingle; a:single; b:single):longint;cdecl;

  dBRngPtr = function (stream:VSLStreamStatePtr; n:longint; r:Pdouble; a:double; b:double):longint;cdecl;

  iBRngPtr = function (stream:VSLStreamStatePtr; n:longint; r:Pdword):longint;cdecl;

  PVSLBRngProperties = ^VSLBRngProperties;
  VSLBRngProperties = record
      StreamStateSize : longint;
      NSeeds : longint;
      IncludesZero : longint;
      WordSize : longint;
      NBits : longint;
      InitStream : InitStreamPtr;
      sBRng : sBRngPtr;
      dBRng : dBRngPtr;
      iBRng : iBRngPtr;
    end;
{
{********** Pointers to callback functions for abstract streams ************ }

  iUpdateFuncPtr = function (stream:VSLStreamStatePtr; n:Plongint; ibuf:Pdword; nmin:Plongint; nmax:Plongint; 
               idx:Plongint):longint;cdecl;

  dUpdateFuncPtr = function (stream:VSLStreamStatePtr; n:Plongint; dbuf:Pdouble; nmin:Plongint; nmax:Plongint; 
               idx:Plongint):longint;cdecl;

  sUpdateFuncPtr = function (stream:VSLStreamStatePtr; n:Plongint; sbuf:Psingle; nmin:Plongint; nmax:Plongint; 
               idx:Plongint):longint;cdecl;
{
//  BASIC RANDOM NUMBER GENERATOR PROPERTIES STRUCTURE
//  The structure describes the properties of given basic generator, e.g. size
//  of the stream state structure, pointers to function implementations, etc.
//
//  BRNG properties structure fields:
//  StreamStateSize - size of the stream state structure (in bytes)
//  WordSize        - size of base word (in bytes). Typically this is 4 bytes.
//  NSeeds          - number of words necessary to describe generator's state
//  NBits           - number of bits actually used in base word. For example,
//                    only 31 least significant bits are actually used in
//                    basic random number generator MCG31m1 with 4-byte base
//                    word. NBits field is useful while interpreting random
//                    words as a sequence of random bits.
//  IncludesZero    - FALSE if 0 cannot be generated in integer-valued
//                    implementation; TRUE if 0 can be potentially generated in
//                    integer-valued implementation.
//  InitStream      - pointer to stream state initialization function
//  sBRng           - pointer to single precision implementation
//  dBRng           - pointer to double precision implementation
//  iBRng           - pointer to integer-value implementation
 }
{ Stream state size (in bytes)  }
{ Number of seeds  }
{ Zero flag  }
{ Size (in bytes) of base word  }
{ Number of actually used bits  }
{ Pointer to InitStream func  }
{ Pointer to S func  }
{ Pointer to D func  }
{ Pointer to I func  }

//++
//  VSL CONTINUOUS DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--





function vdRngCauchy(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngCauchy';





//function VDRNGCAUCHY(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGCAUCHY';
//
//
//
//
//
//function vdrngcauchy(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vdrngcauchy';





function vsRngCauchy(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngCauchy';





//function VSRNGCAUCHY(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGCAUCHY';
//
//
//
//
//
//function vsrngcauchy(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrngcauchy';

{ Uniform distribution  }




function vdRngUniform(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngUniform';





//function VDRNGUNIFORM(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGUNIFORM';
//
//
//
//
//
//function vdrnguniform(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
           //_para6:Pdouble):longint;cdecl;external libmkl name 'vdrnguniform';





function vsRngUniform(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngUniform';





//function VSRNGUNIFORM(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGUNIFORM';
//
//
//
//
//
//function vsrnguniform(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrnguniform';

{ Gaussian distribution  }




function vdRngGaussian(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngGaussian';





//function VDRNGGAUSSIAN(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGGAUSSIAN';
//
//
//
//
//
//function vdrnggaussian(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vdrnggaussian';





function vsRngGaussian(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngGaussian';





//function VSRNGGAUSSIAN(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGGAUSSIAN';
//
//
//
//
//
//function vsrnggaussian(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrnggaussian';

{ GaussianMV distribution  }






function vdRngGaussianMV(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:MKL_INT;
           _para6:MKL_INT; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vdRngGaussianMV';







//function VDRNGGAUSSIANMV(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'VDRNGGAUSSIANMV';
//
//
//
//
//
//
//
//function vdrnggaussianmv(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vdrnggaussianmv';







function vsRngGaussianMV(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:MKL_INT;
           _para6:MKL_INT; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vsRngGaussianMV';







//function VSRNGGAUSSIANMV(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'VSRNGGAUSSIANMV';
//
//
//
//
//
//
//
//function vsrnggaussianmv(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vsrnggaussianmv';

{ Exponential distribution  }




function vdRngExponential(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngExponential';





//function VDRNGEXPONENTIAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGEXPONENTIAL';
//
//
//
//
//
//function vdrngexponential(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vdrngexponential';
//




function vsRngExponential(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngExponential';





//function VSRNGEXPONENTIAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGEXPONENTIAL';
//
//
//
//
//
//function vsrngexponential(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrngexponential';

{ Laplace distribution  }




function vdRngLaplace(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngLaplace';





//function VDRNGLAPLACE(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGLAPLACE';
//
//
//
//
//
//function vdrnglaplace(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vdrnglaplace';





function vsRngLaplace(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngLaplace';





//function VSRNGLAPLACE(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGLAPLACE';
//
//
//
//
//
//function vsrnglaplace(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrnglaplace';

{ Weibull distribution  }





function vdRngWeibull(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double; _para7:double):longint;cdecl;external libmkl name 'vdRngWeibull';






//function VDRNGWEIBULL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'VDRNGWEIBULL';
//
//
//
//
//
//
//function vdrngweibull(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'vdrngweibull';






function vsRngWeibull(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single; _para7:single):longint;cdecl;external libmkl name 'vsRngWeibull';






//function VSRNGWEIBULL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'VSRNGWEIBULL';
//
//
//
//
//
//
//function vsrngweibull(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'vsrngweibull';

{ Rayleigh distribution  }




function vdRngRayleigh(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngRayleigh';





//function VDRNGRAYLEIGH(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGRAYLEIGH';
//
//
//
//
//
//function vdrngrayleigh(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vdrngrayleigh';





function vsRngRayleigh(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngRayleigh';





//function VSRNGRAYLEIGH(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGRAYLEIGH';
//
//
//
//
//
//function vsrngrayleigh(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrngrayleigh';

{ Lognormal distribution  }






function vdRngLognormal(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double; _para7:double; _para8:double):longint;cdecl;external libmkl name 'vdRngLognormal';







//function VDRNGLOGNORMAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'VDRNGLOGNORMAL';
//
//
//
//
//
//
//
//function vdrnglognormal(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vdrnglognormal';







function vsRngLognormal(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single; _para7:single; _para8:single):longint;cdecl;external libmkl name 'vsRngLognormal';







//function VSRNGLOGNORMAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'VSRNGLOGNORMAL';
//
//
//
//
//
//
//
//function vsrnglognormal(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vsrnglognormal';

{ Gumbel distribution  }




function vdRngGumbel(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'vdRngGumbel';





//function VDRNGGUMBEL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VDRNGGUMBEL';
//
//
//
//
//
//function vdrnggumbel(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vdrnggumbel';





function vsRngGumbel(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single):longint;cdecl;external libmkl name 'vsRngGumbel';





//function VSRNGGUMBEL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSRNGGUMBEL';
//
//
//
//
//
//function vsrnggumbel(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vsrnggumbel';

{ Gamma distribution  }





function vdRngGamma(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double; _para7:double):longint;cdecl;external libmkl name 'vdRngGamma';






//function VDRNGGAMMA(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'VDRNGGAMMA';
//
//
//
//
//
//
//function vdrnggamma(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'vdrnggamma';






function vsRngGamma(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single; _para7:single):longint;cdecl;external libmkl name 'vsRngGamma';






//function VSRNGGAMMA(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'VSRNGGAMMA';
//
//
//
//
//
//
//function vsrnggamma(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'vsrnggamma';

{ Beta distribution  }






function vdRngBeta(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:double;
           _para6:double; _para7:double; _para8:double):longint;cdecl;external libmkl name 'vdRngBeta';







//function VDRNGBETA(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'VDRNGBETA';
//
//
//
//
//
//
//
//function vdrngbeta(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vdrngbeta';







function vsRngBeta(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:single;
           _para6:single; _para7:single; _para8:single):longint;cdecl;external libmkl name 'vsRngBeta';







//function VSRNGBETA(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'VSRNGBETA';
//
//
//
//
//
//
//
//function vsrngbeta(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vsrngbeta';

{ Chi-square distribution  }



function vdRngChiSquare(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdouble; _para5:longint):longint;cdecl;external libmkl name 'vdRngChiSquare';




//function VDRNGCHISQUARE(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Plongint):longint;cdecl;external libmkl name 'VDRNGCHISQUARE';
//
//
//
//
//function vdrngchisquare(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdouble; _para5:Plongint):longint;cdecl;external libmkl name 'vdrngchisquare';




function vsRngChiSquare(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Psingle; _para5:longint):longint;cdecl;external libmkl name 'vsRngChiSquare';




//function VSRNGCHISQUARE(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Plongint):longint;cdecl;external libmkl name 'VSRNGCHISQUARE';
//
//
//
//
//function vsrngchisquare(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Psingle; _para5:Plongint):longint;cdecl;external libmkl name 'vsrngchisquare';
//
{
//++
//  VSL DISCRETE DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
 }
{ Bernoulli distribution  }



function viRngBernoulli(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:double):longint;cdecl;external libmkl name 'viRngBernoulli';



//
//function VIRNGBERNOULLI(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'VIRNGBERNOULLI';
//
//
//
//
//function virngbernoulli(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'virngbernoulli';

{ Uniform distribution  }




function viRngUniform(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:longint;
           _para6:longint):longint;cdecl;external libmkl name 'viRngUniform';





//function VIRNGUNIFORM(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Plongint):longint;cdecl;external libmkl name 'VIRNGUNIFORM';
//
//
//
//
//
//function virnguniform(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Plongint):longint;cdecl;external libmkl name 'virnguniform';

{ UniformBits distribution  }


function viRngUniformBits(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'viRngUniformBits';



//function VIRNGUNIFORMBITS(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'VIRNGUNIFORMBITS';
//
//
//
//function virnguniformbits(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'virnguniformbits';

{ UniformBits32 distribution  }


function viRngUniformBits32(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'viRngUniformBits32';



//function VIRNGUNIFORMBITS32(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'VIRNGUNIFORMBITS32';
//
//
//
//function virnguniformbits32(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'virnguniformbits32';

{ UniformBits64 distribution  }


function viRngUniformBits64(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:PMKL_UINT64):longint;cdecl;external libmkl name 'viRngUniformBits64';



//function VIRNGUNIFORMBITS64(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:PMKL_UINT64):longint;cdecl;external libmkl name 'VIRNGUNIFORMBITS64';
//
//
//
//function virnguniformbits64(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:PMKL_UINT64):longint;cdecl;external libmkl name 'virnguniformbits64';

{ Geometric distribution  }



function viRngGeometric(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:double):longint;cdecl;external libmkl name 'viRngGeometric';




//function VIRNGGEOMETRIC(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'VIRNGGEOMETRIC';
//
//
//
//
//function virnggeometric(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'virnggeometric';
//
{ Binomial distribution  }




function viRngBinomial(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:longint;
           _para6:double):longint;cdecl;external libmkl name 'viRngBinomial';





//function VIRNGBINOMIAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VIRNGBINOMIAL';
//
//
//
//
//
//function virngbinomial(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'virngbinomial';

{ Multinomial distribution  }





function viRngMultinomial(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:longint;
           _para6:longint; _para7:Pdouble):longint;cdecl;external libmkl name 'viRngMultinomial';






//function VIRNGMULTINOMIAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Plongint; _para7:Pdouble):longint;cdecl;external libmkl name 'VIRNGMULTINOMIAL';
//
//
//
//
//
//
//function virngmultinomial(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Plongint; _para7:Pdouble):longint;cdecl;external libmkl name 'virngmultinomial';

{ Hypergeometric distribution  }





function viRngHypergeometric(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:longint;
           _para6:longint; _para7:longint):longint;cdecl;external libmkl name 'viRngHypergeometric';






//function VIRNGHYPERGEOMETRIC(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Plongint; _para7:Plongint):longint;cdecl;external libmkl name 'VIRNGHYPERGEOMETRIC';
//
//
//
//
//
//
//function virnghypergeometric(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Plongint;
//           _para6:Plongint; _para7:Plongint):longint;cdecl;external libmkl name 'virnghypergeometric';

{ Negbinomial distribution  }




function viRngNegbinomial(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:double;
           _para6:double):longint;cdecl;external libmkl name 'viRngNegbinomial';





//function viRngNegBinomial(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:double;
//           _para6:double):longint;cdecl;external libmkl name 'viRngNegBinomial';
//
//
//
//
//
//function VIRNGNEGBINOMIAL(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VIRNGNEGBINOMIAL';





//function virngnegbinomial(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'virngnegbinomial';

{ Poisson distribution  }



function viRngPoisson(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:double):longint;cdecl;external libmkl name 'viRngPoisson';




//function VIRNGPOISSON(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'VIRNGPOISSON';
//
//
//
//
//function virngpoisson(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'virngpoisson';

{ PoissonV distribution  }



function viRngPoissonV(_para1:MKL_INT; _para2:VSLStreamStatePtr; _para3:MKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'viRngPoissonV';




//function VIRNGPOISSONV(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'VIRNGPOISSONV';
//
//
//
//
//function virngpoissonv(_para1:PMKL_INT; _para2:PVSLStreamStatePtr; _para3:PMKL_INT; _para4:Plongint; _para5:Pdouble):longint;cdecl;external libmkl name 'virngpoissonv';

{
//++
//  VSL SERVICE FUNCTION DECLARATIONS.
//--
 }
{ NewStream - stream creation/initialization  }


function vslNewStream(_para1:PVSLStreamStatePtr; _para2:MKL_INT; _para3:MKL_UINT):longint;cdecl;external libmkl name 'vslNewStream';



//function vslnewstream(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_UINT):longint;cdecl;external libmkl name 'vslnewstream';
//
//
//
//function VSLNEWSTREAM(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_UINT):longint;cdecl;external libmkl name 'VSLNEWSTREAM';

{ NewStreamEx - advanced stream creation/initialization  }



function vslNewStreamEx(_para1:PVSLStreamStatePtr; _para2:MKL_INT; _para3:MKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'vslNewStreamEx';




//function vslnewstreamex(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'vslnewstreamex';
//
//
//
//
//function VSLNEWSTREAMEX(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Pdword):longint;cdecl;external libmkl name 'VSLNEWSTREAMEX';




function vsliNewAbstractStream(_para1:PVSLStreamStatePtr; _para2:MKL_INT; _para3:Pdword; _para4:iUpdateFuncPtr):longint;cdecl;external libmkl name 'vsliNewAbstractStream';




//function vslinewabstractstream(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:Pdword; _para4:iUpdateFuncPtr):longint;cdecl;external libmkl name 'vslinewabstractstream';
//
//
//
//
//function VSLINEWABSTRACTSTREAM(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:Pdword; _para4:iUpdateFuncPtr):longint;cdecl;external libmkl name 'VSLINEWABSTRACTSTREAM';






function vsldNewAbstractStream(_para1:PVSLStreamStatePtr; _para2:MKL_INT; _para3:Pdouble; _para4:double; _para5:double;
           _para6:dUpdateFuncPtr):longint;cdecl;external libmkl name 'vsldNewAbstractStream';






//function vsldnewabstractstream(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:dUpdateFuncPtr):longint;cdecl;external libmkl name 'vsldnewabstractstream';
//
//
//
//
//
//
//function VSLDNEWABSTRACTSTREAM(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:dUpdateFuncPtr):longint;cdecl;external libmkl name 'VSLDNEWABSTRACTSTREAM';






function vslsNewAbstractStream(_para1:PVSLStreamStatePtr; _para2:MKL_INT; _para3:Psingle; _para4:single; _para5:single;
           _para6:sUpdateFuncPtr):longint;cdecl;external libmkl name 'vslsNewAbstractStream';






//function vslsnewabstractstream(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:sUpdateFuncPtr):longint;cdecl;external libmkl name 'vslsnewabstractstream';
//
//
//
//
//
//
//function VSLSNEWABSTRACTSTREAM(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:sUpdateFuncPtr):longint;cdecl;external libmkl name 'VSLSNEWABSTRACTSTREAM';

{ DeleteStream - delete stream  }
function vslDeleteStream(_para1:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'vslDeleteStream';

//function vsldeletestream(_para1:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'vsldeletestream';
//
//function VSLDELETESTREAM(_para1:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'VSLDELETESTREAM';

{ CopyStream - copy all stream information  }

function vslCopyStream(_para1:PVSLStreamStatePtr; _para2:VSLStreamStatePtr):longint;cdecl;external libmkl name 'vslCopyStream';


//function vslcopystream(_para1:PVSLStreamStatePtr; _para2:VSLStreamStatePtr):longint;cdecl;external libmkl name 'vslcopystream';
//
//
//function VSLCOPYSTREAM(_para1:PVSLStreamStatePtr; _para2:VSLStreamStatePtr):longint;cdecl;external libmkl name 'VSLCOPYSTREAM';

{ CopyStreamState - copy stream state only  }

function vslCopyStreamState(_para1:VSLStreamStatePtr; _para2:VSLStreamStatePtr):longint;cdecl;external libmkl name 'vslCopyStreamState';


//function vslcopystreamstate(_para1:PVSLStreamStatePtr; _para2:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'vslcopystreamstate';
//
//
//function VSLCOPYSTREAMSTATE(_para1:PVSLStreamStatePtr; _para2:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'VSLCOPYSTREAMSTATE';

{ LeapfrogStream - leapfrog method  }


function vslLeapfrogStream(_para1:VSLStreamStatePtr; _para2:MKL_INT; _para3:MKL_INT):longint;cdecl;external libmkl name 'vslLeapfrogStream';



//function vslleapfrogstream(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_INT):longint;cdecl;external libmkl name 'vslleapfrogstream';
//
//
//
//function VSLLEAPFROGSTREAM(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_INT):longint;cdecl;external libmkl name 'VSLLEAPFROGSTREAM';

{ SkipAheadStream - skip-ahead method  }

function vslSkipAheadStream(_para1:VSLStreamStatePtr; _para2:int64):longint;cdecl;external libmkl name 'vslSkipAheadStream';


//function vslskipaheadstream(_para1:PVSLStreamStatePtr; _para2:Pint64):longint;cdecl;external libmkl name 'vslskipaheadstream';
//
//
//function VSLSKIPAHEADSTREAM(_para1:PVSLStreamStatePtr; _para2:Pint64):longint;cdecl;external libmkl name 'VSLSKIPAHEADSTREAM';

{ SkipAheadStreamEx - skip-ahead extended method  }


function vslSkipAheadStreamEx(_para1:VSLStreamStatePtr; _para2:MKL_INT; _para3:PMKL_UINT64):longint;cdecl;external libmkl name 'vslSkipAheadStreamEx';



//function vslskipaheadstreamex(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_UINT64):longint;cdecl;external libmkl name 'vslskipaheadstreamex';
//
//
//
//function VSLSKIPAHEADSTREAMEX(_para1:PVSLStreamStatePtr; _para2:PMKL_INT; _para3:PMKL_UINT64):longint;cdecl;external libmkl name 'VSLSKIPAHEADSTREAMEX';

{ GetStreamStateBrng - get BRNG associated with given stream  }

function vslGetStreamStateBrng(_para1:VSLStreamStatePtr):longint;cdecl;external libmkl name 'vslGetStreamStateBrng';


//function vslgetstreamstatebrng(_para1:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'vslgetstreamstatebrng';
//
//
//function VSLGETSTREAMSTATEBRNG(_para1:PVSLStreamStatePtr):longint;cdecl;external libmkl name 'VSLGETSTREAMSTATEBRNG';

{ GetNumRegBrngs - get number of registered BRNGs  }
function vslGetNumRegBrngs:longint;cdecl;external libmkl name 'vslGetNumRegBrngs';

//function vslgetnumregbrngs:longint;cdecl;external libmkl name 'vslgetnumregbrngs';
//
//function VSLGETNUMREGBRNGS:longint;cdecl;external libmkl name 'VSLGETNUMREGBRNGS';

{ RegisterBrng - register new BRNG  }

function vslRegisterBrng(_para1:PVSLBRngProperties):longint;cdecl;external libmkl name 'vslRegisterBrng';


//function vslregisterbrng(_para1:PVSLBRngProperties):longint;cdecl;external libmkl name 'vslregisterbrng';
//
//
//function VSLREGISTERBRNG(_para1:PVSLBRngProperties):longint;cdecl;external libmkl name 'VSLREGISTERBRNG';

{ GetBrngProperties - get BRNG properties  }

function vslGetBrngProperties(_para1:longint; _para2:PVSLBRngProperties):longint;cdecl;external libmkl name 'vslGetBrngProperties';


//function vslgetbrngproperties(_para1:Plongint; _para2:PVSLBRngProperties):longint;cdecl;external libmkl name 'vslgetbrngproperties';
//
//
//function VSLGETBRNGPROPERTIES(_para1:Plongint; _para2:PVSLBRngProperties):longint;cdecl;external libmkl name 'VSLGETBRNGPROPERTIES';

{ SaveStreamF - save random stream descriptive data to file  }


function vslSaveStreamF(_para1:VSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'vslSaveStreamF';




//function vslsavestreamf(_para1:PVSLStreamStatePtr; _para2:Pchar; _para3:longint):longint;cdecl;external libmkl name 'vslsavestreamf';
//
//
//
//
//function VSLSAVESTREAMF(_para1:PVSLStreamStatePtr; _para2:Pchar; _para3:longint):longint;cdecl;external libmkl name 'VSLSAVESTREAMF';

{ LoadStreamF - load random stream descriptive data from file  }

function vslLoadStreamF(_para1:PVSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'vslLoadStreamF';



//function vslloadstreamf(_para1:PVSLStreamStatePtr; _para2:Pchar; _para3:longint):longint;cdecl;external libmkl name 'vslloadstreamf';
//
//
//
//function VSLLOADSTREAMF(_para1:PVSLStreamStatePtr; _para2:Pchar; _para3:longint):longint;cdecl;external libmkl name 'VSLLOADSTREAMF';

{ SaveStreamM - save random stream descriptive data to memory  }

function vslSaveStreamM(_para1:VSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'vslSaveStreamM';


//function vslsavestreamm(_para1:PVSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'vslsavestreamm';
//
//
//function VSLSAVESTREAMM(_para1:PVSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'VSLSAVESTREAMM';

{ LoadStreamM - load random stream descriptive data from memory  }

function vslLoadStreamM(_para1:PVSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'vslLoadStreamM';


//function vslloadstreamm(_para1:PVSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'vslloadstreamm';
//
//
//function VSLLOADSTREAMM(_para1:PVSLStreamStatePtr; _para2:Pchar):longint;cdecl;external libmkl name 'VSLLOADSTREAMM';

{ GetStreamSize - get size of random stream descriptive data  }

function vslGetStreamSize(_para1:VSLStreamStatePtr):longint;cdecl;external libmkl name 'vslGetStreamSize';


//function vslgetstreamsize(_para1:VSLStreamStatePtr):longint;cdecl;external libmkl name 'vslgetstreamsize';
//
//
//function VSLGETSTREAMSIZE(_para1:VSLStreamStatePtr):longint;cdecl;external libmkl name 'VSLGETSTREAMSIZE';

{
//++
//  VSL CONVOLUTION AND CORRELATION FUNCTION DECLARATIONS.
//--
 }





function vsldConvNewTask(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldConvNewTask';






//function vsldconvnewtask(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvnewtask';
//
//
//
//
//
//
//function VSLDCONVNEWTASK(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVNEWTASK';






function vslsConvNewTask(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslsConvNewTask';






//function vslsconvnewtask(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvnewtask';
//
//
//
//
//
//
//function VSLSCONVNEWTASK(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVNEWTASK';






function vslzConvNewTask(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslzConvNewTask';






//function vslzconvnewtask(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvnewtask';
//
//
//
//
//
//
//function VSLZCONVNEWTASK(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVNEWTASK';






function vslcConvNewTask(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslcConvNewTask';






//function vslcconvnewtask(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvnewtask';
//
//
//
//
//
//
//function VSLCCONVNEWTASK(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVNEWTASK';






function vsldCorrNewTask(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldCorrNewTask';






//function vsldcorrnewtask(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrnewtask';
//
//
//
//
//
//
//function VSLDCORRNEWTASK(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORRNEWTASK';






function vslsCorrNewTask(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslsCorrNewTask';






//function vslscorrnewtask(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrnewtask';
//
//
//
//
//
//
//function VSLSCORRNEWTASK(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORRNEWTASK';






function vslzCorrNewTask(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslzCorrNewTask';






//function vslzcorrnewtask(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrnewtask';
//
//
//
//
//
//
//function VSLZCORRNEWTASK(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORRNEWTASK';






function vslcCorrNewTask(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslcCorrNewTask';






//function vslccorrnewtask(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrnewtask';
//
//
//
//
//
//
//function VSLCCORRNEWTASK(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORRNEWTASK';





function vsldConvNewTask1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vsldConvNewTask1D';





//function vsldconvnewtask1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvnewtask1d';
//
//
//
//
//
//function VSLDCONVNEWTASK1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVNEWTASK1D';





function vslsConvNewTask1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslsConvNewTask1D';





//function vslsconvnewtask1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvnewtask1d';
//
//
//
//
//
//function VSLSCONVNEWTASK1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVNEWTASK1D';





function vslzConvNewTask1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslzConvNewTask1D';





//function vslzconvnewtask1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvnewtask1d';
//
//
//
//
//
//function VSLZCONVNEWTASK1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVNEWTASK1D';





function vslcConvNewTask1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslcConvNewTask1D';





//function vslcconvnewtask1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvnewtask1d';
//
//
//
//
//
//function VSLCCONVNEWTASK1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVNEWTASK1D';
//




function vsldCorrNewTask1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vsldCorrNewTask1D';





//function vsldcorrnewtask1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrnewtask1d';
//
//
//
//
//
//function VSLDCORRNEWTASK1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORRNEWTASK1D';





function vslsCorrNewTask1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslsCorrNewTask1D';





//function vslscorrnewtask1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrnewtask1d';
//
//
//
//
//
//function VSLSCORRNEWTASK1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORRNEWTASK1D';





function vslzCorrNewTask1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslzCorrNewTask1D';





//function vslzcorrnewtask1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrnewtask1d';
//
//
//
//
//
//function VSLZCORRNEWTASK1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORRNEWTASK1D';
//




function vslcCorrNewTask1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslcCorrNewTask1D';





//function vslccorrnewtask1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrnewtask1d';
//
//
//
//
//
//function VSLCCORRNEWTASK1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORRNEWTASK1D';








function vsldConvNewTaskX(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vsldConvNewTaskX';








//function vsldconvnewtaskx(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvnewtaskx';
//
//
//
//
//
//
//
//
//function VSLDCONVNEWTASKX(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVNEWTASKX';








function vslsConvNewTaskX(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslsConvNewTaskX';








//function vslsconvnewtaskx(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvnewtaskx';
//
//
//
//
//
//
//
//
//function VSLSCONVNEWTASKX(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVNEWTASKX';








function vslzConvNewTaskX(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:PMKL_Complex16; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslzConvNewTaskX';








//function vslzconvnewtaskx(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex16; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvnewtaskx';
//
//
//
//
//
//
//
//
//function VSLZCONVNEWTASKX(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex16; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVNEWTASKX';
//







function vslcConvNewTaskX(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:PMKL_Complex8; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslcConvNewTaskX';








//function vslcconvnewtaskx(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex8; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvnewtaskx';
//
//
//
//
//
//
//
//
//function VSLCCONVNEWTASKX(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex8; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVNEWTASKX';
//







function vsldCorrNewTaskX(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vsldCorrNewTaskX';








//function vsldcorrnewtaskx(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrnewtaskx';
//
//
//
//
//
//
//
//
//function VSLDCORRNEWTASKX(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORRNEWTASKX';








function vslsCorrNewTaskX(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslsCorrNewTaskX';








//function vslscorrnewtaskx(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrnewtaskx';
//
//
//
//
//
//
//
//
//function VSLSCORRNEWTASKX(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORRNEWTASKX';








function vslzCorrNewTaskX(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:PMKL_Complex16; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslzCorrNewTaskX';








//function vslzcorrnewtaskx(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex16; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrnewtaskx';
//
//
//
//
//
//
//
//
//function VSLZCORRNEWTASKX(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex16; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORRNEWTASKX';








function vslcCorrNewTaskX(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
           _para6:PMKL_INT; _para7:PMKL_Complex8; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslcCorrNewTaskX';








//function vslccorrnewtaskx(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex8; _para8:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrnewtaskx';
//
//
//
//
//
//
//
//
//function VSLCCORRNEWTASKX(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_INT; _para7:PMKL_Complex8; _para8:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORRNEWTASKX';







function vsldConvNewTaskX1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:Pdouble; _para7:MKL_INT):longint;cdecl;external libmkl name 'vsldConvNewTaskX1D';







//function vsldconvnewtaskx1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvnewtaskx1d';
//
//
//
//
//
//
//
//function VSLDCONVNEWTASKX1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVNEWTASKX1D';







function vslsConvNewTaskX1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:Psingle; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslsConvNewTaskX1D';







//function vslsconvnewtaskx1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvnewtaskx1d';
//
//
//
//
//
//
//
//function VSLSCONVNEWTASKX1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVNEWTASKX1D';







function vslzConvNewTaskX1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:PMKL_Complex16; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslzConvNewTaskX1D';







//function vslzconvnewtaskx1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvnewtaskx1d';
//
//
//
//
//
//
//
//function VSLZCONVNEWTASKX1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVNEWTASKX1D';







function vslcConvNewTaskX1D(_para1:PVSLConvTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:PMKL_Complex8; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslcConvNewTaskX1D';







//function vslcconvnewtaskx1d(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvnewtaskx1d';
//
//
//
//
//
//
//
//function VSLCCONVNEWTASKX1D(_para1:PVSLConvTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVNEWTASKX1D';







function vsldCorrNewTaskX1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:Pdouble; _para7:MKL_INT):longint;cdecl;external libmkl name 'vsldCorrNewTaskX1D';







//function vsldcorrnewtaskx1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrnewtaskx1d';
//
//
//
//
//
//
//
//function VSLDCORRNEWTASKX1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORRNEWTASKX1D';







function vslsCorrNewTaskX1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:Psingle; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslsCorrNewTaskX1D';







//function vslscorrnewtaskx1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrnewtaskx1d';
//
//
//
//
//
//
//
//function VSLSCORRNEWTASKX1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORRNEWTASKX1D';







function vslzCorrNewTaskX1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:PMKL_Complex16; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslzCorrNewTaskX1D';







//function vslzcorrnewtaskx1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrnewtaskx1d';
//
//
//
//
//
//
//
//function VSLZCORRNEWTASKX1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORRNEWTASKX1D';
//






function vslcCorrNewTaskX1D(_para1:PVSLCorrTaskPtr; _para2:MKL_INT; _para3:MKL_INT; _para4:MKL_INT; _para5:MKL_INT;
           _para6:PMKL_Complex8; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslcCorrNewTaskX1D';







//function vslccorrnewtaskx1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrnewtaskx1d';
//
//
//
//
//
//
//
//function VSLCCORRNEWTASKX1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORRNEWTASKX1D';

function vslConvDeleteTask(_para1:PVSLConvTaskPtr):longint;cdecl;external libmkl name 'vslConvDeleteTask';

//function vslconvdeletetask(_para1:PVSLConvTaskPtr):longint;cdecl;external libmkl name 'vslconvdeletetask';
//
//function VSLCONVDeleteTask(_para1:PVSLConvTaskPtr):longint;cdecl;external libmkl name 'VSLCONVDeleteTask';

function vslCorrDeleteTask(_para1:PVSLCorrTaskPtr):longint;cdecl;external libmkl name 'vslCorrDeleteTask';

//function vslcorrdeletetask(_para1:PVSLCorrTaskPtr):longint;cdecl;external libmkl name 'vslcorrdeletetask';
//
//function VSLCORRDeleteTask(_para1:PVSLCorrTaskPtr):longint;cdecl;external libmkl name 'VSLCORRDeleteTask';


function vslConvCopyTask(_para1:PVSLConvTaskPtr; _para2:VSLConvTaskPtr):longint;cdecl;external libmkl name 'vslConvCopyTask';


//function vslconvcopytask(_para1:PVSLConvTaskPtr; _para2:PVSLConvTaskPtr):longint;cdecl;external libmkl name 'vslconvcopytask';
//
//
//function VSLCONVCopyTask(_para1:PVSLConvTaskPtr; _para2:PVSLConvTaskPtr):longint;cdecl;external libmkl name 'VSLCONVCopyTask';


function vslCorrCopyTask(_para1:PVSLCorrTaskPtr; _para2:VSLCorrTaskPtr):longint;cdecl;external libmkl name 'vslCorrCopyTask';


//function vslcorrcopytask(_para1:PVSLCorrTaskPtr; _para2:PVSLCorrTaskPtr):longint;cdecl;external libmkl name 'vslcorrcopytask';
//
//
//function VSLCORRCopyTask(_para1:PVSLCorrTaskPtr; _para2:PVSLCorrTaskPtr):longint;cdecl;external libmkl name 'VSLCORRCopyTask';


function vslConvSetMode(_para1:VSLConvTaskPtr; _para2:MKL_INT):longint;cdecl;external libmkl name 'vslConvSetMode';


//function vslconvsetmode(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslconvsetmode';
//
//
//function VSLCONVSETMODE(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCONVSETMODE';


function vslCorrSetMode(_para1:VSLCorrTaskPtr; _para2:MKL_INT):longint;cdecl;external libmkl name 'vslCorrSetMode';


//function vslcorrsetmode(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslcorrsetmode';
//
//
//function VSLCORRSETMODE(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCORRSETMODE';


function vslConvSetInternalPrecision(_para1:VSLConvTaskPtr; _para2:MKL_INT):longint;cdecl;external libmkl name 'vslConvSetInternalPrecision';


//function vslconvsetinternalprecision(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslconvsetinternalprecision';
//
//
//function VSLCONVSETINTERNALPRECISION(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCONVSETINTERNALPRECISION';


function vslCorrSetInternalPrecision(_para1:VSLCorrTaskPtr; _para2:MKL_INT):longint;cdecl;external libmkl name 'vslCorrSetInternalPrecision';


//function vslcorrsetinternalprecision(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslcorrsetinternalprecision';
//
//
//function VSLCORRSETINTERNALPRECISION(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCORRSETINTERNALPRECISION';


function vslConvSetStart(_para1:VSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslConvSetStart';


//function vslconvsetstart(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslconvsetstart';
//
//
//function VSLCONVSETSTART(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCONVSETSTART';


function vslCorrSetStart(_para1:VSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslCorrSetStart';


//function vslcorrsetstart(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslcorrsetstart';
//
//
//function VSLCORRSETSTART(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCORRSETSTART';


function vslConvSetDecimation(_para1:VSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslConvSetDecimation';


//function vslconvsetdecimation(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslconvsetdecimation';
//
//
//function VSLCONVSETDECIMATION(_para1:PVSLConvTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCONVSETDECIMATION';


function vslCorrSetDecimation(_para1:VSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslCorrSetDecimation';


//function vslcorrsetdecimation(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'vslcorrsetdecimation';
//
//
//function VSLCORRSETDECIMATION(_para1:PVSLCorrTaskPtr; _para2:PMKL_INT):longint;cdecl;external libmkl name 'VSLCORRSETDECIMATION';






function vsldConvExec(_para1:VSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldConvExec';






//function vsldconvexec(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvexec';
//
//
//
//
//
//
//function VSLDCONVEXEC(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVEXEC';






function vslsConvExec(_para1:VSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsConvExec';






//function vslsconvexec(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvexec';
//
//
//
//
//
//
//function VSLSCONVEXEC(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVEXEC';






function vslzConvExec(_para1:VSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzConvExec';






//function vslzconvexec(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvexec';
//
//
//
//
//
//
//function VSLZCONVEXEC(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVEXEC';






function vslcConvExec(_para1:VSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslcConvExec';






//function vslcconvexec(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvexec';
//
//
//
//
//
//
//function VSLCCONVEXEC(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVEXEC';
//





function vsldCorrExec(_para1:VSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldCorrExec';






//function vsldcorrexec(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrexec';
//
//
//
//
//
//
//function VSLDCORREXEC(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORREXEC';






function vslsCorrExec(_para1:VSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsCorrExec';






//function vslscorrexec(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrexec';
//
//
//
//
//
//
//function VSLSCORREXEC(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORREXEC';






function vslzCorrExec(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzCorrExec';






//function vslzcorrexec(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrexec';
//
//
//
//
//
//
//function VSLZCORREXEC(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORREXEC';






function vslcCorrExec(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslcCorrExec';






//function vslccorrexec(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrexec';
//
//
//
//
//
//
//function VSLCCORREXEC(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORREXEC';
//





function vsldConvExec1D(_para1:VSLConvTaskPtr; _para2:Pdouble; _para3:MKL_INT; _para4:Pdouble; _para5:MKL_INT;
           _para6:Pdouble; _para7:MKL_INT):longint;cdecl;external libmkl name 'vsldConvExec1D';






//function vsldconvexec1d(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvexec1d';
//
//
//
//
//
//
//function VSLDCONVEXEC1D(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVEXEC1D';






function vslsConvExec1D(_para1:VSLConvTaskPtr; _para2:Psingle; _para3:MKL_INT; _para4:Psingle; _para5:MKL_INT;
           _para6:Psingle; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslsConvExec1D';






//function vslsconvexec1d(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvexec1d';
//
//
//
//
//
//
//function VSLSCONVEXEC1D(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVEXEC1D';
//





function vslzConvExec1D(_para1:VSLConvTaskPtr; _para2:PMKL_Complex16; _para3:MKL_INT; _para4:PMKL_Complex16; _para5:MKL_INT;
           _para6:PMKL_Complex16; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslzConvExec1D';






//function vslzconvexec1d(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvexec1d';
//
//
//
//
//
//
//function VSLZCONVEXEC1D(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVEXEC1D';






function vslcConvExec1D(_para1:VSLConvTaskPtr; _para2:PMKL_Complex8; _para3:MKL_INT; _para4:PMKL_Complex8; _para5:MKL_INT;
           _para6:PMKL_Complex8; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslcConvExec1D';






//function vslcconvexec1d(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvexec1d';
//
//
//
//
//
//
//function VSLCCONVEXEC1D(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVEXEC1D';






function vsldCorrExec1D(_para1:VSLCorrTaskPtr; _para2:Pdouble; _para3:MKL_INT; _para4:Pdouble; _para5:MKL_INT;
           _para6:Pdouble; _para7:MKL_INT):longint;cdecl;external libmkl name 'vsldCorrExec1D';






//function vsldcorrexec1d(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrexec1d';
//
//
//
//
//
//
//function VSLDCORREXEC1D(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORREXEC1D';






function vslsCorrExec1D(_para1:VSLCorrTaskPtr; _para2:Psingle; _para3:MKL_INT; _para4:Psingle; _para5:MKL_INT;
           _para6:Psingle; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslsCorrExec1D';






//function vslscorrexec1d(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrexec1d';
//
//
//
//
//
//
//function VSLSCORREXEC1D(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORREXEC1D';






function vslzCorrExec1D(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:MKL_INT; _para4:PMKL_Complex16; _para5:MKL_INT;
           _para6:PMKL_Complex16; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslzCorrExec1D';






//function vslzcorrexec1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrexec1d';
//
//
//
//
//
//
//function VSLZCORREXEC1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT;
//           _para6:PMKL_Complex16; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORREXEC1D';






function vslcCorrExec1D(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:MKL_INT; _para4:PMKL_Complex8; _para5:MKL_INT;
           _para6:PMKL_Complex8; _para7:MKL_INT):longint;cdecl;external libmkl name 'vslcCorrExec1D';






//function vslccorrexec1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrexec1d';
//
//
//
//
//
//
//function VSLCCORREXEC1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT;
//           _para6:PMKL_Complex8; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORREXEC1D';




function vsldConvExecX(_para1:VSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldConvExecX';




//function vsldconvexecx(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvexecx';
//
//
//
//
//function VSLDCONVEXECX(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVEXECX';




function vslsConvExecX(_para1:VSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsConvExecX';




//function vslsconvexecx(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvexecx';
//
//
//
//
//function VSLSCONVEXECX(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVEXECX';




function vslzConvExecX(_para1:VSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzConvExecX';




//function vslzconvexecx(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvexecx';
//
//
//
//
//function VSLZCONVEXECX(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVEXECX';
//



function vslcConvExecX(_para1:VSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslcConvExecX';




//function vslcconvexecx(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvexecx';
//
//
//
//
//function VSLCCONVEXECX(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVEXECX';




function vsldCorrExecX(_para1:VSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldCorrExecX';




//function vsldcorrexecx(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrexecx';
//
//
//
//
//function VSLDCORREXECX(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORREXECX';
//



function vslsCorrExecX(_para1:VSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsCorrExecX';




//function vslscorrexecx(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrexecx';
//
//
//
//
//function VSLSCORREXECX(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORREXECX';




function vslzCorrExecX(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzCorrExecX';




//function vslzcorrexecx(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrexecx';
//
//
//
//
//function VSLZCORREXECX(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORREXECX';
//



function vslcCorrExecX(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslcCorrExecX';




//function vslccorrexecx(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrexecx';
//
//
//
//
//function VSLCCORREXECX(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORREXECX';




function vsldConvExecX1D(_para1:VSLConvTaskPtr; _para2:Pdouble; _para3:MKL_INT; _para4:Pdouble; _para5:MKL_INT):longint;cdecl;external libmkl name 'vsldConvExecX1D';




//function vsldconvexecx1d(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldconvexecx1d';
//
//
//
//
//function VSLDCONVEXECX1D(_para1:PVSLConvTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCONVEXECX1D';
//



function vslsConvExecX1D(_para1:VSLConvTaskPtr; _para2:Psingle; _para3:MKL_INT; _para4:Psingle; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslsConvExecX1D';



//
//function vslsconvexecx1d(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsconvexecx1d';
//
//
//
//
//function VSLSCONVEXECX1D(_para1:PVSLConvTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCONVEXECX1D';
//



function vslzConvExecX1D(_para1:VSLConvTaskPtr; _para2:PMKL_Complex16; _para3:MKL_INT; _para4:PMKL_Complex16; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslzConvExecX1D';




//function vslzconvexecx1d(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzconvexecx1d';
//
//
//
//
//function VSLZCONVEXECX1D(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCONVEXECX1D';




function vslcConvExecX1D(_para1:VSLConvTaskPtr; _para2:PMKL_Complex8; _para3:MKL_INT; _para4:PMKL_Complex8; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslcConvExecX1D';




//function vslcconvexecx1d(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslcconvexecx1d';
//
//
//
//
//function VSLCCONVEXECX1D(_para1:PVSLConvTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCONVEXECX1D';
//



function vsldCorrExecX1D(_para1:VSLCorrTaskPtr; _para2:Pdouble; _para3:MKL_INT; _para4:Pdouble; _para5:MKL_INT):longint;cdecl;external libmkl name 'vsldCorrExecX1D';




//function vsldcorrexecx1d(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldcorrexecx1d';
//
//
//
//
//function VSLDCORREXECX1D(_para1:PVSLCorrTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDCORREXECX1D';
//



function vslsCorrExecX1D(_para1:VSLCorrTaskPtr; _para2:Psingle; _para3:MKL_INT; _para4:Psingle; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslsCorrExecX1D';




//function vslscorrexecx1d(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslscorrexecx1d';
//
//
//
//
//function VSLSCORREXECX1D(_para1:PVSLCorrTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSCORREXECX1D';




function vslzCorrExecX1D(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:MKL_INT; _para4:PMKL_Complex16; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslzCorrExecX1D';




//function vslzcorrexecx1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslzcorrexecx1d';
//
//
//
//
//function VSLZCORREXECX1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex16; _para3:PMKL_INT; _para4:PMKL_Complex16; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLZCORREXECX1D';




function vslcCorrExecX1D(_para1:VSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:MKL_INT; _para4:PMKL_Complex8; _para5:MKL_INT):longint;cdecl;external libmkl name 'vslcCorrExecX1D';




//function vslccorrexecx1d(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslccorrexecx1d';
//
//
//
//
//function VSLCCORREXECX1D(_para1:PVSLCorrTaskPtr; _para2:PMKL_Complex8; _para3:PMKL_INT; _para4:PMKL_Complex8; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLCCORREXECX1D';

{
//++
//  SUMMARARY STATTISTICS LIBRARY ROUTINES
//--
 }
{
//  Task constructors
 }






function vsldSSNewTask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:Pdouble;
           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldSSNewTask';







//function vsldssnewtask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vsldssnewtask';
//
//
//
//
//
//
//
//function VSLDSSNEWTASK(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:Pdouble; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSNEWTASK';







function vslsSSNewTask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:Psingle;
           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsSSNewTask';







//function vslsssnewtask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:Psingle;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'vslsssnewtask';
//
//
//
//
//
//
//
//function VSLSSSNEWTASK(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:PMKL_INT; _para5:Psingle;
//           _para6:Psingle; _para7:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSNEWTASK';

{
// Task editors
 }
{
// Editor to modify a task parameter
 }


function vsldSSEditTask(_para1:VSLSSTaskPtr; _para2:MKL_INT; _para3:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditTask';



//function vsldssedittask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble):longint;cdecl;external libmkl name 'vsldssedittask';
//
//
//
//function VSLDSSEDITTASK(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITTASK';



function vslsSSEditTask(_para1:VSLSSTaskPtr; _para2:MKL_INT; _para3:Psingle):longint;cdecl;external libmkl name 'vslsSSEditTask';



//function vslsssedittask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle):longint;cdecl;external libmkl name 'vslsssedittask';
//
//
//
//function VSLSSSEDITTASK(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITTASK';



function vsliSSEditTask(_para1:VSLSSTaskPtr; _para2:MKL_INT; _para3:PMKL_INT):longint;cdecl;external libmkl name 'vsliSSEditTask';



//function vslissedittask(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT):longint;cdecl;external libmkl name 'vslissedittask';
//
//
//
//function VSLISSEDITTASK(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT):longint;cdecl;external libmkl name 'VSLISSEDITTASK';

{
// Task specific editors
 }
{
// Editors to modify moments related parameters
 }
function vsldSSEditMoments(_para1:VSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditMoments';

//function vsldsseditmoments(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vsldsseditmoments';
//
//function VSLDSSEDITMOMENTS(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITMOMENTS';

function vslsSSEditMoments(_para1:VSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:Psingle;
           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vslsSSEditMoments';

//function vslssseditmoments(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vslssseditmoments';
//
//function VSLSSSEDITMOMENTS(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITMOMENTS';

{
// Editors to modify sums related parameters
 }
function vsldSSEditSums(_para1:VSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditSums';

//function vsldsseditsums(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'vsldsseditsums';
//
//function VSLDSSEDITSUMS(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble; _para7:Pdouble; _para8:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITSUMS';

function vslsSSEditSums(_para1:VSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:Psingle;
           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vslsSSEditSums';

//function vslssseditsums(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'vslssseditsums';
//
//function VSLSSSEDITSUMS(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle; _para7:Psingle; _para8:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITSUMS';

{
// Editors to modify variance-covariance/correlation matrix related parameters
 }


function vsldSSEditCovCor(_para1:VSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldSSEditCovCor';



//function vsldsseditcovcor(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldsseditcovcor';
//
//
//
//function VSLDSSEDITCOVCOR(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSEDITCOVCOR';



function vslsSSEditCovCor(_para1:VSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslsSSEditCovCor';



//function vslssseditcovcor(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslssseditcovcor';
//
//
//
//function VSLSSSEDITCOVCOR(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSEDITCOVCOR';

{
// Editors to modify cross-product matrix related parameters
 }

function vsldSSEditCP(_para1:VSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldSSEditCP';


//function vsldsseditcp(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldsseditcp';
//
//
//function VSLDSSEDITCP(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSEDITCP';
//

function vslsSSEditCP(_para1:VSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsSSEditCP';


//function vslssseditcp(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslssseditcp';
//
//
//function VSLSSSEDITCP(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSEDITCP';

{
// Editors to modify partial variance-covariance matrix related parameters
 }







function vsldSSEditPartialCovCor(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT; _para9:Pdouble; _para10:PMKL_INT):longint;cdecl;external libmkl name 'vsldSSEditPartialCovCor';








//function vsldsseditpartialcovcor(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT; _para9:Pdouble; _para10:PMKL_INT):longint;cdecl;external libmkl name 'vsldsseditpartialcovcor';
//
//
//
//
//
//
//
//
//function VSLDSSEDITPARTIALCOVCOR(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT; _para9:Pdouble; _para10:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSEDITPARTIALCOVCOR';








function vslsSSEditPartialCovCor(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT; _para9:Psingle; _para10:PMKL_INT):longint;cdecl;external libmkl name 'vslsSSEditPartialCovCor';








//function vslssseditpartialcovcor(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT; _para9:Psingle; _para10:PMKL_INT):longint;cdecl;external libmkl name 'vslssseditpartialcovcor';
//
//
//
//
//
//
//
//
//function VSLSSSEDITPARTIALCOVCOR(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT; _para9:Psingle; _para10:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSEDITPARTIALCOVCOR';

{
// Editors to modify quantiles related parameters
 }



function vsldSSEditQuantiles(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldSSEditQuantiles';




//function vsldsseditquantiles(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vsldsseditquantiles';
//
//
//
//
//function VSLDSSEDITQUANTILES(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:Pdouble;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSEDITQUANTILES';




function vslsSSEditQuantiles(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:Psingle;
           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslsSSEditQuantiles';




//function vslssseditquantiles(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'vslssseditquantiles';
//
//
//
//
//function VSLSSSEDITQUANTILES(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:Psingle;
//           _para6:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSEDITQUANTILES';

{
// Editors to modify stream data quantiles related parameters
 }




function vsldSSEditStreamQuantiles(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT;
           _para6:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditStreamQuantiles';





//function vsldsseditstreamquantiles(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vsldsseditstreamquantiles';
//
//
//
//
//
//function VSLDSSEDITSTREAMQUANTILES(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITSTREAMQUANTILES';





function vslsSSEditStreamQuantiles(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT;
           _para6:Psingle):longint;cdecl;external libmkl name 'vslsSSEditStreamQuantiles';





//function vslssseditstreamquantiles(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vslssseditstreamquantiles';
//
//
//
//
//
//function VSLSSSEDITSTREAMQUANTILES(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITSTREAMQUANTILES';

{
// Editors to modify pooled/group variance-covariance matrix related parameters
 }


function vsldSSEditPooledCovariance(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT;
           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditPooledCovariance';



//function vsldsseditpooledcovariance(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'vsldsseditpooledcovariance';
//
//
//
//function VSLDSSEDITPOOLEDCOVARIANCE(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble; _para5:PMKL_INT;
//           _para6:Pdouble; _para7:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITPOOLEDCOVARIANCE';



function vslsSSEditPooledCovariance(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT;
           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'vslsSSEditPooledCovariance';



//function vslssseditpooledcovariance(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'vslssseditpooledcovariance';
//
//
//
//function VSLSSSEDITPOOLEDCOVARIANCE(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle; _para5:PMKL_INT;
//           _para6:Psingle; _para7:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITPOOLEDCOVARIANCE';

{
// Editors to modify robust variance-covariance matrix related parameters
 }



function vsldSSEditRobustCovariance(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
           _para6:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditRobustCovariance';




//function vsldsseditrobustcovariance(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'vsldsseditrobustcovariance';
//
//
//
//
//function VSLDSSEDITROBUSTCOVARIANCE(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Pdouble; _para5:Pdouble;
//           _para6:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITROBUSTCOVARIANCE';




function vslsSSEditRobustCovariance(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
           _para6:Psingle):longint;cdecl;external libmkl name 'vslsSSEditRobustCovariance';




//function vslssseditrobustcovariance(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'vslssseditrobustcovariance';
//
//
//
//
//function VSLSSSEDITROBUSTCOVARIANCE(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:PMKL_INT; _para4:Psingle; _para5:Psingle;
//           _para6:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITROBUSTCOVARIANCE';

{
// Editors to modify outliers detection parameters
 }


function vsldSSEditOutliersDetection(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditOutliersDetection';



//function vsldsseditoutliersdetection(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble):longint;cdecl;external libmkl name 'vsldsseditoutliersdetection';
//
//
//
//function VSLDSSEDITOUTLIERSDETECTION(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITOUTLIERSDETECTION';



function vslsSSEditOutliersDetection(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle):longint;cdecl;external libmkl name 'vslsSSEditOutliersDetection';



//function vslssseditoutliersdetection(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle):longint;cdecl;external libmkl name 'vslssseditoutliersdetection';
//
//
//
//function VSLSSSEDITOUTLIERSDETECTION(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITOUTLIERSDETECTION';

{
// Editors to modify missing values support parameters
 }








function vsldSSEditMissingValues(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT; _para9:Pdouble; _para10:PMKL_INT;
           _para11:Pdouble):longint;cdecl;external libmkl name 'vsldSSEditMissingValues';









//function vsldsseditmissingvalues(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT; _para9:Pdouble; _para10:PMKL_INT;
//           _para11:Pdouble):longint;cdecl;external libmkl name 'vsldsseditmissingvalues';
//
//
//
//
//
//
//
//
//
//function VSLDSSEDITMISSINGVALUES(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Pdouble; _para4:PMKL_INT; _para5:Pdouble;
//           _para6:PMKL_INT; _para7:Pdouble; _para8:PMKL_INT; _para9:Pdouble; _para10:PMKL_INT;
//           _para11:Pdouble):longint;cdecl;external libmkl name 'VSLDSSEDITMISSINGVALUES';









function vslsSSEditMissingValues(_para1:VSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT; _para9:Psingle; _para10:PMKL_INT;
           _para11:Psingle):longint;cdecl;external libmkl name 'vslsSSEditMissingValues';









//function vslssseditmissingvalues(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT; _para9:Psingle; _para10:PMKL_INT;
//           _para11:Psingle):longint;cdecl;external libmkl name 'vslssseditmissingvalues';
//
//
//
//
//
//
//
//
//
//function VSLSSSEDITMISSINGVALUES(_para1:PVSLSSTaskPtr; _para2:PMKL_INT; _para3:Psingle; _para4:PMKL_INT; _para5:Psingle;
//           _para6:PMKL_INT; _para7:Psingle; _para8:PMKL_INT; _para9:Psingle; _para10:PMKL_INT;
//           _para11:Psingle):longint;cdecl;external libmkl name 'VSLSSSEDITMISSINGVALUES';

{
// Editors to modify matrixparametrization parameters
 }



function vsldSSEditCorParameterization(_para1:VSLSSTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldSSEditCorParameterization';




//function vsldsseditcorparameterization(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vsldsseditcorparameterization';
//
//
//
//
//function VSLDSSEDITCORPARAMETERIZATION(_para1:PVSLSSTaskPtr; _para2:Pdouble; _para3:PMKL_INT; _para4:Pdouble; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSEDITCORPARAMETERIZATION';




function vslsSSEditCorParameterization(_para1:VSLSSTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslsSSEditCorParameterization';




//function vslssseditcorparameterization(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'vslssseditcorparameterization';
//
//
//
//
//function VSLSSSEDITCORPARAMETERIZATION(_para1:PVSLSSTaskPtr; _para2:Psingle; _para3:PMKL_INT; _para4:Psingle; _para5:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSEDITCORPARAMETERIZATION';

{
// Compute routines
 }


function vsldSSCompute(_para1:VSLSSTaskPtr; _para2:MKL_UINT64; _para3:MKL_INT):longint;cdecl;external libmkl name 'vsldSSCompute';



//function vsldsscompute(_para1:PVSLSSTaskPtr; _para2:PMKL_UINT64; _para3:PMKL_INT):longint;cdecl;external libmkl name 'vsldsscompute';
//
//
//
//function VSLDSSCOMPUTE(_para1:PVSLSSTaskPtr; _para2:PMKL_UINT64; _para3:PMKL_INT):longint;cdecl;external libmkl name 'VSLDSSCOMPUTE';
//


function vslsSSCompute(_para1:VSLSSTaskPtr; _para2:MKL_UINT64; _para3:MKL_INT):longint;cdecl;external libmkl name 'vslsSSCompute';



//function vslssscompute(_para1:PVSLSSTaskPtr; _para2:PMKL_UINT64; _para3:PMKL_INT):longint;cdecl;external libmkl name 'vslssscompute';
//
//
//
//function VSLSSSCOMPUTE(_para1:PVSLSSTaskPtr; _para2:PMKL_UINT64; _para3:PMKL_INT):longint;cdecl;external libmkl name 'VSLSSSCOMPUTE';

{
// Task destructor
 }
function vslSSDeleteTask(_para1:PVSLSSTaskPtr):longint;cdecl;external libmkl name 'vslSSDeleteTask';

//function vslssdeletetask(_para1:PVSLSSTaskPtr):longint;cdecl;external libmkl name 'vslssdeletetask';
//
//function VSLSSDELETETASK(_para1:PVSLSSTaskPtr):longint;cdecl;external libmkl name 'VSLSSDELETETASK';


implementation

end.
