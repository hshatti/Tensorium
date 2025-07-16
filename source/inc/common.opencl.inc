''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains the common defines and type-defs for the CLBlast OpenCL kernels.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Parameters set by the tuner or by the database. Here they are given a basic default value in case'#10
+ '// this file is used outside of the CLBlast library.'#10
+ '#ifndef PRECISION'#10
+ '  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '#ifndef CUDA'#10
+ '  // Enable support for half-precision'#10
+ '  #if PRECISION == 16'#10
+ '    #pragma OPENCL EXTENSION cl_khr_fp16: enable'#10
+ '  #endif'#10
+ '  // Enable support for double-precision'#10
+ '  #if PRECISION == 64 || PRECISION == 6464'#10
+ '    #pragma OPENCL EXTENSION cl_khr_fp64: enable'#10
+ '  #endif'#10
+ '#endif'#10
+ '// Half-precision'#10
+ '#if PRECISION == 16'#10
+ '  typedef half real;'#10
+ '  typedef half2 real2;'#10
+ '  typedef half4 real4;'#10
+ '  typedef half8 real8;'#10
+ '  typedef half16 real16;'#10
+ '  #define ZERO 0'#10
+ '  #define ONE 1'#10
+ '  #define SMALLEST -1.0e14'#10
+ '// Single-precision'#10
+ '#elif PRECISION == 32'#10
+ '  typedef float real;'#10
+ '  typedef float2 real2;'#10
+ '  typedef float4 real4;'#10
+ '  typedef float8 real8;'#10
+ '  typedef float16 real16;'#10
+ '  #define ZERO 0.0f'#10
+ '  #define ONE 1.0f'#10
+ '  #define SMALLEST -1.0e37f'#10
+ '// Double-precision '#10
+ '#elif PRECISION == 64'#10
+ '  typedef double real;'#10
+ '  typedef double2 real2;'#10
+ '  typedef double4 real4;'#10
+ '  typedef double8 real8;'#10
+ '  typedef double16 real16;'#10
+ '  #define ZERO 0.0'#10
+ '  #define ONE 1.0'#10
+ '  #define SMALLEST -1.0e37'#10
+ '// Complex single-precision'#10
+ '#elif PRECISION == 3232'#10
+ '  typedef float2 real;'#10
+ '  typedef struct cfloat2 {real x; real y;} real2;'#10
+ '  typedef struct cfloat4 {real x; real y; real z; real w;} real4;'#10
+ '  typedef struct cfloat8 {real s0; real s1; real s2; real s3;'#10
+ '                          real s4; real s5; real s6; real s7;} real8;'#10
+ '  typedef struct cfloat16 {real s0; real s1; real s2; real s3;'#10
+ '                           real s4; real s5; real s6; real s7;'#10
+ '                           real s8; real s9; real sA; real sB;'#10
+ '                           real sC; real sD; real sE; real sF;} real16;'#10
+ '  #define ZERO 0.0f'#10
+ '  #define ONE 1.0f'#10
+ '  #define SMALLEST -1.0e37f'#10
+ '// Complex double-precision'#10
+ '#elif PRECISION == 6464'#10
+ '  typedef double2 real;'#10
+ '  typedef struct cdouble2 {real x; real y;} real2;'#10
+ '  typedef struct cdouble4 {real x; real y; real z; real w;} real4;'#10
+ '  typedef struct cdouble8 {real s0; real s1; real s2; real s3;'#10
+ '                           real s4; real s5; real s6; real s7;} real8;'#10
+ '  typedef struct cdouble16 {real s0; real s1; real s2; real s3;'#10
+ '                            real s4; real s5; real s6; real s7;'#10
+ '                            real s8; real s9; real sA; real sB;'#10
+ '                            real sC; real sD; real sE; real sF;} real16;'#10
+ '  #define ZERO 0.0'#10
+ '  #define ONE 1.0'#10
+ '  #define SMALLEST -1.0e37'#10
+ '#endif'#10
+ '// Single-element version of a complex number'#10
+ '#if PRECISION == 3232'#10
+ '  typedef float singlereal;'#10
+ '#elif PRECISION == 6464'#10
+ '  typedef double singlereal;'#10
+ '#else'#10
+ '  typedef real singlereal;'#10
+ '#endif'#10
+ '// Converts a ''real argument'' value to a ''real'' value as passed to the kernel. Normally there is no'#10
+ '// conversion, but half-precision is not supported as kernel argument so it is converted from float.'#10
+ '#if PRECISION == 16'#10
+ '  typedef float real_arg;'#10
+ '  #define GetRealArg(x) (half)x'#10
+ '#else'#10
+ '  typedef real real_arg;'#10
+ '  #define GetRealArg(x) x'#10
+ '#endif'#10
+ '// Pointers to local memory objects (using a define because CUDA doesn''t need them)'#10
+ '#ifndef LOCAL_PTR'#10
+ '  #define LOCAL_PTR __local'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Don''t use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific'#10
+ '// devices, this is enabled (see src/routine.cpp).'#10
+ '#ifndef USE_CL_MAD'#10
+ '  #define USE_CL_MAD 0'#10
+ '#endif'#10
+ '// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size '#10
+ '// requirement results in worse performance and is disabled (src/utilities/compile.cpp)'#10
+ '#ifndef RELAX_WORKGROUP_SIZE'#10
+ '  #define RELAX_WORKGROUP_SIZE 0'#10
+ '#endif'#10
+ '// Sets a variable to zero'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define SetToZero(a) a.x = ZERO; a.y = ZERO'#10
+ '#else'#10
+ '  #define SetToZero(a) a = ZERO'#10
+ '#endif'#10
+ '// Sets a variable to zero (only the imaginary part)'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define ImagToZero(a) a.y = ZERO'#10
+ '#else'#10
+ '  #define ImagToZero(a) '#10
+ '#endif'#10
+ '// Sets a variable to one'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define SetToOne(a) a.x = ONE; a.y = ZERO'#10
+ '#else'#10
+ '  #define SetToOne(a) a = ONE'#10
+ '#endif'#10
+ '// Determines whether a variable is zero'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))'#10
+ '#else'#10
+ '  #define IsZero(a) (a == ZERO)'#10
+ '#endif'#10
+ '// The absolute value (component-wise)'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)'#10
+ '#else'#10
+ '  #define AbsoluteValue(value) value = fabs(value)'#10
+ '#endif'#10
+ '// Negation (component-wise)'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define Negate(value) value.x = -(value.x); value.y = -(value.y)'#10
+ '#else'#10
+ '  #define Negate(value) value = -(value)'#10
+ '#endif'#10
+ '// Adds two complex variables'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y'#10
+ '#else'#10
+ '  #define Add(c,a,b) c = a + b'#10
+ '#endif'#10
+ '// Subtracts two complex variables'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y'#10
+ '#else'#10
+ '  #define Subtract(c,a,b) c = a - b'#10
+ '#endif'#10
+ '// Multiply two complex variables (used in the defines below)'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define MulReal(a,b) a.x*b.x - a.y*b.y'#10
+ '  #define MulImag(a,b) a.x*b.y + a.y*b.x'#10
+ '#endif'#10
+ '// The scalar multiply function'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)'#10
+ '#else'#10
+ '  #define Multiply(c,a,b) c = a * b'#10
+ '#endif'#10
+ '// The scalar multiply-add function'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)'#10
+ '#else'#10
+ '  #if USE_CL_MAD == 1'#10
+ '    #define MultiplyAdd(c,a,b) c = mad(a, b, c)'#10
+ '  #else'#10
+ '    #define MultiplyAdd(c,a,b) c += a * b'#10
+ '  #endif'#10
+ '#endif'#10
+ '// The scalar multiply-subtract function'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)'#10
+ '#else'#10
+ '  #define MultiplySubtract(c,a,b) c -= a * b'#10
+ '#endif'#10
+ '// The scalar division function: full division'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom'#10
+ '#else'#10
+ '  #define DivideFull(c,a,b) c = a / b'#10
+ '#endif'#10
+ '// The scalar AXPBY function'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)'#10
+ '#else'#10
+ '  #define AXPBY(e,a,b,c,d) e = a*b + c*d'#10
+ '#endif'#10
+ '// The complex conjugate operation for complex transforms'#10
+ '#if PRECISION == 3232 || PRECISION == 6464'#10
+ '  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y'#10
+ '#else'#10
+ '  #define COMPLEX_CONJUGATE(value) '#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Force inlining functions or not: some compilers don''t support the inline keyword'#10
+ '#ifdef USE_INLINE_KEYWORD'#10
+ '  #define INLINE_FUNC inline'#10
+ '#else'#10
+ '  #define INLINE_FUNC'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is'#10
+ '// enabled (see src/routine.cc).'#10
+ '#ifndef USE_STAGGERED_INDICES'#10
+ '  #define USE_STAGGERED_INDICES 0'#10
+ '#endif'#10
+ '// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula''s are taken from:'#10
+ '// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf'#10
+ '// More details: https://github.com/CNugteren/CLBlast/issues/53'#10
+ '#if USE_STAGGERED_INDICES == 1 && GEMMK == 0'#10
+ '  INLINE_FUNC int GetGroupIDFlat() {'#10
+ '    return get_group_id(0) + get_num_groups(0) * get_group_id(1);'#10
+ '  }'#10
+ '  INLINE_FUNC int GetGroupID1() {'#10
+ '    return (GetGroupIDFlat()) % get_num_groups(1);'#10
+ '  }'#10
+ '  INLINE_FUNC int GetGroupID0() {'#10
+ '    return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);'#10
+ '  }'#10
+ '#else'#10
+ '  INLINE_FUNC int GetGroupID1() { return get_group_id(1); }'#10
+ '  INLINE_FUNC int GetGroupID0() { return get_group_id(0); }'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
