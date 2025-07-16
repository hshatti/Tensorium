''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains the common functions and parameters specific for level 3 BLAS kernels.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Parameters set by the tuner or by the database. Here they are given a basic default value in case'#10
+ '// this kernel file is used outside of the CLBlast library.'#10
+ '// For the ''fast'' copy kernel'#10
+ '#ifndef COPY_DIMX'#10
+ '  #define COPY_DIMX 8      // Local workgroup size in the first dimension (x)'#10
+ '#endif'#10
+ '#ifndef COPY_DIMY'#10
+ '  #define COPY_DIMY 8      // Local workgroup size in the second dimension (y)'#10
+ '#endif'#10
+ '#ifndef COPY_WPT'#10
+ '  #define COPY_WPT 1       // Work per thread in the first dimension (x)'#10
+ '#endif'#10
+ '#ifndef COPY_VW'#10
+ '  #define COPY_VW 1        // Vector width in the second dimension (y)'#10
+ '#endif'#10
+ '// For the padding/copy kernels and the conversion kernels'#10
+ '#ifndef PAD_DIMX'#10
+ '  #define PAD_DIMX 8      // Local workgroup size in the first dimension (x)'#10
+ '#endif'#10
+ '#ifndef PAD_DIMY'#10
+ '  #define PAD_DIMY 8      // Local workgroup size in the second dimension (y)'#10
+ '#endif'#10
+ '#ifndef PAD_WPTX'#10
+ '  #define PAD_WPTX 1      // Work per thread in the first dimension (x)'#10
+ '#endif'#10
+ '#ifndef PAD_WPTY'#10
+ '  #define PAD_WPTY 1      // Work per thread in the second dimension (y)'#10
+ '#endif'#10
+ '// For the ''fast'' transpose kernel'#10
+ '#ifndef TRA_DIM'#10
+ '  #define TRA_DIM 8       // Number of local threads in the two dimensions (x,y)'#10
+ '#endif'#10
+ '#ifndef TRA_WPT'#10
+ '  #define TRA_WPT 1       // Work per thread in one dimension and vector-width in the other'#10
+ '#endif'#10
+ '#ifndef TRA_PAD'#10
+ '  #define TRA_PAD 0       // Padding of the local memory to avoid bank-conflicts'#10
+ '#endif'#10
+ '#ifndef TRA_SHUFFLE'#10
+ '  #define TRA_SHUFFLE 0   // Shuffling of the global indices to avoid global memory bank-conflicts'#10
+ '#endif'#10
+ '// For the padding/transpose kernels'#10
+ '#ifndef PADTRA_TILE'#10
+ '  #define PADTRA_TILE 8   // Number of local threads in the two dimensions (x,y)'#10
+ '#endif'#10
+ '#ifndef PADTRA_WPT'#10
+ '  #define PADTRA_WPT 1    // Amount of work per thread'#10
+ '#endif'#10
+ '#ifndef PADTRA_PAD'#10
+ '  #define PADTRA_PAD 0    // Padding of the local memory to avoid bank-conflicts'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '#if defined(ROUTINE_INVERT) || defined(ROUTINE_TRSM)'#10
+ '__kernel'#10
+ 'void FillMatrix(const int m, const int n, const int ld, const int offset,'#10
+ '                __global real* restrict dest, const real_arg arg_value) {'#10
+ '  const real value = GetRealArg(arg_value);'#10
+ '  const int id_one = get_global_id(0);'#10
+ '  const int id_two = get_global_id(1);'#10
+ '  if (id_one < m && id_two < n) {'#10
+ '    dest[id_two*ld + id_one + offset] = value;'#10
+ '  }'#10
+ '}'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
