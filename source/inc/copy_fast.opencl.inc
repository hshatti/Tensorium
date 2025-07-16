''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains the common kernels shared among different BLAS routines. This file contains'#10
+ '// kernels to copy matrices.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Data-widths'#10
+ '#if COPY_VW == 1'#10
+ '  typedef real realC;'#10
+ '#elif COPY_VW == 2'#10
+ '  typedef real2 realC;'#10
+ '#elif COPY_VW == 4'#10
+ '  typedef real4 realC;'#10
+ '#elif COPY_VW == 8'#10
+ '  typedef real8 realC;'#10
+ '#elif COPY_VW == 16'#10
+ '  typedef real16 realC;'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Fast copy kernel. Requires ''ld'' and the number of threads in dimension 0 to be a multiple of'#10
+ '// COPY_VW. Also requires both matrices to be of the same dimensions and without offset.'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyMatrixFast(const int ld,'#10
+ '                    __global const realC* restrict src,'#10
+ '                    __global realC* dest,'#10
+ '                    const real_arg arg_alpha) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < COPY_WPT; _w_one += 1) {'#10
+ '    const int id_one = get_global_id(0);'#10
+ '    const int id_two = (get_group_id(1)*COPY_WPT + _w_one) * COPY_DIMY + get_local_id(1);'#10
+ '    const int id = id_two*(ld/COPY_VW) + id_one;'#10
+ '    realC result;'#10
+ '    #if COPY_VW == 1'#10
+ '      Multiply(result, alpha, src[id]);'#10
+ '    #elif COPY_VW == 2'#10
+ '      Multiply(result.x, alpha, src[id].x);'#10
+ '      Multiply(result.y, alpha, src[id].y);'#10
+ '    #elif COPY_VW == 4'#10
+ '      Multiply(result.x, alpha, src[id].x);'#10
+ '      Multiply(result.y, alpha, src[id].y);'#10
+ '      Multiply(result.z, alpha, src[id].z);'#10
+ '      Multiply(result.w, alpha, src[id].w);'#10
+ '    #elif COPY_VW == 8'#10
+ '      Multiply(result.s0, alpha, src[id].s0);'#10
+ '      Multiply(result.s1, alpha, src[id].s1);'#10
+ '      Multiply(result.s2, alpha, src[id].s2);'#10
+ '      Multiply(result.s3, alpha, src[id].s3);'#10
+ '      Multiply(result.s4, alpha, src[id].s4);'#10
+ '      Multiply(result.s5, alpha, src[id].s5);'#10
+ '      Multiply(result.s6, alpha, src[id].s6);'#10
+ '      Multiply(result.s7, alpha, src[id].s7);'#10
+ '    #elif COPY_VW == 16'#10
+ '      Multiply(result.s0, alpha, src[id].s0);'#10
+ '      Multiply(result.s1, alpha, src[id].s1);'#10
+ '      Multiply(result.s2, alpha, src[id].s2);'#10
+ '      Multiply(result.s3, alpha, src[id].s3);'#10
+ '      Multiply(result.s4, alpha, src[id].s4);'#10
+ '      Multiply(result.s5, alpha, src[id].s5);'#10
+ '      Multiply(result.s6, alpha, src[id].s6);'#10
+ '      Multiply(result.s7, alpha, src[id].s7);'#10
+ '      Multiply(result.s8, alpha, src[id].s8);'#10
+ '      Multiply(result.s9, alpha, src[id].s9);'#10
+ '      Multiply(result.sA, alpha, src[id].sA);'#10
+ '      Multiply(result.sB, alpha, src[id].sB);'#10
+ '      Multiply(result.sC, alpha, src[id].sC);'#10
+ '      Multiply(result.sD, alpha, src[id].sD);'#10
+ '      Multiply(result.sE, alpha, src[id].sE);'#10
+ '      Multiply(result.sF, alpha, src[id].sF);'#10
+ '    #endif'#10
+ '    dest[id] = result;;'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
