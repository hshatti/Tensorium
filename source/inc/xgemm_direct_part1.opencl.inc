''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This is a generic GEMM kernel that works for all sizes and configurations: it doesn''t require any'#10
+ '// pre and and post-processing kernels.'#10
+ '//'#10
+ '// This kernel is seperated into three files. This is part 1 out of 3.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// Parameters set by the tuner or by the database. Here they are given a basic default value in case'#10
+ '// this kernel file is used outside of the CLBlast library. Note that all parameters here have a'#10
+ '// suffix ''D'' to denote that they are for the ''direct'' version of the GEMM kernel.'#10
+ '#ifndef WGD'#10
+ '  #define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)'#10
+ '#endif'#10
+ '#ifndef MDIMCD'#10
+ '  #define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)'#10
+ '#endif'#10
+ '#ifndef NDIMCD'#10
+ '  #define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)'#10
+ '#endif'#10
+ '#ifndef MDIMAD'#10
+ '  #define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD'#10
+ '#endif'#10
+ '#ifndef NDIMBD'#10
+ '  #define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD'#10
+ '#endif'#10
+ '#ifndef KWID'#10
+ '  #define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)'#10
+ '#endif'#10
+ '#ifndef VWMD'#10
+ '  #define VWMD 1      // Vector width of matrices A and C'#10
+ '#endif'#10
+ '#ifndef VWND'#10
+ '  #define VWND 1      // Vector width of matrix B'#10
+ '#endif'#10
+ '#ifndef PADA'#10
+ '  #define PADA 1      // Local memory padding for matrix A'#10
+ '#endif'#10
+ '#ifndef PADB'#10
+ '  #define PADB 1      // Local memory padding for matrix B'#10
+ '#endif'#10
+ '// Helper parameters based on the above tuning parameters'#10
+ '#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)'#10
+ '#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)'#10
+ '#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD'#10
+ '#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD'#10
+ '#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)'#10
+ '#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)'#10
+ '#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)'#10
+ '#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)'#10
+ '// ================================================================================================='#10
+ '// Data-widths in dimension M'#10
+ '#if VWMD == 1'#10
+ '    typedef real realMD;'#10
+ '#elif VWMD == 2'#10
+ '    typedef real2 realMD;'#10
+ '#elif VWMD == 4'#10
+ '    typedef real4 realMD;'#10
+ '#elif VWMD == 8'#10
+ '    typedef real8 realMD;'#10
+ '#elif VWMD == 16'#10
+ '    typedef real16 realMD;'#10
+ '#endif'#10
+ '// Data-widths in dimension N'#10
+ '#if VWND == 1'#10
+ '    typedef real realND;'#10
+ '#elif VWND == 2'#10
+ '    typedef real2 realND;'#10
+ '#elif VWND == 4'#10
+ '    typedef real4 realND;'#10
+ '#elif VWND == 8'#10
+ '    typedef real8 realND;'#10
+ '#elif VWND == 16'#10
+ '    typedef real16 realND;'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Loads global off-chip memory into thread-private register files. This function is specific for'#10
+ '// loading the A input matrix.'#10
+ 'INLINE_FUNC real GlobalToPrivateDirectA(const __global real* restrict agms, const int _mi,'#10
+ '                                        const int a_ld, const int a_offset, const int idm, const int idk,'#10
+ '                                        const int a_transpose, const int a_conjugate) {'#10
+ '  const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);'#10
+ '  real result = agms[a_index + a_offset];'#10
+ '  if (a_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '  return result;'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC real GlobalToPrivateDirectB(const __global real* restrict bgms, const int _ni,'#10
+ '                                        const int b_ld, const int b_offset, const int idn, const int idk,'#10
+ '                                        const int b_transpose, const int b_conjugate) {'#10
+ '  const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);'#10
+ '  real result = bgms[b_index + b_offset];'#10
+ '  if (b_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '  return result;'#10
+ '}'#10
+ '// Loads global off-chip memory into thread-private register files. This function is specific for'#10
+ '// loading the A input matrix. This is the same as above but now includes a bounds check.'#10
+ 'INLINE_FUNC real GlobalToPrivateCheckedA(const __global real* restrict agms, const int _mi,'#10
+ '                                         const int a_ld, const int a_offset, const int idm, const int idk,'#10
+ '                                         const int a_transpose, const int a_conjugate,'#10
+ '                                         const int kSizeM) {'#10
+ '  real result;'#10
+ '  if (idm + _mi < kSizeM) {'#10
+ '    const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);'#10
+ '    result = agms[a_index + a_offset];'#10
+ '    if (a_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '  }'#10
+ '  else {'#10
+ '    SetToZero(result);'#10
+ '  }'#10
+ '  return result;'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC real GlobalToPrivateCheckedB(const __global real* restrict bgms, const int _ni,'#10
+ '                                         const int b_ld, const int b_offset, const int idn, const int idk,'#10
+ '                                         const int b_transpose, const int b_conjugate,'#10
+ '                                         const int kSizeN) {'#10
+ '  real result;'#10
+ '  if (idn + _ni < kSizeN) {'#10
+ '    const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);'#10
+ '    result = bgms[b_index + b_offset];'#10
+ '    if (b_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '  }'#10
+ '  else {'#10
+ '    SetToZero(result);'#10
+ '  }'#10
+ '  return result;'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Caches on-chip local memory into per-thread private memory (registers). This function is specific'#10
+ '// for caching the A input matrix.'#10
+ 'INLINE_FUNC real LocalToPrivateDirectA(LOCAL_PTR real* alm, const int _mi, const int kg,'#10
+ '                                       const int a_transpose) {'#10
+ '  const int mg = _mi + get_local_id(0)*MWID;'#10
+ '  const int index = (a_transpose) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg;'#10
+ '  return alm[index];'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC real LocalToPrivateDirectB(LOCAL_PTR real* blm, const int _ni, const int kg,'#10
+ '                                       const int b_transpose) {'#10
+ '  const int ng = _ni + get_local_id(1)*NWID;'#10
+ '  const int index = (b_transpose) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng;'#10
+ '  return blm[index];'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication'#10
+ '// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm'#10
+ 'INLINE_FUNC void StoreResultsDirect(__global real* cgm, const real c_value,'#10
+ '                                    const int _mi, const int _ni, const int idm, const int idn,'#10
+ '                                    const real alpha, const real beta,'#10
+ '                                    const int c_ld, const int c_offset, const int c_transpose) {'#10
+ '  // Determines the destination index'#10
+ '  int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);'#10
+ '  // The final multiplication with alpha (in case beta == 0)'#10
+ '  real result;'#10
+ '  if (IsZero(beta)) {'#10
+ '    Multiply(result, alpha, c_value);'#10
+ '  }'#10
+ '  // The final multiplication with alpha and the addition with beta*C'#10
+ '  else {'#10
+ '    AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);'#10
+ '  }'#10
+ '  cgm[c_index + c_offset] = result;'#10
+ '}'#10
+ '// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication'#10
+ '// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm'#10
+ 'INLINE_FUNC void StoreResultsChecked(__global real* cgm, const real c_value,'#10
+ '                                     const int _mi, const int _ni, const int idm, const int idn,'#10
+ '                                     const int kSizeM, const int kSizeN,'#10
+ '                                     const real alpha, const real beta,'#10
+ '                                     const int c_ld, const int c_offset, const int c_transpose) {'#10
+ '  if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) {'#10
+ '    // Deter_mines the destination index'#10
+ '    int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);'#10
+ '    // The final multiplication with alpha (in case beta == 0)'#10
+ '    real result;'#10
+ '    if (IsZero(beta)) {'#10
+ '      Multiply(result, alpha, c_value);'#10
+ '    }'#10
+ '    // The final multiplication with alpha and the addition with beta*C'#10
+ '    else {'#10
+ '      AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);'#10
+ '    }'#10
+ '    cgm[c_index + c_offset] = result;'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
