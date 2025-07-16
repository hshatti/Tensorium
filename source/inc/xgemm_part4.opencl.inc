''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// The upper-triangular and lower-triangular kernels are only used in special cases'#10
+ '#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)'#10
+ '// Main entry point of the kernel. This is the upper-triangular version.'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))'#10
+ '#endif'#10
+ 'void XgemmUpper(const int kSizeN, const int kSizeK,'#10
+ '                const real_arg arg_alpha,'#10
+ '                const real_arg arg_beta,'#10
+ '                const __global realM* restrict agm,'#10
+ '                const __global realN* restrict bgm,'#10
+ '                __global realM* cgm) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  const real beta = GetRealArg(arg_beta);'#10
+ '  // Skip these threads if they do not contain threads contributing to the upper-triangle'#10
+ '  if ((GetGroupID1() + 1)*NWG < GetGroupID0()*MWG) {'#10
+ '    return;'#10
+ '  }'#10
+ '  // Allocates workgroup-private memory (local memory)'#10
+ '  #if SA == 1'#10
+ '    __local realM alm[KWG * MWG/VWM];'#10
+ '  #endif'#10
+ '  #if SB == 1'#10
+ '    __local realN blm[KWG * NWG/VWN];'#10
+ '  #endif'#10
+ '  // Computes the matrix-multiplication and stores the result in global memory'#10
+ '  #if SA == 1 && SB == 1'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);'#10
+ '  #elif SA == 1'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);'#10
+ '  #elif SB == 1'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);'#10
+ '  #else'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);'#10
+ '  #endif'#10
+ '}'#10
+ '// Main entry point of the kernel. This is the lower-triangular version.'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))'#10
+ '#endif'#10
+ 'void XgemmLower(const int kSizeN, const int kSizeK,'#10
+ '                const real_arg arg_alpha,'#10
+ '                const real_arg arg_beta,'#10
+ '                const __global realM* restrict agm,'#10
+ '                const __global realN* restrict bgm,'#10
+ '                __global realM* cgm) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  const real beta = GetRealArg(arg_beta);'#10
+ '  // Skip these threads if they do not contain threads contributing to the lower-triangle'#10
+ '  if (GetGroupID1()*NWG > (GetGroupID0() + 1)*MWG) {'#10
+ '    return;'#10
+ '  }'#10
+ '  // Allocates workgroup-private memory (local memory)'#10
+ '  #if SA == 1'#10
+ '    __local realM alm[KWG * MWG/VWM];'#10
+ '  #endif'#10
+ '  #if SB == 1'#10
+ '    __local realN blm[KWG * NWG/VWN];'#10
+ '  #endif'#10
+ '  // Computes the matrix-multiplication and stores the result in global memory'#10
+ '  #if SA == 1 && SB == 1'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);'#10
+ '  #elif SA == 1'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);'#10
+ '  #elif SB == 1'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);'#10
+ '  #else'#10
+ '    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);'#10
+ '  #endif'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// If not using a triangular version, include the regular kernel'#10
+ '#else'#10
+ '// Main entry point of the kernel. This is the regular full version.'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))'#10
+ '#endif'#10
+ 'void Xgemm(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '           const real_arg arg_alpha,'#10
+ '           const real_arg arg_beta,'#10
+ '           const __global realM* restrict agm,'#10
+ '           const __global realN* restrict bgm,'#10
+ '           __global realM* cgm,'#10
+ '           const int b_offset, const int c_offset) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  const real beta = GetRealArg(arg_beta);'#10
+ '  // Adds the offsets (in case of use of a single temporary buffer for A, B, and C)'#10
+ '  bgm = &bgm[b_offset];'#10
+ '  cgm = &cgm[c_offset];'#10
+ '  // Allocates workgroup-private memory (local memory)'#10
+ '  #if SA == 1'#10
+ '    __local realM alm[KWG * MWG/VWM];'#10
+ '  #endif'#10
+ '  #if SB == 1'#10
+ '    __local realN blm[KWG * NWG/VWN];'#10
+ '  #endif'#10
+ '  // Computes the matrix-multiplication and stores the result in global memory'#10
+ '  #if SA == 1 && SB == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);'#10
+ '  #elif SA == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);'#10
+ '  #elif SB == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);'#10
+ '  #else'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);'#10
+ '  #endif'#10
+ '}'#10
+ '#endif'#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
