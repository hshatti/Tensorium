''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains the batched version of the non-direct GEMM kernel. See part 1 for information'#10
+ '// about the non-batched version of the kernel.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))'#10
+ '#endif'#10
+ 'void XgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                  const __constant real_arg* arg_alphas,'#10
+ '                  const __constant real_arg* arg_betas,'#10
+ '                  const __global realM* restrict agm, const int a_one, const int a_two,'#10
+ '                  const __global realN* restrict bgm, const int b_one, const int b_two,'#10
+ '                  __global realM* cgm, const int c_one, const int c_two) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const real alpha = GetRealArg(arg_alphas[batch]);'#10
+ '  const real beta = GetRealArg(arg_betas[batch]);'#10
+ '  // Sets the offsets'#10
+ '  const int a_offset = batch * a_one * a_two;'#10
+ '  const int b_offset = batch * b_one * b_two;'#10
+ '  const int c_offset = batch * c_one * c_two;'#10
+ '  const __global realM* restrict agm_ = &agm[a_offset / VWM];'#10
+ '  const __global realN* restrict bgm_ = &bgm[b_offset / VWN];'#10
+ '  __global realM* restrict cgm_ = &cgm[c_offset / VWM];'#10
+ '  // Allocates workgroup-private memory (local memory)'#10
+ '  #if SA == 1'#10
+ '    __local realM alm[KWG * MWG/VWM];'#10
+ '  #endif'#10
+ '  #if SB == 1'#10
+ '    __local realN blm[KWG * NWG/VWN];'#10
+ '  #endif'#10
+ '  // Computes the matrix-multiplication and stores the result in global memory'#10
+ '  #if SA == 1 && SB == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm, blm);'#10
+ '  #elif SA == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm);'#10
+ '  #elif SB == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, blm);'#10
+ '  #else'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta);'#10
+ '  #endif'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))'#10
+ '#endif'#10
+ 'void XgemmStridedBatched(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                         const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                         const __global realM* restrict agm, const int a_one, const int a_two,'#10
+ '                         const __global realN* restrict bgm, const int b_one, const int b_two,'#10
+ '                         __global realM* cgm, const int c_one, const int c_two) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  const real beta = GetRealArg(arg_beta);'#10
+ '  // Sets the offsets'#10
+ '  const int a_offset = batch * a_one * a_two;'#10
+ '  const int b_offset = batch * b_one * b_two;'#10
+ '  const int c_offset = batch * c_one * c_two;'#10
+ '  const __global realM* restrict agm_ = &agm[a_offset / VWM];'#10
+ '  const __global realN* restrict bgm_ = &bgm[b_offset / VWN];'#10
+ '  __global realM* restrict cgm_ = &cgm[c_offset / VWM];'#10
+ '  // Allocates workgroup-private memory (local memory)'#10
+ '  #if SA == 1'#10
+ '    __local realM alm[KWG * MWG/VWM];'#10
+ '  #endif'#10
+ '  #if SB == 1'#10
+ '    __local realN blm[KWG * NWG/VWN];'#10
+ '  #endif'#10
+ '  // Computes the matrix-multiplication and stores the result in global memory'#10
+ '  #if SA == 1 && SB == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm, blm);'#10
+ '  #elif SA == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, alm);'#10
+ '  #elif SB == 1'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta, blm);'#10
+ '  #else'#10
+ '    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alpha, beta);'#10
+ '  #endif'#10
+ '}'#10
