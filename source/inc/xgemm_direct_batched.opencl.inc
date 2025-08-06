''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains the batched version of the direct GEMM kernels. See part 1 for information'#10
+ '// about the non-batched version of the kernel.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, non-transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectBatchedNN(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,'#10
+ '                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,'#10
+ '                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,'#10
+ '                          __global real* cgm, const __constant int* c_offsets, const int c_ld,'#10
+ '                          const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const real_arg arg_alpha = arg_alphas[batch];'#10
+ '  const real_arg arg_beta = arg_betas[batch];'#10
+ '  const int a_offset = a_offsets[batch];'#10
+ '  const int b_offset = b_offsets[batch];'#10
+ '  const int c_offset = c_offsets[batch];'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectBatchedNT(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,'#10
+ '                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,'#10
+ '                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,'#10
+ '                          __global real* cgm, const __constant int* c_offsets, const int c_ld,'#10
+ '                          const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const real_arg arg_alpha = arg_alphas[batch];'#10
+ '  const real_arg arg_beta = arg_betas[batch];'#10
+ '  const int a_offset = a_offsets[batch];'#10
+ '  const int b_offset = b_offsets[batch];'#10
+ '  const int c_offset = c_offsets[batch];'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the batched GEMM kernel with [A, B] = [transposed, non-transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectBatchedTN(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,'#10
+ '                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,'#10
+ '                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,'#10
+ '                          __global real* cgm, const __constant int* c_offsets, const int c_ld,'#10
+ '                          const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const real_arg arg_alpha = arg_alphas[batch];'#10
+ '  const real_arg arg_beta = arg_betas[batch];'#10
+ '  const int a_offset = a_offsets[batch];'#10
+ '  const int b_offset = b_offsets[batch];'#10
+ '  const int c_offset = c_offsets[batch];'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the batched GEMM kernel with [A, B] = [transposed, transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectBatchedTT(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,'#10
+ '                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,'#10
+ '                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,'#10
+ '                          __global real* cgm, const __constant int* c_offsets, const int c_ld,'#10
+ '                          const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const real_arg arg_alpha = arg_alphas[batch];'#10
+ '  const real_arg arg_beta = arg_betas[batch];'#10
+ '  const int a_offset = a_offsets[batch];'#10
+ '  const int b_offset = b_offsets[batch];'#10
+ '  const int c_offset = c_offsets[batch];'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, non-transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectStridedBatchedNN(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                                 const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,'#10
+ '                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,'#10
+ '                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,'#10
+ '                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int a_offset_batch = a_offset + a_stride * batch;'#10
+ '  const int b_offset_batch = b_offset + b_stride * batch;'#10
+ '  const int c_offset_batch = c_offset + c_stride * batch;'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,'#10
+ '              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectStridedBatchedNT(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                                 const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,'#10
+ '                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,'#10
+ '                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,'#10
+ '                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int a_offset_batch = a_offset + a_stride * batch;'#10
+ '  const int b_offset_batch = b_offset + b_stride * batch;'#10
+ '  const int c_offset_batch = c_offset + c_stride * batch;'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,'#10
+ '              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, non-transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectStridedBatchedTN(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                                 const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,'#10
+ '                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,'#10
+ '                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,'#10
+ '                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int a_offset_batch = a_offset + a_stride * batch;'#10
+ '  const int b_offset_batch = b_offset + b_stride * batch;'#10
+ '  const int c_offset_batch = c_offset + c_stride * batch;'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,'#10
+ '              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectStridedBatchedTT(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                                 const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,'#10
+ '                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,'#10
+ '                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,'#10
+ '                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int a_offset_batch = a_offset + a_stride * batch;'#10
+ '  const int b_offset_batch = b_offset + b_stride * batch;'#10
+ '  const int c_offset_batch = c_offset + c_stride * batch;'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,'#10
+ '              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
