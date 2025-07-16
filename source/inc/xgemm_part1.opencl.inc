''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains two optimized matrix-multiplication kernels:'#10
+ '// - Kernel 0: inspired by the paper by Matsumoto et al. and the tutorial on'#10
+ '//   http://www.cedricnugteren.nl/tutorial.php'#10
+ '// - Kernel 1: inspired by a Qualcomm optimized GPU kernel with 2D register tiling'#10
+ '//   https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel'#10
+ '// Both are fully configurable (and tunable!) using many parameters. Both kernels support'#10
+ '// different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.'#10
+ '//'#10
+ '// For kernel 0 matrices are accessed as follows:'#10
+ '// A: [k*M + m], with ''k'' ranging from 0:K and ''m'' from 0:M (m,k,m)'#10
+ '// B: [k*N + n], with ''k'' ranging from 0:K and ''n'' from 0:N (n,k,n)'#10
+ '// C: [n*M + m], with ''n'' ranging from 0:N and ''m'' from 0:M (m,n,m)'#10
+ '// For kernel 1, both A and C are transposed w.r.t. the above'#10
+ '//'#10
+ '// Or as an image (assuming column-major)'#10
+ '//       K                      '#10
+ '//    o-------o                 '#10
+ '//    |       |                 '#10
+ '//  N | [B^T] |                 '#10
+ '//    |       |                 '#10
+ '//    o-------o                 '#10
+ '//        K               N     '#10
+ '//    o-------o        o-----o  '#10
+ '//  M |  [A]  |      M | [C] |  '#10
+ '//    |       |        |     |  '#10
+ '//    o-------o        o-----o  '#10
+ '//                              '#10
+ '//'#10
+ '// This kernel is separated into multiple files. This is part 1 out of 4.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// Parameters set by the tuner or by the database. Here they are given a basic default value in case'#10
+ '// this kernel file is used outside of the CLBlast library.'#10
+ '#ifndef GEMMK'#10
+ '  #define GEMMK 0    // Kernel to choose: 0 regular, 1 with 2D register tiling'#10
+ '#endif'#10
+ '#ifndef MWG'#10
+ '  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)'#10
+ '#endif'#10
+ '#ifndef NWG'#10
+ '  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)'#10
+ '#endif'#10
+ '#ifndef KWG'#10
+ '  #define KWG 8      // Tile-size in dimension K (e.g. 8, 16)'#10
+ '#endif'#10
+ '#ifndef MDIMC'#10
+ '  #define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)'#10
+ '#endif'#10
+ '#ifndef NDIMC'#10
+ '  #define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)'#10
+ '#endif'#10
+ '#ifndef MDIMA'#10
+ '  #define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)'#10
+ '#endif'#10
+ '#ifndef NDIMB'#10
+ '  #define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)'#10
+ '#endif'#10
+ '#ifndef KWI'#10
+ '  #define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)'#10
+ '#endif'#10
+ '#ifndef VWM'#10
+ '  #define VWM 1      // Vector width of matrices A and C'#10
+ '#endif'#10
+ '#ifndef VWN'#10
+ '  #define VWN 1      // Vector width of matrix B'#10
+ '#endif'#10
+ '#ifndef STRM'#10
+ '  #define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)'#10
+ '#endif'#10
+ '#ifndef STRN'#10
+ '  #define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)'#10
+ '#endif'#10
+ '#ifndef SA'#10
+ '  #define SA 0       // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)'#10
+ '#endif'#10
+ '#ifndef SB'#10
+ '  #define SB 0       // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)'#10
+ '#endif'#10
+ '#ifndef KREG'#10
+ '  #define KREG 1     // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)'#10
+ '#endif'#10
+ '// Helper parameters based on the above tuning parameters'#10
+ '#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)'#10
+ '#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)'#10
+ '#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA'#10
+ '#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB'#10
+ '#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)'#10
+ '#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)'#10
+ '#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)'#10
+ '#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)'#10
+ '// Settings'#10
+ '#ifndef USE_VECTOR_MAD'#10
+ '  #define USE_VECTOR_MAD 0      // Unroll (0) or don''t (1) unroll the vector MAD manually'#10
+ '#endif'#10
+ '#ifndef GLOBAL_MEM_FENCE'#10
+ '  #define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance'#10
+ '#endif'#10
+ '#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA'#10
+ '  #define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0'#10
+ '#endif'#10
+ '#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA'#10
+ '  #define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0'#10
+ '#endif'#10
+ '#ifndef SUBGROUP_SHUFFLING_INTEL'#10
+ '  #define SUBGROUP_SHUFFLING_INTEL 0'#10
+ '#endif'#10
+ '#ifndef USE_SUBGROUP_SHUFFLING'#10
+ '  #define USE_SUBGROUP_SHUFFLING 0     // Optionally enables subgroup shuffling for Intel GPUs'#10
+ '#endif'#10
+ '// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)'#10
+ '#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1'#10
+ '  #pragma OPENCL EXTENSION cl_intel_subgroups: enable'#10
+ '  #define SUBGROUP_SIZE 8              // Assumes subgroup size is always 8 on Intel GPUs'#10
+ '#endif'#10
+ '// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)'#10
+ '#if USE_SUBGROUP_SHUFFLING == 1'#10
+ '  #if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1'#10
+ '    #define SUBGROUP_SIZE 32            // Assumes subgroup size is always 32 on NVIDIA GPUs'#10
+ '  #endif'#10
+ '#endif'#10
+ '#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE'#10
+ '  #undef USE_SUBGROUP_SHUFFLING'#10
+ '  #define USE_SUBGROUP_SHUFFLING 0     // Disables subgroups in case the assumptions don''t hold'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Data-widths in dimension M'#10
+ '#if VWM == 1'#10
+ '    typedef real realM;'#10
+ '#elif VWM == 2'#10
+ '    typedef real2 realM;'#10
+ '#elif VWM == 4'#10
+ '    typedef real4 realM;'#10
+ '#elif VWM == 8'#10
+ '    typedef real8 realM;'#10
+ '#elif VWM == 16'#10
+ '    typedef real16 realM;'#10
+ '#endif'#10
+ '// Data-widths in dimension N'#10
+ '#if VWN == 1'#10
+ '    typedef real realN;'#10
+ '#elif VWN == 2'#10
+ '    typedef real2 realN;'#10
+ '#elif VWN == 4'#10
+ '    typedef real4 realN;'#10
+ '#elif VWN == 8'#10
+ '    typedef real8 realN;'#10
+ '#elif VWN == 16'#10
+ '    typedef real16 realN;'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Initializes the accumulation registers to zero'#10
+ 'INLINE_FUNC realM InitAccRegisters() {'#10
+ '  realM result;'#10
+ '  #if VWM == 1'#10
+ '    SetToZero(result);'#10
+ '  #elif VWM == 2'#10
+ '    SetToZero(result.x);'#10
+ '    SetToZero(result.y);'#10
+ '  #elif VWM == 4'#10
+ '    SetToZero(result.x);'#10
+ '    SetToZero(result.y);'#10
+ '    SetToZero(result.z);'#10
+ '    SetToZero(result.w);'#10
+ '  #elif VWM == 8'#10
+ '    SetToZero(result.s0);'#10
+ '    SetToZero(result.s1);'#10
+ '    SetToZero(result.s2);'#10
+ '    SetToZero(result.s3);'#10
+ '    SetToZero(result.s4);'#10
+ '    SetToZero(result.s5);'#10
+ '    SetToZero(result.s6);'#10
+ '    SetToZero(result.s7);'#10
+ '  #elif VWM == 16'#10
+ '    SetToZero(result.s0);'#10
+ '    SetToZero(result.s1);'#10
+ '    SetToZero(result.s2);'#10
+ '    SetToZero(result.s3);'#10
+ '    SetToZero(result.s4);'#10
+ '    SetToZero(result.s5);'#10
+ '    SetToZero(result.s6);'#10
+ '    SetToZero(result.s7);'#10
+ '    SetToZero(result.s8);'#10
+ '    SetToZero(result.s9);'#10
+ '    SetToZero(result.sA);'#10
+ '    SetToZero(result.sB);'#10
+ '    SetToZero(result.sC);'#10
+ '    SetToZero(result.sD);'#10
+ '    SetToZero(result.sE);'#10
+ '    SetToZero(result.sF);'#10
+ '  #endif'#10
+ '  return result;'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for'#10
+ '// caching the A input matrix.'#10
+ '#if SA == 1'#10
+ 'INLINE_FUNC void GlobalToLocalA(const __global realM* restrict agm, LOCAL_PTR realM* alm,'#10
+ '                                const int kSizeM, const int tid, const int kwg) {'#10
+ '  const int la0 = tid % MDIMA;'#10
+ '  const int la1 = tid / MDIMA;'#10
+ '  #pragma unroll'#10
+ '  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _kia = 0; _kia < KWA; _kia += 1) {'#10
+ '      // Computes the indices based on strided/non-strided access'#10
+ '      #if STRM == 0'#10
+ '        int mg = _mia + la0*(MWA/VWM);'#10
+ '      #elif STRM == 1'#10
+ '        int mg = la0 + _mia*MDIMA;'#10
+ '      #endif'#10
+ '      // Computes the indices for the global memory'#10
+ '      int kg = _kia + la1*KWA;'#10
+ '      int idm = mg + GetGroupID0() * (MWG/VWM);'#10
+ '      int idk = kg + kwg;'#10
+ '      // Loads the data from global memory (not transposed) into the local memory'#10
+ '      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '#endif'#10
+ '// Same as above, but now for the B input matrix'#10
+ '#if SB == 1'#10
+ 'INLINE_FUNC void GlobalToLocalB(const __global realN* restrict bgm, LOCAL_PTR realN* blm,'#10
+ '                                const int kSizeN, const int tid, const int kwg) {'#10
+ '  const int lb0 = tid % NDIMB;'#10
+ '  const int lb1 = tid / NDIMB;'#10
+ '  #pragma unroll'#10
+ '  for (int _kib = 0; _kib < KWB; _kib += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {'#10
+ '      // Computes the indices based on strided/non-strided access'#10
+ '      #if STRN == 0'#10
+ '        int ng = _nib + lb0*(NWB/VWN);'#10
+ '      #elif STRN == 1'#10
+ '        int ng = lb0 + _nib*NDIMB;'#10
+ '      #endif'#10
+ '      // Computes the indices for the global memory'#10
+ '      int kg = _kib + lb1*KWB;'#10
+ '      int idn = ng + GetGroupID1() * (NWG/VWN);'#10
+ '      int idk = kg + kwg;'#10
+ '      // Loads the data from global memory (transposed) into the local memory'#10
+ '      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Caches global off-chip memory directly into per-thread private memory (registers). This function'#10
+ '// is specific for caching the A input matrix.'#10
+ '#if SA == 0 && GEMMK == 0'#10
+ 'INLINE_FUNC realM GlobalToPrivateA(const __global realM* restrict agm, const int _mi,'#10
+ '                                   const int kSizeM, const int idk, const int kwg) {'#10
+ '  // Computes the indices based on strided/non-strided access'#10
+ '  #if STRM == 0'#10
+ '    int mg = _mi + get_local_id(0)*(MWI/VWM);'#10
+ '  #elif STRM == 1'#10
+ '    int mg = get_local_id(0) + _mi*MDIMC;'#10
+ '  #endif'#10
+ '  // Computes the indices for the global memory'#10
+ '  int idm = mg + GetGroupID0() * (MWG/VWM);'#10
+ '  // Loads the data from global memory (not transposed) and stores into registers'#10
+ '  return agm[idk*(kSizeM/VWM) + idm];'#10
+ '}'#10
+ '#endif'#10
+ '// Same as above, but now for the B input matrix'#10
+ '#if SB == 0 && GEMMK == 0'#10
+ 'INLINE_FUNC realN GlobalToPrivateB(const __global realN* restrict bgm, const int _ni,'#10
+ '                                   const int kSizeN, const int idk) {'#10
+ '  // Computes the indices based on strided/non-strided access'#10
+ '  #if STRN == 0'#10
+ '    int ng = _ni + get_local_id(1)*(NWI/VWN);'#10
+ '  #elif STRN == 1'#10
+ '    int ng = get_local_id(1) + _ni*NDIMC;'#10
+ '  #endif'#10
+ '  // Computes the indices for the global memory'#10
+ '  int idn = ng + GetGroupID1() * (NWG/VWN);'#10
+ '  // Loads the data from global memory (transposed) and stores into registers'#10
+ '  return bgm[idk*(kSizeN/VWN) + idn];'#10
+ '}'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '#if GEMMK == 1'#10
+ '// Caches global off-chip memory directly into per-thread private memory (registers). This function'#10
+ '// is specific for caching the A input matrix for kernel 1.'#10
+ 'INLINE_FUNC realN GlobalToPrivateA2D(const __global real* restrict a_ptr, const int tid_y, const int _ni,'#10
+ '                                     const int kSizeK, const int idk, const int _ki) {'#10
+ '  #if PRECISION == 3232 || PRECISION == 6464'#10
+ '    const int a_index = (tid_y * NWI + _ni) * (kSizeK / VWN) + idk / VWN + _ki;'#10
+ '    const __global realN* restrict agm = (const __global realN* restrict) a_ptr;'#10
+ '    return agm[a_index];'#10
+ '  #else'#10
+ '    const int a_index = (tid_y * NWI + _ni) * kSizeK + idk + _ki * VWN;'#10
+ '    #if VWN == 1'#10
+ '      return a_ptr[a_index];'#10
+ '    #elif VWN == 2'#10
+ '      return vload2(0, a_ptr + a_index);'#10
+ '    #elif VWN == 4'#10
+ '      return vload4(0, a_ptr + a_index);'#10
+ '    #elif VWN == 8'#10
+ '      return vload8(0, a_ptr + a_index);'#10
+ '    #elif VWN == 16'#10
+ '      return vload16(0, a_ptr + a_index);'#10
+ '    #endif'#10
+ '  #endif'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC realM GlobalToPrivateB2D(const __global real* restrict b_ptr, const int tid_x, const int _mi,'#10
+ '                                     const int kSizeN, const int idk, const int _ki) {'#10
+ '  #if PRECISION == 3232 || PRECISION == 6464'#10
+ '    const int b_index = (idk + _ki) * (kSizeN / VWM) + tid_x * (MWI / VWM) + _mi;'#10
+ '    const __global realM* restrict bgm = (const __global realM* restrict) b_ptr;'#10
+ '    return bgm[b_index];'#10
+ '  #else'#10
+ '    const int b_index = (idk + _ki) * kSizeN + tid_x * MWI + _mi * VWM;'#10
+ '    #if VWM == 1'#10
+ '      return b_ptr[b_index];'#10
+ '    #elif VWM == 2'#10
+ '      return vload2(0, b_ptr + b_index);'#10
+ '    #elif VWM == 4'#10
+ '      return vload4(0, b_ptr + b_index);'#10
+ '    #elif VWM == 8'#10
+ '      return vload8(0, b_ptr + b_index);'#10
+ '    #elif VWM == 16'#10
+ '      return vload16(0, b_ptr + b_index);'#10
+ '    #endif'#10
+ '  #endif'#10
+ '}'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Caches on-chip local memory into per-thread private memory (registers). This function is specific'#10
+ '// for caching the A input matrix.'#10
+ '#if SA == 1'#10
+ 'INLINE_FUNC realM LocalToPrivateA(LOCAL_PTR realM* alm, const int _mi, const int kg) {'#10
+ '  #if STRM == 0'#10
+ '    int mg = _mi + get_local_id(0)*(MWI/VWM);'#10
+ '  #elif STRM == 1'#10
+ '    int mg = get_local_id(0) + _mi*MDIMC;'#10
+ '  #endif'#10
+ '  return alm[kg*(MWG/VWM) + mg];'#10
+ '}'#10
+ '#endif'#10
+ '// Same as above, but now for the B input matrix'#10
+ '#if SB == 1'#10
+ 'INLINE_FUNC realN LocalToPrivateB(LOCAL_PTR realN* blm, const int _ni, const int kg) {'#10
+ '  #if STRN == 0'#10
+ '    int ng = _ni + get_local_id(1)*(NWI/VWN);'#10
+ '  #elif STRN == 1'#10
+ '    int ng = get_local_id(1) + _ni*NDIMC;'#10
+ '  #endif'#10
+ '  return blm[kg*(NWG/VWN) + ng];'#10
+ '}'#10
+ '#endif'#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
