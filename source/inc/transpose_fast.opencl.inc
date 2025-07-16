''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains the common kernels shared among different BLAS functions. This file contains'#10
+ '// a kernel to transpose matrices. This is a ''fast'' version with restrictions, see the'#10
+ '// ''padtranspose.opencl'' file for a general transpose kernel.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Data-widths'#10
+ '#if TRA_WPT == 1'#10
+ '  typedef real realT;'#10
+ '#elif TRA_WPT == 2'#10
+ '  typedef real2 realT;'#10
+ '#elif TRA_WPT == 4'#10
+ '  typedef real4 realT;'#10
+ '#elif TRA_WPT == 8'#10
+ '  typedef real8 realT;'#10
+ '#elif TRA_WPT == 16'#10
+ '  typedef real16 realT;'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// Transposes and copies a matrix. Requires both matrices to be of the same dimensions and without'#10
+ '// offset. A more general version is available in ''padtranspose.opencl''.'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(TRA_DIM, TRA_DIM, 1)))'#10
+ '#endif'#10
+ 'void TransposeMatrixFast(const int ld,'#10
+ '                         __global const realT* restrict src,'#10
+ '                         __global realT* dest,'#10
+ '                         const real_arg arg_alpha) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  // Sets the group identifiers. They might be ''shuffled'' around to distribute work in a different'#10
+ '  // way over workgroups, breaking memory-bank dependencies.'#10
+ '  const int gid0 = get_group_id(0);'#10
+ '  #if TRA_SHUFFLE == 1'#10
+ '    const int gid1 = (get_group_id(0) + get_group_id(1)) % get_num_groups(0);'#10
+ '  #else'#10
+ '    const int gid1 = get_group_id(1);'#10
+ '  #endif'#10
+ '  // Local memory to store a tile of the matrix (for coalescing)'#10
+ '  __local realT tile[TRA_WPT*TRA_DIM][TRA_DIM + TRA_PAD];'#10
+ '  // Loops over the work per thread'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {'#10
+ '    // Computes the identifiers for the source matrix. Note that the local and global dimensions'#10
+ '    // do not correspond to each other!'#10
+ '    const int id_one = gid1 * TRA_DIM + get_local_id(0);'#10
+ '    const int id_two = (gid0 * TRA_DIM + get_local_id(1))*TRA_WPT + _w_one;'#10
+ '    // Loads data into the local memory'#10
+ '    realT value = src[id_two*(ld/TRA_WPT) + id_one];'#10
+ '    tile[get_local_id(0)*TRA_WPT + _w_one][get_local_id(1)] = value;'#10
+ '  }'#10
+ '  // Synchronizes all threads in a workgroup'#10
+ '  barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '  // Loads transposed data from the local memory'#10
+ '  #pragma promote_to_registers'#10
+ '  realT vpm[TRA_WPT];'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {'#10
+ '    vpm[_w_one] = tile[get_local_id(1)*TRA_WPT + _w_one][get_local_id(0)];'#10
+ '  }'#10
+ '  // Performs the register-level transpose of the vectorized data'#10
+ '  #pragma promote_to_registers'#10
+ '  realT results[TRA_WPT];'#10
+ '  #if TRA_WPT == 1'#10
+ '    results[0] = vpm[0];'#10
+ '  #elif TRA_WPT == 2'#10
+ '    results[0].x = vpm[0].x; results[0].y = vpm[1].x;'#10
+ '    results[1].x = vpm[0].y; results[1].y = vpm[1].y;'#10
+ '  #elif TRA_WPT == 4'#10
+ '    results[0].x = vpm[0].x; results[0].y = vpm[1].x; results[0].z = vpm[2].x; results[0].w = vpm[3].x;'#10
+ '    results[1].x = vpm[0].y; results[1].y = vpm[1].y; results[1].z = vpm[2].y; results[1].w = vpm[3].y;'#10
+ '    results[2].x = vpm[0].z; results[2].y = vpm[1].z; results[2].z = vpm[2].z; results[2].w = vpm[3].z;'#10
+ '    results[3].x = vpm[0].w; results[3].y = vpm[1].w; results[3].z = vpm[2].w; results[3].w = vpm[3].w;'#10
+ '  #elif TRA_WPT == 8'#10
+ '    results[0].s0 = vpm[0].s0; results[0].s1 = vpm[1].s0; results[0].s2 = vpm[2].s0; results[0].s3 = vpm[3].s0; results[0].s4 = vpm[4].s0; results[0].s5 = vpm[5].s0; results[0].s6 = vpm[6].s0; results[0].s7 = vpm[7].s0;'#10
+ '    results[1].s0 = vpm[0].s1; results[1].s1 = vpm[1].s1; results[1].s2 = vpm[2].s1; results[1].s3 = vpm[3].s1; results[1].s4 = vpm[4].s1; results[1].s5 = vpm[5].s1; results[1].s6 = vpm[6].s1; results[1].s7 = vpm[7].s1;'#10
+ '    results[2].s0 = vpm[0].s2; results[2].s1 = vpm[1].s2; results[2].s2 = vpm[2].s2; results[2].s3 = vpm[3].s2; results[2].s4 = vpm[4].s2; results[2].s5 = vpm[5].s2; results[2].s6 = vpm[6].s2; results[2].s7 = vpm[7].s2;'#10
+ '    results[3].s0 = vpm[0].s3; results[3].s1 = vpm[1].s3; results[3].s2 = vpm[2].s3; results[3].s3 = vpm[3].s3; results[3].s4 = vpm[4].s3; results[3].s5 = vpm[5].s3; results[3].s6 = vpm[6].s3; results[3].s7 = vpm[7].s3;'#10
+ '    results[4].s0 = vpm[0].s4; results[4].s1 = vpm[1].s4; results[4].s2 = vpm[2].s4; results[4].s3 = vpm[3].s4; results[4].s4 = vpm[4].s4; results[4].s5 = vpm[5].s4; results[4].s6 = vpm[6].s4; results[4].s7 = vpm[7].s4;'#10
+ '    results[5].s0 = vpm[0].s5; results[5].s1 = vpm[1].s5; results[5].s2 = vpm[2].s5; results[5].s3 = vpm[3].s5; results[5].s4 = vpm[4].s5; results[5].s5 = vpm[5].s5; results[5].s6 = vpm[6].s5; results[5].s7 = vpm[7].s5;'#10
+ '    results[6].s0 = vpm[0].s6; results[6].s1 = vpm[1].s6; results[6].s2 = vpm[2].s6; results[6].s3 = vpm[3].s6; results[6].s4 = vpm[4].s6; results[6].s5 = vpm[5].s6; results[6].s6 = vpm[6].s6; results[6].s7 = vpm[7].s6;'#10
+ '    results[7].s0 = vpm[0].s7; results[7].s1 = vpm[1].s7; results[7].s2 = vpm[2].s7; results[7].s3 = vpm[3].s7; results[7].s4 = vpm[4].s7; results[7].s5 = vpm[5].s7; results[7].s6 = vpm[6].s7; results[7].s7 = vpm[7].s7;'#10
+ '  #elif TRA_WPT == 16'#10
+ '    results[ 0].s0 = vpm[0].s0; results[ 0].s1 = vpm[1].s0; results[ 0].s2 = vpm[2].s0; results[ 0].s3 = vpm[3].s0; results[ 0].s4 = vpm[4].s0; results[ 0].s5 = vpm[5].s0; results[ 0].s6 = vpm[6].s0; results[ 0].s7 = vpm[7].s0; results[ 0].s8 = vpm[8].s0; results[ 0].s9 = vpm[9].s0; results[ 0].sA = vpm[10].s0; results[ 0].sB = vpm[11].s0; results[ 0].sC = vpm[12].s0; results[ 0].sD = vpm[13].s0; results[ 0].sE = vpm[14].s0; results[ 0].sF = vpm[15].s0;'#10
+ '    results[ 1].s0 = vpm[0].s1; results[ 1].s1 = vpm[1].s1; results[ 1].s2 = vpm[2].s1; results[ 1].s3 = vpm[3].s1; results[ 1].s4 = vpm[4].s1; results[ 1].s5 = vpm[5].s1; results[ 1].s6 = vpm[6].s1; results[ 1].s7 = vpm[7].s1; results[ 1].s8 = vpm[8].s1; results[ 1].s9 = vpm[9].s1; results[ 1].sA = vpm[10].s1; results[ 1].sB = vpm[11].s1; results[ 1].sC = vpm[12].s1; results[ 1].sD = vpm[13].s1; results[ 1].sE = vpm[14].s1; results[ 1].sF = vpm[15].s1;'#10
+ '    results[ 2].s0 = vpm[0].s2; results[ 2].s1 = vpm[1].s2; results[ 2].s2 = vpm[2].s2; results[ 2].s3 = vpm[3].s2; results[ 2].s4 = vpm[4].s2; results[ 2].s5 = vpm[5].s2; results[ 2].s6 = vpm[6].s2; results[ 2].s7 = vpm[7].s2; results[ 2].s8 = vpm[8].s2; results[ 2].s9 = vpm[9].s2; results[ 2].sA = vpm[10].s2; results[ 2].sB = vpm[11].s2; results[ 2].sC = vpm[12].s2; results[ 2].sD = vpm[13].s2; results[ 2].sE = vpm[14].s2; results[ 2].sF = vpm[15].s2;'#10
+ '    results[ 3].s0 = vpm[0].s3; results[ 3].s1 = vpm[1].s3; results[ 3].s2 = vpm[2].s3; results[ 3].s3 = vpm[3].s3; results[ 3].s4 = vpm[4].s3; results[ 3].s5 = vpm[5].s3; results[ 3].s6 = vpm[6].s3; results[ 3].s7 = vpm[7].s3; results[ 3].s8 = vpm[8].s3; results[ 3].s9 = vpm[9].s3; results[ 3].sA = vpm[10].s3; results[ 3].sB = vpm[11].s3; results[ 3].sC = vpm[12].s3; results[ 3].sD = vpm[13].s3; results[ 3].sE = vpm[14].s3; results[ 3].sF = vpm[15].s3;'#10
+ '    results[ 4].s0 = vpm[0].s4; results[ 4].s1 = vpm[1].s4; results[ 4].s2 = vpm[2].s4; results[ 4].s3 = vpm[3].s4; results[ 4].s4 = vpm[4].s4; results[ 4].s5 = vpm[5].s4; results[ 4].s6 = vpm[6].s4; results[ 4].s7 = vpm[7].s4; results[ 4].s8 = vpm[8].s4; results[ 4].s9 = vpm[9].s4; results[ 4].sA = vpm[10].s4; results[ 4].sB = vpm[11].s4; results[ 4].sC = vpm[12].s4; results[ 4].sD = vpm[13].s4; results[ 4].sE = vpm[14].s4; results[ 4].sF = vpm[15].s4;'#10
+ '    results[ 5].s0 = vpm[0].s5; results[ 5].s1 = vpm[1].s5; results[ 5].s2 = vpm[2].s5; results[ 5].s3 = vpm[3].s5; results[ 5].s4 = vpm[4].s5; results[ 5].s5 = vpm[5].s5; results[ 5].s6 = vpm[6].s5; results[ 5].s7 = vpm[7].s5; results[ 5].s8 = vpm[8].s5; results[ 5].s9 = vpm[9].s5; results[ 5].sA = vpm[10].s5; results[ 5].sB = vpm[11].s5; results[ 5].sC = vpm[12].s5; results[ 5].sD = vpm[13].s5; results[ 5].sE = vpm[14].s5; results[ 5].sF = vpm[15].s5;'#10
+ '    results[ 6].s0 = vpm[0].s6; results[ 6].s1 = vpm[1].s6; results[ 6].s2 = vpm[2].s6; results[ 6].s3 = vpm[3].s6; results[ 6].s4 = vpm[4].s6; results[ 6].s5 = vpm[5].s6; results[ 6].s6 = vpm[6].s6; results[ 6].s7 = vpm[7].s6; results[ 6].s8 = vpm[8].s6; results[ 6].s9 = vpm[9].s6; results[ 6].sA = vpm[10].s6; results[ 6].sB = vpm[11].s6; results[ 6].sC = vpm[12].s6; results[ 6].sD = vpm[13].s6; results[ 6].sE = vpm[14].s6; results[ 6].sF = vpm[15].s6;'#10
+ '    results[ 7].s0 = vpm[0].s7; results[ 7].s1 = vpm[1].s7; results[ 7].s2 = vpm[2].s7; results[ 7].s3 = vpm[3].s7; results[ 7].s4 = vpm[4].s7; results[ 7].s5 = vpm[5].s7; results[ 7].s6 = vpm[6].s7; results[ 7].s7 = vpm[7].s7; results[ 7].s8 = vpm[8].s7; results[ 7].s9 = vpm[9].s7; results[ 7].sA = vpm[10].s7; results[ 7].sB = vpm[11].s7; results[ 7].sC = vpm[12].s7; results[ 7].sD = vpm[13].s7; results[ 7].sE = vpm[14].s7; results[ 7].sF = vpm[15].s7;'#10
+ '    results[ 8].s0 = vpm[0].s8; results[ 8].s1 = vpm[1].s8; results[ 8].s2 = vpm[2].s8; results[ 8].s3 = vpm[3].s8; results[ 8].s4 = vpm[4].s8; results[ 8].s5 = vpm[5].s8; results[ 8].s6 = vpm[6].s8; results[ 8].s7 = vpm[7].s8; results[ 8].s8 = vpm[8].s8; results[ 8].s9 = vpm[9].s8; results[ 8].sA = vpm[10].s8; results[ 8].sB = vpm[11].s8; results[ 8].sC = vpm[12].s8; results[ 8].sD = vpm[13].s8; results[ 8].sE = vpm[14].s8; results[ 8].sF = vpm[15].s8;'#10
+ '    results[ 9].s0 = vpm[0].s9; results[ 9].s1 = vpm[1].s9; results[ 9].s2 = vpm[2].s9; results[ 9].s3 = vpm[3].s9; results[ 9].s4 = vpm[4].s9; results[ 9].s5 = vpm[5].s9; results[ 9].s6 = vpm[6].s9; results[ 9].s7 = vpm[7].s9; results[ 9].s8 = vpm[8].s9; results[ 9].s9 = vpm[9].s9; results[ 9].sA = vpm[10].s9; results[ 9].sB = vpm[11].s9; results[ 9].sC = vpm[12].s9; results[ 9].sD = vpm[13].s9; results[ 9].sE = vpm[14].s9; results[ 9].sF = vpm[15].s9;'#10
+ '    results[10].s0 = vpm[0].sA; results[10].s1 = vpm[1].sA; results[10].s2 = vpm[2].sA; results[10].s3 = vpm[3].sA; results[10].s4 = vpm[4].sA; results[10].s5 = vpm[5].sA; results[10].s6 = vpm[6].sA; results[10].s7 = vpm[7].sA; results[10].s8 = vpm[8].sA; results[10].s9 = vpm[9].sA; results[10].sA = vpm[10].sA; results[10].sB = vpm[11].sA; results[10].sC = vpm[12].sA; results[10].sD = vpm[13].sA; results[10].sE = vpm[14].sA; results[10].sF = vpm[15].sA;'#10
+ '    results[11].s0 = vpm[0].sB; results[11].s1 = vpm[1].sB; results[11].s2 = vpm[2].sB; results[11].s3 = vpm[3].sB; results[11].s4 = vpm[4].sB; results[11].s5 = vpm[5].sB; results[11].s6 = vpm[6].sB; results[11].s7 = vpm[7].sB; results[11].s8 = vpm[8].sB; results[11].s9 = vpm[9].sB; results[11].sA = vpm[10].sB; results[11].sB = vpm[11].sB; results[11].sC = vpm[12].sB; results[11].sD = vpm[13].sB; results[11].sE = vpm[14].sB; results[11].sF = vpm[15].sB;'#10
+ '    results[12].s0 = vpm[0].sC; results[12].s1 = vpm[1].sC; results[12].s2 = vpm[2].sC; results[12].s3 = vpm[3].sC; results[12].s4 = vpm[4].sC; results[12].s5 = vpm[5].sC; results[12].s6 = vpm[6].sC; results[12].s7 = vpm[7].sC; results[12].s8 = vpm[8].sC; results[12].s9 = vpm[9].sC; results[12].sA = vpm[10].sC; results[12].sB = vpm[11].sC; results[12].sC = vpm[12].sC; results[12].sD = vpm[13].sC; results[12].sE = vpm[14].sC; results[12].sF = vpm[15].sC;'#10
+ '    results[13].s0 = vpm[0].sD; results[13].s1 = vpm[1].sD; results[13].s2 = vpm[2].sD; results[13].s3 = vpm[3].sD; results[13].s4 = vpm[4].sD; results[13].s5 = vpm[5].sD; results[13].s6 = vpm[6].sD; results[13].s7 = vpm[7].sD; results[13].s8 = vpm[8].sD; results[13].s9 = vpm[9].sD; results[13].sA = vpm[10].sD; results[13].sB = vpm[11].sD; results[13].sC = vpm[12].sD; results[13].sD = vpm[13].sD; results[13].sE = vpm[14].sD; results[13].sF = vpm[15].sD;'#10
+ '    results[14].s0 = vpm[0].sE; results[14].s1 = vpm[1].sE; results[14].s2 = vpm[2].sE; results[14].s3 = vpm[3].sE; results[14].s4 = vpm[4].sE; results[14].s5 = vpm[5].sE; results[14].s6 = vpm[6].sE; results[14].s7 = vpm[7].sE; results[14].s8 = vpm[8].sE; results[14].s9 = vpm[9].sE; results[14].sA = vpm[10].sE; results[14].sB = vpm[11].sE; results[14].sC = vpm[12].sE; results[14].sD = vpm[13].sE; results[14].sE = vpm[14].sE; results[14].sF = vpm[15].sE;'#10
+ '    results[15].s0 = vpm[0].sF; results[15].s1 = vpm[1].sF; results[15].s2 = vpm[2].sF; results[15].s3 = vpm[3].sF; results[15].s4 = vpm[4].sF; results[15].s5 = vpm[5].sF; results[15].s6 = vpm[6].sF; results[15].s7 = vpm[7].sF; results[15].s8 = vpm[8].sF; results[15].s9 = vpm[9].sF; results[15].sA = vpm[10].sF; results[15].sB = vpm[11].sF; results[15].sC = vpm[12].sF; results[15].sD = vpm[13].sF; results[15].sE = vpm[14].sF; results[15].sF = vpm[15].sF;'#10
+ '  #endif'#10
+ '  // Multiplies by alpha and then stores the results into the destination matrix'#10
+ '  #pragma unroll'#10
+ '  for (int _w_two = 0; _w_two < TRA_WPT; _w_two += 1) {'#10
+ '    realT result;'#10
+ '    #if TRA_WPT == 1'#10
+ '      Multiply(result, alpha, results[_w_two]);'#10
+ '    #elif TRA_WPT == 2'#10
+ '      Multiply(result.x, alpha, results[_w_two].x);'#10
+ '      Multiply(result.y, alpha, results[_w_two].y);'#10
+ '    #elif TRA_WPT == 4'#10
+ '      Multiply(result.x, alpha, results[_w_two].x);'#10
+ '      Multiply(result.y, alpha, results[_w_two].y);'#10
+ '      Multiply(result.z, alpha, results[_w_two].z);'#10
+ '      Multiply(result.w, alpha, results[_w_two].w);'#10
+ '    #elif TRA_WPT == 8'#10
+ '      Multiply(result.s0, alpha, results[_w_two].s0);'#10
+ '      Multiply(result.s1, alpha, results[_w_two].s1);'#10
+ '      Multiply(result.s2, alpha, results[_w_two].s2);'#10
+ '      Multiply(result.s3, alpha, results[_w_two].s3);'#10
+ '      Multiply(result.s4, alpha, results[_w_two].s4);'#10
+ '      Multiply(result.s5, alpha, results[_w_two].s5);'#10
+ '      Multiply(result.s6, alpha, results[_w_two].s6);'#10
+ '      Multiply(result.s7, alpha, results[_w_two].s7);'#10
+ '    #elif TRA_WPT == 16'#10
+ '      Multiply(result.s0, alpha, results[_w_two].s0);'#10
+ '      Multiply(result.s1, alpha, results[_w_two].s1);'#10
+ '      Multiply(result.s2, alpha, results[_w_two].s2);'#10
+ '      Multiply(result.s3, alpha, results[_w_two].s3);'#10
+ '      Multiply(result.s4, alpha, results[_w_two].s4);'#10
+ '      Multiply(result.s5, alpha, results[_w_two].s5);'#10
+ '      Multiply(result.s6, alpha, results[_w_two].s6);'#10
+ '      Multiply(result.s7, alpha, results[_w_two].s7);'#10
+ '      Multiply(result.s8, alpha, results[_w_two].s8);'#10
+ '      Multiply(result.s9, alpha, results[_w_two].s9);'#10
+ '      Multiply(result.sA, alpha, results[_w_two].sA);'#10
+ '      Multiply(result.sB, alpha, results[_w_two].sB);'#10
+ '      Multiply(result.sC, alpha, results[_w_two].sC);'#10
+ '      Multiply(result.sD, alpha, results[_w_two].sD);'#10
+ '      Multiply(result.sE, alpha, results[_w_two].sE);'#10
+ '      Multiply(result.sF, alpha, results[_w_two].sF);'#10
+ '    #endif'#10
+ '    const int id_one = gid0*TRA_DIM + get_local_id(0);'#10
+ '    const int id_two = (gid1*TRA_DIM + get_local_id(1))*TRA_WPT + _w_two;'#10
+ '    dest[id_two*(ld/TRA_WPT) + id_one] = result;'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
