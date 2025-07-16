''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// A common interface for subgroup functions'#10
+ '#if USE_SUBGROUP_SHUFFLING == 1'#10
+ 'INLINE_FUNC int clblast_get_sub_group_local_id() {'#10
+ '  // Intel extension '#10
+ '  #if SUBGROUP_SHUFFLING_INTEL == 1'#10
+ '  return get_sub_group_local_id();'#10
+ '  '#10
+ '  // Nvidia inline PTX'#10
+ '  #elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1'#10
+ '  int ret;'#10
+ '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret) );'#10
+ '  return ret;'#10
+ '  #endif '#10
+ '}'#10
+ 'INLINE_FUNC realN clblast_sub_group_shuffle(realN reg, int src) {'#10
+ '  // Intel extension '#10
+ '  #if SUBGROUP_SHUFFLING_INTEL == 1'#10
+ '  return intel_sub_group_shuffle(reg, src);'#10
+ '  '#10
+ '  // Nvidia inline PTX'#10
+ '  // Volta and later requires .sync shuffle instructions with an extra mask arg'#10
+ '  #elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1'#10
+ '  realN ret;'#10
+ '    #if SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1'#10
+ '    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(ret): "f"(reg), "r"(src));'#10
+ '    #else'#10
+ '    asm volatile("shfl.idx.b32 %0, %1, %2, 0x1f;" : "=f"(ret): "f"(reg), "r"(src));'#10
+ '    #endif'#10
+ '  return ret;'#10
+ '  #endif'#10
+ '}'#10
+ '#endif'#10
+ '// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.'#10
+ 'INLINE_FUNC void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                           const __global realM* restrict agm, const __global realN* restrict bgm,'#10
+ '                           __global realM* cgm, const real alpha, const real beta'#10
+ '                           #if SA == 1 && SB == 1'#10
+ '                             , LOCAL_PTR realM* alm, LOCAL_PTR realN* blm'#10
+ '                           #elif SA == 1'#10
+ '                             , LOCAL_PTR realM* alm'#10
+ '                           #elif SB == 1'#10
+ '                             , LOCAL_PTR realN* blm'#10
+ '                           #endif'#10
+ '                           ) {'#10
+ '  // Allocates workitem-private memory (registers)'#10
+ '  #if GEMMK == 0'#10
+ '    #pragma promote_to_registers'#10
+ '    realM apm[MWI/VWM]; // MWI * 1'#10
+ '    #pragma promote_to_registers'#10
+ '    realN bpm[NWI/VWN]; // 1 * NWI'#10
+ '  #elif GEMMK == 1'#10
+ '    #if USE_SUBGROUP_SHUFFLING == 1'#10
+ '      #pragma promote_to_registers'#10
+ '      realN apm[KREG/VWN]; // KREG (subgroup shuffling in NWI dimension)'#10
+ '    #else'#10
+ '      #pragma promote_to_registers'#10
+ '      realN apm[NWI*(KREG/VWN)]; // NWI * KREG'#10
+ '    #endif'#10
+ '    #pragma promote_to_registers'#10
+ '    realM bpm[KREG*(MWI/VWM)]; // KREG * MWI'#10
+ '  #endif'#10
+ '  #pragma promote_to_registers'#10
+ '  realM cpm[NWI*(MWI/VWM)]; // NWI * MWI'#10
+ '  #if GEMMK == 1'#10
+ '    const __global real* restrict a_ptr = (const __global real* restrict) &agm[0];'#10
+ '    const __global real* restrict b_ptr = (const __global real* restrict) &bgm[0];'#10
+ '    const int tid_x = get_local_id(0) + MDIMC * GetGroupID0();'#10
+ '    const int tid_y = get_local_id(1) + NDIMC * GetGroupID1();'#10
+ '  #endif'#10
+ '  // Combined thread identifier (volatile to disable caching)'#10
+ '  #if SA == 1 || SB == 1'#10
+ '    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);'#10
+ '  #endif'#10
+ '  // Initializes the accumulation registers'#10
+ '  #pragma unroll'#10
+ '  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _ni = 0; _ni < NWI; _ni += 1) {'#10
+ '      cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();'#10
+ '    }'#10
+ '  }'#10
+ '  // Loops over all workgroup tiles'#10
+ '  for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {'#10
+ '    // Loads data: off-chip --> local (matrix A)'#10
+ '    #if SA == 1'#10
+ '      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);'#10
+ '    #endif'#10
+ '    // Loads data: off-chip --> local (matrix B)'#10
+ '    #if SB == 1'#10
+ '      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);'#10
+ '    #endif'#10
+ '    #if SA == 1 || SB == 1'#10
+ '      barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    #endif'#10
+ '    // Loops over all workitem tiles, unrolled by a factor KWI'#10
+ '    for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {'#10
+ '      #pragma unroll'#10
+ '      for (int _pit = 0; _pit < KWI*KREG; _pit += KREG) {'#10
+ '        #if SA == 0 || SB == 0'#10
+ '          int idk = kwg + pwi + _pit;'#10
+ '        #endif'#10
+ '        #if SA == 1 || SB == 1'#10
+ '          int kg = pwi + _pit;'#10
+ '        #endif'#10
+ '        // Loads matrix A (kernel 0) or matrix B (kernel 1)'#10
+ '        #pragma unroll'#10
+ '        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {'#10
+ '          // Loads data: local --> private (matrix A)'#10
+ '          #if GEMMK == 0 && SA == 1'#10
+ '            apm[_mi] = LocalToPrivateA(alm, _mi, kg);'#10
+ '          // Loads data: off-chip --> private (matrix A)'#10
+ '          #elif GEMMK == 0 && SA == 0'#10
+ '            apm[_mi] = GlobalToPrivateA(agm, _mi, kSizeM, idk, kwg);'#10
+ '          // Loads data: 2D global --> 2D private (matrix B)'#10
+ '          #elif GEMMK == 1'#10
+ '            #pragma unroll'#10
+ '            for (int _ki = 0; _ki < KREG; _ki += 1) {'#10
+ '              bpm[_ki * (MWI/VWM) + _mi] = GlobalToPrivateB2D(b_ptr, tid_x, _mi, kSizeN, idk, _ki);'#10
+ '            }'#10
+ '          #endif'#10
+ '        }'#10
+ '        // Loads matrix B (kernel 0) or matrix A (kernel 1)'#10
+ '        #if GEMMK == 0'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {'#10
+ '            // Loads data: local --> private (matrix B)'#10
+ '            #if SB == 1'#10
+ '              bpm[_ni] = LocalToPrivateB(blm, _ni, kg);'#10
+ '            // Loads data: off-chip --> private (matrix B)'#10
+ '            #else'#10
+ '              bpm[_ni] = GlobalToPrivateB(bgm, _ni, kSizeN, idk);'#10
+ '            #endif'#10
+ '          }'#10
+ '        #elif GEMMK == 1'#10
+ '          // Loads data: 2D global --> 2D private (matrix A). Partly, shuffled later among subgroups'#10
+ '          #if USE_SUBGROUP_SHUFFLING == 1'#10
+ '            const int _ni = clblast_get_sub_group_local_id();'#10
+ '            #pragma unroll'#10
+ '            for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {'#10
+ '              apm[_ki] = GlobalToPrivateA2D(a_ptr, tid_y, _ni, kSizeK, idk, _ki);'#10
+ '            }'#10
+ '          // Loads data: 2D global --> 2D private (matrix A)'#10
+ '          #else'#10
+ '            #pragma unroll'#10
+ '            for (int _ni = 0; _ni < NWI; _ni += 1) {'#10
+ '              #pragma unroll'#10
+ '              for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {'#10
+ '                apm[_ni * (KREG/VWN) + _ki] = GlobalToPrivateA2D(a_ptr, tid_y, _ni, kSizeK, idk, _ki);'#10
+ '              }'#10
+ '            }'#10
+ '          #endif'#10
+ '        #endif'#10
+ '        // Performs the accumulation (Cpm += Apm * Bpm)'#10
+ '        #if GEMMK == 0'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {'#10
+ '            #pragma unroll'#10
+ '            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {'#10
+ '              const realM aval = apm[_mi];'#10
+ '              #if VWN == 1'#10
+ '                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni]);'#10
+ '              #elif VWN == 2'#10
+ '                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);'#10
+ '                cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);'#10
+ '              #elif VWN == 4'#10
+ '                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);'#10
+ '                cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);'#10
+ '                cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].z);'#10
+ '                cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].w);'#10
+ '              #elif VWN == 8'#10
+ '                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].s0);'#10
+ '                cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].s1);'#10
+ '                cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].s2);'#10
+ '                cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].s3);'#10
+ '                cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[_ni].s4);'#10
+ '                cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[_ni].s5);'#10
+ '                cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[_ni].s6);'#10
+ '                cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[_ni].s7);'#10
+ '              #elif VWN == 16'#10
+ '                cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi], aval, bpm[_ni].s0);'#10
+ '                cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi], aval, bpm[_ni].s1);'#10
+ '                cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi], aval, bpm[_ni].s2);'#10
+ '                cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi], aval, bpm[_ni].s3);'#10
+ '                cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi], aval, bpm[_ni].s4);'#10
+ '                cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi], aval, bpm[_ni].s5);'#10
+ '                cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi], aval, bpm[_ni].s6);'#10
+ '                cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi], aval, bpm[_ni].s7);'#10
+ '                cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi], aval, bpm[_ni].s8);'#10
+ '                cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi], aval, bpm[_ni].s9);'#10
+ '                cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi], aval, bpm[_ni].sA);'#10
+ '                cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi], aval, bpm[_ni].sB);'#10
+ '                cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi], aval, bpm[_ni].sC);'#10
+ '                cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi], aval, bpm[_ni].sD);'#10
+ '                cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi], aval, bpm[_ni].sE);'#10
+ '                cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi], aval, bpm[_ni].sF);'#10
+ '              #endif'#10
+ '            }'#10
+ '          }'#10
+ '        #elif GEMMK == 1'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWI; _ni += 1) {'#10
+ '            #pragma unroll'#10
+ '            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {'#10
+ '              #pragma unroll'#10
+ '              for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {'#10
+ '                #if USE_SUBGROUP_SHUFFLING == 1'#10
+ '                  const realN aval = clblast_sub_group_shuffle(apm[_ki], _ni);'#10
+ '                #else'#10
+ '                  const realN aval = apm[_ni * (KREG/VWN) + _ki];'#10
+ '                #endif'#10
+ '                #if VWN == 1'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval);'#10
+ '                #elif VWN == 2'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.x);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.y);'#10
+ '                #elif VWN == 4'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.x);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.y);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2) * (MWI/VWM) + _mi], aval.z);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3) * (MWI/VWM) + _mi], aval.w);'#10
+ '                #elif VWN == 8'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.s0);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.s1);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2) * (MWI/VWM) + _mi], aval.s2);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3) * (MWI/VWM) + _mi], aval.s3);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 4) * (MWI/VWM) + _mi], aval.s4);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 5) * (MWI/VWM) + _mi], aval.s5);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 6) * (MWI/VWM) + _mi], aval.s6);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 7) * (MWI/VWM) + _mi], aval.s7);'#10
+ '                #elif VWN == 16'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0 ) * (MWI/VWM) + _mi], aval.s0);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1 ) * (MWI/VWM) + _mi], aval.s1);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2 ) * (MWI/VWM) + _mi], aval.s2);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3 ) * (MWI/VWM) + _mi], aval.s3);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 4 ) * (MWI/VWM) + _mi], aval.s4);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 5 ) * (MWI/VWM) + _mi], aval.s5);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 6 ) * (MWI/VWM) + _mi], aval.s6);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 7 ) * (MWI/VWM) + _mi], aval.s7);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 8 ) * (MWI/VWM) + _mi], aval.s8);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 9 ) * (MWI/VWM) + _mi], aval.s9);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 10) * (MWI/VWM) + _mi], aval.sA);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 11) * (MWI/VWM) + _mi], aval.sB);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 12) * (MWI/VWM) + _mi], aval.sC);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 13) * (MWI/VWM) + _mi], aval.sD);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 14) * (MWI/VWM) + _mi], aval.sE);'#10
+ '                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 15) * (MWI/VWM) + _mi], aval.sF);'#10
+ '                #endif'#10
+ '              }'#10
+ '            }'#10
+ '          }'#10
+ '        #endif'#10
+ '      }'#10
+ '    }'#10
+ '    #if SA == 1 || SB == 1'#10
+ '      barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    #endif'#10
+ '  }'#10
+ '  #if GLOBAL_MEM_FENCE == 1'#10
+ '    barrier(CLK_GLOBAL_MEM_FENCE);'#10
+ '  #endif'#10
+ '  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta'#10
+ '  #if GEMMK == 0'#10
+ '    const int cld = kSizeM;'#10
+ '  #elif GEMMK == 1'#10
+ '    const int cld = kSizeN;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _ni = 0; _ni < NWI; _ni += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {'#10
+ '      StoreResults(cgm, cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, alpha, beta);'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
