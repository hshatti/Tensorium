''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Main body of the kernel. This is the direct version without pre/post processing and restrictions.'#10
+ 'INLINE_FUNC void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                             const real_arg arg_alpha,'#10
+ '                             const real_arg arg_beta,'#10
+ '                             const __global realMD* restrict agm, const int a_offset, const int a_ld,'#10
+ '                             const __global realND* restrict bgm, const int b_offset, const int b_ld,'#10
+ '                             __global real* cgm, const int c_offset, const int c_ld,'#10
+ '                             LOCAL_PTR real* alm, LOCAL_PTR real* blm,'#10
+ '                             const int a_transpose, const int b_transpose, const int c_transpose,'#10
+ '                             const int a_conjugate, const int b_conjugate) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  const real beta = GetRealArg(arg_beta);'#10
+ '  // Extra pointers to scalar versions of global memory'#10
+ '  const __global real* restrict agms = (const __global real* restrict) agm;'#10
+ '  const __global real* restrict bgms = (const __global real* restrict) bgm;'#10
+ '  // Allocates workitem-private memory (registers)'#10
+ '  #pragma promote_to_registers'#10
+ '  real apd[MWID];'#10
+ '  #pragma promote_to_registers'#10
+ '  real bpd[NWID];'#10
+ '  #pragma promote_to_registers'#10
+ '  real cpd[NWID * MWID];'#10
+ '  // Initializes the accumulation registers'#10
+ '  #pragma unroll'#10
+ '  for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '      SetToZero(cpd[_ni * MWID + _mi]);'#10
+ '    }'#10
+ '  }'#10
+ '  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section'#10
+ '  // processes only the main parts: output blocks of WGD by WGD.'#10
+ '  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;'#10
+ '  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;'#10
+ '  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {'#10
+ '    // Loops over all complete workgroup tiles (K-dimension)'#10
+ '    int kwg = 0;'#10
+ '    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {'#10
+ '      // Loads data: off-chip --> local (matrix A and B)'#10
+ '      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {'#10
+ '        GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);'#10
+ '      }'#10
+ '      else {'#10
+ '        GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);'#10
+ '      }'#10
+ '      if (b_ld % VWND == 0 && b_offset % VWND == 0) {'#10
+ '        GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);'#10
+ '      }'#10
+ '      else {'#10
+ '        GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);'#10
+ '      }'#10
+ '      barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '      // Loops over all workitem tiles, unrolled by a factor KWID'#10
+ '      for (int pwi = 0; pwi < WGD; pwi += KWID) {'#10
+ '        #pragma unroll'#10
+ '        for (int _pit = 0; _pit < KWID; _pit += 1) {'#10
+ '          int kg = pwi + _pit;'#10
+ '          // Loads data: local --> private (matrix A and B)'#10
+ '          #pragma unroll'#10
+ '          for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);'#10
+ '          }'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);'#10
+ '          }'#10
+ '          // Performs the accumulation (Cpmd += Apmd * Bpmd)'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '            #pragma unroll'#10
+ '            for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);'#10
+ '            }'#10
+ '          }'#10
+ '        }'#10
+ '      }'#10
+ '      barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    }'#10
+ '    // Loop over the remaining part (incomplete tile in K-dimension)'#10
+ '    for (; kwg < kSizeK; ++kwg) {'#10
+ '      // Loads data: off-chip --> private (matrix A and B)'#10
+ '      #pragma unroll'#10
+ '      for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '        apd[_mi] = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);'#10
+ '      }'#10
+ '      #pragma unroll'#10
+ '      for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '        bpd[_ni] = GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);'#10
+ '      }'#10
+ '      // Performs the accumulation (Cpmd += Apmd * Bpmd)'#10
+ '      #pragma unroll'#10
+ '      for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '        #pragma unroll'#10
+ '        for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '    // Stores a tile of results and performs the multiplication with alpha and beta'#10
+ '    #pragma unroll'#10
+ '    for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '      #pragma unroll'#10
+ '      for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '        StoreResultsDirect(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,'#10
+ '                           alpha, beta, c_ld, c_offset, c_transpose);'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)'#10
+ '  else {'#10
+ '    // Loops over all complete workgroup tiles (K-dimension)'#10
+ '    int kwg = 0;'#10
+ '    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {'#10
+ '      // Loads data: off-chip --> local (matrix A and B)'#10
+ '      GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);'#10
+ '      GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);'#10
+ '      barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '      // Loops over all workitem tiles, unrolled by a factor KWID'#10
+ '      for (int pwi = 0; pwi < WGD; pwi += KWID) {'#10
+ '        #pragma unroll'#10
+ '        for (int _pit = 0; _pit < KWID; _pit += 1) {'#10
+ '          int kg = pwi + _pit;'#10
+ '          // Loads data: local --> private (matrix A and B)'#10
+ '          #pragma unroll'#10
+ '          for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);'#10
+ '          }'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);'#10
+ '          }'#10
+ '          // Performs the accumulation (C += A * B)'#10
+ '          #pragma unroll'#10
+ '          for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '            #pragma unroll'#10
+ '            for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);'#10
+ '            }'#10
+ '          }'#10
+ '        }'#10
+ '      }'#10
+ '      barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    }'#10
+ '    // Loop over the remaining part (incomplete tile in K-dimension)'#10
+ '    for (; kwg < kSizeK; ++kwg) {'#10
+ '      // Loads data: off-chip --> private (matrix A and B)'#10
+ '      #pragma unroll'#10
+ '      for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '        apd[_mi] = GlobalToPrivateCheckedA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);'#10
+ '      }'#10
+ '      #pragma unroll'#10
+ '      for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '        bpd[_ni] = GlobalToPrivateCheckedB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);'#10
+ '      }'#10
+ '      // Performs the accumulation (C += A * B)'#10
+ '      #pragma unroll'#10
+ '      for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '        #pragma unroll'#10
+ '        for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '    // Stores a tile of results and performs the multiplication with alpha and beta'#10
+ '    #pragma unroll'#10
+ '    for (int _ni = 0; _ni < NWID; _ni += 1) {'#10
+ '      #pragma unroll'#10
+ '      for (int _mi = 0; _mi < MWID; _mi += 1) {'#10
+ '        StoreResultsChecked(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,'#10
+ '                            alpha, beta, c_ld, c_offset, c_transpose);'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                            const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                            const __global realMD* restrict agm, const int a_offset, const int a_ld,'#10
+ '                            const __global realND* restrict bgm, const int b_offset, const int b_ld,'#10
+ '                            __global real* cgm, const int c_offset, const int c_ld,'#10
+ '                            const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                            const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                            const __global realMD* restrict agm, const int a_offset, const int a_ld,'#10
+ '                            const __global realND* restrict bgm, const int b_offset, const int b_ld,'#10
+ '                            __global real* cgm, const int c_offset, const int c_ld,'#10
+ '                            const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                            const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                            const __global realMD* restrict agm, const int a_offset, const int a_ld,'#10
+ '                            const __global realND* restrict bgm, const int b_offset, const int b_ld,'#10
+ '                            __global real* cgm, const int c_offset, const int c_ld,'#10
+ '                            const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))'#10
+ '#endif'#10
+ 'void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,'#10
+ '                            const real_arg arg_alpha, const real_arg arg_beta,'#10
+ '                            const __global realMD* restrict agm, const int a_offset, const int a_ld,'#10
+ '                            const __global realND* restrict bgm, const int b_offset, const int b_ld,'#10
+ '                            __global real* cgm, const int c_offset, const int c_ld,'#10
+ '                            const int c_transpose, const int a_conjugate, const int b_conjugate) {'#10
+ '  __local real alm[WGD * (WGD + PADA)];'#10
+ '  __local real blm[WGD * (WGD + PADB)];'#10
+ '  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,'#10
+ '              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,'#10
+ '              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
