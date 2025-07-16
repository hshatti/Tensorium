''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This is part 2 of 3 of the GEMM kernel. See part 1 for more information.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for'#10
+ '// caching the A input matrix.'#10
+ 'INLINE_FUNC void GlobalToLocalDirectA(const __global realMD* restrict agm, LOCAL_PTR real* alm,'#10
+ '                                      const int a_ld, const int a_offset, const int kwg,'#10
+ '                                      const int a_transpose, const int a_conjugate) {'#10
+ '  #if MDIMCD == MDIMAD'#10
+ '    const int la0 = get_local_id(0);'#10
+ '    const int la1 = get_local_id(1);'#10
+ '  #else'#10
+ '    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);'#10
+ '    const int la0 = tid % MDIMAD;'#10
+ '    const int la1 = tid / MDIMAD;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _mia = 0; _mia < MWAD/VWMD; _mia += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _kia = 0; _kia < KWAD; _kia += 1) {'#10
+ '      // Computes the indices for the global memory'#10
+ '      int mg = _mia + la0*(MWAD/VWMD);'#10
+ '      int kg = _kia + la1*KWAD;'#10
+ '      int idm = (a_transpose) ? mg + kwg/VWMD : mg + GetGroupID0()*(WGD/VWMD);'#10
+ '      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;'#10
+ '      // Loads the data from global memory into the local memory'#10
+ '      const realMD avec = agm[idk*(a_ld/VWMD) + idm + (a_offset/VWMD)];'#10
+ '      #if VWMD == 1'#10
+ '         alm[kg*(WGD + PADA) + mg] = avec;'#10
+ '      #elif VWMD == 2'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;'#10
+ '      #elif VWMD == 4'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.z;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.w;'#10
+ '      #elif VWMD == 8'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;'#10
+ '      #elif VWMD == 16'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 8] = avec.s8;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 9] = avec.s9;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 10] = avec.sA;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 11] = avec.sB;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 12] = avec.sC;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 13] = avec.sD;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 14] = avec.sE;'#10
+ '         alm[kg*(WGD + PADA) + mg*VWMD + 15] = avec.sF;'#10
+ '      #endif'#10
+ '      if (a_conjugate) {'#10
+ '        for (int vm=0; vm<VWMD; ++vm) {'#10
+ '          COMPLEX_CONJUGATE(alm[kg*(WGD + PADA) + mg*VWMD + vm]);'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC void GlobalToLocalDirectB(const __global realND* restrict bgm, LOCAL_PTR real* blm,'#10
+ '                                      const int b_ld, const int b_offset, const int kwg,'#10
+ '                                      const int b_transpose, const int b_conjugate) {'#10
+ '  #if MDIMCD == NDIMBD'#10
+ '    const int lb0 = get_local_id(0);'#10
+ '    const int lb1 = get_local_id(1);'#10
+ '  #else'#10
+ '    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);'#10
+ '    const int lb0 = tid % NDIMBD;'#10
+ '    const int lb1 = tid / NDIMBD;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _kib = 0; _kib < KWBD; _kib += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _nib = 0; _nib < NWBD/VWND; _nib += 1) {'#10
+ '      // Computes the indices for the global memory'#10
+ '      int ng = _nib + lb0*(NWBD/VWND);'#10
+ '      int kg = _kib + lb1*KWBD;'#10
+ '      int idn = (b_transpose) ? ng + kwg/VWND : ng + GetGroupID1()*(WGD/VWND);'#10
+ '      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;'#10
+ '      // Loads the data from global memory into the local memory'#10
+ '      const realND bvec = bgm[idk*(b_ld/VWND) + idn + (b_offset/VWND)];'#10
+ '      #if VWND == 1'#10
+ '         blm[kg*(WGD + PADB) + ng] = bvec;'#10
+ '      #elif VWND == 2'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;'#10
+ '      #elif VWND == 4'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.z;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.w;'#10
+ '      #elif VWND == 8'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;'#10
+ '      #elif VWND == 16'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 8] = bvec.s8;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 9] = bvec.s9;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 10] = bvec.sA;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 11] = bvec.sB;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 12] = bvec.sC;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 13] = bvec.sD;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 14] = bvec.sE;'#10
+ '         blm[kg*(WGD + PADB) + ng*VWND + 15] = bvec.sF;'#10
+ '      #endif'#10
+ '      if (b_conjugate) {'#10
+ '        #pragma unroll'#10
+ '        for (int _vn = 0; _vn < VWND; _vn += 1) {'#10
+ '          COMPLEX_CONJUGATE(blm[kg*(WGD + PADB) + ng*VWND + _vn]);'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for'#10
+ '// caching the A input matrix. In contrast to the functions above, this function performs doesn''t'#10
+ '// use the vector data-types.'#10
+ 'INLINE_FUNC void GlobalToLocalScalarA(const __global real* restrict agms, LOCAL_PTR real* alm,'#10
+ '                                      const int a_ld, const int a_offset, const int kwg,'#10
+ '                                      const int a_transpose, const int a_conjugate) {'#10
+ '  #if MDIMCD == MDIMAD'#10
+ '    const int la0 = get_local_id(0);'#10
+ '    const int la1 = get_local_id(1);'#10
+ '  #else'#10
+ '    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);'#10
+ '    const int la0 = tid % MDIMAD;'#10
+ '    const int la1 = tid / MDIMAD;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _mia = 0; _mia < MWAD; _mia += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _kia = 0; _kia < KWAD; _kia += 1) {'#10
+ '      // Computes the indices for the global memory'#10
+ '      int mg = _mia + la0*MWAD;'#10
+ '      int kg = _kia + la1*KWAD;'#10
+ '      int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;'#10
+ '      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;'#10
+ '      // Loads the data from global memory into the local memory'#10
+ '      real result = agms[idk*a_ld + idm + a_offset];'#10
+ '      if (a_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '      alm[kg*(WGD + PADA) + mg] = result;'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC void GlobalToLocalScalarB(const __global real* restrict bgms, LOCAL_PTR real* blm,'#10
+ '                                      const int b_ld, const int b_offset, const int kwg,'#10
+ '                                      const int b_transpose, const int b_conjugate) {'#10
+ '  #if MDIMCD == NDIMBD'#10
+ '    const int lb0 = get_local_id(0);'#10
+ '    const int lb1 = get_local_id(1);'#10
+ '  #else'#10
+ '    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);'#10
+ '    const int lb0 = tid % NDIMBD;'#10
+ '    const int lb1 = tid / NDIMBD;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _kib = 0; _kib < KWBD; _kib += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _nib = 0; _nib < NWBD; _nib += 1) {'#10
+ '      // Computes the indices for the global memory'#10
+ '      int ng = _nib + lb0*NWBD;'#10
+ '      int kg = _kib + lb1*KWBD;'#10
+ '      int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;'#10
+ '      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;'#10
+ '      // Loads the data from global memory into the local memory'#10
+ '      real result = bgms[idk*b_ld + idn + b_offset];'#10
+ '      if (b_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '      blm[kg*(WGD + PADB) + ng] = result;'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for'#10
+ '// caching the A input matrix. In contrast to the functions above, this function performs bounds'#10
+ '// checks and doesn''t use the vector data-types.'#10
+ 'INLINE_FUNC void GlobalToLocalCheckedA(const __global real* restrict agms, LOCAL_PTR real* alm,'#10
+ '                                       const int a_ld, const int a_offset, const int kwg,'#10
+ '                                       const int a_transpose, const int a_conjugate,'#10
+ '                                       const int kSizeM, const int kSizeK) {'#10
+ '  #if MDIMCD == MDIMAD'#10
+ '    const int la0 = get_local_id(0);'#10
+ '    const int la1 = get_local_id(1);'#10
+ '  #else'#10
+ '    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);'#10
+ '    const int la0 = tid % MDIMAD;'#10
+ '    const int la1 = tid / MDIMAD;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _mia = 0; _mia < MWAD; _mia += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _kia = 0; _kia < KWAD; _kia += 1) {'#10
+ '      // Computes the indices for the global memory'#10
+ '      int mg = _mia + la0*MWAD;'#10
+ '      int kg = _kia + la1*KWAD;'#10
+ '      int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;'#10
+ '      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;'#10
+ '      // Loads the data from global memory into the local memory'#10
+ '      int condition = (a_transpose) ? (idm < kSizeK) && (idk < kSizeM) :'#10
+ '                                      (idm < kSizeM) && (idk < kSizeK);'#10
+ '      if (condition) {'#10
+ '        real result = agms[idk*a_ld + idm + a_offset];'#10
+ '        if (a_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '        alm[kg*(WGD + PADA) + mg] = result;'#10
+ '      }'#10
+ '      else {'#10
+ '        SetToZero(alm[kg*(WGD + PADA) + mg]);'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Same as above, but now for the B input matrix'#10
+ 'INLINE_FUNC void GlobalToLocalCheckedB(const __global real* restrict bgms, LOCAL_PTR real* blm,'#10
+ '                                       const int b_ld, const int b_offset, const int kwg,'#10
+ '                                       const int b_transpose, const int b_conjugate,'#10
+ '                                       const int kSizeN, const int kSizeK) {'#10
+ '  #if MDIMCD == NDIMBD'#10
+ '    const int lb0 = get_local_id(0);'#10
+ '    const int lb1 = get_local_id(1);'#10
+ '  #else'#10
+ '    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);'#10
+ '    const int lb0 = tid % NDIMBD;'#10
+ '    const int lb1 = tid / NDIMBD;'#10
+ '  #endif'#10
+ '  #pragma unroll'#10
+ '  for (int _kib = 0; _kib < KWBD; _kib += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _nib = 0; _nib < NWBD; _nib += 1) {'#10
+ '      // Computes the indices for the global memory'#10
+ '      int ng = _nib + lb0*NWBD;'#10
+ '      int kg = _kib + lb1*KWBD;'#10
+ '      int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;'#10
+ '      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;'#10
+ '      // Loads the data from global memory into the local memory'#10
+ '      int condition = (b_transpose) ? (idn < kSizeK) && (idk < kSizeN) :'#10
+ '                                      (idn < kSizeN) && (idk < kSizeK);'#10
+ '      if (condition) {'#10
+ '        real result = bgms[idk*b_ld + idn + b_offset];'#10
+ '        if (b_conjugate) { COMPLEX_CONJUGATE(result); }'#10
+ '        blm[kg*(WGD + PADB) + ng] = result;'#10
+ '      }'#10
+ '      else {'#10
+ '        SetToZero(blm[kg*(WGD + PADB) + ng]);'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
