''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// The vectorised multiply-add function'#10
+ 'INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {'#10
+ '  #if USE_VECTOR_MAD == 1'#10
+ '    cvec += avec * bval;'#10
+ '  #else'#10
+ '    #if VWM == 1'#10
+ '      MultiplyAdd(cvec,    avec,    bval);'#10
+ '    #elif VWM == 2'#10
+ '      MultiplyAdd(cvec.x , avec.x,  bval);'#10
+ '      MultiplyAdd(cvec.y , avec.y,  bval);'#10
+ '    #elif VWM == 4'#10
+ '      MultiplyAdd(cvec.x , avec.x,  bval);'#10
+ '      MultiplyAdd(cvec.y , avec.y,  bval);'#10
+ '      MultiplyAdd(cvec.z , avec.z,  bval);'#10
+ '      MultiplyAdd(cvec.w , avec.w,  bval);'#10
+ '    #elif VWM == 8'#10
+ '      MultiplyAdd(cvec.s0, avec.s0, bval);'#10
+ '      MultiplyAdd(cvec.s1, avec.s1, bval);'#10
+ '      MultiplyAdd(cvec.s2, avec.s2, bval);'#10
+ '      MultiplyAdd(cvec.s3, avec.s3, bval);'#10
+ '      MultiplyAdd(cvec.s4, avec.s4, bval);'#10
+ '      MultiplyAdd(cvec.s5, avec.s5, bval);'#10
+ '      MultiplyAdd(cvec.s6, avec.s6, bval);'#10
+ '      MultiplyAdd(cvec.s7, avec.s7, bval);'#10
+ '    #elif VWM == 16'#10
+ '      MultiplyAdd(cvec.s0, avec.s0, bval);'#10
+ '      MultiplyAdd(cvec.s1, avec.s1, bval);'#10
+ '      MultiplyAdd(cvec.s2, avec.s2, bval);'#10
+ '      MultiplyAdd(cvec.s3, avec.s3, bval);'#10
+ '      MultiplyAdd(cvec.s4, avec.s4, bval);'#10
+ '      MultiplyAdd(cvec.s5, avec.s5, bval);'#10
+ '      MultiplyAdd(cvec.s6, avec.s6, bval);'#10
+ '      MultiplyAdd(cvec.s7, avec.s7, bval);'#10
+ '      MultiplyAdd(cvec.s8, avec.s8, bval);'#10
+ '      MultiplyAdd(cvec.s9, avec.s9, bval);'#10
+ '      MultiplyAdd(cvec.sA, avec.sA, bval);'#10
+ '      MultiplyAdd(cvec.sB, avec.sB, bval);'#10
+ '      MultiplyAdd(cvec.sC, avec.sC, bval);'#10
+ '      MultiplyAdd(cvec.sD, avec.sD, bval);'#10
+ '      MultiplyAdd(cvec.sE, avec.sE, bval);'#10
+ '      MultiplyAdd(cvec.sF, avec.sF, bval);'#10
+ '    #endif'#10
+ '  #endif'#10
+ '  return cvec;'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication'#10
+ '// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm'#10
+ 'INLINE_FUNC void StoreResults(__global realM* cgm, realM c_value, const int _mi, const int _ni,'#10
+ '                              const int kSizeM, const real alpha, const real beta) {'#10
+ '  #if STRM == 0'#10
+ '    int mg = _mi + get_local_id(0)*(MWI/VWM);'#10
+ '  #elif STRM == 1'#10
+ '    int mg = get_local_id(0) + _mi*MDIMC;'#10
+ '  #endif'#10
+ '  #if STRN == 0'#10
+ '    int ng = _ni + get_local_id(1)*NWI;'#10
+ '  #elif STRN == 1'#10
+ '    int ng = _ni%VWN + get_local_id(1)*VWN + (_ni/VWN)*VWN*NDIMC;'#10
+ '  #endif'#10
+ '  int idm = mg + GetGroupID0() * (MWG/VWM);'#10
+ '  int idn = ng + GetGroupID1() * NWG;'#10
+ '  int index = idn*(kSizeM/VWM) + idm;'#10
+ '  realM result;'#10
+ '  realM xval = c_value;'#10
+ '  // The final multiplication with alpha (in case beta == 0)'#10
+ '  if (IsZero(beta)) {'#10
+ '    #if VWM == 1'#10
+ '      Multiply(result, alpha, xval);'#10
+ '    #elif VWM == 2'#10
+ '      Multiply(result.x, alpha, xval.x);'#10
+ '      Multiply(result.y, alpha, xval.y);'#10
+ '    #elif VWM == 4'#10
+ '      Multiply(result.x, alpha, xval.x);'#10
+ '      Multiply(result.y, alpha, xval.y);'#10
+ '      Multiply(result.z, alpha, xval.z);'#10
+ '      Multiply(result.w, alpha, xval.w);'#10
+ '    #elif VWM == 8'#10
+ '      Multiply(result.s0, alpha, xval.s0);'#10
+ '      Multiply(result.s1, alpha, xval.s1);'#10
+ '      Multiply(result.s2, alpha, xval.s2);'#10
+ '      Multiply(result.s3, alpha, xval.s3);'#10
+ '      Multiply(result.s4, alpha, xval.s4);'#10
+ '      Multiply(result.s5, alpha, xval.s5);'#10
+ '      Multiply(result.s6, alpha, xval.s6);'#10
+ '      Multiply(result.s7, alpha, xval.s7);'#10
+ '    #elif VWM == 16'#10
+ '      Multiply(result.s0, alpha, xval.s0);'#10
+ '      Multiply(result.s1, alpha, xval.s1);'#10
+ '      Multiply(result.s2, alpha, xval.s2);'#10
+ '      Multiply(result.s3, alpha, xval.s3);'#10
+ '      Multiply(result.s4, alpha, xval.s4);'#10
+ '      Multiply(result.s5, alpha, xval.s5);'#10
+ '      Multiply(result.s6, alpha, xval.s6);'#10
+ '      Multiply(result.s7, alpha, xval.s7);'#10
+ '      Multiply(result.s8, alpha, xval.s8);'#10
+ '      Multiply(result.s9, alpha, xval.s9);'#10
+ '      Multiply(result.sA, alpha, xval.sA);'#10
+ '      Multiply(result.sB, alpha, xval.sB);'#10
+ '      Multiply(result.sC, alpha, xval.sC);'#10
+ '      Multiply(result.sD, alpha, xval.sD);'#10
+ '      Multiply(result.sE, alpha, xval.sE);'#10
+ '      Multiply(result.sF, alpha, xval.sF);'#10
+ '    #endif'#10
+ '  }'#10
+ '  // The final multiplication with alpha and the addition with beta*C'#10
+ '  else {'#10
+ '    realM yval = cgm[index];'#10
+ '    #if VWM == 1'#10
+ '      AXPBY(result, alpha, xval, beta, yval);'#10
+ '    #elif VWM == 2'#10
+ '      AXPBY(result.x, alpha, xval.x, beta, yval.x);'#10
+ '      AXPBY(result.y, alpha, xval.y, beta, yval.y);'#10
+ '    #elif VWM == 4'#10
+ '      AXPBY(result.x, alpha, xval.x, beta, yval.x);'#10
+ '      AXPBY(result.y, alpha, xval.y, beta, yval.y);'#10
+ '      AXPBY(result.z, alpha, xval.z, beta, yval.z);'#10
+ '      AXPBY(result.w, alpha, xval.w, beta, yval.w);'#10
+ '    #elif VWM == 8'#10
+ '      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);'#10
+ '      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);'#10
+ '      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);'#10
+ '      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);'#10
+ '      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);'#10
+ '      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);'#10
+ '      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);'#10
+ '      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);'#10
+ '    #elif VWM == 16'#10
+ '      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);'#10
+ '      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);'#10
+ '      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);'#10
+ '      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);'#10
+ '      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);'#10
+ '      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);'#10
+ '      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);'#10
+ '      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);'#10
+ '      AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);'#10
+ '      AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);'#10
+ '      AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);'#10
+ '      AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);'#10
+ '      AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);'#10
+ '      AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);'#10
+ '      AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);'#10
+ '      AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);'#10
+ '    #endif'#10
+ '  }'#10
+ '  cgm[index] = result;'#10
+ '}'#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
