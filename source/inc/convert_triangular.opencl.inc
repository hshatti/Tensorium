''
+ '// ================================================================================================='#10
+ '// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This'#10
+ '// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-'#10
+ '// width of 100 characters per line.'#10
+ '//'#10
+ '// Author(s):'#10
+ '//   Cedric Nugteren <www.cedricnugteren.nl>'#10
+ '//'#10
+ '// This file contains kernels to convert triangular matrices to/from general matrices.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '#if defined(ROUTINE_TRMM)'#10
+ '// Kernel to populate a squared triangular matrix, given that the triangle which holds the data is'#10
+ '// stored as the lower-triangle of the input matrix. This uses the padding kernel''s parameters.'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void TriaLowerToSquared(const int src_dim,'#10
+ '                        const int src_ld, const int src_offset,'#10
+ '                        __global const real* restrict src,'#10
+ '                        const int dest_dim,'#10
+ '                        const int dest_ld, const int dest_offset,'#10
+ '                        __global real* dest,'#10
+ '                        const int unit_diagonal) {'#10
+ '  // Loops over the work per thread in both dimensions'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {'#10
+ '    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {'#10
+ '      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);'#10
+ '      if (id_two < dest_dim && id_one < dest_dim) {'#10
+ '        // Loads data from the lower-triangular matrix'#10
+ '        real result;'#10
+ '        SetToZero(result);'#10
+ '        if (id_two < src_dim && id_one < src_dim) {'#10
+ '          if (id_two <= id_one) { result = src[id_two*src_ld + id_one + src_offset]; }'#10
+ '          if (id_two == id_one && unit_diagonal) { SetToOne(result); }'#10
+ '          // Else: result is zero'#10
+ '        }'#10
+ '        // Stores the result in the destination matrix'#10
+ '        dest[id_two*dest_ld + id_one + dest_offset] = result;'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Same as above, but now the matrix'' data is stored in the upper-triangle'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void TriaUpperToSquared(const int src_dim,'#10
+ '                        const int src_ld, const int src_offset,'#10
+ '                        __global const real* restrict src,'#10
+ '                        const int dest_dim,'#10
+ '                        const int dest_ld, const int dest_offset,'#10
+ '                        __global real* dest,'#10
+ '                        const int unit_diagonal) {'#10
+ '  // Loops over the work per thread in both dimensions'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {'#10
+ '    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {'#10
+ '      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);'#10
+ '      if (id_two < dest_dim && id_one < dest_dim) {'#10
+ '        // Loads data from the upper-triangular matrix'#10
+ '        real result;'#10
+ '        SetToZero(result);'#10
+ '        if (id_two < src_dim && id_one < src_dim) {'#10
+ '          if (id_one <= id_two) { result = src[id_two*src_ld + id_one + src_offset]; }'#10
+ '          if (id_one == id_two && unit_diagonal) { SetToOne(result); }'#10
+ '          // Else: result is zero'#10
+ '        }'#10
+ '        // Stores the result in the destination matrix'#10
+ '        dest[id_two*dest_ld + id_one + dest_offset] = result;'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ '// End of the C++11 raw string literal'#10
+ '// ================================================================================================='#10
