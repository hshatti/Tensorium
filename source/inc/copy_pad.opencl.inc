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
+ '// kernels to copy and pad matrices in various ways, including:'#10
+ '// 1) copying into a larger matrix by adding padding'#10
+ '// 2) copying into a smaller matrix by optionally removing padding. This is the general version'#10
+ '//    without restrictions, see the ''copy.opencl'' file for a faster but more restricted copy kernel.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Copies a matrix from source to destination. The output is padded with zero values in case the'#10
+ '// destination matrix dimensions are larger than the source matrix dimensions. Additionally, the ld'#10
+ '// value and offset can be different.'#10
+ 'INLINE_FUNC void _CopyPadMatrix(const int src_one, const int src_two,'#10
+ '                                const int src_ld, const int src_offset,'#10
+ '                                __global const real* restrict src,'#10
+ '                                const int dest_one, const int dest_two,'#10
+ '                                const int dest_ld, const int dest_offset,'#10
+ '                                __global real* dest,'#10
+ '                                const real alpha,'#10
+ '                                const int do_conjugate) {'#10
+ '  // Loops over the work per thread in both dimensions'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {'#10
+ '    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {'#10
+ '      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);'#10
+ '      if (id_two < dest_two && id_one < dest_one) {'#10
+ '        // Loads data if the thread IDs are within bounds of the source matrix. Otherwise, set the'#10
+ '        // value to be written to zero.'#10
+ '        real value;'#10
+ '        SetToZero(value);'#10
+ '        if (id_two < src_two && id_one < src_one) {'#10
+ '          value = src[id_two*src_ld + id_one + src_offset];'#10
+ '        }'#10
+ '        // Stores the value in the destination matrix'#10
+ '        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }'#10
+ '        Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Interface to the above function'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyPadMatrix(const int src_one, const int src_two,'#10
+ '                   const int src_ld, const int src_offset,'#10
+ '                   __global const real* restrict src,'#10
+ '                   const int dest_one, const int dest_two,'#10
+ '                   const int dest_ld, const int dest_offset,'#10
+ '                   __global real* dest,'#10
+ '                   const real_arg arg_alpha,'#10
+ '                   const int do_conjugate) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  _CopyPadMatrix(src_one, src_two, src_ld, src_offset, src,'#10
+ '                 dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '                 alpha, do_conjugate);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Same as above, but now un-pads a matrix. This kernel reads data from a padded source matrix, but'#10
+ '// writes only the actual data back to the destination matrix. Again, the ld value and offset can'#10
+ '// be different.'#10
+ 'INLINE_FUNC void _CopyMatrix(const int src_one, const int src_two,'#10
+ '                             const int src_ld, const int src_offset,'#10
+ '                             __global const real* restrict src,'#10
+ '                             const int dest_one, const int dest_two,'#10
+ '                             const int dest_ld, const int dest_offset,'#10
+ '                             __global real* dest,'#10
+ '                             const real alpha,'#10
+ '                             const int upper, const int lower,'#10
+ '                             const int diagonal_imag_zero) {'#10
+ '  // Loops over the work per thread in both dimensions'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {'#10
+ '    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {'#10
+ '      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);'#10
+ '      // Masking in case of triangular matrices: updates only the upper or lower part'#10
+ '      bool condition = true;'#10
+ '      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)'#10
+ '        if (upper == 1) { condition = (id_two >= id_one); }'#10
+ '        else if (lower == 1) { condition = (id_two <= id_one); }'#10
+ '      #endif'#10
+ '      if (condition) {'#10
+ '        // Copies the value into the destination matrix. This is always within bounds of the source'#10
+ '        // matrix, as we know that the destination matrix is smaller or equal to the source.'#10
+ '        if (id_two < dest_two && id_one < dest_one) {'#10
+ '          real value = src[id_two*src_ld + id_one + src_offset];'#10
+ '          if (diagonal_imag_zero == 1 && id_one == id_two) { ImagToZero(value); }'#10
+ '          Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Interface to the above function'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyMatrix(const int src_one, const int src_two,'#10
+ '                const int src_ld, const int src_offset,'#10
+ '                __global const real* restrict src,'#10
+ '                const int dest_one, const int dest_two,'#10
+ '                const int dest_ld, const int dest_offset,'#10
+ '                __global real* dest,'#10
+ '                const real_arg arg_alpha,'#10
+ '                const int upper, const int lower,'#10
+ '                const int diagonal_imag_zero) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  _CopyMatrix(src_one, src_two, src_ld, src_offset, src,'#10
+ '              dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '              alpha, upper, lower, diagonal_imag_zero);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyPadMatrixBatched(const int src_one, const int src_two,'#10
+ '                          const int src_ld, const __constant int* src_offsets,'#10
+ '                          __global const real* restrict src,'#10
+ '                          const int dest_one, const int dest_two,'#10
+ '                          const int dest_ld, const __constant int* dest_offsets,'#10
+ '                          __global real* dest,'#10
+ '                          const int do_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset = src_offsets[batch];'#10
+ '  const int dest_offset = dest_offsets[batch];'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  _CopyPadMatrix(src_one, src_two, src_ld, src_offset, src,'#10
+ '                 dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '                 alpha, do_conjugate);'#10
+ '}'#10
+ '// Batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyMatrixBatched(const int src_one, const int src_two,'#10
+ '                       const int src_ld, const __constant int* src_offsets,'#10
+ '                       __global const real* restrict src,'#10
+ '                       const int dest_one, const int dest_two,'#10
+ '                       const int dest_ld, const __constant int* dest_offsets,'#10
+ '                       __global real* dest) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset = src_offsets[batch];'#10
+ '  const int dest_offset = dest_offsets[batch];'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  _CopyMatrix(src_one, src_two, src_ld, src_offset, src,'#10
+ '              dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '              alpha, 0, 0, 0);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Strided-batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyPadMatrixStridedBatched(const int src_one, const int src_two,'#10
+ '                                 const int src_ld, const int src_offset,'#10
+ '                                 const int src_stride, __global const real* restrict src,'#10
+ '                                 const int dest_one, const int dest_two,'#10
+ '                                 const int dest_ld, const int dest_offset,'#10
+ '                                 const int dest_stride, __global real* dest,'#10
+ '                                 const int do_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset_batch = src_offset + src_stride * batch;'#10
+ '  const int dest_offset_batch = dest_offset + dest_stride * batch;'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  _CopyPadMatrix(src_one, src_two, src_ld, src_offset_batch, src,'#10
+ '                 dest_one, dest_two, dest_ld, dest_offset_batch, dest,'#10
+ '                 alpha, do_conjugate);'#10
+ '}'#10
+ '// Strided-batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))'#10
+ '#endif'#10
+ 'void CopyMatrixStridedBatched(const int src_one, const int src_two,'#10
+ '                              const int src_ld, const int src_offset,'#10
+ '                              const int src_stride, __global const real* restrict src,'#10
+ '                              const int dest_one, const int dest_two,'#10
+ '                              const int dest_ld, const int dest_offset,'#10
+ '                              const int dest_stride, __global real* dest) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset_batch = src_offset + src_stride * batch;'#10
+ '  const int dest_offset_batch = dest_offset + dest_stride * batch;'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  _CopyMatrix(src_one, src_two, src_ld, src_offset_batch, src,'#10
+ '              dest_one, dest_two, dest_ld, dest_offset_batch, dest,'#10
+ '              alpha, 0, 0, 0);'#10
+ '}'#10
