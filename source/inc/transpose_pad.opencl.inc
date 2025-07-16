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
+ '// kernels to transpose matrices in various ways, including:'#10
+ '// 1) transposing into a larger matrix by adding padding'#10
+ '// 2) transposing into a smaller matrix by optionally removing padding. This is the general version'#10
+ '//    without restrictions, see the ''transpose.opencl'' file for a faster but more restricted'#10
+ '//    transpose kernel.'#10
+ '//'#10
+ '// ================================================================================================='#10
+ '// Enables loading of this file using the C++ pre-processor''s #include (C++11 standard raw string'#10
+ '// literal). Comment-out this line for syntax-highlighting when developing.'#10
+ '// ================================================================================================='#10
+ '// Transposes a matrix from source to destination. The output is padded with zero values in case the'#10
+ '// destination matrix dimensions are larger than the transposed source matrix dimensions.'#10
+ 'INLINE_FUNC void _TransposePadMatrix(LOCAL_PTR real* tile,'#10
+ '                                     const int src_one, const int src_two,'#10
+ '                                     const int src_ld, const int src_offset,'#10
+ '                                     __global const real* restrict src,'#10
+ '                                     const int dest_one, const int dest_two,'#10
+ '                                     const int dest_ld, const int dest_offset,'#10
+ '                                     __global real* dest,'#10
+ '                                     const real alpha,'#10
+ '                                     const int do_conjugate) {'#10
+ '  // Loop over the work per thread'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {'#10
+ '      // Computes the identifiers for the source matrix. Note that the local and global dimensions'#10
+ '      // do not correspond to each other!'#10
+ '      const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);'#10
+ '      const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);'#10
+ '      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.'#10
+ '      // Otherwise, set the local memory value to zero.'#10
+ '      real value;'#10
+ '      SetToZero(value);'#10
+ '      if (id_src_two < src_two && id_src_one < src_one) {'#10
+ '        value = src[id_src_two*src_ld + id_src_one + src_offset];'#10
+ '      }'#10
+ '      const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;'#10
+ '      const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;'#10
+ '      tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;'#10
+ '    }'#10
+ '  }'#10
+ '  // Synchronizes all threads in a workgroup'#10
+ '  barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '  // Loop over the work per thread'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {'#10
+ '      // Computes the identifiers for the destination matrix'#10
+ '      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);'#10
+ '      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);'#10
+ '      // Stores the transposed value in the destination matrix'#10
+ '      if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {'#10
+ '        const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;'#10
+ '        const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;'#10
+ '        real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];'#10
+ '        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }'#10
+ '        Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Interface to the above function'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))'#10
+ '#endif'#10
+ 'void TransposePadMatrix(const int src_one, const int src_two,'#10
+ '                        const int src_ld, const int src_offset,'#10
+ '                        __global const real* restrict src,'#10
+ '                        const int dest_one, const int dest_two,'#10
+ '                        const int dest_ld, const int dest_offset,'#10
+ '                        __global real* dest,'#10
+ '                        const real_arg arg_alpha,'#10
+ '                        const int do_conjugate) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];'#10
+ '  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,'#10
+ '                      dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '                      alpha, do_conjugate);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Transposes a matrix, while considering possible padding in the source matrix. Data is read from a'#10
+ '// padded source matrix, but only the actual data is written back to the transposed destination'#10
+ '// matrix. This kernel optionally checks for upper/lower triangular matrices.'#10
+ 'INLINE_FUNC void _TransposeMatrix(LOCAL_PTR real* tile,'#10
+ '                                  const int src_one, const int src_two,'#10
+ '                                  const int src_ld, const int src_offset,'#10
+ '                                  __global const real* restrict src,'#10
+ '                                  const int dest_one, const int dest_two,'#10
+ '                                  const int dest_ld, const int dest_offset,'#10
+ '                                  __global real* dest,'#10
+ '                                  const real alpha,'#10
+ '                                  const int upper, const int lower,'#10
+ '                                  const int diagonal_imag_zero) {'#10
+ '  // Loop over the work per thread'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {'#10
+ '      // Computes the identifiers for the source matrix. Note that the local and global dimensions'#10
+ '      // do not correspond to each other!'#10
+ '      const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);'#10
+ '      const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);'#10
+ '      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.'#10
+ '      if ((id_src_one < src_one) && (id_src_two < src_two)) {'#10
+ '        real value = src[id_src_two*src_ld + id_src_one + src_offset];'#10
+ '        const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;'#10
+ '        const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;'#10
+ '        tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '  // Synchronizes all threads in a workgroup'#10
+ '  barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '  // Loop over the work per thread'#10
+ '  #pragma unroll'#10
+ '  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {'#10
+ '    #pragma unroll'#10
+ '    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {'#10
+ '      // Computes the identifiers for the destination matrix'#10
+ '      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);'#10
+ '      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);'#10
+ '      // Masking in case of triangular matrices: updates only the upper or lower part'#10
+ '      bool condition = true;'#10
+ '      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)'#10
+ '        if (upper == 1) { condition = (id_dest_one >= id_dest_two); }'#10
+ '        else if (lower == 1) { condition = (id_dest_one <= id_dest_two); }'#10
+ '      #endif'#10
+ '      if (condition) {'#10
+ '        // Stores the transposed value in the destination matrix'#10
+ '        if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {'#10
+ '          const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;'#10
+ '          const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;'#10
+ '          real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];'#10
+ '          if (diagonal_imag_zero == 1 && id_dest_one == id_dest_two) { ImagToZero(value); }'#10
+ '          Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// Interface to the above function'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))'#10
+ '#endif'#10
+ 'void TransposeMatrix(const int src_one, const int src_two,'#10
+ '                     const int src_ld, const int src_offset,'#10
+ '                     __global const real* restrict src,'#10
+ '                     const int dest_one, const int dest_two,'#10
+ '                     const int dest_ld, const int dest_offset,'#10
+ '                     __global real* dest,'#10
+ '                     const real_arg arg_alpha,'#10
+ '                     const int upper, const int lower,'#10
+ '                     const int diagonal_imag_zero) {'#10
+ '  const real alpha = GetRealArg(arg_alpha);'#10
+ '  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];'#10
+ '  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,'#10
+ '                   dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '                   alpha, upper, lower, diagonal_imag_zero);'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))'#10
+ '#endif'#10
+ 'void TransposePadMatrixBatched(const int src_one, const int src_two,'#10
+ '                               const int src_ld, const __constant int* src_offsets,'#10
+ '                               __global const real* restrict src,'#10
+ '                               const int dest_one, const int dest_two,'#10
+ '                               const int dest_ld, const __constant int* dest_offsets,'#10
+ '                               __global real* dest,'#10
+ '                               const int do_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset = src_offsets[batch];'#10
+ '  const int dest_offset = dest_offsets[batch];'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];'#10
+ '  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,'#10
+ '                      dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '                      alpha, do_conjugate);'#10
+ '}'#10
+ '// Batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))'#10
+ '#endif'#10
+ 'void TransposeMatrixBatched(const int src_one, const int src_two,'#10
+ '                            const int src_ld, const __constant int* src_offsets,'#10
+ '                            __global const real* restrict src,'#10
+ '                            const int dest_one, const int dest_two,'#10
+ '                            const int dest_ld, const __constant int* dest_offsets,'#10
+ '                            __global real* dest) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset = src_offsets[batch];'#10
+ '  const int dest_offset = dest_offsets[batch];'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];'#10
+ '  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,'#10
+ '                   dest_one, dest_two, dest_ld, dest_offset, dest,'#10
+ '                   alpha, 0, 0, 0);'#10
+ '}'#10
+ '// ================================================================================================='#10+ '// Strided-batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))'#10
+ '#endif'#10
+ 'void TransposePadMatrixStridedBatched(const int src_one, const int src_two,'#10
+ '                                      const int src_ld, const int src_offset,'#10
+ '                                      const int src_stride, __global const real* restrict src,'#10
+ '                                      const int dest_one, const int dest_two,'#10
+ '                                      const int dest_ld, const int dest_offset,'#10
+ '                                      const int dest_stride, __global real* dest,'#10
+ '                                      const int do_conjugate) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset_batch = src_offset + src_stride * batch;'#10
+ '  const int dest_offset_batch = dest_offset + dest_stride * batch;'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];'#10
+ '  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset_batch, src,'#10
+ '                      dest_one, dest_two, dest_ld, dest_offset_batch, dest,'#10
+ '                      alpha, do_conjugate);'#10
+ '}'#10
+ '// Strided-batched version of the above'#10
+ '#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '#else'#10
+ '  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))'#10
+ '#endif'#10
+ 'void TransposeMatrixStridedBatched(const int src_one, const int src_two,'#10
+ '                                   const int src_ld, const int src_offset,'#10
+ '                                   const int src_stride, __global const real* restrict src,'#10
+ '                                   const int dest_one, const int dest_two,'#10
+ '                                   const int dest_ld, const int dest_offset,'#10
+ '                                   const int dest_stride, __global real* dest) {'#10
+ '  const int batch = get_group_id(2);'#10
+ '  const int src_offset_batch = src_offset + src_stride * batch;'#10
+ '  const int dest_offset_batch = dest_offset + dest_stride * batch;'#10
+ '  real alpha; SetToOne(alpha);'#10
+ '  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];'#10
+ '  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset_batch, src,'#10
+ '                   dest_one, dest_two, dest_ld, dest_offset_batch, dest,'#10
+ '                   alpha, 0, 0, 0);'#10
+ '}'#10
