''
+ '#pragma OPENCL EXTENSION cl_khr_fp16 : enable'#10
+ '//#define nfloat  float'#10
+ '#define sEPSILON 0.0000001f'#10
+ '/*'#10
+ 'inline nfloat mapX(const nfloat x){'#10
+ '  return x*3-2.1F;'#10
+ '}'#10
+ '// Same purpose as mapX'#10
+ '// [0, 1] -> [-1.25, 1.25]'#10
+ 'inline nfloat mapY(const nfloat y){'#10
+ '  return y*3 - 1.5F;'#10
+ '}'#10
+ '#define max_iteration  10000'#10
+ '#define _max           4.0f'#10
+ '__kernel void mandel(__global uchar *buf, const int w, const int h){'#10
+ '  const nfloat lnxp1_max_iteration = log1p((nfloat)max_iteration);'#10
+ '  int y = get_global_id(0);'#10
+ '  int x = get_global_id(1);'#10
+ '  nfloat xx = mapX(x/(nfloat)w);'#10
+ '  nfloat yy = mapY(y/(nfloat)h);'#10
+ '  y *= w * sizeof(uint);'#10
+ '  x *= sizeof(uint);'#10
+ '  nfloat x0 = 0.0f; nfloat y0 = 0.0f;'#10
+ '  int iteration = 0;'#10
+ '  nfloat oldAbs = 0.0f;'#10
+ '  nfloat coverageNum = max_iteration;'#10
+ '  buf += y;'#10
+ '  while (iteration < max_iteration) {'#10
+ '      nfloat xtemp = x0 * x0 - y0 * y0;'#10
+ '      y0 = 2 * x0 * y0;'#10
+ '      x0 = xtemp;'#10
+ '      x0 = x0 + xx;'#10
+ '      y0 = y0 + yy;'#10
+ '      nfloat currentAbs = x0*x0 + y0*y0;'#10
+ '      if (currentAbs>4.0f){'#10
+ '         nfloat diffToLast  = currentAbs - oldAbs;'#10
+ '         nfloat diffToMax   =       _max - oldAbs;'#10
+ '         coverageNum = iteration + diffToMax/diffToLast;'#10
+ '         break;'#10
+ '      }'#10
+ '      oldAbs = currentAbs;'#10
+ '      iteration++;'#10
+ '  }'#10
+ '  if (iteration == max_iteration)'#10
+ '#if defined(__MACH__)'#10
+ '  {'#10
+ '      buf[x] = 0xff;'#10
+ '      buf[x+1] = 0;'#10
+ '      buf[x+2] = 0;'#10
+ '      buf[x+3] = 0;'#10
+ '  } else'#10
+ '  {'#10
+ '      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;'#10
+ '      buf[x+0] = 0xff;'#10
+ '      buf[x+1] = c;'#10
+ '      buf[x+2] = c;'#10
+ '      buf[x+3] = c;'#10
+ '   }'#10
+ '#else'#10
+ '  {'#10
+ '      buf[x] = 0;'#10
+ '      buf[x+1] = 0;'#10
+ '      buf[x+2] = 0;'#10
+ '      buf[x+3] = 0xff;'#10
+ '  } else'#10
+ '  {'#10
+ '      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;'#10
+ '      buf[x+0] = c;'#10
+ '      buf[x+1] = c;'#10
+ '      buf[x+2] = c;'#10
+ '      buf[x+3] = 0xff;'#10
+ '  }'#10
+ '#endif'#10
+ '}'#10
+ '*/'#10
+ '  //     K          N          N'#10
+ '  //   [...]      [...]      [...]'#10
+ '  // M [.A.]  X K [.B.] => M [.C.]'#10
+ '  //   [...]      [...]      [...]'#10
+ 'typedef enum {'#10
+ '    acLOGISTIC, acRELU, acRELU6, acRELIE, acLINEAR, acRAMP, acTANH, acPLSE,'#10
+ '    acREVLEAKY, acLEAKY, acELU, acLOGGY, acSTAIR, acHARDTAN, acLHTAN, acSELU, acSOFTMAX,'#10
+ '    acGELU, acSWISH, acMISH, acHARD_MISH, acNORM_CHAN, acNORM_CHAN_SOFTMAX,'#10
+ '    acNORM_CHAN_SOFTMAX_MAXVAL'#10
+ '  } ActivationType;'#10
+ 'nfloat sumv(const n_int N, __global const nfloat* v, const n_int stride){'#10
+ '  nfloat sum=0;'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0;i<N; i++)'#10
+ '    sum += v[i*stride];'#10
+ '  return sum;'#10
+ '}'#10
+ 'nfloat maxv(const n_int N, __global const nfloat* v, const n_int stride){'#10
+ '  nfloat m=v[0];'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=1;i<N; i++)'#10
+ '    m = max(m, v[i*stride]);'#10
+ '  return m;'#10
+ '}'#10
+ 'nfloat minv(const n_int N, __global const nfloat* v, const n_int stride){'#10
+ '  nfloat m=v[0];'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=1;i<N; i++)'#10
+ '    m = min(m, v[i*stride]);'#10
+ '  return m;'#10
+ '}'#10
+ 'nfloat rssv(const n_int N, const nfloat mean, __global const nfloat* src, const n_int stride){'#10
+ '  nfloat sum=0;'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0;i<N; i++){'#10
+ '    const nfloat v = src[i*stride] - mean;'#10
+ '    sum += v*v;'#10
+ '  }'#10
+ '  return sum;'#10
+ '}'#10
+ 'nfloat sqr(const nfloat x){'#10
+ '  return x*x;'#10
+ '}'#10
+ '//#define VW 8'#10
+ '//nfloat sumv_simd(const n_int N, __global nfloat* v){'#10
+ '//  float8 sum4 = 0;'#10
+ '//  nfloat sum = 0;'#10
+ '//  n_int n = N / VW;'#10
+ '//'#10
+ '//  #pragma unroll 8'#10
+ '//  for (n_int i=0;i<n; i++)'#10
+ '//    sum4 += vload8(i, v);'#10
+ '//  v += n*VW;'#10
+ '//'#10
+ '//  #pragma unroll 8'#10
+ '//  for (n_int i=0; i<N%VW;i++)'#10
+ '//    sum4[i] += v[i];'#10
+ '//  return sum4[0] + sum4[1] + sum4[2] + sum4[3] + sum4[4] + sum4[5] + sum4[6] + sum4[7] ;'#10
+ '//}'#10
+ 'nfloat dotv(const n_int N, __global nfloat* a, const n_int inca,  __global nfloat* b, const n_int incb){'#10
+ '  nfloat d = 0;'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0; i<N;i++)'#10
+ '    d += a[i*inca]*b[i*incb];'#10
+ '  return d;'#10
+ '}'#10
+ '//nfloat dotv_simd(const n_int N, __global nfloat* a,  __global nfloat* b){'#10
+ '//'#10
+ '//  float8 d = 0;'#10
+ '//  n_int n = N / VW;'#10
+ '//  #pragma unroll 8'#10
+ '//  for (n_int i=0; i<n;i++)'#10
+ '//    d += vload8(i, a) * vload8(i, b);'#10
+ '//  a += n*VW;'#10
+ '//  b += n*VW;'#10
+ '//'#10
+ '//  #pragma unroll 8'#10
+ '//  for (n_int i=0; i<N%VW;i++)'#10
+ '//    d.x += a[i]*b[i];'#10
+ '//'#10
+ '//  return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] ;'#10
+ '//}'#10
+ '#define TSM 128                // The tile-size in dimension M'#10
+ '#define TSN 128                // The tile-size in dimension N'#10
+ '#define TSK 16                 // The tile-size in dimension K'#10
+ '#define WPTM 8                 // The work-per-thread in dimension M'#10
+ '#define WPTN 8                 // The work-per-thread in dimension N'#10
+ '#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M'#10
+ '#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N'#10
+ '#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A'#10
+ '#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B'#10
+ '              // naive GEMM with unrolling for now'#10
+ '__kernel void sgemm1_nn(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,'#10
+ '                      __global nfloat* A, const n_int aOffset, const n_int lda,'#10
+ '                      __global nfloat* B, const n_int bOffset, const n_int ldb,'#10
+ '                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {'#10
+ '    const n_int globalRow = get_global_id(0); // Row ID of C (0..M)'#10
+ '    const n_int globalCol = get_global_id(1); // Col ID of C (0..N)'#10
+ '    A += globalRow*lda + aOffset ;'#10
+ '    B += globalCol + bOffset;'#10
+ '    C += globalRow*ldc + globalCol + cOffset;'#10
+ '    *C *= BETA;'#10
+ '    *C += dotv(K, A, 1, B, ldb) * ALPHA;'#10
+ '}'#10
+ '__kernel void sgemm2_nn(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,'#10
+ '                      __global nfloat* A, const n_int aOffset, const n_int lda,'#10
+ '                      __global nfloat* B, const n_int bOffset, const n_int ldb,'#10
+ '                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {'#10
+ '    const n_int globalRow = get_global_id(1); // Row ID of C (0..M)'#10
+ '    const n_int globalCol = get_global_id(0); // Col ID of C (0..N)'#10
+ '    A += globalRow*lda + aOffset ;'#10
+ '    B += globalCol + bOffset;'#10
+ '    C += globalRow*ldc + globalCol + cOffset;'#10
+ '    *C *= BETA;'#10
+ '    *C += dotv(K, A, 1, B, ldb) * ALPHA;'#10
+ '}'#10
+ '__kernel void sgemm1_nt(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,'#10
+ '                      __global nfloat* A, const n_int aOffset, const n_int lda,'#10
+ '                      __global nfloat* B, const n_int bOffset, const n_int ldb,'#10
+ '                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {'#10
+ '    const n_int globalRow = get_global_id(0);    // M'#10
+ '    const n_int globalCol = get_global_id(1);    // N'#10
+ '    A += globalRow*lda + aOffset;'#10
+ '    B += globalCol*ldb + bOffset;'#10
+ '    C += globalRow*ldc + globalCol + cOffset;'#10
+ '    *C *= BETA;'#10
+ '    //nfloat acc =0;'#10
+ '    //#pragma unroll 8'#10
+ '    //for (n_int k=0; k<K; k++)'#10
+ '    //    acc += A[k] * B[k];'#10
+ '    //*C += acc * ALPHA;'#10
+ '    //*C += dotv(K, A, 1, B, 1) * ALPHA;'#10
+ '    *C += dotv(K, A, 1, B, 1) * ALPHA;'#10
+ '            //}'#10
+ '}'#10
+ '__kernel void sgemm1_tn(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,'#10
+ '                      __global nfloat* A, const n_int aOffset, const n_int lda,'#10
+ '                      __global nfloat* B, const n_int bOffset, const n_int ldb,'#10
+ '                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {'#10
+ '    const n_int row = get_global_id(0); // Row ID of C (0..M)'#10
+ '    const n_int col = get_global_id(1); // Col ID of C (0..N)'#10
+ '    A += row           +  aOffset;'#10
+ '    B += col           +  bOffset;'#10
+ '    C += row*ldc + col +  cOffset;'#10
+ '    *C *= BETA;'#10
+ '    //nfloat acc = 0;'#10
+ '    //#pragma unroll 8'#10
+ '    //for (n_int k=0; k<K; k++)'#10
+ '    //    acc += A[k * lda] * B[k * ldb] ;'#10
+ '    //*C += acc * ALPHA ;'#10
+ '    *C += dotv(K, A, lda, B, ldb) * ALPHA ;'#10
+ '}'#10
+ '__kernel void forward_bias(const int reshape, __global nfloat* a,  const n_int aOffset, /*const n_int blockSize, */__global nfloat* b, const n_int bOffset, const n_int incb)'#10
+ '{'#10
+ '  n_int N, blockSize, i, k, j;'#10
+ '  switch (reshape) {'#10
+ '    case 0:'#10
+ '      N = get_global_size(0);'#10
+ '      blockSize = get_global_size(1);'#10
+ '      i = get_global_id(0);'#10
+ '      k = get_global_id(2);'#10
+ '      j = get_global_id(1);'#10
+ '      break;'#10
+ '    case 1:'#10
+ '      N = get_global_size(1);'#10
+ '      blockSize = get_global_size(0);'#10
+ '      i = get_global_id(1);'#10
+ '      k = get_global_id(2);'#10
+ '      j = get_global_id(0);'#10
+ '      break;'#10
+ '  }'#10
+ '  a += aOffset;'#10
+ '  //if (i==0 && k==0){'#10
+ '    //printf("        N = %ld\n", N);'#10
+ '    //printf("blockSize = %ld\n", blockSize);'#10
+ '  //}'#10
+ '  //for (i = 0; i<N; i++)'#10
+ '  //  for (k = 0; k<batch; k++){'#10
+ '  a += (k*N + i)*blockSize;'#10
+ '  nfloat bb = b[i * incb];'#10
+ '  //#pragma unroll 8'#10
+ '  //    for (n_int j=0; j<blockSize; j++)'#10
+ '        a[j] += bb;'#10
+ '  //}'#10
+ '}'#10
+ 'nfloat stair_activate(const nfloat x)'#10
+ '{'#10
+ '  n_int n = floor(x);'#10
+ '  if (n % 2 == 0) return floor(x/ 2);'#10
+ '  return (x - n) + floor(x/2);'#10
+ '}'#10
+ 'nfloat hardtan_activate(const nfloat x)'#10
+ '{'#10
+ '    if (x < -1) return -1;'#10
+ '    if (x > 1) return 1;'#10
+ '    return (x);'#10
+ '}'#10
+ 'nfloat linear_activate(const nfloat x)'#10
+ '{'#10
+ '  return x;'#10
+ '}'#10
+ 'nfloat logistic_activate(const nfloat x)'#10
+ '{'#10
+ '  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))'#10
+ '  return 1/(1 + exp(-x));'#10
+ '}'#10
+ 'nfloat loggy_activate(const nfloat x)'#10
+ '{'#10
+ '  //result := 2/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp))) - 1;'#10
+ '  return 2/(1 + exp(-x)) - 1;'#10
+ '}'#10
+ 'nfloat relu_activate(const nfloat x)'#10
+ '{'#10
+ '  //return x*n_int(x>0);'#10
+ '  if (x<0) return 0;'#10
+ '  return x;'#10
+ '}'#10
+ 'nfloat relu6_activate(const nfloat x)'#10
+ '{'#10
+ '  //min_val_cmp(max_val_cmp(x, 0), 6)'#10
+ '  //result := EnsureRange(x,0,6);'#10
+ '  return  x*(x>0) * (x<=6);'#10
+ '}'#10
+ 'nfloat elu_activate(const nfloat x)'#10
+ '{'#10
+ '  return (x >= 0)*x + (x < 0)*(exp(x)-1);'#10
+ '}'#10
+ 'nfloat selu_activate(const nfloat x)'#10
+ '{'#10
+ '  return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);'#10
+ '}'#10
+ 'nfloat gelu_activate(const nfloat x)'#10
+ '{'#10
+ '  return 0.5f*x*(1 + tanh(0.797885f*x + 0.035677f*pow(x, 3)));'#10
+ '}'#10
+ 'nfloat relie_activate(const nfloat x)'#10
+ '{'#10
+ '  if (x>0) return x;'#10
+ '  else return 0.01f*x;'#10
+ '}'#10
+ 'nfloat ramp_activate(const nfloat x)'#10
+ '{'#10
+ '  return  x*(x>0)+0.1f*x;'#10
+ '}'#10
+ 'nfloat leaky_activate(const nfloat x)'#10
+ '{'#10
+ '  if (x>0) return  x;'#10
+ '  else return  0.1f*x;'#10
+ '}'#10
+ 'nfloat tanh_activate(const nfloat x)'#10
+ '{'#10
+ '  //const nfloat px = exp(x);'#10
+ '  //const nfloat nx = exp(-x);'#10
+ '  //return (px - nx)/(px + nx);'#10
+ '  //return 2 / (1 + exp(ensureRange(-2 * x, minSingleExp, maxSingleExp))) - 1'#10
+ '  return 2/ (1+exp(-2*x)) - 1 ;'#10
+ '//  return  (exp(2*x)-1)/(exp(2*x)+1);'#10
+ '}'#10
+ 'nfloat softplus_activate(const nfloat x, const nfloat threshold)'#10
+ '{'#10
+ '    if (x > threshold)'#10
+ '      return (x);                // too large'#10
+ '    else if (x < -threshold)'#10
+ '      return (exp(x));    // too small'#10
+ '    //return (log(exp(x) + 1));'#10
+ '    return log1p(exp(x));'#10
+ '}'#10
+ 'nfloat plse_activate(const nfloat x)'#10
+ '{'#10
+ '    if (x < -4 ) return( 0.01f * (x + 4));'#10
+ '    if (x > 4 ) return( 0.01f * (x - 4) + 1);'#10
+ '    return  0.125f*x + 0.5f;'#10
+ '}'#10
+ 'nfloat lhtan_activate(const nfloat x)'#10
+ '{'#10
+ '    if(x < 0) return (0.001f*x);'#10
+ '    if(x > 1) return (0.001f*(x-1) + 1);'#10
+ '    return  x;'#10
+ '}'#10
+ 'nfloat silu_activate(const nfloat x)'#10
+ '{'#10
+ '    return x * logistic_activate(x) ;'#10
+ '}'#10
+ '#define  MISH_THRESHOLD 20.0f'#10
+ 'nfloat mish_activate(const nfloat x)'#10
+ '{'#10
+ '    return x*tanh_activate(softplus_activate(x, MISH_THRESHOLD));'#10
+ '}'#10
+ '//void softmax_activate(const N:SizeInt; const x: PSingle);'#10
+ '//{'#10
+ '//  n_int i;'#10
+ '//  nfloat mx := TSingleTensor.maxv(N, Pointer(x), 1);//MaxValue(x, N);'#10
+ '//  for i:=0 to N-1 do'#10
+ '//    //x[i] := Exp(EnsureRange(x[i]-mx, minSingleExp, maxSingleExp));'#10
+ '//    x[i] := Exp(x[i]-mx);'#10
+ '//'#10
+ '//  mx := TSingleTensor.Sumv(N, pointer(x), 1);'#10
+ '//  //r:=copy(x);'#10
+ '//  //r.Exp();'#10
+ '//  for i :=0 to N-1 do'#10
+ '//    x[i] := x[i] / mx'#10
+ '//}'#10
+ '__kernel void activate_array( __global nfloat* x, n_int const offset, const ActivationType a)'#10
+ '{'#10
+ '      const n_int i = get_global_id(0);'#10
+ '      //int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);'#10
+ '      x += offset;'#10
+ '      switch (a) {'#10
+ '          case acLOGISTIC:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = logistic_activate(x[i]);'#10
+ '            break;'#10
+ '          case acRELU:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = relu_activate(x[i]);'#10
+ '            break;'#10
+ '          case acRELU6:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = relu6_activate(x[i]);'#10
+ '            break;'#10
+ '          case acRELIE:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = relie_activate(x[i]);'#10
+ '            break;'#10
+ '          case acLINEAR:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '            //    x[i] = linear_activate(x[i])'#10
+ '            break;'#10
+ '          case acRAMP:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = ramp_activate(x[i]);'#10
+ '            break;'#10
+ '          case acTANH:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = tanh_activate(x[i]);'#10
+ '            break;'#10
+ '          case acPLSE:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = plse_activate(x[i]);'#10
+ '            break;'#10
+ '          case acREVLEAKY: case acLEAKY:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '             if (x[i]<0) x[i] = 0.1f*x[i];'#10
+ '              //x[i] = leaky_activate(x[i]);'#10
+ '            break;'#10
+ '          case acELU:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = elu_activate(x[i]);'#10
+ '            break;'#10
+ '          case acLOGGY:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = loggy_activate(x[i]);'#10
+ '            break;'#10
+ '          case acSTAIR:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = stair_activate(x[i]);'#10
+ '            break;'#10
+ '          case acHARDTAN:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = hardtan_activate(x[i]);'#10
+ '            break;'#10
+ '          case acLHTAN:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = lhtan_activate(x[i]);'#10
+ '            break;'#10
+ '          case acSELU:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = selu_activate(x[i]);'#10
+ '            break;'#10
+ '          case acGELU:'#10
+ '            //for (i = 0; i< N; i++)'#10
+ '                x[i] = gelu_activate(x[i]);'#10
+ '            break;'#10
+ '          case acSWISH:'#10
+ '                x[i] = silu_activate(x[i]);'#10
+ '            break;'#10
+ '          case acMISH:'#10
+ '                x[i] = mish_activate(x[i]);'#10
+ '            break;'#10
+ '          //case acSOFTMAX:'#10
+ '            //softmax_activate(N, x);'#10
+ '            //break'#10
+ '          default:'#10
+ '            if (i==0) printf("[Activation] %d: not Implemented\n", (int)a);'#10
+ '      }'#10
+ '   //printf("%ld, ", i);'#10
+ '}'#10
+ '__kernel void array_activate_swish(__global nfloat* x, n_int const offset, __global nfloat* output, __global nfloat* output2)'#10
+ '{'#10
+ '    n_int i = get_global_id(0);'#10
+ '    x += offset;'#10
+ '    output += offset;'#10
+ '    output2 += offset;'#10
+ '    nfloat x_val       = x[i];'#10
+ '    nfloat sigmoid     = logistic_activate(x_val);'#10
+ '    output[i]         = sigmoid;'#10
+ '    output2[i]        = x_val * sigmoid;'#10
+ '}'#10
+ 'nfloat lhtan_gradient(const nfloat x)'#10
+ '{'#10
+ '    if ((x > 0) &&  (x < 1))'#10
+ '      return 1;'#10
+ '    return 0.001f;'#10
+ '}'#10
+ 'nfloat hardtan_gradient(const nfloat x)'#10
+ '{'#10
+ '    if ((x > -1) && (x < 1))'#10
+ '      return 1;'#10
+ '    return 0;'#10
+ '}'#10
+ 'nfloat linear_gradient(const nfloat x)'#10
+ '{'#10
+ '    return 1;'#10
+ '}'#10
+ 'nfloat logistic_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (1-x)*x;'#10
+ '}'#10
+ 'nfloat loggy_gradient(const nfloat x)'#10
+ '{'#10
+ '    nfloat y = (x+1.0f)/2.0f;'#10
+ '    return 2.0f*(1.0f-y)*y;'#10
+ '}'#10
+ 'nfloat stair_gradient(const nfloat x)'#10
+ '{'#10
+ '    if (floor(x) == x) return( 0);'#10
+ '    return 1;'#10
+ '}'#10
+ 'nfloat relu_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x>0?1:0);'#10
+ '}'#10
+ 'nfloat relu6_gradient(const nfloat x)'#10
+ '{'#10
+ '    return ((x>0) && (x<6)?1:0);'#10
+ '}'#10
+ 'nfloat elu_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x >= 0?1:0) + (x < 0?1:0)*(x + 1);'#10
+ '}'#10
+ 'nfloat selu_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x >= 0?1:0)*1.0507f + (x < 0?1:0)*(x + 1.0507f*1.6732f);'#10
+ '}'#10
+ 'nfloat relie_gradient(const nfloat x)'#10
+ '{'#10
+ '    if (x>0) return 1;'#10
+ '    else return 0.01f;'#10
+ '}'#10
+ 'nfloat ramp_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x>0?1:0) + 0.1f;'#10
+ '}'#10
+ 'nfloat leaky_gradient(const nfloat x)'#10
+ '{'#10
+ '    if (x>0) return 1;'#10
+ '    else return 0.1f;'#10
+ '}'#10
+ 'nfloat tanh_gradient(const nfloat x)'#10
+ '{'#10
+ '    return 1.0f-x*x;'#10
+ '}'#10
+ 'nfloat sech(const nfloat x)'#10
+ '{'#10
+ '    return 2.0f / (exp(x) + exp(-x));'#10
+ '}'#10
+ 'nfloat gelu_gradient(const nfloat x)'#10
+ '{'#10
+ '    nfloat x3 = x*x*x;'#10
+ '    nfloat sec = sech(0.0356774f*x3 + 0.797885f*x);'#10
+ '    return 0.5f*tanh(0.0356774f*x3 + 0.797885f*x) + (0.0535161f*x3 + 0.398942f*x) * sec*sec + 0.5f ;'#10
+ '}'#10
+ 'nfloat plse_gradient(const nfloat x)'#10
+ '{'#10
+ '  if ((x < 0) || (x > 1))'#10
+ '    return  0.01f;'#10
+ '  else'#10
+ '    return 0.125f;'#10
+ '}'#10
+ '__kernel void gradient_array(__global nfloat* x, n_int const offset, const ActivationType a, __global nfloat* delta)'#10
+ '{'#10
+ '    n_int i = get_global_id(0);'#10
+ '    x += offset;'#10
+ '    delta += offset;'#10
+ '    switch (a) {'#10
+ '        case acLOGISTIC:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= logistic_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acRELU:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= x[i]>0?1:0;//relu_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acRELU6:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= relu6_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acRELIE:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= relie_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acLINEAR:'#10
+ '          //////for (i = 0; i<N;i++)'#10
+ '          //    delta[i] *= linear_gradient(x[i])'#10
+ '          //;'#10
+ '              break;'#10
+ '        case acRAMP:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= ramp_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acTANH:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= tanh_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acPLSE:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= plse_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acREVLEAKY: case acLEAKY:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= leaky_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acELU:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= elu_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acLOGGY:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= loggy_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acSTAIR:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= stair_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acHARDTAN:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= hardtan_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acLHTAN:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= lhtan_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acSELU:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= selu_gradient(x[i]);'#10
+ '              break;'#10
+ '        case acGELU:'#10
+ '          //for (i = 0; i<N;i++)'#10
+ '              delta[i] *= gelu_gradient(x[i]);'#10
+ '              break;'#10
+ '    //   case acSWISH:'#10
+ '    //               ;'#10
+ '    //'#10
+ '    //   case acMISH:'#10
+ '    //               ;'#10
+ '    //'#10
+ '    //   case acHARD_MISH:'#10
+ '    //               ;'#10
+ '    //'#10
+ '    //   case acNORM_CHAN:'#10
+ '    //               ;'#10
+ '    //'#10
+ '    //   case acNORM_CHAN_SOFTMAX:'#10
+ '    //               ;'#10
+ '    //'#10
+ '    //   case acNORM_CHAN_SOFTMAX_MAXVAL:'#10
+ '    //'#10
+ '        default:'#10
+ '            if (i==0) printf("[Gradient] : not Implemented %d\n", (int)a);'#10
+ '    }'#10
+ '}'#10
+ '// #define BLOCK 512'#10
+ '__kernel void backward_bias(__global nfloat* dst, const n_int blockSize, __global nfloat* src, const n_int srcOffset, const n_int batch)'#10
+ '{'#10
+ '    //const n_int filter = get_group_id(0);'#10
+ '    //const n_int p = get_local_id(0);'#10
+ '    //const n_int N = get_global_size(0);'#10
+ '    //'#10
+ '    //int i,b;'#10
+ '    //local float part[BLOCK];'#10
+ '    //'#10
+ '    //src += srcOffset;'#10
+ '    //float sum = 0;'#10
+ '    //for(b = 0; b < batch; ++b){'#10
+ '    //    for(i = 0; i < blockSize; i += BLOCK){'#10
+ '    //        int index = p + i + blockSize*(filter + N*b);'#10
+ '    //        sum += (p+i < blockSize) ? src[index] : 0;'#10
+ '    //    }'#10
+ '    //}'#10
+ '    //part[p] = sum;'#10
+ '    //'#10
+ '    ////__syncthreads();'#10
+ '    //barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    //if (p == 0) {'#10
+ '    //    printf("summing up :\n");'#10
+ '    //    for(i = 0; i < BLOCK; ++i){'#10
+ '    //      dst[filter] += part[i];'#10
+ '    //      printf("%f\n", part[i]);'#10
+ '    //    }'#10
+ '    //}'#10
+ '    const n_int i = get_global_id(0);//if (i==0) printf("n_int %ull\n", sizeof(n_int));'#10
+ '    const n_int N = get_global_size(0);'#10
+ '    //for (n_int i=0 ; i<N ;i++) {'#10
+ '      nfloat sum = 0;'#10
+ '      src += i * blockSize + srcOffset;'#10
+ '      const n_int incbias = N*blockSize;'#10
+ '      // take a shortcut'#10
+ '      if(blockSize==1){'#10
+ '        sum = sumv(batch, src, N);'#10
+ '        dst[i] +=sum;'#10
+ '        return;'#10
+ '      }'#10
+ '      #pragma unroll 8'#10
+ '      for (n_int j=0; j<batch; j++){'#10
+ '        sum += sumv(blockSize, src, 1);'#10
+ '        src += incbias;'#10
+ '      }'#10
+ '      dst[i] +=sum;'#10
+ '}'#10
+ '__kernel void addv( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* dst, const n_int dstOffset, const n_int incc){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] + src2[i*incb + src2Offset];'#10
+ '}'#10
+ '__kernel void subv( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* dst, const n_int dstOffset, const n_int incc){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] - src2[i*incb + src2Offset];'#10
+ '}'#10
+ '__kernel void axpy(const nfloat a, __global nfloat* x, const n_int xOffset, const n_int incx, __global nfloat* y, const n_int yOffset, const n_int incy){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   x += xOffset;'#10
+ '   y += yOffset;'#10
+ '   y[i*incy] += a*x[i*incx];'#10
+ '}'#10
+ '__kernel void scale(const nfloat a, __global nfloat* x, const n_int incx){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   x[i*incx] *= a;'#10
+ '}'#10
+ '__kernel void crossEntropyLogistics(__global const nfloat* pred, __global const nfloat* truth, __global nfloat* delta, __global nfloat* error){'#10
+ '  const n_int i = get_global_id(0);'#10
+ '  nfloat t = truth[i];'#10
+ '  nfloat p = pred[i];'#10
+ '  error[i] = -t*log(max(p, sEPSILON)) - (1-t) * log(max(1 - p, sEPSILON));'#10
+ '  //error[i] = -t*log(p) - (1-t) * log(1 - p);'#10
+ '  delta[i] = t - p;'#10
+ '   //printf("%ld, ", i);'#10
+ '}'#10
+ '__kernel void fill(__global nfloat* x, const n_int offset, const nfloat val, const n_int stride){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   //x += offset;'#10
+ '   x[i*stride + offset] = val;'#10
+ '}'#10
+ '// naive copy for now'#10
+ '__kernel void copy('#10
+ '    __global nfloat* src, const n_int srcOffset, const n_int srcInc'#10
+ '  , __global nfloat* dst, const n_int dstOffset, const n_int dstInc){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   //src += srcOffset; dst += dstOffset;'#10
+ '   //if (srcInc==1 && dstInc==1){'#10
+ '   //  dst[i+dstOffset] = src[i+srcOffset];'#10
+ '   //  return;'#10
+ '   //}'#10
+ '   dst[i*dstInc + dstOffset] = src[i*srcInc + srcOffset];'#10
+ '}'#10
+ '__kernel void forward_maxpool('#10
+ '     __global nfloat* input'#10
+ '     , const n_int c, const n_int h, const n_int w'#10
+ '     , const n_int stride_x, const n_int stride_y, const n_int padding, const n_int kernelSize'#10
+ '     , __global n_int* indexes, __global nfloat* output){'#10
+ '  const n_int w_offset = -padding / 2;'#10
+ '  const n_int h_offset = -padding / 2;'#10
+ '  //const n_int outC = get_global_size(0);'#10
+ '  const n_int outH = get_global_size(1);'#10
+ '  const n_int outW = get_global_size(2);'#10
+ '  n_int k = get_global_id(0);'#10
+ '  n_int y = get_global_id(1);'#10
+ '  n_int x = get_global_id(2);'#10
+ '  n_int out_index = x + outW*(y + outH*k) ;//+ outW*outH*outC*b;'#10
+ '  nfloat max = -FLT_MAX;'#10
+ '  n_int max_i = -1;'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int n=0; n<kernelSize; n++)'#10
+ '      #pragma unroll 8'#10
+ '      for (n_int m=0; m<kernelSize; m++){'#10
+ '          n_int cur_h = h_offset+y * stride_y+n;'#10
+ '          n_int cur_w = w_offset+x * stride_x+m;'#10
+ '          n_int index = cur_w + w*(cur_h + h*k) ;//+ w*h*outC*b;'#10
+ '          nfloat val = (cur_h >= 0) && (cur_h < h) && (cur_w >= 0) && (cur_w < w)? input[index]: -FLT_MAX;'#10
+ '          if (val > max){'#10
+ '            max_i = index;'#10
+ '            max = val;'#10
+ '          }'#10
+ '      }'#10
+ '  output[out_index] = max;'#10
+ '  if (indexes)'#10
+ '      indexes[out_index] = max_i;'#10
+ '}'#10
+ '__kernel void backward_maxpool( __global nfloat* output, __global const n_int* indexes, __global const nfloat* delta){'#10
+ '        const n_int i = get_global_id(0);'#10
+ '        const n_int j = get_global_id(1);'#10
+ '        const n_int id = i*get_global_size(1) + j;'#10
+ '        const n_int index = indexes[id];'#10
+ '        output[index] += delta[id];'#10
+ '}'#10
+ 'void softmax(const n_int n, __global nfloat* input, const n_int stride, const nfloat temp, __global nfloat* output){'#10
+ '  nfloat largest = maxv(n, input, stride);'#10
+ '  nfloat sum = 0;'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0;i<n;i++) {'#10
+ '      nfloat e = exp((input[i*stride] - largest)/temp);'#10
+ '      sum += e;'#10
+ '      output[i*stride] = e;'#10
+ '  }'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0; i<n; i++)'#10
+ '      output[i*stride]/=sum;'#10
+ '}'#10
+ '__kernel void softmaxBatch(__global nfloat* input, const n_int iOffset, const n_int n'#10
+ '  , const n_int batch_size, const n_int group_size, const n_int stride'#10
+ '  , const nfloat temp, __global nfloat* output, const n_int oOffset){'#10
+ '  const n_int b = get_global_id(0);'#10
+ '  const n_int g = get_global_id(1);'#10
+ '  softmax(n'#10
+ '  , input + iOffset + b*batch_size + g*group_size'#10
+ '  , stride'#10
+ '  , temp'#10
+ '  , output + oOffset + b*batch_size + g*group_size);'#10
+ '}'#10
+ 'void move(const __global nfloat* src, __global nfloat* dst , const n_int count){'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0 ; i<count; i++) dst[i] = src[i];'#10
+ '}'#10
+ '__kernel void crossEntropySoftmax(const __global nfloat* pred, const __global nfloat* truth, __global nfloat* delta, __global nfloat* error){'#10
+ '  const n_int i = get_global_id(0);'#10
+ '  nfloat t = truth[i];'#10
+ '  nfloat p = pred[i];'#10
+ '  if (t!=0)'#10
+ '      error[i] = -log(max(p, sEPSILON));'#10
+ '      //error[i] = -log(p);'#10
+ '  else'#10
+ '      error[i] = 0;'#10
+ '  delta[i] = t - p;'#10
+ '}'#10
+ '__kernel void im2col(const n_int aHeight, const n_int aWidth'#10
+ '  , const n_int kernelHeight, const n_int kernelWidth, const n_int padHeight, const n_int padWidth'#10
+ '  , const n_int strideY, const n_int strideX, const n_int dilationY, const n_int dilationX'#10
+ '  , __global nfloat* im , const n_int imOffset'#10
+ '  , __global nfloat* col, const n_int colOffset, const n_int batch){'#10
+ '  n_int aChannels = get_global_size(0);'#10
+ '  n_int chan = get_global_id(0);'#10
+ '  n_int k = get_global_id(1);'#10
+ '  const n_int kernelSize = kernelHeight*kernelWidth;'#10
+ '  const n_int outWidth = (aWidth + 2 * padWidth - (dilationX * (kernelWidth - 1) + 1)) / strideX + 1;'#10
+ '  const n_int outHeight = (aHeight + 2 * padHeight - (dilationY * (kernelHeight - 1) + 1)) / strideY + 1;'#10
+ '  const n_int outSize = outWidth * outHeight;'#10
+ '  const n_int inSize = aWidth * aHeight;'#10
+ '  const n_int sizeX = outWidth - 2 * padWidth;'#10
+ '    //for (n_int k=0 ; k<kernelWidth*kernelHeight; k++)'#10
+ '    {'#10
+ '      n_int kernelRow = k / kernelWidth;'#10
+ '      n_int kernelCol = k % kernelWidth;'#10
+ '      //const n_int kernelRow = get_global_id(1);'#10
+ '      //const n_int kernelCol = get_global_id(2);'#10
+ '      #pragma unroll 8'#10
+ '      for (n_int b=0 ; b<batch; b++)'#10
+ '      {'#10
+ '        n_int i = (b*aChannels + chan)*inSize + aWidth*(kernelRow*dilationY - padHeight) + kernelCol*dilationX - padWidth;'#10
+ '        __global nfloat* im1 = imOffset + im + i;'#10
+ '        __global nfloat* col1 = colOffset + col + padWidth * outWidth + outSize*kernelSize*(chan + b*aChannels) + outSize*(kernelRow * kernelWidth + kernelCol) ;'#10
+ '        #pragma unroll 8'#10
+ '        for (n_int outRow=padHeight ; outRow<outHeight - padHeight ; outRow++)'#10
+ '        {'#10
+ '          n_int j = outRow * aWidth * strideY + padWidth * strideX ;'#10
+ '          if (strideX == 1)'#10
+ '            move(im1 + j, col1 + padWidth, sizeX);'#10
+ '          else'#10
+ '            #pragma unroll 8'#10
+ '            for (n_int outCol=padWidth ;  outCol<outWidth - padWidth; outCol++)'#10
+ '            {'#10
+ '              //j := outRow * aWidth * strideY + outCol * strideX;'#10
+ '              col1[outCol] = im1[j];'#10
+ '              j += strideX;'#10
+ '            }'#10
+ '          col1 += outWidth;'#10
+ '        }'#10
+ '      }'#10
+ '    }'#10
+ '}'#10
+ 'void fills(__global nfloat* dst, const n_int N, const nfloat val){'#10
+ '  #pragma unroll 8'#10
+ '  for (n_int i=0 ; i<N; i++)'#10
+ '    dst[i] = val;'#10
+ '}'#10
+ '//col2im :'#10
+ '//https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/col2im.opencl'#10
+ '// Work-group size parameters re-used from the ''copy'' kernel'#10
+ '#ifndef COPY_DIMX'#10
+ '  #define COPY_DIMX 8      // Local workgroup size in the first dimension (w)'#10
+ '#endif'#10
+ '#ifndef COPY_DIMY'#10
+ '  #define COPY_DIMY 8      // Local workgroup size in the second dimension (h)'#10
+ '#endif'#10
+ '// ================================================================================================='#10
+ 'n_int grid_ceil(const n_int x, const n_int step) {'#10
+ '  return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;'#10
+ '}'#10
+ 'void xim2col(const n_int input_h, const n_int input_w, const n_int channels,'#10
+ '                         const n_int output_h, const n_int output_w,'#10
+ '                         const n_int kernel_h, const n_int kernel_w,'#10
+ '                         const n_int pad_h, const n_int pad_w,'#10
+ '                         const n_int stride_h, const n_int stride_w,'#10
+ '                         const n_int dilation_h, const n_int dilation_w,'#10
+ '                         const bool kernel_flip,'#10
+ '                         const __global nfloat* restrict im_buffer, const n_int im_offset,'#10
+ '                         __global nfloat* col_buffer, const n_int col_offset) {'#10
+ '  // Thread IDs'#10
+ '  const n_int w_id = get_global_id(0); // image width, max ''output_w'''#10
+ '  const n_int h_id = ((n_int)get_global_id(1)) % output_h; // image height, max ''output_h'''#10
+ '  const n_int c_id = ((n_int)get_global_id(1)) / output_h; // input channels'#10
+ '  if (h_id < output_h && w_id < output_w && c_id < channels) {'#10
+ '    #pragma unroll 8'#10
+ '    for (n_int kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height'#10
+ '      #pragma unroll 8'#10
+ '      for (n_int kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width'#10
+ '        // Retrieves the input value'#10
+ '        const n_int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;'#10
+ '        const n_int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;'#10
+ '        nfloat val;'#10
+ '        if (h_index >= 0 && h_index < input_h &&'#10
+ '            w_index >= 0 && w_index < input_w) {'#10
+ '          const n_int input_index = w_index + input_w * (h_index + input_h * c_id);'#10
+ '          val = im_buffer[input_index + im_offset];'#10
+ '        }'#10
+ '        else {'#10
+ '          val = 0;'#10
+ '        }'#10
+ '        // Sets the output value'#10
+ '        const n_int kernel_index = (kernel_flip)'#10
+ '                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1'#10
+ '                               : kw_id + kernel_w * kh_id;'#10
+ '        const n_int patch_index = w_id + output_w * h_id;'#10
+ '        const n_int output_index = patch_index + kernel_index * output_w * output_h +'#10
+ '                                  c_id * output_w * output_h * kernel_h * kernel_w;'#10
+ '        col_buffer[output_index + col_offset] = val;'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Kernel flip version of the Xim2col kernel (for convolution)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '//#else'#10
+ '//  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xim2colKernelFlip(const n_int input_h, const n_int input_w, const n_int channels,'#10
+ '                       const n_int output_h, const n_int output_w,'#10
+ '                       const n_int kernel_h, const n_int kernel_w,'#10
+ '                       const n_int pad_h, const n_int pad_w,'#10
+ '                       const n_int stride_h, const n_int stride_w,'#10
+ '                       const n_int dilation_h, const n_int dilation_w,'#10
+ '                       const __global nfloat* restrict im_buffer, const n_int im_offset,'#10
+ '                       __global nfloat* col_buffer, const n_int col_offset) {'#10
+ '  const bool kernel_flip = true;'#10
+ '  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          kernel_flip,'#10
+ '          im_buffer, im_offset, col_buffer, col_offset);'#10
+ '}'#10
+ '// Normal version of the Xim2col kernel (for cross-correlation)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '//#else'#10
+ '//  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xim2colKernelNormal(const n_int input_h, const n_int input_w, const n_int channels,'#10
+ '                         const n_int output_h, const n_int output_w,'#10
+ '                         const n_int kernel_h, const n_int kernel_w,'#10
+ '                         const n_int pad_h, const n_int pad_w,'#10
+ '                         const n_int stride_h, const n_int stride_w,'#10
+ '                         const n_int dilation_h, const n_int dilation_w,'#10
+ '                         const __global nfloat* restrict im_buffer, const n_int im_offset,'#10
+ '                         __global nfloat* col_buffer, const n_int col_offset) {'#10
+ '  const bool kernel_flip = false;'#10
+ '  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          kernel_flip,'#10
+ '          im_buffer, im_offset, col_buffer, col_offset);'#10
+ '}'#10
+ 'void xcol2im(const n_int input_h, const n_int input_w, const n_int channels,'#10
+ '                         const n_int output_h, const n_int output_w,'#10
+ '                         const n_int kernel_h, const n_int kernel_w,'#10
+ '                         const n_int pad_h, const n_int pad_w,'#10
+ '                         const n_int stride_h, const n_int stride_w,'#10
+ '                         const n_int dilation_h, const n_int dilation_w,'#10
+ '                         const n_int stride_bez_h, const n_int stride_bez_w,'#10
+ '                         const n_int dilation_bez_h, const n_int dilation_bez_w,'#10
+ '                         const n_int gcd_h, const n_int gcd_w,'#10
+ '                         const bool kernel_flip,'#10
+ '                         const __global nfloat* restrict col_buffer, const n_int col_offset,'#10
+ '                         __global nfloat* im_buffer, const n_int im_offset) {'#10
+ '  const n_int input_h_scaled = (input_h - 1) / gcd_h + 1;'#10
+ '  // Thread IDs'#10
+ '  const n_int gcd_scale_w = get_global_id(0) + (pad_w - 1) / gcd_w + 1;'#10
+ '  const n_int gcd_scale_h = ((n_int) get_global_id(1)) % input_h_scaled + (pad_h - 1) / gcd_h + 1;'#10
+ '  const n_int c_id = ((n_int) get_global_id(1)) / input_h_scaled;'#10
+ '  const n_int w_index = gcd_scale_w * gcd_w - pad_w;'#10
+ '  const n_int h_index = gcd_scale_h * gcd_h - pad_h;'#10
+ '  const n_int th_step = stride_h * dilation_h / gcd_h;'#10
+ '  const n_int th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,'#10
+ '                                     (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),'#10
+ '                                 th_step);'#10
+ '  const n_int th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,'#10
+ '                         (dilation_bez_h * gcd_scale_h + 1) * dilation_h);'#10
+ '  const n_int tw_step = stride_w * dilation_w / gcd_w;'#10
+ '  const n_int tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,'#10
+ '                                     (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),'#10
+ '                                 tw_step);'#10
+ '  const n_int tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,'#10
+ '                         (dilation_bez_w * gcd_scale_w + 1) * dilation_w);'#10
+ '  if (w_index < input_w && c_id < channels) {'#10
+ '    nfloat val = 0;'#10
+ '    #pragma unroll 8'#10
+ '    for (n_int th = th_begin; th < th_end; th += th_step) {'#10
+ '      #pragma unroll 8'#10
+ '      for (n_int tw = tw_begin; tw < tw_end; tw += tw_step) {'#10
+ '        const n_int kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;'#10
+ '        const n_int kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;'#10
+ '        const n_int h_id = th / stride_h + stride_bez_h * gcd_scale_h;'#10
+ '        const n_int w_id = tw / stride_w + stride_bez_w * gcd_scale_w;'#10
+ '        const n_int kernel_index = (kernel_flip)'#10
+ '                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1'#10
+ '                               : kw_id + kernel_w * kh_id;'#10
+ '        const n_int patch_index = w_id + output_w * h_id;'#10
+ '        const n_int output_index = patch_index + kernel_index * output_w * output_h +'#10
+ '                                 c_id * output_w * output_h * kernel_h * kernel_w;'#10
+ '        val += col_buffer[output_index + col_offset];'#10
+ '      }'#10
+ '    }'#10
+ '    // Accumulates the resulting value with the existing im-buffer (+= val)'#10
+ '    const n_int input_index = w_index + input_w * (h_index + input_h * c_id);'#10
+ '    nfloat im_buffer_value = im_buffer[input_index + im_offset];'#10
+ '    im_buffer[input_index + im_offset] = im_buffer_value + val;'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Kernel flip version of the Xcol2im kernel (for convolution)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '//#else'#10
+ '  //__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xcol2imKernelFlip(const n_int input_h, const n_int input_w, const n_int channels,'#10
+ '                       const n_int output_h, const n_int output_w,'#10
+ '                       const n_int kernel_h, const n_int kernel_w,'#10
+ '                       const n_int pad_h, const n_int pad_w,'#10
+ '                       const n_int stride_h, const n_int stride_w,'#10
+ '                       const n_int dilation_h, const n_int dilation_w,'#10
+ '                       const n_int stride_bez_h, const n_int stride_bez_w,'#10
+ '                       const n_int dilation_bez_h, const n_int dilation_bez_w,'#10
+ '                       const n_int gcd_h, const n_int gcd_w,'#10
+ '                       const __global nfloat* restrict col_buffer, const n_int col_offset,'#10
+ '                       __global nfloat* im_buffer, const n_int im_offset) {'#10
+ '  const bool kernel_flip = true;'#10
+ '  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,'#10
+ '          kernel_flip,'#10
+ '          col_buffer, col_offset, im_buffer, im_offset);'#10
+ '}'#10
+ '// Normal version of the Xcol2im kernel (for cross-correlation)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  __kernel'#10
+ '//#else'#10
+ '  //__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xcol2imKernelNormal(const n_int input_h, const n_int input_w, const n_int channels,'#10
+ '                         const n_int output_h, const n_int output_w,'#10
+ '                         const n_int kernel_h, const n_int kernel_w,'#10
+ '                         const n_int pad_h, const n_int pad_w,'#10
+ '                         const n_int stride_h, const n_int stride_w,'#10
+ '                         const n_int dilation_h, const n_int dilation_w,'#10
+ '                         const n_int stride_bez_h, const n_int stride_bez_w,'#10
+ '                         const n_int dilation_bez_h, const n_int dilation_bez_w,'#10
+ '                         const n_int gcd_h, const n_int gcd_w,'#10
+ '                         const __global nfloat* restrict col_buffer, const n_int col_offset,'#10
+ '                         __global nfloat* im_buffer, const n_int im_offset) {'#10
+ '  const bool kernel_flip = false;'#10
+ '  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,'#10
+ '          kernel_flip,'#10
+ '          col_buffer, col_offset, im_buffer, im_offset);'#10
+ '}'#10
+ '__kernel void upsample(__global nfloat* in, const n_int stride, const int isForward, const nfloat scale, __global nfloat* out, const int zero){'#10
+ '   const n_int c = get_global_id(0);'#10
+ '   const n_int y = get_global_id(1);'#10
+ '   const n_int x = get_global_id(2);'#10
+ '   const n_int h = get_global_size(1)/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??'#10
+ '   const n_int w = get_global_size(2)/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??'#10
+ '   const n_int in_index   = (c*h + (y / stride))*w + x / stride;'#10
+ '   const n_int out_index  = (c*h*stride + y)*stride*w + x;   // <-- why having to adjust by stride instead of remving it !!!'#10
+ '   if (isForward){'#10
+ '     out[out_index] = scale*in[in_index];'#10
+ '     return;'#10
+ '   }'#10
+ '   if (zero) in[in_index] = 0;'#10
+ '   in[in_index] += scale*out[out_index]; // ok seems like, it''s because of trying to add adjust all input pixels in the stride a in here'#10
+ '}'#10
+ '__kernel void fmavss(__global nfloat* src, const n_int offset, const nfloat scalar, const nfloat bias, __global nfloat* dst){'#10
+ '  //const n_int w = get_global_size(1);'#10
+ '  const n_int y = get_global_id(0);'#10
+ '  //const n_int x = get_global_id(1);'#10
+ '  //const n_int idx = y*w + x;'#10
+ '  src += offset;'#10
+ '  dst += offset;'#10
+ '  //dst[idx] = mad(src[idx], scalar, bias);'#10
+ '  //dst[idx] = src[idx]*scalar + bias;'#10
+ '  dst[y] = src[y]*scalar + bias;'#10
+ '}'#10
+ '__kernel void means(const n_int blocksize, const n_int groups,  __global nfloat* src, const n_int offset, __global nfloat* means){'#10
+ '    local float buff[BLOCK];'#10
+ '    const n_int N  = get_num_groups(0);'#10
+ '    const n_int id = get_local_id(0);'#10
+ '    buff[id] = 0;'#10
+ '    n_int filter = get_group_id(0);'#10
+ '    src += offset;'#10
+ '    n_int i, j;'#10
+ '    #pragma unroll 8'#10
+ '        for(j = 0; j < groups; ++j){'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            n_int index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            buff[id] += (i+id < blocksize) ? src[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    if(id == 0){'#10
+ '        float mean_tmp = 0;'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            mean_tmp += buff[i];'#10
+ '        }'#10
+ '        mean_tmp /= blocksize * groups;'#10
+ '        means[filter] = mean_tmp;'#10
+ '    }'#10
+ '}'#10
+ '__kernel void vars(const n_int blocksize, const n_int groups, __global nfloat* src, const n_int offset, __global nfloat* means, __global nfloat* vars){'#10
+ '    local float buf[BLOCK];'#10
+ '    const n_int N  = get_num_groups(0);'#10
+ '    const n_int id = get_local_id(0);'#10
+ '    buf[id] = 0;'#10
+ '    n_int filter = get_group_id(0);'#10
+ '    src += offset;'#10
+ '    n_int i, j;'#10
+ '    #pragma unroll 8'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            n_int index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            buf[id] += (i+id < blocksize) ? sqr(src[index] - means[filter]) : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    if(id == 0){'#10
+ '        float variance_tmp = 0;'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            variance_tmp += buf[i];'#10
+ '        }'#10
+ '        variance_tmp /= (blocksize * groups);'#10
+ '        vars[filter] = variance_tmp;'#10
+ '    }'#10
+ '}'#10
+ '__kernel void means_vars(const n_int blocksize, const n_int groups, __global nfloat* src, const n_int offset, __global nfloat* means, __global nfloat* vars){'#10
+ '    local float buf[BLOCK];'#10
+ '    const n_int N  = get_num_groups(0);'#10
+ '    const n_int id = get_local_id(0);//threadIdx.x;'#10
+ '    n_int filter = get_group_id(0);//blockIdx.x;'#10
+ '    buf[id] = 0;'#10
+ '    src += offset;'#10
+ '    n_int i, j;'#10
+ '    #pragma unroll 8'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            n_int index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            buf[id] += (i+id < blocksize) ? src[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    //__syncthreads();'#10
+ '    barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    if(id == 0){'#10
+ '        float mean_tmp = 0;'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            mean_tmp += buf[i];'#10
+ '        }'#10
+ '        mean_tmp /= blocksize * groups;'#10
+ '        means[filter] = mean_tmp;'#10
+ '    }'#10
+ '    barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    //src += offset;'#10
+ '    //n_int i, j;'#10
+ '    buf[id] = 0;'#10
+ '    #pragma unroll 8'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            n_int index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            buf[id] += (i+id < blocksize) ? sqr(src[index] - means[filter]) : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    //__syncthreads();'#10
+ '    barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    if(id == 0){'#10
+ '        float variance_tmp = 0;'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            variance_tmp += buf[i];'#10
+ '        }'#10
+ '        variance_tmp /= (blocksize * groups);'#10
+ '        vars[filter] = variance_tmp;'#10
+ '    }'#10
+ '    //nfloat m = 0;'#10
+ '    //nfloat v = 0;'#10
+ '    //const n_int i = get_global_id(0);'#10
+ '    //const n_int N = get_global_size(0);'#10
+ '    //const n_int S = groups*blocksize;'#10
+ '    //src += offset;'#10
+ '    //// take a shortcut'#10
+ '    //if(blocksize==1){'#10
+ '    //  m = sumv(groups, src+i, N);'#10
+ '    //  m /= S;'#10
+ '    //  v = rssv(groups, m, src+i, N);'#10
+ '    //  means[i] = m;'#10
+ '    //  vars[i]  = v / (S-1);'#10
+ '    //  return;'#10
+ '    //}'#10
+ '    //'#10
+ '    //#pragma unroll'#10
+ '    //for (n_int b=0; b<groups; b++){'#10
+ '    //    const n_int idx = (i + b*N)*blocksize;'#10
+ '    //    m += sumv(blocksize, src + idx, 1);'#10
+ '    //}'#10
+ '    //m /= S;'#10
+ '    //#pragma unroll'#10
+ '    //for (n_int b=0; b<groups; b++){'#10
+ '    //    const n_int idx = (i + b*N)*blocksize;'#10
+ '    //    v += rssv(blocksize, m, src + idx, 1);'#10
+ '    //}'#10
+ '    //means[i] = m;'#10
+ '    //vars[i]  = v / (S-1);'#10
+ '}'#10
+ '__kernel void normvv(__global nfloat* mean, const n_int mean_stride, __global nfloat* variance, const n_int variance_stride, __global nfloat* dst, const n_int dst_stride)'#10
+ '{'#10
+ '  const n_int i = get_global_id(0);'#10
+ '  dst[i * dst_stride] = (dst[i*dst_stride] - mean[i*mean_stride])/sqrt(max(variance[i*variance_stride], sEPSILON));'#10
+ '}'#10
+ '__kernel void normvs(__global nfloat* src, const nfloat mean, const nfloat variance)'#10
+ '{'#10
+ '  const n_int i = get_global_id(0);'#10
+ '  src[i] = (src[i] - mean)/sqrt(max(variance, sEPSILON));'#10
+ '}'#10
+ '__kernel void normblkvv(__global nfloat* means, const n_int means_stride, __global nfloat* vars, const n_int vars_stride, __global nfloat* dst, const n_int offset)'#10
+ '{'#10
+ '  const n_int blocksize = get_global_size(1);'#10
+ '  const n_int batchsize = get_global_size(0)*blocksize;'#10
+ '  const n_int i = get_global_id(0); // means vars pos'#10
+ '  const n_int b = get_global_id(2); // batch batch pos'#10
+ '  const n_int j = get_global_id(1)+ b*batchsize + i*blocksize + offset; // block pos'#10
+ '  const nfloat v = sqrt(max(vars[i*vars_stride], sEPSILON));'#10
+ '  const nfloat m = means[i*means_stride];'#10
+ '  //dst += b*batchsize + i*blocksize + offset;'#10
+ '  dst[j] = (dst[j] - m)/v;'#10
+ '}'#10
+ '__kernel void means_vars_delta(const n_int groups, const n_int blocksize,'#10
+ '         __global nfloat* delta, __global nfloat* x, const n_int offset,'#10
+ '         __global nfloat* means, __global nfloat* vars,'#10
+ '         __global nfloat* means_delta, __global nfloat* vars_delta){'#10
+ '  local float buf[BLOCK];'#10
+ '  const n_int N = get_num_groups(0);'#10
+ '  n_int id = get_local_id(0);'#10
+ '  n_int filter = get_group_id(0);'#10
+ '  buf[id] = 0;'#10
+ '  x     += offset;'#10
+ '  delta += offset;'#10
+ '  n_int i, j;'#10
+ '  #pragma unroll 8'#10
+ '  for(j = 0; j < groups; ++j){'#10
+ '      #pragma unroll 8'#10
+ '      for(i = 0; i < blocksize; i += BLOCK){'#10
+ '          n_int index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '          buf[id] += (i+id < blocksize) ? delta[index] : 0;'#10
+ '      }'#10
+ '  }'#10
+ '  barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '  if(id == 0){'#10
+ '      means_delta[filter] = 0;'#10
+ '      #pragma unroll 8'#10
+ '      for(i = 0; i < BLOCK; ++i){'#10
+ '          means_delta[filter] += buf[i];'#10
+ '      }'#10
+ '      means_delta[filter] *= (-1.0f/sqrt(max(vars[filter] , sEPSILON)));'#10
+ '  }'#10
+ '  barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '  buf[id] = 0;'#10
+ '  #pragma unroll 8'#10
+ '  for(j = 0; j < groups; ++j){'#10
+ '      #pragma unroll 8'#10
+ '      for(i = 0; i < blocksize; i += BLOCK){'#10
+ '          n_int index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '          buf[id] += (i+id < blocksize) ? delta[index]*(x[index] - means[filter]) : 0;'#10
+ '      }'#10
+ '  }'#10
+ '  barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '  if(id == 0){'#10
+ '      vars_delta[filter] = 0;'#10
+ '      #pragma unroll 8'#10
+ '      for(i = 0; i < BLOCK; ++i){'#10
+ '          vars_delta[filter] += buf[i];'#10
+ '      }'#10
+ '      vars_delta[filter] *= -0.5f * pow(max(vars[filter] , sEPSILON), -1.5f);'#10
+ '  }'#10
+ '  //nfloat m = 0;'#10
+ '  //nfloat v = 0;'#10
+ '  //const n_int i = get_global_id(0);'#10
+ '  //const n_int ndst = get_global_size(0);'#10
+ '  //x     += offset;'#10
+ '  //delta += offset;'#10
+ '  //// take a shortcut'#10
+ '  //if(blocksize==1){'#10
+ '  //  #pragma unroll'#10
+ '  //  for (n_int j=0 ;j<groups; j++){'#10
+ '  //    const n_int index = i+j*ndst;'#10
+ '  //    m += delta[index];'#10
+ '  //    v += delta[index] * (x[index] - means[i]);'#10
+ '  //  }'#10
+ '  //  means_delta[i] = m * (-1.0f / sqrt(fmax(vars[i], sEPSILON)));'#10
+ '  //  vars_delta[i]  = v * -0.5f * pow(fmax(vars[i], sEPSILON), -1.5f);'#10
+ '  //  return;'#10
+ '  //}'#10
+ '  //#pragma unroll'#10
+ '  //for (n_int j=0 ;j<groups; j++)'#10
+ '  //  #pragma unroll'#10
+ '  //  for (n_int k=0; k<blocksize; k++){'#10
+ '  //    const n_int index = (i + j*ndst)*blocksize + k;'#10
+ '  //    m += delta[index];'#10
+ '  //    v += delta[index] * (x[index] - means[i]);'#10
+ '  //  }'#10
+ '  //means_delta[i] = m * (-1.0f / sqrt(fmax(vars[i], sEPSILON)));'#10
+ '  //vars_delta[i]  = v * -0.5f * pow(fmax(vars[i], sEPSILON), -1.5f);'#10
+ '  ////means_delta[i] = m * (-1.0f / sqrt(vars[i]));'#10
+ '  ////vars_delta[i]  = v * -0.5f / (vars[i]*sqrt(vars[i]));'#10
+ '}'#10
+ '__kernel void norm_delta(__global nfloat* x, const n_int offset, __global nfloat* means, __global nfloat* vars, __global nfloat* means_delta, __global nfloat* vars_delta, __global nfloat* delta){'#10
+ '  const n_int j = get_global_id(2);'#10
+ '  const n_int i = get_global_id(0);'#10
+ '  const n_int k = get_global_id(1);'#10
+ '  const n_int groups    = get_global_size(2);'#10
+ '  const n_int N         = get_global_size(0);'#10
+ '  const n_int blocksize = get_global_size(1);'#10
+ '  const n_int batchsize = blocksize * groups;'#10
+ '  const n_int index = (i + j*N) * blocksize +k;'#10
+ '  delta += offset;'#10
+ '  x     += offset;'#10
+ '  delta[index] ='#10
+ '    delta[index] / (sqrt(max(vars[i], sEPSILON))) +'#10
+ '    //delta[index] / (sqrt(vars[i])) +'#10
+ '    (2.0f * vars_delta[i] * (x[index] - means[i]) + means_delta[i]) / batchsize;'#10
+ '}'#10
+ '__kernel void add_dots(const n_int groups, const n_int blocksize, __global nfloat* src1, __global nfloat* src2, const n_int srcOffset, __global nfloat* dst){'#10
+ '    const n_int filter = get_group_id(0);'#10
+ '    const n_int p = get_local_id(0);'#10
+ '    const n_int N = get_num_groups(0);'#10
+ '    local float part[BLOCK];'#10
+ '    src1 += srcOffset;'#10
+ '    src2 += srcOffset;'#10
+ '    n_int i,b;'#10
+ '    float sum = 0;'#10
+ '    #pragma unroll 8'#10
+ '    for(b = 0; b < groups; ++b){'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            int index = p + i + blocksize*(filter + N*b);'#10
+ '            sum += (p+i < blocksize) ? src1[index]*src2[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    part[p] = sum;'#10
+ '    barrier(CLK_LOCAL_MEM_FENCE);'#10
+ '    if (p == 0) {'#10
+ '        #pragma unroll 8'#10
+ '        for(i = 0; i < BLOCK; ++i) dst[filter] += part[i];'#10
+ '    }'#10
+ '    //const n_int i = get_global_id(0);'#10
+ '    //const n_int ndst = get_global_size(0);'#10
+ '    //nfloat sum = 0;'#10
+ '    //src1 += srcOffset;'#10
+ '    //src2 += srcOffset;'#10
+ '    //// take a shortcut'#10
+ '    //if (blocksize==1){'#10
+ '    //  sum = dotv(groups, src1 + i, ndst, src2 + i, ndst);'#10
+ '    //  dst[i] += sum;'#10
+ '    //  return;'#10
+ '    //}'#10
+ '    //#pragma unroll'#10
+ '    //for (n_int b=0; b<groups; b++){'#10
+ '    //  const n_int idx = (i + b * ndst) * blocksize;'#10
+ '    //  sum += dotv(blocksize, src1 + idx, 1, src2 + idx, 1);'#10
+ '    //}'#10
+ '    //dst[i] += sum;'#10
+ '}'#10
+ '__kernel void forward_scale(const int reshape, __global nfloat* output,  const n_int outputOffset, __global nfloat* scale, const n_int scaleOffset, const n_int incb)'#10
+ '{'#10
+ '  n_int N, blockSize, i, k, j;'#10
+ '  switch (reshape) {'#10
+ '    case 0:'#10
+ '      N = get_global_size(0);'#10
+ '      blockSize = get_global_size(1);'#10
+ '      i = get_global_id(0);'#10
+ '      k = get_global_id(2);'#10
+ '      j = get_global_id(1);'#10
+ '      break;'#10
+ '    case 1:'#10
+ '      N = get_global_size(1);'#10
+ '      blockSize = get_global_size(0);'#10
+ '      i = get_global_id(1);'#10
+ '      k = get_global_id(2);'#10
+ '      j = get_global_id(0);'#10
+ '      break;'#10
+ '  }'#10
+ '  //if (i==0 && k==0){'#10
+ '    //printf("        N = %ld\n", N);'#10
+ '    //printf("blockSize = %ld\n", blockSize);'#10
+ '  //}'#10
+ '  //for (i = 0; i<N; i++)'#10
+ '  //  for (k = 0; k<batch; k++){'#10
+ '  output += (k*N + i)*blockSize + outputOffset;'#10
+ '  nfloat bb = scale[i * incb];'#10
+ '  //#pragma unroll 8'#10
+ '  //    for (n_int j=0; j<blockSize; j++)'#10
+ '        output[j] *= bb;'#10
+ '  //}'#10
+ '}'#10
+ '__kernel void forward_scale_add(const int reshape, __global nfloat* a,  const n_int aOffset, __global nfloat* s, __global nfloat* b, const n_int bOffset, const n_int incb)'#10
+ '{'#10
+ '  n_int N, blockSize, i, k, j;'#10
+ '  switch (reshape) {'#10
+ '    case 0:'#10
+ '      N = get_global_size(0);'#10
+ '      blockSize = get_global_size(1);'#10
+ '      i = get_global_id(0);'#10
+ '      k = get_global_id(2);'#10
+ '      j = get_global_id(1);'#10
+ '      break;'#10
+ '    case 1:'#10
+ '      N = get_global_size(1);'#10
+ '      blockSize = get_global_size(0);'#10
+ '      i = get_global_id(1);'#10
+ '      k = get_global_id(2);'#10
+ '      j = get_global_id(0);'#10
+ '      break;'#10
+ '  }'#10
+ '  //if (i==0 && k==0){'#10
+ '    //printf("        N = %ld\n", N);'#10
+ '    //printf("blockSize = %ld\n", blockSize);'#10
+ '  //}'#10
+ '  //for (i = 0; i<N; i++)'#10
+ '  //  for (k = 0; k<batch; k++){'#10
+ '  a += (k*N + i)*blockSize + aOffset;'#10
+ '  nfloat ss = s[i * incb];'#10
+ '  nfloat bb = b[i * incb];'#10
+ '  //#pragma unroll 8'#10
+ '  //    for (n_int j=0; j<blockSize; j++)'#10
+ '        a[j] *= ss;'#10
+ '        a[j] += bb;'#10
+ '  //}'#10
+ '}'#10
+ '#define RANDOM_MAX 0x10000000u'#10
+ '// https://en.wikipedia.org/wiki/Xorshift'#10
+ 'unsigned n_int rand_xorshift(const unsigned n_int seed){'#10
+ '  //un_int res = ((seed + get_global_linear_id()) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);'#10
+ '  //return (res >> 16) % RANDOM_MAX;'#10
+ '  //      uint x = seed + get_global_linear_id();'#10
+ '    unsigned n_int x = seed;'#10
+ '    x ^= x >> 12;'#10
+ '    x ^= x << 25;'#10
+ '    x ^= x >> 27;'#10
+ '    x *= 0x2545F4914F6CDD1DUL;'#10
+ '    //seed = x;'#10
+ '    return x%RANDOM_MAX;'#10
+ '}'#10
+ '__kernel void forward_dropout(const unsigned int seed, const nfloat probability, const nfloat scale, __global const nfloat* src, __global nfloat* rnd, __global nfloat* dst)'#10
+ '{'#10
+ '  unsigned n_int i  = get_global_id(0);'#10
+ '  //mt_state state;'#10
+ '  //init_mt(&state, seed + i);'#10
+ '  //rnd[i]  = rand_mt(&state);'#10
+ '  rnd[i]  = rand_xorshift(seed+i);'#10
+ '  rnd[i] /= RANDOM_MAX;'#10
+ '  dst[i]  = rnd[i] < probability? 0: src[i]*scale;'#10
+ '}'#10
+ 'kernel void backward_dropout(const nfloat probability, const nfloat scale, global const nfloat* src, global const nfloat* rnd, global nfloat* dst)'#10
+ '{'#10
+ '  n_int i = get_global_id(0);'#10
+ '  dst[i] = rnd[i] < probability? 0: src[i]*scale;'#10
+ '}'#10
+ 'kernel void cost_l2(global const nfloat* pred, global const nfloat* truth, global nfloat* delta, global nfloat* error){'#10
+ '  n_int i = get_global_id(0);'#10
+ '  nfloat r = truth[i] - pred[i];'#10
+ '  delta[i] = r ;'#10
+ '  error[i] = r*r;'#10
+ '}'#10
+ '__kernel void mulv( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* dst, const n_int dstOffset, const n_int incc){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset];'#10
+ '}'#10
+ '__kernel void fmav( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* src3, const n_int src3Offset, const n_int incc, __global nfloat* dst, const n_int dstOffset, const n_int incd){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   dst[i*incd + dstOffset] = mad(src1[i*inca + src1Offset], src2[i*incb + src2Offset], src3[i*incc + src3Offset]);'#10
+ '   //dst[i*incd + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset] + src3[i*incc + src3Offset];'#10
+ '}'#10
+ '__kernel void power(__global nfloat* base, const n_int srcOffset, const n_int srcStride, const nfloat expo, __global nfloat* dst, const n_int dstOffset, const n_int dstStride){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   dst[i*dstStride + dstOffset] = pow(base[i*srcStride + srcOffset], expo);'#10
+ '}'#10
+ '__kernel void clip(const nfloat alpha, __global const nfloat* src, __global nfloat* dst, const n_int stride, const n_int offset ){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   src += i*stride + offset;'#10
+ '   dst += i*stride + offset;'#10
+ '   *dst = max(min(*src, alpha), -alpha);'#10
+ '}'#10
+ '__kernel void inverse_sqrt(__global const nfloat* src, __global nfloat* dst, const n_int stride, const n_int offset){'#10
+ '   const n_int i = get_global_id(0);'#10
+ '   src += i*stride + offset;'#10
+ '   dst += i*stride + offset;'#10
+ '   *dst= 1/sqrt(max(*src, sEPSILON));'#10
+ '}'#10
+ '//__kernel void halftest(__global half* a, __global half* b, __global half* c){'#10
+ '//   const n_int i = get_global_id(0);'#10
+ '//  c[i] = a[i]+b[i];'#10
+ '//}'#10
