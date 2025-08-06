''
+ '//#define nfloat  float'#10
+ '#define sEPSILON 0.000001f'#10
+ '//#include <stdio.h>'#10
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
+ '#ifndef BLOCK'#10
+ '#define BLOCK 512'#10
+ '#endif'#10
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
+ '__device__ nfloat sqr(const nfloat x){'#10
+ '  return x*x;'#10
+ '}'#10
+ '__device__ nfloat sumv(const long N,  const nfloat* v, const long stride){'#10
+ '  nfloat sum=0;'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0;i<N; i++)'#10
+ '    sum += v[i*stride];'#10
+ '  return sum;'#10
+ '}'#10
+ '__device__ nfloat maxv(const long N,  const nfloat* v, const long stride){'#10
+ '  nfloat m=v[0];'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=1;i<N; i++)'#10
+ '    m = max(m, v[i*stride]);'#10
+ '  return m;'#10
+ '}'#10
+ '__device__ nfloat minv(const long N,  const nfloat* v, const long stride){'#10
+ '  nfloat m=v[0];'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=1;i<N; i++)'#10
+ '    m = min(m, v[i*stride]);'#10
+ '  return m;'#10
+ '}'#10
+ '__device__ nfloat rssv(const long N, const nfloat mean,  const nfloat* src, const long stride){'#10
+ '  nfloat sum=0;'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0;i<N; i++){'#10
+ '    const nfloat v = src[i*stride] - mean;'#10
+ '    sum += v*v;'#10
+ '  }'#10
+ '  return sum;'#10
+ '}'#10
+ '//#define VW 8'#10
+ '//nfloat sumv_simd(const long N,  nfloat* v){'#10
+ '//  float8 sum4 = 0;'#10
+ '//  nfloat sum = 0;'#10
+ '//  long n = N / VW;'#10
+ '//'#10
+ '//  #pragma unroll 8'#10
+ '//  for (long i=0;i<n; i++)'#10
+ '//    sum4 += vload8(i, v);'#10
+ '//  v += n*VW;'#10
+ '//'#10
+ '//  #pragma unroll 8'#10
+ '//  for (long i=0; i<N%VW;i++)'#10
+ '//    sum4[i] += v[i];'#10
+ '//  return sum4[0] + sum4[1] + sum4[2] + sum4[3] + sum4[4] + sum4[5] + sum4[6] + sum4[7] ;'#10
+ '//}'#10
+ '__device__ nfloat dotv(const long N,  nfloat* a, const long inca,   nfloat* b, const long incb){'#10
+ '  nfloat d = 0;'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0; i<N;i++)'#10
+ '    d += a[i*inca]*b[i*incb];'#10
+ '  return d;'#10
+ '}'#10
+ '//nfloat dotv_simd(const long N,  nfloat* a,   nfloat* b){'#10
+ '//'#10
+ '//  float8 d = 0;'#10
+ '//  long n = N / VW;'#10
+ '//  #pragma unroll 8'#10
+ '//  for (long i=0; i<n;i++)'#10
+ '//    d += vload8(i, a) * vload8(i, b);'#10
+ '//  a += n*VW;'#10
+ '//  b += n*VW;'#10
+ '//'#10
+ '//  #pragma unroll 8'#10
+ '//  for (long i=0; i<N%VW;i++)'#10
+ '//    d.x += a[i]*b[i];'#10
+ '//'#10
+ '//  return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] ;'#10
+ '//}'#10
+ '#define WIDTH 4'#10
+ '              // naive GEMM with unrolling for now'#10
+ 'extern "C" __global__ void sgemm1_nn(const long M, const long N, const long K, const nfloat ALPHA ,'#10
+ '                       nfloat* A, const long aOffset, const long lda,'#10
+ '                       nfloat* B, const long bOffset, const long ldb,'#10
+ '                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {'#10
+ '    const long globalRow = blockDim.x * blockIdx.x + threadIdx.x; // Row ID of C (0..M)'#10
+ '    const long globalCol = blockDim.y * blockIdx.y + threadIdx.y; // Col ID of C (0..N)'#10
+ '    if (globalRow >= M || globalCol >= N) return;'#10
+ '    A += globalRow*lda + aOffset ;'#10
+ '    B += globalCol + bOffset;'#10
+ '    C += globalRow*ldc + globalCol + cOffset;'#10
+ '    *C *= BETA;'#10
+ '    //nfloat acc =0;'#10
+ '    //'#10
+ '    //#pragma unroll 8'#10
+ '    //for (long k=0; k<K; k++)'#10
+ '    //  acc += A[k]*B[k*ldb];'#10
+ '    //*C += acc * ALPHA;'#10
+ '    *C += dotv(K, A, 1, B, ldb) * ALPHA;'#10
+ '}'#10
+ 'extern "C" __global__ void sgemm2_nn(const long M, const long N, const long K, const nfloat ALPHA ,'#10
+ '                       nfloat* A, const long aOffset, const long lda,'#10
+ '                       nfloat* B, const long bOffset, const long ldb,'#10
+ '                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {'#10
+ '    const long globalRow = blockDim.y * blockIdx.y + threadIdx.y; // Row ID of C (0..M)'#10
+ '    const long globalCol = blockDim.x * blockIdx.x + threadIdx.x; // Col ID of C (0..N)'#10
+ '    if (globalRow >= M || globalCol >= N) return;'#10
+ '    A += globalRow*lda + aOffset ;'#10
+ '    B += globalCol + bOffset;'#10
+ '    C += globalRow*ldc + globalCol + cOffset;'#10
+ '    *C *= BETA;'#10
+ '    //nfloat acc =0;'#10
+ '    //#pragma unroll 8'#10
+ '    //for (long k=0; k<K; k++)'#10
+ '    //    acc += A[k]*B[k*ldb];'#10
+ '    //*C += acc * ALPHA;'#10
+ '    *C += dotv(K, A, 1, B, ldb) * ALPHA;'#10
+ '}'#10
+ 'extern "C" __global__ void sgemm1_nt(const long M, const long N, const long K, const nfloat ALPHA ,'#10
+ '                       nfloat* A, const long aOffset, const long lda,'#10
+ '                       nfloat* B, const long bOffset, const long ldb,'#10
+ '                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {'#10
+ '    const long globalRow = blockDim.x * blockIdx.x + threadIdx.x;    // M'#10
+ '    const long globalCol = blockDim.y * blockIdx.y + threadIdx.y;    // N'#10
+ '    if (globalRow >= M || globalCol >= N) return;'#10
+ '    A += globalRow*lda + aOffset;'#10
+ '    B += globalCol*ldb + bOffset;'#10
+ '    C += globalRow*ldc + globalCol + cOffset;'#10
+ '    *C *= BETA;'#10
+ '    //nfloat acc =0;'#10
+ '    //#pragma unroll 8'#10
+ '    //for (long k=0; k<K; k++)'#10
+ '    //    acc += A[k] * B[k];'#10
+ '    //*C += acc * ALPHA;'#10
+ '    //*C += dotv(K, A, 1, B, 1) * ALPHA;'#10
+ '    *C += dotv(K, A, 1, B, 1) * ALPHA;'#10
+ '            //}'#10
+ '}'#10
+ 'extern "C" __global__ void sgemm1_tn(const long M, const long N, const long K, const nfloat ALPHA ,'#10
+ '                       nfloat* A, const long aOffset, const long lda,'#10
+ '                       nfloat* B, const long bOffset, const long ldb,'#10
+ '                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {'#10
+ '    const long row = blockDim.x * blockIdx.x + threadIdx.x; // Row ID of C (0..M)'#10
+ '    const long col = blockDim.y * blockIdx.y + threadIdx.y; // Col ID of C (0..N)'#10
+ '    if (row >= M || col >= N) return;'#10
+ '    A += row           +  aOffset;'#10
+ '    B += col           +  bOffset;'#10
+ '    C += row*ldc + col +  cOffset;'#10
+ '    *C *= BETA;'#10
+ '    //nfloat acc = 0;'#10
+ '    //#pragma unroll 8'#10
+ '    //for (long k=0; k<K; k++)'#10
+ '    //    acc += A[k * lda] * B[k * ldb] ;'#10
+ '    //*C += acc * ALPHA ;'#10
+ '    *C += dotv(K, A, lda, B, ldb) * ALPHA ;'#10
+ '}'#10
+ 'extern "C" __global__ void forward_bias(const long N, const long biasSize, const long blockSize, nfloat* output,  const long outputOffset, nfloat* bias, const long biasOffset, const long incb)'#10
+ '{'#10
+ '    const int i = blockIdx.x*blockDim.x + threadIdx.x;'#10
+ '    if (i >= N) return;'#10
+ '    const int f = (i / blockSize) % biasSize;'#10
+ '    output[i + outputOffset] += bias[f];'#10
+ '}'#10
+ '__device__ nfloat stair_activate(const nfloat x)'#10
+ '{'#10
+ '  long n = floor(x);'#10
+ '  if (n % 2 == 0) return floor(x/ 2);'#10
+ '  return (x - n) + floor(x/2);'#10
+ '}'#10
+ '__device__ nfloat hardtan_activate(const nfloat x)'#10
+ '{'#10
+ '    if (x < -1) return -1;'#10
+ '    if (x > 1) return 1;'#10
+ '    return (x);'#10
+ '}'#10
+ '__device__ nfloat linear_activate(const nfloat x)'#10
+ '{'#10
+ '  return x;'#10
+ '}'#10
+ '__device__ nfloat logistic_activate(const nfloat x)'#10
+ '{'#10
+ '  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))'#10
+ '  return 1/(1 + exp(-x));'#10
+ '}'#10
+ '__device__ nfloat loggy_activate(const nfloat x)'#10
+ '{'#10
+ '  //result := 2/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp))) - 1;'#10
+ '  return 2/(1 + exp(-x)) - 1;'#10
+ '}'#10
+ '__device__ nfloat relu_activate(const nfloat x)'#10
+ '{'#10
+ '  //return x*long(x>0);'#10
+ '  if (x<0) return 0;'#10
+ '  return x;'#10
+ '}'#10
+ '__device__ nfloat relu6_activate(const nfloat x)'#10
+ '{'#10
+ '  //min_val_cmp(max_val_cmp(x, 0), 6)'#10
+ '  //result := EnsureRange(x,0,6);'#10
+ '  return  x*(x>0) * (x<=6);'#10
+ '}'#10
+ '__device__ nfloat elu_activate(const nfloat x)'#10
+ '{'#10
+ '  return (x >= 0)*x + (x < 0)*(exp(x)-1);'#10
+ '}'#10
+ '__device__ nfloat selu_activate(const nfloat x)'#10
+ '{'#10
+ '  return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);'#10
+ '}'#10
+ '__device__ nfloat gelu_activate(const nfloat x)'#10
+ '{'#10
+ '  return 0.5f*x*(1 + tanh(0.797885f*x + 0.035677f*pow(x, 3)));'#10
+ '}'#10
+ '__device__ nfloat relie_activate(const nfloat x)'#10
+ '{'#10
+ '  if (x>0) return x;'#10
+ '  else return 0.01f*x;'#10
+ '}'#10
+ '__device__ nfloat ramp_activate(const nfloat x)'#10
+ '{'#10
+ '  return  x*(x>0)+0.1f*x;'#10
+ '}'#10
+ '__device__ nfloat leaky_activate(const nfloat x)'#10
+ '{'#10
+ '  if (x>0) return  x;'#10
+ '  else return  0.1f*x;'#10
+ '}'#10
+ '__device__ nfloat tanh_activate(const nfloat x)'#10
+ '{'#10
+ '  //const nfloat px = exp(x);'#10
+ '  //const nfloat nx = exp(-x);'#10
+ '  //return (px - nx)/(px + nx);'#10
+ '  //return 2 / (1 + exp(ensureRange(-2 * x, minSingleExp, maxSingleExp))) - 1'#10
+ '  return 2/ (1+exp(-2*x)) - 1 ;'#10
+ '//  return  (exp(2*x)-1)/(exp(2*x)+1);'#10
+ '}'#10
+ '__device__ nfloat softplus_activate(const nfloat x, const nfloat threshold)'#10
+ '{'#10
+ '    if (x > threshold)'#10
+ '      return (x);                // too large'#10
+ '    else if (x < -threshold)'#10
+ '      return (exp(x));    // too small'#10
+ '    //return (log(exp(x) + 1));'#10
+ '    return log1p(exp(x));'#10
+ '}'#10
+ '__device__ nfloat plse_activate(const nfloat x)'#10
+ '{'#10
+ '    if (x < -4 ) return( 0.01f * (x + 4));'#10
+ '    if (x > 4 ) return( 0.01f * (x - 4) + 1);'#10
+ '    return  0.125f*x + 0.5f;'#10
+ '}'#10
+ '__device__ nfloat lhtan_activate(const nfloat x)'#10
+ '{'#10
+ '    if(x < 0) return (0.001f*x);'#10
+ '    if(x > 1) return (0.001f*(x-1) + 1);'#10
+ '    return  x;'#10
+ '}'#10
+ '__device__ nfloat silu_activate(const nfloat x)'#10
+ '{'#10
+ '    return x * logistic_activate(x) ;'#10
+ '}'#10
+ '#define  MISH_THRESHOLD 20.0f'#10
+ '__device__ nfloat mish_activate(const nfloat x)'#10
+ '{'#10
+ '    return x*tanh_activate(softplus_activate(x, MISH_THRESHOLD));'#10
+ '}'#10
+ '//void softmax_activate(const N:SizeInt; const x: PSingle);'#10
+ '//{'#10
+ '//  long i;'#10
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
+ 'extern "C" __global__ void activate_array(const long N, nfloat* x, long const offset, const ActivationType a)'#10
+ '{'#10
+ '      const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '      //int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);'#10
+ '      if (i>=N) return;'#10
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
+ '            //if (i==0) printf("[Activation] %d: not Implemented\n", (int)a);'#10
+ '	    ;'#10
+ '      }'#10
+ '   //printf("%ld, ", i);'#10
+ '}'#10
+ 'extern "C" __global__ void array_activate_swish(const long N, nfloat* x, long const offset,  nfloat* output,  nfloat* output2)'#10
+ '{'#10
+ '    long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '    if (i>=N) return;'#10
+ '    x += offset;'#10
+ '    output += offset;'#10
+ '    output2 += offset;'#10
+ '    nfloat x_val       = x[i];'#10
+ '    nfloat sigmoid     = logistic_activate(x_val);'#10
+ '    output[i]         = sigmoid;'#10
+ '    output2[i]        = x_val * sigmoid;'#10
+ '}'#10
+ '__device__ nfloat lhtan_gradient(const nfloat x)'#10
+ '{'#10
+ '    if ((x > 0) &&  (x < 1))'#10
+ '      return 1;'#10
+ '    return 0.001f;'#10
+ '}'#10
+ '__device__ nfloat hardtan_gradient(const nfloat x)'#10
+ '{'#10
+ '    if ((x > -1) && (x < 1))'#10
+ '      return 1;'#10
+ '    return 0;'#10
+ '}'#10
+ '__device__ nfloat linear_gradient(const nfloat x)'#10
+ '{'#10
+ '    return 1;'#10
+ '}'#10
+ '__device__ nfloat logistic_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (1-x)*x;'#10
+ '}'#10
+ '__device__ nfloat loggy_gradient(const nfloat x)'#10
+ '{'#10
+ '    nfloat y = (x+1.0f)/2.0f;'#10
+ '    return 2.0f*(1.0f-y)*y;'#10
+ '}'#10
+ '__device__ nfloat stair_gradient(const nfloat x)'#10
+ '{'#10
+ '    if (floor(x) == x) return( 0);'#10
+ '    return 1;'#10
+ '}'#10
+ '__device__ nfloat relu_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x>0?1:0);'#10
+ '}'#10
+ '__device__ nfloat relu6_gradient(const nfloat x)'#10
+ '{'#10
+ '    return ((x>0) && (x<6)?1:0);'#10
+ '}'#10
+ '__device__ nfloat elu_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x >= 0?1:0) + (x < 0?1:0)*(x + 1);'#10
+ '}'#10
+ '__device__ nfloat selu_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x >= 0?1:0)*1.0507f + (x < 0?1:0)*(x + 1.0507f*1.6732f);'#10
+ '}'#10
+ '__device__ nfloat relie_gradient(const nfloat x)'#10
+ '{'#10
+ '    if (x>0) return 1;'#10
+ '    else return 0.01f;'#10
+ '}'#10
+ '__device__ nfloat ramp_gradient(const nfloat x)'#10
+ '{'#10
+ '    return (x>0?1:0) + 0.1f;'#10
+ '}'#10
+ '__device__ nfloat leaky_gradient(const nfloat x)'#10
+ '{'#10
+ '    if (x>0) return 1;'#10
+ '    else return 0.1f;'#10
+ '}'#10
+ '__device__ nfloat tanh_gradient(const nfloat x)'#10
+ '{'#10
+ '    return 1.0f-x*x;'#10
+ '}'#10
+ '__device__ nfloat sech(const nfloat x)'#10
+ '{'#10
+ '    return 2.0f / (exp(x) + exp(-x));'#10
+ '}'#10
+ '__device__ nfloat gelu_gradient(const nfloat x)'#10
+ '{'#10
+ '    nfloat x3 = x*x*x;'#10
+ '    nfloat sec = sech(0.0356774f*x3 + 0.797885f*x);'#10
+ '    return 0.5f*tanh(0.0356774f*x3 + 0.797885f*x) + (0.0535161f*x3 + 0.398942f*x) * sec*sec + 0.5f ;'#10
+ '}'#10
+ '__device__ nfloat plse_gradient(const nfloat x)'#10
+ '{'#10
+ '  if ((x < 0) || (x > 1))'#10
+ '    return  0.01f;'#10
+ '  else'#10
+ '    return 0.125f;'#10
+ '}'#10
+ 'extern "C" __global__ void gradient_array(const long N, nfloat* x, long const offset, const ActivationType a,  nfloat* delta)'#10
+ '{'#10
+ '    long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '    if (i>=N) return;'#10
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
+ '            //if (i==0) printf("[Gradient] : not Implemented %d\n", (int)a);'#10
+ '	    ;'#10
+ '    }'#10
+ '}'#10
+ '//#define BLOCK 0x20'#10
+ 'extern "C" __global__ void backward_bias('#10
+ '       const long N,'#10
+ '       nfloat* dst,'#10
+ '       const long blockSize,'#10
+ '       nfloat* src,'#10
+ '       const long srcOffset,'#10
+ '       const long batch)'#10
+ '{'#10
+ '    const long filter = blockIdx.x;'#10
+ '    const long p = threadIdx.x;'#10
+ '    //if(blockDim.x * filter + p>=N) return;'#10
+ '    long i,b;'#10
+ '    __shared__ float part[BLOCK];'#10
+ '    src += srcOffset;'#10
+ '    float sum = 0;'#10
+ '    for(b = 0; b < batch; ++b){'#10
+ '        for(i = 0; i < blockSize; i += BLOCK){'#10
+ '            int index = p + i + blockSize*(filter + N*b);'#10
+ '            sum += (p+i < blockSize) ? src[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    part[p] = sum;'#10
+ '    __syncthreads();'#10
+ '    if (p == 0) {'#10
+ '        for(i = 0; i < BLOCK; ++i)'#10
+ '          dst[filter] += part[i];'#10
+ '    }'#10
+ '    //const long i = blockDim.x * blockIdx.x + threadIdx.x;//if (i==0) printf("long %ull\n", sizeof(long));'#10
+ '    //if (i>=N) return;'#10
+ '    //nfloat sum = 0;'#10
+ '    ////const long N = blockDim.x * gridDim.x;'#10
+ '    ////for (long i=0 ; i<N ;i++) {'#10
+ '    //  src += i * blockSize + srcOffset;'#10
+ '    //   //take a shortcut'#10
+ '    //  if(blockSize==1){'#10
+ '    //    dst[i] += sumv(batch, src, N);'#10
+ '    //    //#pragma unroll 8'#10
+ '    //    //for (long j=0; j<batch; j++)'#10
+ '    //    //  sum += src[j*N];'#10
+ '    //    //dst[i] +=sum;'#10
+ '    //    return;'#10
+ '    //  }'#10
+ '    //'#10
+ '    //  const long incbias = N*blockSize;'#10
+ '    //  #pragma unroll 8'#10
+ '    //  for (long j=0; j<batch; j++){'#10
+ '    //    sum += sumv(blockSize, src, 1);'#10
+ '    //    src += incbias;'#10
+ '    //  }'#10
+ '    //  dst[i] +=sum;'#10
+ '}'#10
+ 'extern "C" __global__ void addv(const long N, nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* dst, const long dstOffset, const long incc){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i>=N) return;'#10
+ '   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] + src2[i*incb + src2Offset];'#10
+ '}'#10
+ 'extern "C" __global__ void subv(const long N, nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* dst, const long dstOffset, const long incc){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i>=N) return;'#10
+ '   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] - src2[i*incb + src2Offset];'#10
+ '}'#10
+ 'extern "C" __global__ void axpy(const long N, const nfloat a,  nfloat* x, const long xOffset, const long incx,  nfloat* y, const long yOffset, const long incy){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i >= N) return;'#10
+ '   x += xOffset;'#10
+ '   y += yOffset;'#10
+ '   y[i*incy] += a*x[i*incx];'#10
+ '}'#10
+ 'extern "C" __global__ void scale(const long N, const nfloat a,  nfloat* x, const long incx){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i >= N) return;'#10
+ '   x[i*incx] *= a;'#10
+ '}'#10
+ 'extern "C" __global__ void crossEntropyLogistics('#10
+ '           const long N,'#10
+ '           const nfloat* pred,'#10
+ '           const nfloat* truth,'#10
+ '           nfloat* delta,'#10
+ '           nfloat* error){'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i >= N) return;'#10
+ '  nfloat t = truth[i];'#10
+ '  nfloat p = pred[i];'#10
+ '  error[i] = -t*log(max(p, sEPSILON)) - (1-t) * log(max(1 - p, sEPSILON));'#10
+ '  //error[i] = -t*log(p) - (1-t) * log(1 - p);'#10
+ '  delta[i] = t - p;'#10
+ '   //printf("%ld, ", i);'#10
+ '}'#10
+ 'extern "C" __global__ void fill(const long N, nfloat* x, const long offset, const nfloat val, const long stride){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i >= N) return;'#10
+ '   //x += offset;'#10
+ '   x[i*stride + offset] = val;'#10
+ '}'#10
+ '// naive copy for now'#10
+ 'extern "C" __global__ void copy(const long N,'#10
+ '     nfloat* src, const long srcOffset, const long srcInc'#10
+ '  ,  nfloat* dst, const long dstOffset, const long dstInc){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i >= N) return;'#10
+ '   //src += srcOffset; dst += dstOffset;'#10
+ '   //if (srcInc==1 && dstInc==1){'#10
+ '   //  dst[i+dstOffset] = src[i+srcOffset];'#10
+ '   //  return;'#10
+ '   //}'#10
+ '   dst[i*dstInc + dstOffset] = src[i*srcInc + srcOffset];'#10
+ '}'#10
+ '#define FLT_MAX 3.402823466e+38F'#10
+ 'extern "C" __global__ void forward_maxpool('#10
+ '     const long N'#10
+ '     , const long outH'#10
+ '     , const long outW'#10
+ '     , nfloat* input'#10
+ '     , const long c'#10
+ '     , const long h'#10
+ '     , const long w'#10
+ '     , const long stride_x'#10
+ '     , const long stride_y'#10
+ '     , const long padding'#10
+ '     , const long kernelSize'#10
+ '     , size_t* indexes'#10
+ '     , nfloat* output'#10
+ '     ){'#10
+ '  const long w_offset = -padding / 2;'#10
+ '  const long h_offset = -padding / 2;'#10
+ '  //const long outC = blockDim.x * gridDim.x;'#10
+ '  //const long outH = blockDim.y * gridDim.y;'#10
+ '  //const long outW = blockDim.z * gridDim.z;'#10
+ '  long k = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  long y = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  long x = blockDim.z * blockIdx.z + threadIdx.z;'#10
+ '  if (k>=N || y>=outH || x>=outW) return;'#10
+ '  long out_index = x + outW*(y + outH*k) ;//+ outW*outH*outC*b;'#10
+ '  nfloat max = -FLT_MAX;'#10
+ '  long max_i = -1;'#10
+ '  #pragma unroll 8'#10
+ '  for (long n=0; n<kernelSize; n++)'#10
+ '      #pragma unroll 8'#10
+ '      for (long m=0; m<kernelSize; m++){'#10
+ '          long cur_h = h_offset+y * stride_y+n;'#10
+ '          long cur_w = w_offset+x * stride_x+m;'#10
+ '          long index = cur_w + w*(cur_h + h*k) ;//+ w*h*outC*b;'#10
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
+ 'extern "C" __global__ void backward_maxpool(const long M, const long N,  nfloat* output,  const size_t* indexes,  const nfloat* delta){'#10
+ '        const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '        const long j = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '        //const long id = i*blockDim.y * gridDim.y + j;'#10
+ '        if (i>=M || j>=N) return;'#10
+ '        const long id = i*N + j;'#10
+ '        const long index = indexes[id];'#10
+ '        output[index] += delta[id];'#10
+ '}'#10
+ '__device__ void softmax(const long n,  nfloat* input, const long stride, const nfloat temp,  nfloat* output){'#10
+ '  nfloat largest = maxv(n, input, stride);'#10
+ '  nfloat sum = 0;'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0;i<n;i++) {'#10
+ '      nfloat e = expf((input[i*stride] - largest)/temp);'#10
+ '      sum += e;'#10
+ '      output[i*stride] = e;'#10
+ '  }'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0; i<n; i++)'#10
+ '      output[i*stride]/=sum;'#10
+ '}'#10
+ 'extern "C" __global__ void softmaxBatch(const long batch, const long groups, nfloat* input, const long iOffset, const long n'#10
+ '  , const long batch_size, const long group_size, const long stride'#10
+ '  , const nfloat temp,  nfloat* output, const long oOffset){'#10
+ '  const long b = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  const long g = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (b>=batch || g>=groups) return;'#10
+ '  softmax(n'#10
+ '  , input + iOffset + b*batch_size + g*group_size'#10
+ '  , stride'#10
+ '  , temp'#10
+ '  , output + oOffset + b*batch_size + g*group_size);'#10
+ '}'#10
+ '__device__ void move(const nfloat* src,  nfloat* dst , const long count){'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0 ; i<count; i++) dst[i] = src[i];'#10
+ '}'#10
+ 'extern "C" __global__ void crossEntropySoftmax('#10
+ '  const long N,'#10
+ '  const nfloat* pred,'#10
+ '  const nfloat* truth,'#10
+ '  nfloat* delta,'#10
+ '  nfloat* error){'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i>=N) return;'#10
+ '  nfloat t = truth[i];'#10
+ '  nfloat p = pred[i];'#10
+ '  if (t!=0)'#10
+ '      error[i] = -log(max(p, sEPSILON));'#10
+ '      //error[i] = -log(p);'#10
+ '  else'#10
+ '      error[i] = 0;'#10
+ '  delta[i] = t - p;'#10
+ '}'#10
+ 'extern "C" __global__ void im2col(const long aHeight, const long aWidth'#10
+ '  , const long kernelHeight, const long kernelWidth, const long padHeight, const long padWidth'#10
+ '  , const long strideY, const long strideX, const long dilationY, const long dilationX'#10
+ '  ,  nfloat* im , const long imOffset'#10
+ '  ,  nfloat* col, const long colOffset, const long batch){'#10
+ '  long aChannels = blockDim.x * gridDim.x;'#10
+ '  long chan = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  long k = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  const long kernelSize = kernelHeight*kernelWidth;'#10
+ '  const long outWidth = (aWidth + 2 * padWidth - (dilationX * (kernelWidth - 1) + 1)) / strideX + 1;'#10
+ '  const long outHeight = (aHeight + 2 * padHeight - (dilationY * (kernelHeight - 1) + 1)) / strideY + 1;'#10
+ '  const long outSize = outWidth * outHeight;'#10
+ '  const long inSize = aWidth * aHeight;'#10
+ '  const long sizeX = outWidth - 2 * padWidth;'#10
+ '    //for (long k=0 ; k<kernelWidth*kernelHeight; k++)'#10
+ '    {'#10
+ '      long kernelRow = k / kernelWidth;'#10
+ '      long kernelCol = k % kernelWidth;'#10
+ '      //const long kernelRow = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '      //const long kernelCol = blockDim.z * blockIdx.z + threadIdx.z;'#10
+ '      #pragma unroll 8'#10
+ '      for (long b=0 ; b<batch; b++)'#10
+ '      {'#10
+ '        long i = (b*aChannels + chan)*inSize + aWidth*(kernelRow*dilationY - padHeight) + kernelCol*dilationX - padWidth;'#10
+ '         nfloat* im1 = imOffset + im + i;'#10
+ '         nfloat* col1 = colOffset + col + padWidth * outWidth + outSize*kernelSize*(chan + b*aChannels) + outSize*(kernelRow * kernelWidth + kernelCol) ;'#10
+ '        #pragma unroll 8'#10
+ '        for (long outRow=padHeight ; outRow<outHeight - padHeight ; outRow++)'#10
+ '        {'#10
+ '          long j = outRow * aWidth * strideY + padWidth * strideX ;'#10
+ '          if (strideX == 1)'#10
+ '            move(im1 + j, col1 + padWidth, sizeX);'#10
+ '          else'#10
+ '            #pragma unroll 8'#10
+ '            for (long outCol=padWidth ;  outCol<outWidth - padWidth; outCol++)'#10
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
+ '__device__ void fills( nfloat* dst, const long N, const nfloat val){'#10
+ '  #pragma unroll 8'#10
+ '  for (long i=0 ; i<N; i++)'#10
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
+ '__device__ long grid_ceil(const long x, const long step) {'#10
+ '  return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;'#10
+ '}'#10
+ '__device__ void xim2col(const long input_h, const long input_w, const long channels,'#10
+ '                         const long output_h, const long output_w,'#10
+ '                         const long kernel_h, const long kernel_w,'#10
+ '                         const long pad_h, const long pad_w,'#10
+ '                         const long stride_h, const long stride_w,'#10
+ '                         const long dilation_h, const long dilation_w,'#10
+ '                         const bool kernel_flip,'#10
+ '                         const  nfloat* __restrict__ im_buffer, const long im_offset,'#10
+ '                          nfloat* col_buffer, const long col_offset) {'#10
+ '  // Thread IDs'#10
+ '  const long w_id = blockDim.x * blockIdx.x + threadIdx.x; // image width, max ''output_w'''#10
+ '  const long h_id = ((long)blockDim.y * blockIdx.y + threadIdx.y) % output_h; // image height, max ''output_h'''#10
+ '  const long c_id = ((long)blockDim.y * blockIdx.y + threadIdx.y) / output_h; // input channels'#10
+ '  if (h_id < output_h && w_id < output_w && c_id < channels) {'#10
+ '    #pragma unroll 8'#10
+ '    for (long kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height'#10
+ '      #pragma unroll 8'#10
+ '      for (long kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width'#10
+ '        // Retrieves the input value'#10
+ '        const long h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;'#10
+ '        const long w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;'#10
+ '        nfloat val;'#10
+ '        if (h_index >= 0 && h_index < input_h &&'#10
+ '            w_index >= 0 && w_index < input_w) {'#10
+ '          const long input_index = w_index + input_w * (h_index + input_h * c_id);'#10
+ '          val = im_buffer[input_index + im_offset];'#10
+ '        }'#10
+ '        else {'#10
+ '          val = 0;'#10
+ '        }'#10
+ '        // Sets the output value'#10
+ '        const long kernel_index = (kernel_flip)'#10
+ '                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1'#10
+ '                               : kw_id + kernel_w * kh_id;'#10
+ '        const long patch_index = w_id + output_w * h_id;'#10
+ '        const long output_index = patch_index + kernel_index * output_w * output_h +'#10
+ '                                  c_id * output_w * output_h * kernel_h * kernel_w;'#10
+ '        col_buffer[output_index + col_offset] = val;'#10
+ '      }'#10
+ '    }'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Kernel flip version of the Xim2col kernel (for convolution)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  extern "C" __global__'#10
+ '//#else'#10
+ '//  extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xim2colKernelFlip(const long M, const long N,'#10
+ '                       const long input_h, const long input_w, const long channels,'#10
+ '                       const long output_h, const long output_w,'#10
+ '                       const long kernel_h, const long kernel_w,'#10
+ '                       const long pad_h, const long pad_w,'#10
+ '                       const long stride_h, const long stride_w,'#10
+ '                       const long dilation_h, const long dilation_w,'#10
+ '                       const  nfloat* __restrict__ im_buffer, const long im_offset,'#10
+ '                        nfloat* col_buffer, const long col_offset) {'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  const long j = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (i>=M || j>=N) return;'#10
+ '  const bool kernel_flip = true;'#10
+ '  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          kernel_flip,'#10
+ '          im_buffer, im_offset, col_buffer, col_offset);'#10
+ '}'#10
+ '// Normal version of the Xim2col kernel (for cross-correlation)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  extern "C" __global__'#10
+ '//#else'#10
+ '//  extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xim2colKernelNormal(const long M, const long N,'#10
+ '                         const long input_h, const long input_w, const long channels,'#10
+ '                         const long output_h, const long output_w,'#10
+ '                         const long kernel_h, const long kernel_w,'#10
+ '                         const long pad_h, const long pad_w,'#10
+ '                         const long stride_h, const long stride_w,'#10
+ '                         const long dilation_h, const long dilation_w,'#10
+ '                         const  nfloat* __restrict__ im_buffer, const long im_offset,'#10
+ '                          nfloat* col_buffer, const long col_offset) {'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  const long j = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (i>=M || j>=N) return;'#10
+ '  const bool kernel_flip = false;'#10
+ '  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          kernel_flip,'#10
+ '          im_buffer, im_offset, col_buffer, col_offset);'#10
+ '}'#10
+ '__device__ void xcol2im(const long input_h, const long input_w, const long channels,'#10
+ '                         const long output_h, const long output_w,'#10
+ '                         const long kernel_h, const long kernel_w,'#10
+ '                         const long pad_h, const long pad_w,'#10
+ '                         const long stride_h, const long stride_w,'#10
+ '                         const long dilation_h, const long dilation_w,'#10
+ '                         const long stride_bez_h, const long stride_bez_w,'#10
+ '                         const long dilation_bez_h, const long dilation_bez_w,'#10
+ '                         const long gcd_h, const long gcd_w,'#10
+ '                         const bool kernel_flip,'#10
+ '                         const  nfloat* __restrict__ col_buffer, const long col_offset,'#10
+ '                          nfloat* im_buffer, const long im_offset) {'#10
+ '  const long input_h_scaled = (input_h - 1) / gcd_h + 1;'#10
+ '  // Thread IDs'#10
+ '  const long gcd_scale_w = blockDim.x * blockIdx.x + threadIdx.x + (pad_w - 1) / gcd_w + 1;'#10
+ '  const long gcd_scale_h = ((long) blockDim.y * blockIdx.y + threadIdx.y) % input_h_scaled + (pad_h - 1) / gcd_h + 1;'#10
+ '  const long c_id = ((long) blockDim.y * blockIdx.y + threadIdx.y) / input_h_scaled;'#10
+ '  const long w_index = gcd_scale_w * gcd_w - pad_w;'#10
+ '  const long h_index = gcd_scale_h * gcd_h - pad_h;'#10
+ '  const long th_step = stride_h * dilation_h / gcd_h;'#10
+ '  const long th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,'#10
+ '                                     (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),'#10
+ '                                 th_step);'#10
+ '  const long th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,'#10
+ '                         (dilation_bez_h * gcd_scale_h + 1) * dilation_h);'#10
+ '  const long tw_step = stride_w * dilation_w / gcd_w;'#10
+ '  const long tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,'#10
+ '                                     (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),'#10
+ '                                 tw_step);'#10
+ '  const long tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,'#10
+ '                         (dilation_bez_w * gcd_scale_w + 1) * dilation_w);'#10
+ '  if (w_index < input_w && c_id < channels) {'#10
+ '    nfloat val = 0;'#10
+ '    #pragma unroll 8'#10
+ '    for (long th = th_begin; th < th_end; th += th_step) {'#10
+ '      #pragma unroll 8'#10
+ '      for (long tw = tw_begin; tw < tw_end; tw += tw_step) {'#10
+ '        const long kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;'#10
+ '        const long kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;'#10
+ '        const long h_id = th / stride_h + stride_bez_h * gcd_scale_h;'#10
+ '        const long w_id = tw / stride_w + stride_bez_w * gcd_scale_w;'#10
+ '        const long kernel_index = (kernel_flip)'#10
+ '                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1'#10
+ '                               : kw_id + kernel_w * kh_id;'#10
+ '        const long patch_index = w_id + output_w * h_id;'#10
+ '        const long output_index = patch_index + kernel_index * output_w * output_h +'#10
+ '                                 c_id * output_w * output_h * kernel_h * kernel_w;'#10
+ '        val += col_buffer[output_index + col_offset];'#10
+ '      }'#10
+ '    }'#10
+ '    // Accumulates the resulting value with the existing im-buffer (+= val)'#10
+ '    const long input_index = w_index + input_w * (h_index + input_h * c_id);'#10
+ '    nfloat im_buffer_value = im_buffer[input_index + im_offset];'#10
+ '    im_buffer[input_index + im_offset] = im_buffer_value + val;'#10
+ '  }'#10
+ '}'#10
+ '// ================================================================================================='#10
+ '// Kernel flip version of the Xcol2im kernel (for convolution)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  extern "C" __global__'#10
+ '//#else'#10
+ '  //extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xcol2imKernelFlip(const long M, const long N,'#10
+ '                       const long input_h, const long input_w, const long channels,'#10
+ '                       const long output_h, const long output_w,'#10
+ '                       const long kernel_h, const long kernel_w,'#10
+ '                       const long pad_h, const long pad_w,'#10
+ '                       const long stride_h, const long stride_w,'#10
+ '                       const long dilation_h, const long dilation_w,'#10
+ '                       const long stride_bez_h, const long stride_bez_w,'#10
+ '                       const long dilation_bez_h, const long dilation_bez_w,'#10
+ '                       const long gcd_h, const long gcd_w,'#10
+ '                       const  nfloat* __restrict__ col_buffer, const long col_offset,'#10
+ '                        nfloat* im_buffer, const long im_offset) {'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  const long j = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (i>=M || j>=N) return;'#10
+ '  const bool kernel_flip = true;'#10
+ '  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,'#10
+ '          kernel_flip,'#10
+ '          col_buffer, col_offset, im_buffer, im_offset);'#10
+ '}'#10
+ '// Normal version of the Xcol2im kernel (for cross-correlation)'#10
+ '//#if RELAX_WORKGROUP_SIZE == 1'#10
+ '  extern "C" __global__'#10
+ '//#else'#10
+ '  //extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))'#10
+ '//#endif'#10
+ 'void Xcol2imKernelNormal(const long M, const long N,'#10
+ '                         const long input_h, const long input_w, const long channels,'#10
+ '                         const long output_h, const long output_w,'#10
+ '                         const long kernel_h, const long kernel_w,'#10
+ '                         const long pad_h, const long pad_w,'#10
+ '                         const long stride_h, const long stride_w,'#10
+ '                         const long dilation_h, const long dilation_w,'#10
+ '                         const long stride_bez_h, const long stride_bez_w,'#10
+ '                         const long dilation_bez_h, const long dilation_bez_w,'#10
+ '                         const long gcd_h, const long gcd_w,'#10
+ '                         const  nfloat* __restrict__ col_buffer, const long col_offset,'#10
+ '                          nfloat* im_buffer, const long im_offset) {'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  const long j = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (i>=M || j>=N) return;'#10
+ '  const bool kernel_flip = false;'#10
+ '  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,'#10
+ '          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,'#10
+ '          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,'#10
+ '          kernel_flip,'#10
+ '          col_buffer, col_offset, im_buffer, im_offset);'#10
+ '}'#10
+ 'extern "C" __global__ void upsample(const long N, const long M, const long K,'#10
+ '                                    nfloat* in, const long stride, const int isForward, const nfloat scale,  nfloat* out, const int zero){'#10
+ '   const long c = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (c>=N) return;'#10
+ '   const long y = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '   if (y>=M) return;'#10
+ '   const long x = blockDim.z * blockIdx.z + threadIdx.z;'#10
+ '   if (x>=K) return;'#10
+ '   const long h = M/stride; //blockDim.y * gridDim.y/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??'#10
+ '   const long w = K/stride; //blockDim.z * gridDim.z/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??'#10
+ '   const long in_index   = (c*h + (y / stride))*w + x / stride;'#10
+ '   const long out_index  = (c*h*stride + y)*stride*w + x;   // <-- why having to adjust by stride instead of remving it !!!'#10
+ '   if (isForward){'#10
+ '     out[out_index] = scale*in[in_index];'#10
+ '     return;'#10
+ '   }'#10
+ '   if (zero) in[in_index] = 0;'#10
+ '   in[in_index] += scale*out[out_index]; // ok seems like, it''s because of trying to add adjust all input pixels in the stride a in here'#10
+ '}'#10
+ 'extern "C" __global__ void fmavss(const long N, nfloat* src, const long offset, const nfloat scalar, const nfloat bias,  nfloat* dst){'#10
+ '  //const long w = blockDim.y * gridDim.y;'#10
+ '  const long y = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  //const long x = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (y>=N) return;'#10
+ '  //const long idx = y*w + x;'#10
+ '  src += offset;'#10
+ '  dst += offset;'#10
+ '  //dst[idx] = mad(src[idx], scalar, bias);'#10
+ '  //dst[idx] = src[idx]*scalar + bias;'#10
+ '  dst[y] = src[y]*scalar + bias;'#10
+ '}'#10
+ 'extern "C" __global__ void means_vars(const long N, const long blocksize, const long groups,  nfloat* src, const long offset,  nfloat* means,  nfloat* vars){'#10
+ '    nfloat m = 0;'#10
+ '    nfloat v = 0;'#10
+ '    const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '    if (i>=N) return;'#10
+ '    //const long N = blockDim.x * gridDim.x;'#10
+ '    const long S = groups*blocksize;'#10
+ '    src += offset;'#10
+ '    // take a shortcut'#10
+ '    if(blocksize==1){'#10
+ '      m = sumv(groups, src+i, N);'#10
+ '      m /= S;'#10
+ '      v = rssv(groups, m, src+i, N);'#10
+ '      means[i] = m;'#10
+ '      vars[i]  = v / (S-1);'#10
+ '      return;'#10
+ '    }'#10
+ '    #pragma unroll'#10
+ '    for (long b=0; b<groups; b++){'#10
+ '        const long idx = (i + b*N)*blocksize;'#10
+ '        m += sumv(blocksize, src + idx, 1);'#10
+ '    }'#10
+ '    m /= S;'#10
+ '    #pragma unroll'#10
+ '    for (long b=0; b<groups; b++){'#10
+ '        const long idx = (i + b*N)*blocksize;'#10
+ '        v += rssv(blocksize, m, src + idx, 1);'#10
+ '    }'#10
+ '    means[i] = m;'#10
+ '    vars[i]  = v / (S-1);'#10
+ '}'#10
+ 'extern "C" __global__ void normvv( nfloat* mean, const long mean_stride,  nfloat* variance, const long variance_stride,  nfloat* dst, const long dst_stride)'#10
+ '{'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  dst[i * dst_stride] = (dst[i*dst_stride] - mean[i*mean_stride])/sqrt(max(variance[i*variance_stride], sEPSILON));'#10
+ '}'#10
+ 'extern "C" __global__ void normvs( nfloat* src, const nfloat mean, const nfloat variance)'#10
+ '{'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  src[i] = (src[i] - mean)/sqrt(max(variance, sEPSILON));'#10
+ '}'#10
+ 'extern "C" __global__ void normblkvv( const long M, const long N, const long K,'#10
+ '                                      nfloat* means, const long means_stride,  nfloat* vars, const long vars_stride,  nfloat* dst, const long offset)'#10
+ '{'#10
+ '  const long blocksize = N;                      //blockDim.y * gridDim.y;'#10
+ '  const long batchsize = M*blocksize;            //blockDim.x * gridDim.x*blocksize;'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x; // means vars pos'#10
+ '  const long b = blockDim.z * blockIdx.z + threadIdx.z; // batch batch pos'#10
+ '  long j = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (i>=M || j>=N || b>=K) return;'#10
+ '  j += b*batchsize + i*blocksize + offset; // block pos'#10
+ '  const nfloat v = sqrt(max(vars[i*vars_stride], sEPSILON));'#10
+ '  const nfloat m = means[i*means_stride];'#10
+ '  //dst += b*batchsize + i*blocksize + offset;'#10
+ '  dst[j] = (dst[j] - m)/v;'#10
+ '}'#10
+ 'extern "C" __global__ void means_vars_delta('#10
+ '          const long N, const long groups, const long blocksize,'#10
+ '          nfloat* delta,  nfloat* x, const long offset,'#10
+ '          nfloat* means,  nfloat* vars,'#10
+ '          nfloat* means_delta,  nfloat* vars_delta){'#10
+ '  nfloat m = 0;'#10
+ '  nfloat v = 0;'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i>=N) return;'#10
+ '  //const long ndst = blockDim.x * gridDim.x;'#10
+ '  x     += offset;'#10
+ '  delta += offset;'#10
+ '  // take a shortcut'#10
+ '  if(blocksize==1){'#10
+ '    #pragma unroll'#10
+ '    for (long j=0 ;j<groups; j++){'#10
+ '      const long index = i+j*N;'#10
+ '      m += delta[index];'#10
+ '      v += delta[index] * (x[index] - means[i]);'#10
+ '    }'#10
+ '    means_delta[i] = m * (-1.0f / sqrt(max(vars[i], sEPSILON)));'#10
+ '    vars_delta[i]  = v * -0.5f * pow(max(vars[i], sEPSILON), -1.5f);'#10
+ '    return;'#10
+ '  }'#10
+ '  #pragma unroll'#10
+ '  for (long j=0 ;j<groups; j++)'#10
+ '    #pragma unroll'#10
+ '    for (long k=0; k<blocksize; k++){'#10
+ '      const long index = (i + j*N)*blocksize + k;'#10
+ '      m += delta[index];'#10
+ '      v += delta[index] * (x[index] - means[i]);'#10
+ '    }'#10
+ '  means_delta[i] = m * (-1.0f / sqrt(max(vars[i], sEPSILON)));'#10
+ '  vars_delta[i]  = v * -0.5f * pow(max(vars[i], sEPSILON), -1.5f);'#10
+ '  //means_delta[i] = m * (-1.0f / sqrt(vars[i]));'#10
+ '  //vars_delta[i]  = v * -0.5f / (vars[i]*sqrt(vars[i]));'#10
+ '}'#10
+ 'extern "C" __global__ void norm_delta(const long N, const long blockSize, const long groups,'#10
+ '                                      nfloat* x, const long offset,  nfloat* means,  nfloat* vars,  nfloat* means_delta,  nfloat* vars_delta,  nfloat* delta){'#10
+ '  const long j = blockDim.z * blockIdx.z + threadIdx.z;'#10
+ '  if (j>=groups) return;'#10
+ '  const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i>=N) return;'#10
+ '  const long k = blockDim.y * blockIdx.y + threadIdx.y;'#10
+ '  if (k>=blockSize) return;'#10
+ '  //const long groups    = blockDim.z * gridDim.z;'#10
+ '  //const long N         = blockDim.x * gridDim.x;'#10
+ '  //const long blocksize = blockDim.y * gridDim.y;'#10
+ '  const long batchsize = blockSize * groups;'#10
+ '  const long index = (i + j*N) * blockSize +k;'#10
+ '  delta += offset;'#10
+ '  x     += offset;'#10
+ '  delta[index] ='#10
+ '    delta[index] / (sqrt(fmax(vars[i], sEPSILON))) +'#10
+ '    //delta[index] / (sqrt(vars[i])) +'#10
+ '    (2.0f * vars_delta[i] * (x[index] - means[i]) + means_delta[i]) / batchsize;'#10
+ '}'#10
+ 'extern "C" __global__ void add_dots('#10
+ '       const long N'#10
+ '       , const long groups'#10
+ '       , const long blockSize'#10
+ '       , nfloat* src1'#10
+ '       , nfloat* src2'#10
+ '       , const long srcOffset'#10
+ '       , nfloat* dst){'#10
+ '    const long filter = blockIdx.x;'#10
+ '    const long p = threadIdx.x;'#10
+ '    //if (blockDim.x*filter + p>=N) return;'#10
+ '    __shared__ float part[BLOCK];'#10
+ '    src1 += srcOffset;'#10
+ '    src2 += srcOffset;'#10
+ '    long i,b;'#10
+ '    float sum = 0;'#10
+ '    for(b = 0; b < groups; ++b){'#10
+ '        for(i = 0; i < blockSize; i += BLOCK){'#10
+ '            int index = p + i + blockSize*(filter + N*b);'#10
+ '            sum += (p+i < blockSize) ? src1[index]*src2[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    part[p] = sum;'#10
+ '    __syncthreads();'#10
+ '    if (p == 0) {'#10
+ '        for(i = 0; i < BLOCK; ++i) dst[filter] += part[i];'#10
+ '    }'#10
+ '    //const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '    //if (i>=N) return;'#10
+ '    //'#10
+ '    ////const long ndst = blockDim.x * gridDim.x;'#10
+ '    //nfloat sum = 0;'#10
+ '    //src1 += srcOffset;'#10
+ '    //src2 += srcOffset;'#10
+ '    //// take a shortcut'#10
+ '    //if (blockSize==1){'#10
+ '    //  sum = dotv(groups, src1 + i, N, src2 + i, N);'#10
+ '    //  dst[i] += sum;'#10
+ '    //  return;'#10
+ '    //}'#10
+ '    //#pragma unroll'#10
+ '    //for (long b=0; b<groups; b++){'#10
+ '    //  const long idx = (i + b * N) * blockSize;'#10
+ '    //  sum += dotv(blockSize, src1 + idx, 1, src2 + idx, 1);'#10
+ '    //}'#10
+ '    //dst[i] += sum;'#10
+ '}'#10
+ 'extern "C" __global__ void forward_scale('#10
+ '       const long N,'#10
+ '       const long scaleSize,'#10
+ '       const long blockSize,'#10
+ '       nfloat* output,'#10
+ '       const long outputOffset,'#10
+ '       nfloat* scale,'#10
+ '       const long scaleOffset,'#10
+ '       const long incb)'#10
+ '{'#10
+ '    const int i = blockIdx.x*blockDim.x + threadIdx.x;'#10
+ '    if (i >= N) return;'#10
+ '    const int f = (i / blockSize) % scaleSize;'#10
+ '    output[i + outputOffset] *= scale[f];'#10
+ '}'#10
+ 'extern "C" __global__ void forward_scale_add('#10
+ '       const long N,'#10
+ '       const long scaleSize,'#10
+ '       const long blockSize,'#10
+ '       nfloat* output,'#10
+ '       const long outputOffset,'#10
+ '       nfloat* scale,'#10
+ '       nfloat* bias,'#10
+ '       const long sOffset,'#10
+ '       const long incb)'#10
+ '{'#10
+ '    const int i = blockIdx.x*blockDim.x + threadIdx.x;'#10
+ '    if (i >= N) return;'#10
+ '    const int f = (i / blockSize) % scaleSize;'#10
+ '    output[i + outputOffset] = output[i + outputOffset]*scale[f] + bias[f];'#10
+ '}'#10
+ '//  Pseudorandom number generators'#10
+ '// https://en.wikipedia.org/wiki/Mersenne_Twister'#10
+ '#define NR 624'#10
+ '#define MR 397'#10
+ '#define wR 32'#10
+ '#define RR 31'#10
+ '#define UMASK (0xffffffffUL << RR)'#10
+ '#define LMASK (0xffffffffUL >> (wR-RR))'#10
+ '#define AR 0x9908b0dfUL'#10
+ '#define UR 11'#10
+ '#define SR 7'#10
+ '#define TR 15'#10
+ '#define LR 18'#10
+ '#define BR 0x9d2c5680UL'#10
+ '#define CR 0xefc60000UL'#10
+ '#define FR 1812433253UL'#10
+ '//typedef unsigned long long uint64_t;'#10
+ '//typedef unsigned long uint32_t;'#10
+ 'typedef struct'#10
+ '{'#10
+ '    unsigned int state_array[NR];         // the array for the state vector'#10
+ '    int state_index;                 // index into state vector array, 0 <= state_index <= NR-1   always'#10
+ '} mt_state;'#10
+ '__device__ void init_mt(mt_state* state, unsigned int seed)'#10
+ '{'#10
+ '    unsigned int* state_array = &(state->state_array[0]);'#10
+ '    state_array[0] = seed;                          // suggested initial seed = 19650218UL'#10
+ '    for (int i=1; i<NR; i++)'#10
+ '    {'#10
+ '        seed = FR * (seed ^ (seed >> (wR-2))) + i;    // Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.'#10
+ '        state_array[i] = seed;'#10
+ '    }'#10
+ '    state->state_index = 0;'#10
+ '}'#10
+ '__device__ unsigned int rand_mt(mt_state* state)'#10
+ '{'#10
+ '    unsigned int* state_array = &(state->state_array[0]);'#10
+ '    int k = state->state_index;      // point to current state location'#10
+ '                                     // 0 <= state_index <= NR-1   always'#10
+ '//  int k = k - NR;                   // point to state NR iterations before'#10
+ '//  if (k < 0) k += NR;               // modulo NR circular indexing'#10
+ '                                     // the previous 2 lines actually do nothing'#10
+ '                                     //  for illustration only'#10
+ '    int j = k - (NR-1);               // point to state NR-1 iterations before'#10
+ '    if (j < 0) j += NR;               // modulo NR circular indexing'#10
+ '    unsigned int x = (state_array[k] & UMASK) | (state_array[j] & LMASK);'#10
+ '    unsigned int xA = x >> 1;'#10
+ '    if (x & 0x00000001UL) xA ^= AR;'#10
+ '    j = k - (NR-MR);                   // point to state NR-MR iterations before'#10
+ '    if (j < 0) j += NR;               // modulo NR circular indexing'#10
+ '    x = state_array[j] ^ xA;         // compute next value in the state'#10
+ '    state_array[k++] = x;            // update new state value'#10
+ '    if (k >= NR) k = 0;               // modulo NR circular indexing'#10
+ '    state->state_index = k;'#10
+ '    unsigned int y = x ^ (x >> UR);       // tempering'#10
+ '             y = y ^ ((y << SR) & BR);'#10
+ '             y = y ^ ((y << TR) & CR);'#10
+ '    unsigned int z = y ^ (y >> LR);'#10
+ '    return z;'#10
+ '}'#10
+ '#define RANDOM_MAX 0x10000000ull'#10
+ '// https://en.wikipedia.org/wiki/Xorshift'#10
+ '__device__ unsigned long long rand_xorshift(const unsigned long long seed){'#10
+ '  //ulong res = ((seed + get_global_linear_id()) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);'#10
+ '  //return (res >> 16) % RANDOM_MAX;'#10
+ '  //      uint x = seed + get_global_linear_id();'#10
+ '    unsigned long long x = seed;'#10
+ '    x ^= x >> 12;'#10
+ '    x ^= x << 25;'#10
+ '    x ^= x >> 27;'#10
+ '    x *= 0x2545F4914F6CDD1DULL;'#10
+ '    //seed = x;'#10
+ '    return x%RANDOM_MAX;'#10
+ '}'#10
+ 'extern "C" __global__ void forward_dropout(const long N, const unsigned int seed, const nfloat probability, const nfloat scale, const nfloat* src, nfloat* rnd, nfloat* dst)'#10
+ '{'#10
+ '  long i  = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i>=N) return;'#10
+ '  //mt_state state;'#10
+ '  //init_mt(&state, seed + i);'#10
+ '  //rnd[i]  = rand_mt(&state);'#10
+ '  rnd[i]  = rand_xorshift(seed+i);'#10
+ '  rnd[i] /= RANDOM_MAX;'#10
+ '  dst[i]  = rnd[i] < probability? 0: src[i]*scale;'#10
+ '}'#10
+ 'extern "C" __global__ void backward_dropout(const long N, const nfloat probability, const nfloat scale, const nfloat* src, const nfloat* rnd, nfloat* dst)'#10
+ '{'#10
+ '  long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i>=N) return;'#10
+ '  dst[i] = rnd[i] < probability? 0: src[i]*scale;'#10
+ '}'#10
+ 'extern "C" __global__ void cost_l2(const long N, const nfloat* pred, const nfloat* truth, nfloat* delta, nfloat* error){'#10
+ '  long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '  if (i>=N) return;'#10
+ '  nfloat r = truth[i] - pred[i];'#10
+ '  delta[i] = r ;'#10
+ '  error[i] = r*r;'#10
+ '}'#10
+ 'extern "C" __global__ void mulv(const long N, nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* dst, const long dstOffset, const long incc){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if (i>=N) return;'#10
+ '   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset];'#10
+ '}'#10
+ 'extern "C" __global__ void fmav(  nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* src3, const long src3Offset, const long incc,  nfloat* dst, const long dstOffset, const long incd){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   //dst[i*incd + dstOffset] = mad(src1[i*inca + src1Offset], src2[i*incb + src2Offset], src3[i*incc + src3Offset]);'#10
+ '   dst[i*incd + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset] + src3[i*incc + src3Offset];'#10
+ '}'#10
+ 'extern "C" __global__ void power(const long N, nfloat* base, const long srcOffset, const long srcStride, const nfloat expo,  nfloat* dst, const long dstOffset, const long dstStride){'#10
+ '   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '   if(i>=N) return;'#10
+ '   dst[i*dstStride + dstOffset] = pow(base[i*srcStride + srcOffset], expo);'#10
+ '}'#10
+ '//extern "C" __global__ void halftest( half* a,  half* b, __global half* c){'#10
+ '//   const long i = blockDim.x * blockIdx.x + threadIdx.x;'#10
+ '//  c[i] = a[i]+b[i];'#10
+ '//}'#10
+ 'extern "C" __global__ void means(const long N, const long blocksize, const long groups,  nfloat* src, const long offset,  nfloat* means){'#10
+ '    __shared__ float local[BLOCK];'#10
+ '    const long id = threadIdx.x;'#10
+ '    local[id] = 0;'#10
+ '    long filter = blockIdx.x;'#10
+ '    src += offset;'#10
+ '    long i, j;'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            long index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            local[id] += (i+id < blocksize) ? src[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    __syncthreads();'#10
+ '    if(id == 0){'#10
+ '        float mean_tmp = 0;'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            mean_tmp += local[i];'#10
+ '        }'#10
+ '        mean_tmp /= blocksize * groups;'#10
+ '        means[filter] = mean_tmp;'#10
+ '    }'#10
+ '}'#10
+ 'extern "C" __global__ void vars(const long N, const long blocksize, const long groups,  nfloat* src, const long offset,  nfloat* means,  nfloat* vars){'#10
+ '    __shared__ float local[BLOCK];'#10
+ '    const long id = threadIdx.x;'#10
+ '    local[id] = 0;'#10
+ '    long filter = blockIdx.x;'#10
+ '    src += offset;'#10
+ '    long i, j;'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            long index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            //local[id] += (i+id < blocksize) ? pow((src[index] - means[filter]), 2.0f) : 0; // pow will not work always when -use_fast_math compiler switch'#10
+ '            local[id] += (i+id < blocksize) ? sqr(src[index] - means[filter]) : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    __syncthreads();'#10
+ '    if(id == 0){'#10
+ '        float variance_tmp = 0;'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            variance_tmp += local[i];'#10
+ '        }'#10
+ '        variance_tmp /= (blocksize * groups);'#10
+ '        vars[filter] = variance_tmp;'#10
+ '    }'#10
+ '}'#10
+ 'extern "C" __global__ void means_vars_delta_fast('#10
+ '          const long N, const long groups, const long blocksize,'#10
+ '          nfloat* delta,  nfloat* x, const long offset,'#10
+ '          nfloat* means,  nfloat* vars,'#10
+ '          nfloat* means_delta,  nfloat* vars_delta){'#10
+ '    __shared__ float local[BLOCK];'#10
+ '    long id = threadIdx.x;'#10
+ '    local[id] = 0;'#10
+ '    long filter = blockIdx.x;'#10
+ '    x     += offset;'#10
+ '    delta += offset;'#10
+ '    long i, j;'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            long index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            local[id] += (i+id < blocksize) ? delta[index] : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    __syncthreads();'#10
+ '    if(id == 0){'#10
+ '        means_delta[filter] = 0;'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            means_delta[filter] += local[i];'#10
+ '        }'#10
+ '        means_delta[filter] *= (-1.0f/sqrt(max(vars[filter] , sEPSILON)));'#10
+ '    }'#10
+ '    __syncthreads();'#10
+ '    local[id] = 0;'#10
+ '    for(j = 0; j < groups; ++j){'#10
+ '        for(i = 0; i < blocksize; i += BLOCK){'#10
+ '            long index = j*blocksize*N + filter*blocksize + i + id;'#10
+ '            local[id] += (i+id < blocksize) ? delta[index]*(x[index] - means[filter]) : 0;'#10
+ '        }'#10
+ '    }'#10
+ '    __syncthreads();'#10
+ '    if(id == 0){'#10
+ '        vars_delta[filter] = 0;'#10
+ '        for(i = 0; i < BLOCK; ++i){'#10
+ '            vars_delta[filter] += local[i];'#10
+ '        }'#10
+ '        vars_delta[filter] *= -0.5f * pow(max(vars[filter] , sEPSILON), -1.5f);'#10
+ '    }'#10
+ '}'#10
+ 'extern "C" __global__ void clamp(const long N, const nfloat alpha, const nfloat* src, nfloat* dst, const long stride, const long offset ){'#10
+ '   const int i = blockIdx.x*blockDim.x + threadIdx.x;'#10
+ '   if(i>=N) return;'#10
+ '   src += i*stride + offset;'#10
+ '   dst += i*stride + offset;'#10
+ '   *dst = min(alpha, max(*src, -alpha));'#10
+ '}'#10
+ 'extern "C" __global__ void inverse_sqrt(const long N, const nfloat* src, nfloat* dst, const long stride, const long offset){'#10
+ '   const int i = blockIdx.x*blockDim.x + threadIdx.x;'#10
+ '   if(i>=N) return;'#10
+ '   src += i*stride + offset;'#10
+ '   dst += i*stride + offset;'#10
+ '   *dst = 1/sqrt(max(*src, sEPSILON));'#10
+ '}'#10
