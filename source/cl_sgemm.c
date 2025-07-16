#pragma OPENCL EXTENSION cl_khr_fp16 : enable
//#define nfloat  float
#define sEPSILON 0.0000001f


/*
inline nfloat mapX(const nfloat x){
  return x*3-2.1F;
}
// Same purpose as mapX
// [0, 1] -> [-1.25, 1.25]
inline nfloat mapY(const nfloat y){
  return y*3 - 1.5F;
}

#define max_iteration  10000
#define _max           4.0f


__kernel void mandel(__global uchar *buf, const int w, const int h){

  const nfloat lnxp1_max_iteration = log1p((nfloat)max_iteration);

  int y = get_global_id(0);
  int x = get_global_id(1);
  nfloat xx = mapX(x/(nfloat)w);
  nfloat yy = mapY(y/(nfloat)h);

  y *= w * sizeof(uint);
  x *= sizeof(uint);

  nfloat x0 = 0.0f; nfloat y0 = 0.0f;
  int iteration = 0;
  nfloat oldAbs = 0.0f;
  nfloat coverageNum = max_iteration;
  buf += y;
  while (iteration < max_iteration) {
      nfloat xtemp = x0 * x0 - y0 * y0;
      y0 = 2 * x0 * y0;
      x0 = xtemp;
      x0 = x0 + xx;
      y0 = y0 + yy;
      nfloat currentAbs = x0*x0 + y0*y0;
      if (currentAbs>4.0f){
         nfloat diffToLast  = currentAbs - oldAbs;
         nfloat diffToMax   =       _max - oldAbs;
         coverageNum = iteration + diffToMax/diffToLast;
         break;
      }
      oldAbs = currentAbs;
      iteration++;
  }
  if (iteration == max_iteration)
#if defined(__MACH__)
  {
      buf[x] = 0xff;
      buf[x+1] = 0;
      buf[x+2] = 0;
      buf[x+3] = 0;
  } else
  {
      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;
      buf[x+0] = 0xff;
      buf[x+1] = c;
      buf[x+2] = c;
      buf[x+3] = c;
   }
#else
  {
      buf[x] = 0;
      buf[x+1] = 0;
      buf[x+2] = 0;
      buf[x+3] = 0xff;
  } else
  {
      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;
      buf[x+0] = c;
      buf[x+1] = c;
      buf[x+2] = c;
      buf[x+3] = 0xff;
  }
#endif
}
*/


  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]

typedef enum {
    acLOGISTIC, acRELU, acRELU6, acRELIE, acLINEAR, acRAMP, acTANH, acPLSE,
    acREVLEAKY, acLEAKY, acELU, acLOGGY, acSTAIR, acHARDTAN, acLHTAN, acSELU, acSOFTMAX,
    acGELU, acSWISH, acMISH, acHARD_MISH, acNORM_CHAN, acNORM_CHAN_SOFTMAX,
    acNORM_CHAN_SOFTMAX_MAXVAL
  } ActivationType;


nfloat sumv(const n_int N, __global const nfloat* v, const n_int stride){
  nfloat sum=0;
  #pragma unroll 8
  for (n_int i=0;i<N; i++)
    sum += v[i*stride];
  return sum;
}

nfloat maxv(const n_int N, __global const nfloat* v, const n_int stride){
  nfloat m=v[0];
  #pragma unroll 8
  for (n_int i=1;i<N; i++)
    m = max(m, v[i*stride]);
  return m;
}

nfloat minv(const n_int N, __global const nfloat* v, const n_int stride){
  nfloat m=v[0];
  #pragma unroll 8
  for (n_int i=1;i<N; i++)
    m = min(m, v[i*stride]);
  return m;
}

nfloat rssv(const n_int N, const nfloat mean, __global const nfloat* src, const n_int stride){
  nfloat sum=0;
  #pragma unroll 8
  for (n_int i=0;i<N; i++){
    const nfloat v = src[i*stride] - mean;
    sum += v*v;
  }
  return sum;
}

nfloat sqr(const nfloat x){
  return x*x;
}

//#define VW 8
//nfloat sumv_simd(const n_int N, __global nfloat* v){
//  float8 sum4 = 0;
//  nfloat sum = 0;
//  n_int n = N / VW;
//
//  #pragma unroll 8
//  for (n_int i=0;i<n; i++)
//    sum4 += vload8(i, v);
//  v += n*VW;
//
//  #pragma unroll 8
//  for (n_int i=0; i<N%VW;i++)
//    sum4[i] += v[i];
//  return sum4[0] + sum4[1] + sum4[2] + sum4[3] + sum4[4] + sum4[5] + sum4[6] + sum4[7] ;
//}

nfloat dotv(const n_int N, __global nfloat* a, const n_int inca,  __global nfloat* b, const n_int incb){

  nfloat d = 0;
  #pragma unroll 8
  for (n_int i=0; i<N;i++)
    d += a[i*inca]*b[i*incb];
  return d;
}

//nfloat dotv_simd(const n_int N, __global nfloat* a,  __global nfloat* b){
//
//  float8 d = 0;
//  n_int n = N / VW;
//  #pragma unroll 8
//  for (n_int i=0; i<n;i++)
//    d += vload8(i, a) * vload8(i, b);
//  a += n*VW;
//  b += n*VW;
//
//  #pragma unroll 8
//  for (n_int i=0; i<N%VW;i++)
//    d.x += a[i]*b[i];
//
//  return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] ;
//}

#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

              // naive GEMM with unrolling for now
__kernel void sgemm1_nn(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,
                      __global nfloat* A, const n_int aOffset, const n_int lda,
                      __global nfloat* B, const n_int bOffset, const n_int ldb,
                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {

    const n_int globalRow = get_global_id(0); // Row ID of C (0..M)
    const n_int globalCol = get_global_id(1); // Col ID of C (0..N)
    A += globalRow*lda + aOffset ;
    B += globalCol + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;
    *C += dotv(K, A, 1, B, ldb) * ALPHA;
}

__kernel void sgemm2_nn(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,
                      __global nfloat* A, const n_int aOffset, const n_int lda,
                      __global nfloat* B, const n_int bOffset, const n_int ldb,
                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {

    const n_int globalRow = get_global_id(1); // Row ID of C (0..M)
    const n_int globalCol = get_global_id(0); // Col ID of C (0..N)

    A += globalRow*lda + aOffset ;
    B += globalCol + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;
    *C += dotv(K, A, 1, B, ldb) * ALPHA;
}

__kernel void sgemm1_nt(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,
                      __global nfloat* A, const n_int aOffset, const n_int lda,
                      __global nfloat* B, const n_int bOffset, const n_int ldb,
                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {

    const n_int globalRow = get_global_id(0);    // M
    const n_int globalCol = get_global_id(1);    // N
    A += globalRow*lda + aOffset;
    B += globalCol*ldb + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;

    //nfloat acc =0;

    //#pragma unroll 8
    //for (n_int k=0; k<K; k++)
    //    acc += A[k] * B[k];
    //*C += acc * ALPHA;
    //*C += dotv(K, A, 1, B, 1) * ALPHA;
    *C += dotv(K, A, 1, B, 1) * ALPHA;
            //}
}

__kernel void sgemm1_tn(const n_int M, const n_int N, const n_int K, const nfloat ALPHA ,
                      __global nfloat* A, const n_int aOffset, const n_int lda,
                      __global nfloat* B, const n_int bOffset, const n_int ldb,
                      const nfloat BETA, __global nfloat* C, const n_int cOffset, const n_int ldc) {

    const n_int row = get_global_id(0); // Row ID of C (0..M)
    const n_int col = get_global_id(1); // Col ID of C (0..N)

    A += row           +  aOffset;
    B += col           +  bOffset;
    C += row*ldc + col +  cOffset;

    *C *= BETA;
    //nfloat acc = 0;

    //#pragma unroll 8
    //for (n_int k=0; k<K; k++)
    //    acc += A[k * lda] * B[k * ldb] ;
    //*C += acc * ALPHA ;
    *C += dotv(K, A, lda, B, ldb) * ALPHA ;
}

__kernel void forward_bias(const int reshape, __global nfloat* a,  const n_int aOffset, /*const n_int blockSize, */__global nfloat* b, const n_int bOffset, const n_int incb)
{
  n_int N, blockSize, i, k, j;
  switch (reshape) {
    case 0:
      N = get_global_size(0);
      blockSize = get_global_size(1);
      i = get_global_id(0);
      k = get_global_id(2);
      j = get_global_id(1);
      break;
    case 1:
      N = get_global_size(1);
      blockSize = get_global_size(0);
      i = get_global_id(1);
      k = get_global_id(2);
      j = get_global_id(0);
      break;
  }
  a += aOffset;
  //if (i==0 && k==0){
    //printf("        N = %ld\n", N);
    //printf("blockSize = %ld\n", blockSize);
  //}
  //for (i = 0; i<N; i++)
  //  for (k = 0; k<batch; k++){
  a += (k*N + i)*blockSize;
  nfloat bb = b[i * incb];
  //#pragma unroll 8
  //    for (n_int j=0; j<blockSize; j++)
        a[j] += bb;
  //}
}

nfloat stair_activate(const nfloat x)
{
  n_int n = floor(x);
  if (n % 2 == 0) return floor(x/ 2);
  return (x - n) + floor(x/2);
}

nfloat hardtan_activate(const nfloat x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return (x);
}

nfloat linear_activate(const nfloat x)
{
  return x;
}

nfloat logistic_activate(const nfloat x)
{
  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))
  return 1/(1 + exp(-x));
}

nfloat loggy_activate(const nfloat x)
{
  //result := 2/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp))) - 1;
  return 2/(1 + exp(-x)) - 1;
}

nfloat relu_activate(const nfloat x)
{
  //return x*n_int(x>0);
  if (x<0) return 0;
  return x;
}

nfloat relu6_activate(const nfloat x)
{
  //min_val_cmp(max_val_cmp(x, 0), 6)
  //result := EnsureRange(x,0,6);
  return  x*(x>0) * (x<=6);
}

nfloat elu_activate(const nfloat x)
{
  return (x >= 0)*x + (x < 0)*(exp(x)-1);
}

nfloat selu_activate(const nfloat x)
{
  return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);
}

nfloat gelu_activate(const nfloat x)
{
  return 0.5f*x*(1 + tanh(0.797885f*x + 0.035677f*pow(x, 3)));
}

nfloat relie_activate(const nfloat x)
{
  if (x>0) return x;
  else return 0.01f*x;
}

nfloat ramp_activate(const nfloat x)
{
  return  x*(x>0)+0.1f*x;
}

nfloat leaky_activate(const nfloat x)
{
  if (x>0) return  x;
  else return  0.1f*x;
}

nfloat tanh_activate(const nfloat x)
{
  //const nfloat px = exp(x);
  //const nfloat nx = exp(-x);
  //return (px - nx)/(px + nx);
  //return 2 / (1 + exp(ensureRange(-2 * x, minSingleExp, maxSingleExp))) - 1
  return 2/ (1+exp(-2*x)) - 1 ;
//  return  (exp(2*x)-1)/(exp(2*x)+1);
}

nfloat softplus_activate(const nfloat x, const nfloat threshold)
{
    if (x > threshold)
      return (x);                // too large
    else if (x < -threshold)
      return (exp(x));    // too small
    //return (log(exp(x) + 1));
    return log1p(exp(x));
}

nfloat plse_activate(const nfloat x)
{
    if (x < -4 ) return( 0.01f * (x + 4));
    if (x > 4 ) return( 0.01f * (x - 4) + 1);
    return  0.125f*x + 0.5f;
}

nfloat lhtan_activate(const nfloat x)
{
    if(x < 0) return (0.001f*x);
    if(x > 1) return (0.001f*(x-1) + 1);
    return  x;
}

nfloat silu_activate(const nfloat x)
{
    return x * logistic_activate(x) ;
}

#define  MISH_THRESHOLD 20.0f
nfloat mish_activate(const nfloat x)
{
    return x*tanh_activate(softplus_activate(x, MISH_THRESHOLD));
}
//void softmax_activate(const N:SizeInt; const x: PSingle);
//{
//  n_int i;
//  nfloat mx := TSingleTensor.maxv(N, Pointer(x), 1);//MaxValue(x, N);
//  for i:=0 to N-1 do
//    //x[i] := Exp(EnsureRange(x[i]-mx, minSingleExp, maxSingleExp));
//    x[i] := Exp(x[i]-mx);
//
//  mx := TSingleTensor.Sumv(N, pointer(x), 1);
//  //r:=copy(x);
//  //r.Exp();
//  for i :=0 to N-1 do
//    x[i] := x[i] / mx
//}


__kernel void activate_array( __global nfloat* x, n_int const offset, const ActivationType a)
{
      const n_int i = get_global_id(0);
      //int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
      x += offset;
      switch (a) {
          case acLOGISTIC:
            //for (i = 0; i< N; i++)
                x[i] = logistic_activate(x[i]);
            break;
          case acRELU:
            //for (i = 0; i< N; i++)
                x[i] = relu_activate(x[i]);
            break;
          case acRELU6:
            //for (i = 0; i< N; i++)
                x[i] = relu6_activate(x[i]);
            break;
          case acRELIE:
            //for (i = 0; i< N; i++)
                x[i] = relie_activate(x[i]);
            break;
          case acLINEAR:
            //for (i = 0; i< N; i++)
            //    x[i] = linear_activate(x[i])
            break;
          case acRAMP:
            //for (i = 0; i< N; i++)
                x[i] = ramp_activate(x[i]);
            break;
          case acTANH:
            //for (i = 0; i< N; i++)
                x[i] = tanh_activate(x[i]);
            break;
          case acPLSE:
            //for (i = 0; i< N; i++)
                x[i] = plse_activate(x[i]);
            break;
          case acREVLEAKY: case acLEAKY:
            //for (i = 0; i< N; i++)
             if (x[i]<0) x[i] = 0.1f*x[i];
              //x[i] = leaky_activate(x[i]);
            break;
          case acELU:
            //for (i = 0; i< N; i++)
                x[i] = elu_activate(x[i]);
            break;
          case acLOGGY:
            //for (i = 0; i< N; i++)
                x[i] = loggy_activate(x[i]);
            break;
          case acSTAIR:
            //for (i = 0; i< N; i++)
                x[i] = stair_activate(x[i]);
            break;
          case acHARDTAN:
            //for (i = 0; i< N; i++)
                x[i] = hardtan_activate(x[i]);
            break;
          case acLHTAN:
            //for (i = 0; i< N; i++)
                x[i] = lhtan_activate(x[i]);
            break;
          case acSELU:
            //for (i = 0; i< N; i++)
                x[i] = selu_activate(x[i]);
            break;
          case acGELU:
            //for (i = 0; i< N; i++)
                x[i] = gelu_activate(x[i]);
            break;
          case acSWISH:
                x[i] = silu_activate(x[i]);
            break;
          case acMISH:
                x[i] = mish_activate(x[i]);
            break;
          //case acSOFTMAX:
            //softmax_activate(N, x);
            //break
          default:
            if (i==0) printf("[Activation] %d: not Implemented\n", (int)a);

      }
   //printf("%ld, ", i);

}

__kernel void array_activate_swish(__global nfloat* x, n_int const offset, __global nfloat* output, __global nfloat* output2)
{
    n_int i = get_global_id(0);
    x += offset;
    output += offset;
    output2 += offset;
    nfloat x_val       = x[i];
    nfloat sigmoid     = logistic_activate(x_val);
    output[i]         = sigmoid;
    output2[i]        = x_val * sigmoid;
}




nfloat lhtan_gradient(const nfloat x)
{
    if ((x > 0) &&  (x < 1))
      return 1;
    return 0.001f;
}


nfloat hardtan_gradient(const nfloat x)
{
    if ((x > -1) && (x < 1))
      return 1;
    return 0;
}

nfloat linear_gradient(const nfloat x)
{
    return 1;
}

nfloat logistic_gradient(const nfloat x)
{
    return (1-x)*x;
}

nfloat loggy_gradient(const nfloat x)
{
    nfloat y = (x+1.0f)/2.0f;
    return 2.0f*(1.0f-y)*y;
}

nfloat stair_gradient(const nfloat x)
{
    if (floor(x) == x) return( 0);
    return 1;
}

nfloat relu_gradient(const nfloat x)
{
    return (x>0?1:0);
}

nfloat relu6_gradient(const nfloat x)
{
    return ((x>0) && (x<6)?1:0);
}

nfloat elu_gradient(const nfloat x)
{
    return (x >= 0?1:0) + (x < 0?1:0)*(x + 1);
}

nfloat selu_gradient(const nfloat x)
{
    return (x >= 0?1:0)*1.0507f + (x < 0?1:0)*(x + 1.0507f*1.6732f);
}

nfloat relie_gradient(const nfloat x)
{
    if (x>0) return 1;
    else return 0.01f;
}

nfloat ramp_gradient(const nfloat x)
{
    return (x>0?1:0) + 0.1f;
}

nfloat leaky_gradient(const nfloat x)
{
    if (x>0) return 1;
    else return 0.1f;
}

nfloat tanh_gradient(const nfloat x)
{
    return 1.0f-x*x;
}

nfloat sech(const nfloat x)
{
    return 2.0f / (exp(x) + exp(-x));
}

nfloat gelu_gradient(const nfloat x)
{
    nfloat x3 = x*x*x;
    nfloat sec = sech(0.0356774f*x3 + 0.797885f*x);
    return 0.5f*tanh(0.0356774f*x3 + 0.797885f*x) + (0.0535161f*x3 + 0.398942f*x) * sec*sec + 0.5f ;
}

nfloat plse_gradient(const nfloat x)
{

  if ((x < 0) || (x > 1))
    return  0.01f;
  else
    return 0.125f;
}

__kernel void gradient_array(__global nfloat* x, n_int const offset, const ActivationType a, __global nfloat* delta)
{
    n_int i = get_global_id(0);

    x += offset;
    delta += offset;
    switch (a) {
        case acLOGISTIC:
          //for (i = 0; i<N;i++)
              delta[i] *= logistic_gradient(x[i]);
              break;
        case acRELU:
          //for (i = 0; i<N;i++)
              delta[i] *= x[i]>0?1:0;//relu_gradient(x[i]);
              break;
        case acRELU6:
          //for (i = 0; i<N;i++)
              delta[i] *= relu6_gradient(x[i]);
              break;
        case acRELIE:
          //for (i = 0; i<N;i++)
              delta[i] *= relie_gradient(x[i]);
              break;
        case acLINEAR:
          //////for (i = 0; i<N;i++)
          //    delta[i] *= linear_gradient(x[i])
          //;
              break;
        case acRAMP:
          //for (i = 0; i<N;i++)
              delta[i] *= ramp_gradient(x[i]);
              break;
        case acTANH:
          //for (i = 0; i<N;i++)
              delta[i] *= tanh_gradient(x[i]);
              break;
        case acPLSE:
          //for (i = 0; i<N;i++)
              delta[i] *= plse_gradient(x[i]);
              break;
        case acREVLEAKY: case acLEAKY:
          //for (i = 0; i<N;i++)
              delta[i] *= leaky_gradient(x[i]);
              break;
        case acELU:
          //for (i = 0; i<N;i++)
              delta[i] *= elu_gradient(x[i]);
              break;
        case acLOGGY:
          //for (i = 0; i<N;i++)
              delta[i] *= loggy_gradient(x[i]);
              break;
        case acSTAIR:
          //for (i = 0; i<N;i++)
              delta[i] *= stair_gradient(x[i]);
              break;
        case acHARDTAN:
          //for (i = 0; i<N;i++)
              delta[i] *= hardtan_gradient(x[i]);
              break;
        case acLHTAN:
          //for (i = 0; i<N;i++)
              delta[i] *= lhtan_gradient(x[i]);
              break;
        case acSELU:
          //for (i = 0; i<N;i++)
              delta[i] *= selu_gradient(x[i]);
              break;
        case acGELU:
          //for (i = 0; i<N;i++)
              delta[i] *= gelu_gradient(x[i]);
              break;
    //   case acSWISH:
    //               ;
    //
    //   case acMISH:
    //               ;
    //
    //   case acHARD_MISH:
    //               ;
    //
    //   case acNORM_CHAN:
    //               ;
    //
    //   case acNORM_CHAN_SOFTMAX:
    //               ;
    //
    //   case acNORM_CHAN_SOFTMAX_MAXVAL:
    //
        default:
            if (i==0) printf("[Gradient] : not Implemented %d\n", (int)a);

    }

}

// #define BLOCK 512
__kernel void backward_bias(__global nfloat* dst, const n_int blockSize, __global nfloat* src, const n_int srcOffset, const n_int batch)
{
    //const n_int filter = get_group_id(0);
    //const n_int p = get_local_id(0);
    //const n_int N = get_global_size(0);
    //
    //int i,b;
    //local float part[BLOCK];
    //
    //src += srcOffset;
    //float sum = 0;
    //for(b = 0; b < batch; ++b){
    //    for(i = 0; i < blockSize; i += BLOCK){
    //        int index = p + i + blockSize*(filter + N*b);
    //        sum += (p+i < blockSize) ? src[index] : 0;
    //    }
    //}
    //part[p] = sum;
    //
    ////__syncthreads();
    //barrier(CLK_LOCAL_MEM_FENCE);
    //if (p == 0) {
    //    printf("summing up :\n");
    //    for(i = 0; i < BLOCK; ++i){
    //      dst[filter] += part[i];
    //      printf("%f\n", part[i]);
    //    }
    //}
    const n_int i = get_global_id(0);//if (i==0) printf("n_int %ull\n", sizeof(n_int));
    const n_int N = get_global_size(0);
    //for (n_int i=0 ; i<N ;i++) {
      nfloat sum = 0;
      src += i * blockSize + srcOffset;
      const n_int incbias = N*blockSize;
      // take a shortcut
      if(blockSize==1){
        sum = sumv(batch, src, N);
        dst[i] +=sum;
        return;
      }

      #pragma unroll 8
      for (n_int j=0; j<batch; j++){
        sum += sumv(blockSize, src, 1);
        src += incbias;
      }
      dst[i] +=sum;
}

__kernel void addv( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* dst, const n_int dstOffset, const n_int incc){

   const n_int i = get_global_id(0);

   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] + src2[i*incb + src2Offset];
}

__kernel void subv( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* dst, const n_int dstOffset, const n_int incc){

   const n_int i = get_global_id(0);

   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] - src2[i*incb + src2Offset];
}

__kernel void axpy(const nfloat a, __global nfloat* x, const n_int xOffset, const n_int incx, __global nfloat* y, const n_int yOffset, const n_int incy){

   const n_int i = get_global_id(0);
   x += xOffset;
   y += yOffset;
   y[i*incy] += a*x[i*incx];

}

__kernel void scale(const nfloat a, __global nfloat* x, const n_int incx){

   const n_int i = get_global_id(0);
   x[i*incx] *= a;

}

__kernel void crossEntropyLogistics(__global const nfloat* pred, __global const nfloat* truth, __global nfloat* delta, __global nfloat* error){

  const n_int i = get_global_id(0);
  nfloat t = truth[i];
  nfloat p = pred[i];
  error[i] = -t*log(max(p, sEPSILON)) - (1-t) * log(max(1 - p, sEPSILON));
  //error[i] = -t*log(p) - (1-t) * log(1 - p);
  delta[i] = t - p;
   //printf("%ld, ", i);

}

__kernel void fill(__global nfloat* x, const n_int offset, const nfloat val, const n_int stride){

   const n_int i = get_global_id(0);
   //x += offset;
   x[i*stride + offset] = val;
}

// naive copy for now
__kernel void copy(
    __global nfloat* src, const n_int srcOffset, const n_int srcInc
  , __global nfloat* dst, const n_int dstOffset, const n_int dstInc){

   const n_int i = get_global_id(0);
   //src += srcOffset; dst += dstOffset;
   //if (srcInc==1 && dstInc==1){
   //  dst[i+dstOffset] = src[i+srcOffset];
   //  return;
   //}
   dst[i*dstInc + dstOffset] = src[i*srcInc + srcOffset];
}

__kernel void forward_maxpool(
     __global nfloat* input
     , const n_int c, const n_int h, const n_int w
     , const n_int stride_x, const n_int stride_y, const n_int padding, const n_int kernelSize
     , __global n_int* indexes, __global nfloat* output){

  const n_int w_offset = -padding / 2;
  const n_int h_offset = -padding / 2;

  //const n_int outC = get_global_size(0);
  const n_int outH = get_global_size(1);
  const n_int outW = get_global_size(2);
  n_int k = get_global_id(0);
  n_int y = get_global_id(1);
  n_int x = get_global_id(2);

  n_int out_index = x + outW*(y + outH*k) ;//+ outW*outH*outC*b;
  nfloat max = -FLT_MAX;
  n_int max_i = -1;
  #pragma unroll 8
  for (n_int n=0; n<kernelSize; n++)
      #pragma unroll 8
      for (n_int m=0; m<kernelSize; m++){
          n_int cur_h = h_offset+y * stride_y+n;
          n_int cur_w = w_offset+x * stride_x+m;
          n_int index = cur_w + w*(cur_h + h*k) ;//+ w*h*outC*b;
          nfloat val = (cur_h >= 0) && (cur_h < h) && (cur_w >= 0) && (cur_w < w)? input[index]: -FLT_MAX;
          if (val > max){
            max_i = index;
            max = val;
          }
      }
  output[out_index] = max;
  if (indexes)
      indexes[out_index] = max_i;
}

__kernel void backward_maxpool( __global nfloat* output, __global const n_int* indexes, __global const nfloat* delta){
        const n_int i = get_global_id(0);
        const n_int j = get_global_id(1);
        const n_int id = i*get_global_size(1) + j;

        const n_int index = indexes[id];
        output[index] += delta[id];
}


void softmax(const n_int n, __global nfloat* input, const n_int stride, const nfloat temp, __global nfloat* output){

  nfloat largest = maxv(n, input, stride);
  nfloat sum = 0;
  #pragma unroll 8
  for (n_int i=0;i<n;i++) {
      nfloat e = exp((input[i*stride] - largest)/temp);
      sum += e;
      output[i*stride] = e;
  }

  #pragma unroll 8
  for (n_int i=0; i<n; i++)
      output[i*stride]/=sum;
}

__kernel void softmaxBatch(__global nfloat* input, const n_int iOffset, const n_int n
  , const n_int batch_size, const n_int group_size, const n_int stride
  , const nfloat temp, __global nfloat* output, const n_int oOffset){

  const n_int b = get_global_id(0);
  const n_int g = get_global_id(1);

  softmax(n
  , input + iOffset + b*batch_size + g*group_size
  , stride
  , temp
  , output + oOffset + b*batch_size + g*group_size);

}

void move(const __global nfloat* src, __global nfloat* dst , const n_int count){
  #pragma unroll 8
  for (n_int i=0 ; i<count; i++) dst[i] = src[i];
}

__kernel void crossEntropySoftmax(const __global nfloat* pred, const __global nfloat* truth, __global nfloat* delta, __global nfloat* error){

  const n_int i = get_global_id(0);

  nfloat t = truth[i];
  nfloat p = pred[i];
  if (t!=0)
      error[i] = -log(max(p, sEPSILON));
      //error[i] = -log(p);
  else
      error[i] = 0;
  delta[i] = t - p;

}

__kernel void im2col(const n_int aHeight, const n_int aWidth
  , const n_int kernelHeight, const n_int kernelWidth, const n_int padHeight, const n_int padWidth
  , const n_int strideY, const n_int strideX, const n_int dilationY, const n_int dilationX
  , __global nfloat* im , const n_int imOffset
  , __global nfloat* col, const n_int colOffset, const n_int batch){

  n_int aChannels = get_global_size(0);
  n_int chan = get_global_id(0);
  n_int k = get_global_id(1);
  const n_int kernelSize = kernelHeight*kernelWidth;

  const n_int outWidth = (aWidth + 2 * padWidth - (dilationX * (kernelWidth - 1) + 1)) / strideX + 1;
  const n_int outHeight = (aHeight + 2 * padHeight - (dilationY * (kernelHeight - 1) + 1)) / strideY + 1;
  const n_int outSize = outWidth * outHeight;
  const n_int inSize = aWidth * aHeight;
  const n_int sizeX = outWidth - 2 * padWidth;

    //for (n_int k=0 ; k<kernelWidth*kernelHeight; k++)
    {
      n_int kernelRow = k / kernelWidth;
      n_int kernelCol = k % kernelWidth;
      //const n_int kernelRow = get_global_id(1);
      //const n_int kernelCol = get_global_id(2);

      #pragma unroll 8
      for (n_int b=0 ; b<batch; b++)
      {
        n_int i = (b*aChannels + chan)*inSize + aWidth*(kernelRow*dilationY - padHeight) + kernelCol*dilationX - padWidth;
        __global nfloat* im1 = imOffset + im + i;
        __global nfloat* col1 = colOffset + col + padWidth * outWidth + outSize*kernelSize*(chan + b*aChannels) + outSize*(kernelRow * kernelWidth + kernelCol) ;
        #pragma unroll 8
        for (n_int outRow=padHeight ; outRow<outHeight - padHeight ; outRow++)
        {
          n_int j = outRow * aWidth * strideY + padWidth * strideX ;
          if (strideX == 1)
            move(im1 + j, col1 + padWidth, sizeX);
          else
            #pragma unroll 8
            for (n_int outCol=padWidth ;  outCol<outWidth - padWidth; outCol++)
            {
              //j := outRow * aWidth * strideY + outCol * strideX;
              col1[outCol] = im1[j];
              j += strideX;
            }
          col1 += outWidth;
        }
      }
    }
}

void fills(__global nfloat* dst, const n_int N, const nfloat val){
  #pragma unroll 8
  for (n_int i=0 ; i<N; i++)
    dst[i] = val;
}


//col2im :
//https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/col2im.opencl

// Work-group size parameters re-used from the 'copy' kernel
#ifndef COPY_DIMX
  #define COPY_DIMX 8      // Local workgroup size in the first dimension (w)
#endif
#ifndef COPY_DIMY
  #define COPY_DIMY 8      // Local workgroup size in the second dimension (h)
#endif

// =================================================================================================

n_int grid_ceil(const n_int x, const n_int step) {
  return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;
}

void xim2col(const n_int input_h, const n_int input_w, const n_int channels,
                         const n_int output_h, const n_int output_w,
                         const n_int kernel_h, const n_int kernel_w,
                         const n_int pad_h, const n_int pad_w,
                         const n_int stride_h, const n_int stride_w,
                         const n_int dilation_h, const n_int dilation_w,
                         const bool kernel_flip,
                         const __global nfloat* restrict im_buffer, const n_int im_offset,
                         __global nfloat* col_buffer, const n_int col_offset) {

  // Thread IDs
  const n_int w_id = get_global_id(0); // image width, max 'output_w'
  const n_int h_id = ((n_int)get_global_id(1)) % output_h; // image height, max 'output_h'
  const n_int c_id = ((n_int)get_global_id(1)) / output_h; // input channels
  if (h_id < output_h && w_id < output_w && c_id < channels) {
    #pragma unroll 8
    for (n_int kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height
      #pragma unroll 8
      for (n_int kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width

        // Retrieves the input value
        const n_int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
        const n_int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;
        nfloat val;
        if (h_index >= 0 && h_index < input_h &&
            w_index >= 0 && w_index < input_w) {
          const n_int input_index = w_index + input_w * (h_index + input_h * c_id);
          val = im_buffer[input_index + im_offset];
        }
        else {
          val = 0;
        }

        // Sets the output value
        const n_int kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const n_int patch_index = w_id + output_w * h_id;
        const n_int output_index = patch_index + kernel_index * output_w * output_h +
                                  c_id * output_w * output_h * kernel_h * kernel_w;
        col_buffer[output_index + col_offset] = val;
      }
    }
  }
}

// =================================================================================================

// Kernel flip version of the Xim2col kernel (for convolution)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
//  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xim2colKernelFlip(const n_int input_h, const n_int input_w, const n_int channels,
                       const n_int output_h, const n_int output_w,
                       const n_int kernel_h, const n_int kernel_w,
                       const n_int pad_h, const n_int pad_w,
                       const n_int stride_h, const n_int stride_w,
                       const n_int dilation_h, const n_int dilation_w,
                       const __global nfloat* restrict im_buffer, const n_int im_offset,
                       __global nfloat* col_buffer, const n_int col_offset) {
  const bool kernel_flip = true;
  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          kernel_flip,
          im_buffer, im_offset, col_buffer, col_offset);
}

// Normal version of the Xim2col kernel (for cross-correlation)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
//  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xim2colKernelNormal(const n_int input_h, const n_int input_w, const n_int channels,
                         const n_int output_h, const n_int output_w,
                         const n_int kernel_h, const n_int kernel_w,
                         const n_int pad_h, const n_int pad_w,
                         const n_int stride_h, const n_int stride_w,
                         const n_int dilation_h, const n_int dilation_w,
                         const __global nfloat* restrict im_buffer, const n_int im_offset,
                         __global nfloat* col_buffer, const n_int col_offset) {
  const bool kernel_flip = false;
  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          kernel_flip,
          im_buffer, im_offset, col_buffer, col_offset);
}

void xcol2im(const n_int input_h, const n_int input_w, const n_int channels,
                         const n_int output_h, const n_int output_w,
                         const n_int kernel_h, const n_int kernel_w,
                         const n_int pad_h, const n_int pad_w,
                         const n_int stride_h, const n_int stride_w,
                         const n_int dilation_h, const n_int dilation_w,
                         const n_int stride_bez_h, const n_int stride_bez_w,
                         const n_int dilation_bez_h, const n_int dilation_bez_w,
                         const n_int gcd_h, const n_int gcd_w,
                         const bool kernel_flip,
                         const __global nfloat* restrict col_buffer, const n_int col_offset,
                         __global nfloat* im_buffer, const n_int im_offset) {

  const n_int input_h_scaled = (input_h - 1) / gcd_h + 1;

  // Thread IDs
  const n_int gcd_scale_w = get_global_id(0) + (pad_w - 1) / gcd_w + 1;
  const n_int gcd_scale_h = ((n_int) get_global_id(1)) % input_h_scaled + (pad_h - 1) / gcd_h + 1;
  const n_int c_id = ((n_int) get_global_id(1)) / input_h_scaled;

  const n_int w_index = gcd_scale_w * gcd_w - pad_w;
  const n_int h_index = gcd_scale_h * gcd_h - pad_h;
  const n_int th_step = stride_h * dilation_h / gcd_h;
  const n_int th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,
                                     (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),
                                 th_step);
  const n_int th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,
                         (dilation_bez_h * gcd_scale_h + 1) * dilation_h);
  const n_int tw_step = stride_w * dilation_w / gcd_w;
  const n_int tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,
                                     (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),
                                 tw_step);
  const n_int tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,
                         (dilation_bez_w * gcd_scale_w + 1) * dilation_w);
  if (w_index < input_w && c_id < channels) {
    nfloat val = 0;
    #pragma unroll 8
    for (n_int th = th_begin; th < th_end; th += th_step) {
      #pragma unroll 8
      for (n_int tw = tw_begin; tw < tw_end; tw += tw_step) {
        const n_int kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;
        const n_int kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;
        const n_int h_id = th / stride_h + stride_bez_h * gcd_scale_h;
        const n_int w_id = tw / stride_w + stride_bez_w * gcd_scale_w;
        const n_int kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const n_int patch_index = w_id + output_w * h_id;
        const n_int output_index = patch_index + kernel_index * output_w * output_h +
                                 c_id * output_w * output_h * kernel_h * kernel_w;
        val += col_buffer[output_index + col_offset];
      }
    }

    // Accumulates the resulting value with the existing im-buffer (+= val)
    const n_int input_index = w_index + input_w * (h_index + input_h * c_id);
    nfloat im_buffer_value = im_buffer[input_index + im_offset];
    im_buffer[input_index + im_offset] = im_buffer_value + val;
  }
}

// =================================================================================================

// Kernel flip version of the Xcol2im kernel (for convolution)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
  //__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xcol2imKernelFlip(const n_int input_h, const n_int input_w, const n_int channels,
                       const n_int output_h, const n_int output_w,
                       const n_int kernel_h, const n_int kernel_w,
                       const n_int pad_h, const n_int pad_w,
                       const n_int stride_h, const n_int stride_w,
                       const n_int dilation_h, const n_int dilation_w,
                       const n_int stride_bez_h, const n_int stride_bez_w,
                       const n_int dilation_bez_h, const n_int dilation_bez_w,
                       const n_int gcd_h, const n_int gcd_w,
                       const __global nfloat* restrict col_buffer, const n_int col_offset,
                       __global nfloat* im_buffer, const n_int im_offset) {
  const bool kernel_flip = true;
  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}

// Normal version of the Xcol2im kernel (for cross-correlation)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
  //__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xcol2imKernelNormal(const n_int input_h, const n_int input_w, const n_int channels,
                         const n_int output_h, const n_int output_w,
                         const n_int kernel_h, const n_int kernel_w,
                         const n_int pad_h, const n_int pad_w,
                         const n_int stride_h, const n_int stride_w,
                         const n_int dilation_h, const n_int dilation_w,
                         const n_int stride_bez_h, const n_int stride_bez_w,
                         const n_int dilation_bez_h, const n_int dilation_bez_w,
                         const n_int gcd_h, const n_int gcd_w,
                         const __global nfloat* restrict col_buffer, const n_int col_offset,
                         __global nfloat* im_buffer, const n_int im_offset) {
  const bool kernel_flip = false;
  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}


__kernel void upsample(__global nfloat* in, const n_int stride, const int isForward, const nfloat scale, __global nfloat* out, const int zero){

   const n_int c = get_global_id(0);
   const n_int y = get_global_id(1);
   const n_int x = get_global_id(2);
   const n_int h = get_global_size(1)/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??
   const n_int w = get_global_size(2)/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??

   const n_int in_index   = (c*h + (y / stride))*w + x / stride;
   const n_int out_index  = (c*h*stride + y)*stride*w + x;   // <-- why having to adjust by stride instead of remving it !!!
   if (isForward){
     out[out_index] = scale*in[in_index];
     return;
   }
   if (zero) in[in_index] = 0;
   in[in_index] += scale*out[out_index]; // ok seems like, it's because of trying to add adjust all input pixels in the stride a in here

}

__kernel void fmavss(__global nfloat* src, const n_int offset, const nfloat scalar, const nfloat bias, __global nfloat* dst){

  //const n_int w = get_global_size(1);
  const n_int y = get_global_id(0);
  //const n_int x = get_global_id(1);
  //const n_int idx = y*w + x;
  src += offset;
  dst += offset;
  //dst[idx] = mad(src[idx], scalar, bias);
  //dst[idx] = src[idx]*scalar + bias;
  dst[y] = src[y]*scalar + bias;

}

__kernel void means(const n_int blocksize, const n_int groups,  __global nfloat* src, const n_int offset, __global nfloat* means){

    local float buff[BLOCK];
    const n_int N  = get_num_groups(0);
    const n_int id = get_local_id(0);
    buff[id] = 0;

    n_int filter = get_group_id(0);
    src += offset;
    n_int i, j;
    #pragma unroll 8
        for(j = 0; j < groups; ++j){
        #pragma unroll 8
        for(i = 0; i < blocksize; i += BLOCK){
            n_int index = j*blocksize*N + filter*blocksize + i + id;
            buff[id] += (i+id < blocksize) ? src[index] : 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        float mean_tmp = 0;
        #pragma unroll 8
        for(i = 0; i < BLOCK; ++i){
            mean_tmp += buff[i];
        }
        mean_tmp /= blocksize * groups;
        means[filter] = mean_tmp;
    }

}

__kernel void vars(const n_int blocksize, const n_int groups, __global nfloat* src, const n_int offset, __global nfloat* means, __global nfloat* vars){

    local float buf[BLOCK];
    const n_int N  = get_num_groups(0);
    const n_int id = get_local_id(0);
    buf[id] = 0;

    n_int filter = get_group_id(0);
    src += offset;
    n_int i, j;
    #pragma unroll 8
    for(j = 0; j < groups; ++j){
        #pragma unroll 8
        for(i = 0; i < blocksize; i += BLOCK){
            n_int index = j*blocksize*N + filter*blocksize + i + id;
            buf[id] += (i+id < blocksize) ? sqr(src[index] - means[filter]) : 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        float variance_tmp = 0;
        #pragma unroll 8
        for(i = 0; i < BLOCK; ++i){
            variance_tmp += buf[i];
        }
        variance_tmp /= (blocksize * groups);
        vars[filter] = variance_tmp;
    }
}


__kernel void means_vars(const n_int blocksize, const n_int groups, __global nfloat* src, const n_int offset, __global nfloat* means, __global nfloat* vars){

    local float buf[BLOCK];
    const n_int N  = get_num_groups(0);
    const n_int id = get_local_id(0);//threadIdx.x;
    n_int filter = get_group_id(0);//blockIdx.x;

    buf[id] = 0;

    src += offset;
    n_int i, j;

    #pragma unroll 8
    for(j = 0; j < groups; ++j){
        #pragma unroll 8
        for(i = 0; i < blocksize; i += BLOCK){
            n_int index = j*blocksize*N + filter*blocksize + i + id;
            buf[id] += (i+id < blocksize) ? src[index] : 0;
        }
    }
    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        float mean_tmp = 0;
        #pragma unroll 8
        for(i = 0; i < BLOCK; ++i){
            mean_tmp += buf[i];
        }
        mean_tmp /= blocksize * groups;
        means[filter] = mean_tmp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //src += offset;
    //n_int i, j;
    buf[id] = 0;
    #pragma unroll 8
    for(j = 0; j < groups; ++j){
        #pragma unroll 8
        for(i = 0; i < blocksize; i += BLOCK){
            n_int index = j*blocksize*N + filter*blocksize + i + id;
            buf[id] += (i+id < blocksize) ? sqr(src[index] - means[filter]) : 0;
        }
    }
    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);

    if(id == 0){
        float variance_tmp = 0;
        #pragma unroll 8
        for(i = 0; i < BLOCK; ++i){
            variance_tmp += buf[i];
        }
        variance_tmp /= (blocksize * groups);
        vars[filter] = variance_tmp;
    }

    //nfloat m = 0;
    //nfloat v = 0;
    //const n_int i = get_global_id(0);
    //const n_int N = get_global_size(0);
    //const n_int S = groups*blocksize;
    //src += offset;
    //// take a shortcut
    //if(blocksize==1){
    //  m = sumv(groups, src+i, N);
    //  m /= S;
    //  v = rssv(groups, m, src+i, N);
    //  means[i] = m;
    //  vars[i]  = v / (S-1);
    //  return;
    //}
    //
    //#pragma unroll
    //for (n_int b=0; b<groups; b++){
    //    const n_int idx = (i + b*N)*blocksize;
    //    m += sumv(blocksize, src + idx, 1);
    //}
    //m /= S;
    //#pragma unroll
    //for (n_int b=0; b<groups; b++){
    //    const n_int idx = (i + b*N)*blocksize;
    //    v += rssv(blocksize, m, src + idx, 1);
    //}
    //means[i] = m;
    //vars[i]  = v / (S-1);
}

__kernel void normvv(__global nfloat* mean, const n_int mean_stride, __global nfloat* variance, const n_int variance_stride, __global nfloat* dst, const n_int dst_stride)
{
  const n_int i = get_global_id(0);
  dst[i * dst_stride] = (dst[i*dst_stride] - mean[i*mean_stride])/sqrt(max(variance[i*variance_stride], sEPSILON));
}

__kernel void normvs(__global nfloat* src, const nfloat mean, const nfloat variance)
{
  const n_int i = get_global_id(0);
  src[i] = (src[i] - mean)/sqrt(max(variance, sEPSILON));
}

__kernel void normblkvv(__global nfloat* means, const n_int means_stride, __global nfloat* vars, const n_int vars_stride, __global nfloat* dst, const n_int offset)
{
  const n_int blocksize = get_global_size(1);
  const n_int batchsize = get_global_size(0)*blocksize;
  const n_int i = get_global_id(0); // means vars pos
  const n_int b = get_global_id(2); // batch batch pos
  const n_int j = get_global_id(1)+ b*batchsize + i*blocksize + offset; // block pos
  const nfloat v = sqrt(max(vars[i*vars_stride], sEPSILON));
  const nfloat m = means[i*means_stride];
  //dst += b*batchsize + i*blocksize + offset;
  dst[j] = (dst[j] - m)/v;
}

__kernel void means_vars_delta(const n_int groups, const n_int blocksize,
         __global nfloat* delta, __global nfloat* x, const n_int offset,
         __global nfloat* means, __global nfloat* vars,
         __global nfloat* means_delta, __global nfloat* vars_delta){

  local float buf[BLOCK];

  const n_int N = get_num_groups(0);
  n_int id = get_local_id(0);
  n_int filter = get_group_id(0);

  buf[id] = 0;


  x     += offset;
  delta += offset;

  n_int i, j;
  #pragma unroll 8
  for(j = 0; j < groups; ++j){
      #pragma unroll 8
      for(i = 0; i < blocksize; i += BLOCK){
          n_int index = j*blocksize*N + filter*blocksize + i + id;
          buf[id] += (i+id < blocksize) ? delta[index] : 0;
      }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if(id == 0){
      means_delta[filter] = 0;
      #pragma unroll 8
      for(i = 0; i < BLOCK; ++i){
          means_delta[filter] += buf[i];
      }
      means_delta[filter] *= (-1.0f/sqrt(max(vars[filter] , sEPSILON)));
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  buf[id] = 0;
  #pragma unroll 8
  for(j = 0; j < groups; ++j){
      #pragma unroll 8
      for(i = 0; i < blocksize; i += BLOCK){
          n_int index = j*blocksize*N + filter*blocksize + i + id;

          buf[id] += (i+id < blocksize) ? delta[index]*(x[index] - means[filter]) : 0;
      }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if(id == 0){
      vars_delta[filter] = 0;
      #pragma unroll 8
      for(i = 0; i < BLOCK; ++i){
          vars_delta[filter] += buf[i];
      }
      vars_delta[filter] *= -0.5f * pow(max(vars[filter] , sEPSILON), -1.5f);
  }

  //nfloat m = 0;
  //nfloat v = 0;
  //const n_int i = get_global_id(0);
  //const n_int ndst = get_global_size(0);
  //x     += offset;
  //delta += offset;
  //// take a shortcut
  //if(blocksize==1){
  //  #pragma unroll
  //  for (n_int j=0 ;j<groups; j++){
  //    const n_int index = i+j*ndst;
  //    m += delta[index];
  //    v += delta[index] * (x[index] - means[i]);
  //  }
  //  means_delta[i] = m * (-1.0f / sqrt(fmax(vars[i], sEPSILON)));
  //  vars_delta[i]  = v * -0.5f * pow(fmax(vars[i], sEPSILON), -1.5f);
  //  return;
  //}
  //#pragma unroll
  //for (n_int j=0 ;j<groups; j++)
  //  #pragma unroll
  //  for (n_int k=0; k<blocksize; k++){
  //    const n_int index = (i + j*ndst)*blocksize + k;
  //    m += delta[index];
  //    v += delta[index] * (x[index] - means[i]);
  //  }
  //means_delta[i] = m * (-1.0f / sqrt(fmax(vars[i], sEPSILON)));
  //vars_delta[i]  = v * -0.5f * pow(fmax(vars[i], sEPSILON), -1.5f);
  ////means_delta[i] = m * (-1.0f / sqrt(vars[i]));
  ////vars_delta[i]  = v * -0.5f / (vars[i]*sqrt(vars[i]));

}

__kernel void norm_delta(__global nfloat* x, const n_int offset, __global nfloat* means, __global nfloat* vars, __global nfloat* means_delta, __global nfloat* vars_delta, __global nfloat* delta){
  const n_int j = get_global_id(2);
  const n_int i = get_global_id(0);
  const n_int k = get_global_id(1);
  const n_int groups    = get_global_size(2);
  const n_int N         = get_global_size(0);
  const n_int blocksize = get_global_size(1);

  const n_int batchsize = blocksize * groups;
  const n_int index = (i + j*N) * blocksize +k;
  delta += offset;
  x     += offset;
  delta[index] =
    delta[index] / (sqrt(max(vars[i], sEPSILON))) +
    //delta[index] / (sqrt(vars[i])) +
    (2.0f * vars_delta[i] * (x[index] - means[i]) + means_delta[i]) / batchsize;

}

__kernel void add_dots(const n_int groups, const n_int blocksize, __global nfloat* src1, __global nfloat* src2, const n_int srcOffset, __global nfloat* dst){

    const n_int filter = get_group_id(0);
    const n_int p = get_local_id(0);
    const n_int N = get_num_groups(0);

    local float part[BLOCK];
    src1 += srcOffset;
    src2 += srcOffset;

    n_int i,b;
    float sum = 0;
    #pragma unroll 8
    for(b = 0; b < groups; ++b){
        #pragma unroll 8
        for(i = 0; i < blocksize; i += BLOCK){
            int index = p + i + blocksize*(filter + N*b);
            sum += (p+i < blocksize) ? src1[index]*src2[index] : 0;
        }
    }
    part[p] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (p == 0) {
        #pragma unroll 8
        for(i = 0; i < BLOCK; ++i) dst[filter] += part[i];
    }

    //const n_int i = get_global_id(0);
    //const n_int ndst = get_global_size(0);
    //nfloat sum = 0;
    //src1 += srcOffset;
    //src2 += srcOffset;
    //// take a shortcut
    //if (blocksize==1){
    //  sum = dotv(groups, src1 + i, ndst, src2 + i, ndst);
    //  dst[i] += sum;
    //  return;
    //}
    //#pragma unroll
    //for (n_int b=0; b<groups; b++){
    //  const n_int idx = (i + b * ndst) * blocksize;
    //  sum += dotv(blocksize, src1 + idx, 1, src2 + idx, 1);
    //}
    //dst[i] += sum;
}

__kernel void forward_scale(const int reshape, __global nfloat* output,  const n_int outputOffset, __global nfloat* scale, const n_int scaleOffset, const n_int incb)
{
  n_int N, blockSize, i, k, j;
  switch (reshape) {
    case 0:
      N = get_global_size(0);
      blockSize = get_global_size(1);
      i = get_global_id(0);
      k = get_global_id(2);
      j = get_global_id(1);
      break;
    case 1:
      N = get_global_size(1);
      blockSize = get_global_size(0);
      i = get_global_id(1);
      k = get_global_id(2);
      j = get_global_id(0);
      break;
  }
  //if (i==0 && k==0){
    //printf("        N = %ld\n", N);
    //printf("blockSize = %ld\n", blockSize);
  //}
  //for (i = 0; i<N; i++)
  //  for (k = 0; k<batch; k++){
  output += (k*N + i)*blockSize + outputOffset;
  nfloat bb = scale[i * incb];
  //#pragma unroll 8
  //    for (n_int j=0; j<blockSize; j++)
        output[j] *= bb;
  //}
}

__kernel void forward_scale_add(const int reshape, __global nfloat* a,  const n_int aOffset, __global nfloat* s, __global nfloat* b, const n_int bOffset, const n_int incb)
{
  n_int N, blockSize, i, k, j;
  switch (reshape) {
    case 0:
      N = get_global_size(0);
      blockSize = get_global_size(1);
      i = get_global_id(0);
      k = get_global_id(2);
      j = get_global_id(1);
      break;
    case 1:
      N = get_global_size(1);
      blockSize = get_global_size(0);
      i = get_global_id(1);
      k = get_global_id(2);
      j = get_global_id(0);
      break;
  }
  //if (i==0 && k==0){
    //printf("        N = %ld\n", N);
    //printf("blockSize = %ld\n", blockSize);
  //}
  //for (i = 0; i<N; i++)
  //  for (k = 0; k<batch; k++){
  a += (k*N + i)*blockSize + aOffset;
  nfloat ss = s[i * incb];
  nfloat bb = b[i * incb];
  //#pragma unroll 8
  //    for (n_int j=0; j<blockSize; j++)

        a[j] *= ss;
        a[j] += bb;
  //}
}

#define RANDOM_MAX 0x10000000u

// https://en.wikipedia.org/wiki/Xorshift
unsigned n_int rand_xorshift(const unsigned n_int seed){
  //un_int res = ((seed + get_global_linear_id()) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  //return (res >> 16) % RANDOM_MAX;
  //      uint x = seed + get_global_linear_id();
    unsigned n_int x = seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x *= 0x2545F4914F6CDD1DUL;
    //seed = x;
    return x%RANDOM_MAX;
}


__kernel void forward_dropout(const unsigned int seed, const nfloat probability, const nfloat scale, __global const nfloat* src, __global nfloat* rnd, __global nfloat* dst)
{
  unsigned n_int i  = get_global_id(0);
  //mt_state state;
  //init_mt(&state, seed + i);
  //rnd[i]  = rand_mt(&state);
  rnd[i]  = rand_xorshift(seed+i);
  rnd[i] /= RANDOM_MAX;
  dst[i]  = rnd[i] < probability? 0: src[i]*scale;

}

kernel void backward_dropout(const nfloat probability, const nfloat scale, global const nfloat* src, global const nfloat* rnd, global nfloat* dst)
{
  n_int i = get_global_id(0);
  dst[i] = rnd[i] < probability? 0: src[i]*scale;

}

kernel void cost_l2(global const nfloat* pred, global const nfloat* truth, global nfloat* delta, global nfloat* error){
  n_int i = get_global_id(0);
  nfloat r = truth[i] - pred[i];
  delta[i] = r ;
  error[i] = r*r;

}

__kernel void mulv( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* dst, const n_int dstOffset, const n_int incc){

   const n_int i = get_global_id(0);

   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset];
}

__kernel void fmav( __global nfloat* src1, const n_int src1Offset, const n_int inca, __global nfloat* src2, const n_int src2Offset, const n_int incb, __global nfloat* src3, const n_int src3Offset, const n_int incc, __global nfloat* dst, const n_int dstOffset, const n_int incd){

   const n_int i = get_global_id(0);

   dst[i*incd + dstOffset] = mad(src1[i*inca + src1Offset], src2[i*incb + src2Offset], src3[i*incc + src3Offset]);
   //dst[i*incd + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset] + src3[i*incc + src3Offset];
}

__kernel void power(__global nfloat* base, const n_int srcOffset, const n_int srcStride, const nfloat expo, __global nfloat* dst, const n_int dstOffset, const n_int dstStride){

   const n_int i = get_global_id(0);
   dst[i*dstStride + dstOffset] = pow(base[i*srcStride + srcOffset], expo);
}

__kernel void clip(const nfloat alpha, __global const nfloat* src, __global nfloat* dst, const n_int stride, const n_int offset ){
   const n_int i = get_global_id(0);
   src += i*stride + offset;
   dst += i*stride + offset;
   *dst = max(min(*src, alpha), -alpha);
}

__kernel void inverse_sqrt(__global const nfloat* src, __global nfloat* dst, const n_int stride, const n_int offset){
   const n_int i = get_global_id(0);
   src += i*stride + offset;
   dst += i*stride + offset;
   *dst= 1/sqrt(max(*src, sEPSILON));
}

//__kernel void halftest(__global half* a, __global half* b, __global half* c){
//   const n_int i = get_global_id(0);
//  c[i] = a[i]+b[i];
//}
