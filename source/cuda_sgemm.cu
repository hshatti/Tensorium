#define nfloat  float
#define sEPSILON 0.000001f
//#include <stdio.h>
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

#ifndef BLOCK
#define BLOCK 512
#endif

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


__device__ nfloat sqr(const nfloat x){
  return x*x;
}

__device__ nfloat sumv(const long N,  const nfloat* v, const long stride){
  nfloat sum=0;
  #pragma unroll 8
  for (long i=0;i<N; i++)
    sum += v[i*stride];
  return sum;
}

__device__ nfloat maxv(const long N,  const nfloat* v, const long stride){
  nfloat m=v[0];
  #pragma unroll 8
  for (long i=1;i<N; i++)
    m = max(m, v[i*stride]);
  return m;
}

__device__ nfloat minv(const long N,  const nfloat* v, const long stride){
  nfloat m=v[0];
  #pragma unroll 8
  for (long i=1;i<N; i++)
    m = min(m, v[i*stride]);
  return m;
}

__device__ nfloat rssv(const long N, const nfloat mean,  const nfloat* src, const long stride){
  nfloat sum=0;
  #pragma unroll 8
  for (long i=0;i<N; i++){
    const nfloat v = src[i*stride] - mean;
    sum += v*v;
  }
  return sum;
}

//#define VW 8
//nfloat sumv_simd(const long N,  nfloat* v){
//  float8 sum4 = 0;
//  nfloat sum = 0;
//  long n = N / VW;
//
//  #pragma unroll 8
//  for (long i=0;i<n; i++)
//    sum4 += vload8(i, v);
//  v += n*VW;
//
//  #pragma unroll 8
//  for (long i=0; i<N%VW;i++)
//    sum4[i] += v[i];
//  return sum4[0] + sum4[1] + sum4[2] + sum4[3] + sum4[4] + sum4[5] + sum4[6] + sum4[7] ;
//}

__device__ nfloat dotv(const long N,  nfloat* a, const long inca,   nfloat* b, const long incb){

  nfloat d = 0;
  #pragma unroll 8
  for (long i=0; i<N;i++)
    d += a[i*inca]*b[i*incb];
  return d;
}

//nfloat dotv_simd(const long N,  nfloat* a,   nfloat* b){
//
//  float8 d = 0;
//  long n = N / VW;
//  #pragma unroll 8
//  for (long i=0; i<n;i++)
//    d += vload8(i, a) * vload8(i, b);
//  a += n*VW;
//  b += n*VW;
//
//  #pragma unroll 8
//  for (long i=0; i<N%VW;i++)
//    d.x += a[i]*b[i];
//
//  return d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] ;
//}

#define WIDTH 4
              // naive GEMM with unrolling for now
extern "C" __global__ void sgemm1_nn(const long M, const long N, const long K, const nfloat ALPHA ,
                       nfloat* A, const long aOffset, const long lda,
                       nfloat* B, const long bOffset, const long ldb,
                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {

    const long globalRow = blockDim.x * blockIdx.x + threadIdx.x; // Row ID of C (0..M)
    const long globalCol = blockDim.y * blockIdx.y + threadIdx.y; // Col ID of C (0..N)
    if (globalRow >= M || globalCol >= N) return;

    A += globalRow*lda + aOffset ;
    B += globalCol + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;

    //nfloat acc =0;
    //
    //#pragma unroll 8
    //for (long k=0; k<K; k++)
    //  acc += A[k]*B[k*ldb];
    //*C += acc * ALPHA;
    *C += dotv(K, A, 1, B, ldb) * ALPHA;
}

extern "C" __global__ void sgemm2_nn(const long M, const long N, const long K, const nfloat ALPHA ,
                       nfloat* A, const long aOffset, const long lda,
                       nfloat* B, const long bOffset, const long ldb,
                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {

    const long globalRow = blockDim.y * blockIdx.y + threadIdx.y; // Row ID of C (0..M)
    const long globalCol = blockDim.x * blockIdx.x + threadIdx.x; // Col ID of C (0..N)
    if (globalRow >= M || globalCol >= N) return;

    A += globalRow*lda + aOffset ;
    B += globalCol + bOffset;
    C += globalRow*ldc + globalCol + cOffset;

    *C *= BETA;
    //nfloat acc =0;

    //#pragma unroll 8
    //for (long k=0; k<K; k++)
    //    acc += A[k]*B[k*ldb];
    //*C += acc * ALPHA;
    *C += dotv(K, A, 1, B, ldb) * ALPHA;
}

extern "C" __global__ void sgemm1_nt(const long M, const long N, const long K, const nfloat ALPHA ,
                       nfloat* A, const long aOffset, const long lda,
                       nfloat* B, const long bOffset, const long ldb,
                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {

    const long globalRow = blockDim.x * blockIdx.x + threadIdx.x;    // M
    const long globalCol = blockDim.y * blockIdx.y + threadIdx.y;    // N
    if (globalRow >= M || globalCol >= N) return;

    A += globalRow*lda + aOffset;
    B += globalCol*ldb + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;

    //nfloat acc =0;

    //#pragma unroll 8
    //for (long k=0; k<K; k++)
    //    acc += A[k] * B[k];
    //*C += acc * ALPHA;
    //*C += dotv(K, A, 1, B, 1) * ALPHA;
    *C += dotv(K, A, 1, B, 1) * ALPHA;
            //}
}

extern "C" __global__ void sgemm1_tn(const long M, const long N, const long K, const nfloat ALPHA ,
                       nfloat* A, const long aOffset, const long lda,
                       nfloat* B, const long bOffset, const long ldb,
                      const nfloat BETA,  nfloat* C, const long cOffset, const long ldc) {

    const long row = blockDim.x * blockIdx.x + threadIdx.x; // Row ID of C (0..M)
    const long col = blockDim.y * blockIdx.y + threadIdx.y; // Col ID of C (0..N)
    if (row >= M || col >= N) return;
    A += row           +  aOffset;
    B += col           +  bOffset;
    C += row*ldc + col +  cOffset;

    *C *= BETA;
    //nfloat acc = 0;

    //#pragma unroll 8
    //for (long k=0; k<K; k++)
    //    acc += A[k * lda] * B[k * ldb] ;
    //*C += acc * ALPHA ;
    *C += dotv(K, A, lda, B, ldb) * ALPHA ;
}

extern "C" __global__ void forward_bias(const long N, const long biasSize, const long blockSize, nfloat* output,  const long outputOffset, nfloat* bias, const long biasOffset, const long incb)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int f = (i / blockSize) % biasSize;
    output[i + outputOffset] += bias[f];
}

__device__ nfloat stair_activate(const nfloat x)
{
  long n = floor(x);
  if (n % 2 == 0) return floor(x/ 2);
  return (x - n) + floor(x/2);
}

__device__ nfloat hardtan_activate(const nfloat x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return (x);
}

__device__ nfloat linear_activate(const nfloat x)
{
  return x;
}

__device__ nfloat logistic_activate(const nfloat x)
{
  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))
  return 1/(1 + exp(-x));
}

__device__ nfloat loggy_activate(const nfloat x)
{
  //result := 2/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp))) - 1;
  return 2/(1 + exp(-x)) - 1;
}

__device__ nfloat relu_activate(const nfloat x)
{
  //return x*long(x>0);
  if (x<0) return 0;
  return x;
}

__device__ nfloat relu6_activate(const nfloat x)
{
  //min_val_cmp(max_val_cmp(x, 0), 6)
  //result := EnsureRange(x,0,6);
  return  x*(x>0) * (x<=6);
}

__device__ nfloat elu_activate(const nfloat x)
{
  return (x >= 0)*x + (x < 0)*(exp(x)-1);
}

__device__ nfloat selu_activate(const nfloat x)
{
  return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);
}

__device__ nfloat gelu_activate(const nfloat x)
{
  return 0.5f*x*(1 + tanh(0.797885f*x + 0.035677f*pow(x, 3)));
}

__device__ nfloat relie_activate(const nfloat x)
{
  if (x>0) return x;
  else return 0.01f*x;
}

__device__ nfloat ramp_activate(const nfloat x)
{
  return  x*(x>0)+0.1f*x;
}

__device__ nfloat leaky_activate(const nfloat x)
{
  if (x>0) return  x;
  else return  0.1f*x;
}

__device__ nfloat tanh_activate(const nfloat x)
{
  //const nfloat px = exp(x);
  //const nfloat nx = exp(-x);
  //return (px - nx)/(px + nx);
  //return 2 / (1 + exp(ensureRange(-2 * x, minSingleExp, maxSingleExp))) - 1
  return 2/ (1+exp(-2*x)) - 1 ;
//  return  (exp(2*x)-1)/(exp(2*x)+1);
}

__device__ nfloat softplus_activate(const nfloat x, const nfloat threshold)
{
    if (x > threshold)
      return (x);                // too large
    else if (x < -threshold)
      return (exp(x));    // too small
    //return (log(exp(x) + 1));
    return log1p(exp(x));
}

__device__ nfloat plse_activate(const nfloat x)
{
    if (x < -4 ) return( 0.01f * (x + 4));
    if (x > 4 ) return( 0.01f * (x - 4) + 1);
    return  0.125f*x + 0.5f;
}

__device__ nfloat lhtan_activate(const nfloat x)
{
    if(x < 0) return (0.001f*x);
    if(x > 1) return (0.001f*(x-1) + 1);
    return  x;
}

__device__ nfloat silu_activate(const nfloat x)
{
    return x * logistic_activate(x) ;
}

#define  MISH_THRESHOLD 20.0f
__device__ nfloat mish_activate(const nfloat x)
{
    return x*tanh_activate(softplus_activate(x, MISH_THRESHOLD));
}
//void softmax_activate(const N:SizeInt; const x: PSingle);
//{
//  long i;
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


extern "C" __global__ void activate_array(const long N, nfloat* x, long const offset, const ActivationType a)
{
      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      //int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
      if (i>=N) return;
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
            //if (i==0) printf("[Activation] %d: not Implemented\n", (int)a);
	    ;
      }
   //printf("%ld, ", i);

}

extern "C" __global__ void array_avtivate_swish(const long N, nfloat* x, long const offset,  nfloat* output,  nfloat* output2)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=N) return;
    x += offset;
    output += offset;
    output2 += offset;
    nfloat x_val       = x[i];
    nfloat sigmoid     = logistic_activate(x_val);
    output[i]         = sigmoid;
    output2[i]        = x_val * sigmoid;
}




__device__ nfloat lhtan_gradient(const nfloat x)
{
    if ((x > 0) &&  (x < 1))
      return 1;
    return 0.001f;
}


__device__ nfloat hardtan_gradient(const nfloat x)
{
    if ((x > -1) && (x < 1))
      return 1;
    return 0;
}

__device__ nfloat linear_gradient(const nfloat x)
{
    return 1;
}

__device__ nfloat logistic_gradient(const nfloat x)
{
    return (1-x)*x;
}

__device__ nfloat loggy_gradient(const nfloat x)
{
    nfloat y = (x+1.0f)/2.0f;
    return 2.0f*(1.0f-y)*y;
}

__device__ nfloat stair_gradient(const nfloat x)
{
    if (floor(x) == x) return( 0);
    return 1;
}

__device__ nfloat relu_gradient(const nfloat x)
{
    return (x>0?1:0);
}

__device__ nfloat relu6_gradient(const nfloat x)
{
    return ((x>0) && (x<6)?1:0);
}

__device__ nfloat elu_gradient(const nfloat x)
{
    return (x >= 0?1:0) + (x < 0?1:0)*(x + 1);
}

__device__ nfloat selu_gradient(const nfloat x)
{
    return (x >= 0?1:0)*1.0507f + (x < 0?1:0)*(x + 1.0507f*1.6732f);
}

__device__ nfloat relie_gradient(const nfloat x)
{
    if (x>0) return 1;
    else return 0.01f;
}

__device__ nfloat ramp_gradient(const nfloat x)
{
    return (x>0?1:0) + 0.1f;
}

__device__ nfloat leaky_gradient(const nfloat x)
{
    if (x>0) return 1;
    else return 0.1f;
}

__device__ nfloat tanh_gradient(const nfloat x)
{
    return 1.0f-x*x;
}

__device__ nfloat sech(const nfloat x)
{
    return 2.0f / (exp(x) + exp(-x));
}

__device__ nfloat gelu_gradient(const nfloat x)
{
    nfloat x3 = x*x*x;
    nfloat sec = sech(0.0356774f*x3 + 0.797885f*x);
    return 0.5f*tanh(0.0356774f*x3 + 0.797885f*x) + (0.0535161f*x3 + 0.398942f*x) * sec*sec + 0.5f ;
}

__device__ nfloat plse_gradient(const nfloat x)
{

  if ((x < 0) || (x > 1))
    return  0.01f;
  else
    return 0.125f;
}

extern "C" __global__ void gradient_array(const long N, nfloat* x, long const offset, const ActivationType a,  nfloat* delta)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=N) return;
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
            //if (i==0) printf("[Gradient] : not Implemented %d\n", (int)a);
	    ;
    }

}

//#define BLOCK 0x20
extern "C" __global__ void backward_bias(
       const long N,
       nfloat* dst,
       const long blockSize,
       nfloat* src,
       const long srcOffset,
       const long batch)
{


    const long filter = blockIdx.x;
    const long p = threadIdx.x;
    //if(blockDim.x * filter + p>=N) return;

    long i,b;
    __shared__ float part[BLOCK];

    src += srcOffset;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < blockSize; i += BLOCK){
            int index = p + i + blockSize*(filter + N*b);
            sum += (p+i < blockSize) ? src[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i)
          dst[filter] += part[i];
    }

    //const long i = blockDim.x * blockIdx.x + threadIdx.x;//if (i==0) printf("long %ull\n", sizeof(long));
    //if (i>=N) return;
    //nfloat sum = 0;
    ////const long N = blockDim.x * gridDim.x;
    ////for (long i=0 ; i<N ;i++) {
    //  src += i * blockSize + srcOffset;
    //   //take a shortcut
    //  if(blockSize==1){
    //    dst[i] += sumv(batch, src, N);
    //    //#pragma unroll 8
    //    //for (long j=0; j<batch; j++)
    //    //  sum += src[j*N];
    //    //dst[i] +=sum;
    //    return;
    //  }
    //
    //  const long incbias = N*blockSize;
    //  #pragma unroll 8
    //  for (long j=0; j<batch; j++){
    //    sum += sumv(blockSize, src, 1);
    //    src += incbias;
    //  }
    //  dst[i] +=sum;
}

extern "C" __global__ void addv(const long N, nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* dst, const long dstOffset, const long incc){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i>=N) return;
   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] + src2[i*incb + src2Offset];
}

extern "C" __global__ void subv(const long N, nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* dst, const long dstOffset, const long incc){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i>=N) return;
   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] - src2[i*incb + src2Offset];
}

extern "C" __global__ void axpy(const long N, const nfloat a,  nfloat* x, const long xOffset, const long incx,  nfloat* y, const long yOffset, const long incy){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= N) return;
   x += xOffset;
   y += yOffset;
   y[i*incy] += a*x[i*incx];

}

extern "C" __global__ void scale(const long N, const nfloat a,  nfloat* x, const long incx){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= N) return;
   x[i*incx] *= a;

}

extern "C" __global__ void crossEntropyLogistics(
           const long N,
           const nfloat* pred,
           const nfloat* truth,
           nfloat* delta,
           nfloat* error){

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  nfloat t = truth[i];
  nfloat p = pred[i];
  error[i] = -t*log(max(p, sEPSILON)) - (1-t) * log(max(1 - p, sEPSILON));
  //error[i] = -t*log(p) - (1-t) * log(1 - p);
  delta[i] = t - p;
   //printf("%ld, ", i);

}

extern "C" __global__ void fill(const long N, nfloat* x, const long offset, const nfloat val, const long stride){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= N) return;
   //x += offset;
   x[i*stride + offset] = val;
}

// naive copy for now
extern "C" __global__ void copy(const long N,
     nfloat* src, const long srcOffset, const long srcInc
  ,  nfloat* dst, const long dstOffset, const long dstInc){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= N) return;
   //src += srcOffset; dst += dstOffset;
   //if (srcInc==1 && dstInc==1){
   //  dst[i+dstOffset] = src[i+srcOffset];
   //  return;
   //}
   dst[i*dstInc + dstOffset] = src[i*srcInc + srcOffset];
}

#define FLT_MAX 3.402823466e+38F

extern "C" __global__ void forward_maxpool(
     const long N
     , const long outH
     , const long outW
     , nfloat* input
     , const long c
     , const long h
     , const long w
     , const long stride_x
     , const long stride_y
     , const long padding
     , const long kernelSize
     , size_t* indexes
     , nfloat* output
     ){

  const long w_offset = -padding / 2;
  const long h_offset = -padding / 2;

  //const long outC = blockDim.x * gridDim.x;
  //const long outH = blockDim.y * gridDim.y;
  //const long outW = blockDim.z * gridDim.z;
  long k = blockDim.x * blockIdx.x + threadIdx.x;
  long y = blockDim.y * blockIdx.y + threadIdx.y;
  long x = blockDim.z * blockIdx.z + threadIdx.z;
  if (k>=N || y>=outH || x>=outW) return;

  long out_index = x + outW*(y + outH*k) ;//+ outW*outH*outC*b;
  nfloat max = -FLT_MAX;
  long max_i = -1;
  #pragma unroll 8
  for (long n=0; n<kernelSize; n++)
      #pragma unroll 8
      for (long m=0; m<kernelSize; m++){
          long cur_h = h_offset+y * stride_y+n;
          long cur_w = w_offset+x * stride_x+m;
          long index = cur_w + w*(cur_h + h*k) ;//+ w*h*outC*b;
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

extern "C" __global__ void backward_maxpool(const long M, const long N,  nfloat* output,  const size_t* indexes,  const nfloat* delta){
        const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        //const long id = i*blockDim.y * gridDim.y + j;
        if (i>=M || j>=N) return;

        const long id = i*N + j;

        const long index = indexes[id];
        output[index] += delta[id];
}


__device__ void softmax(const long n,  nfloat* input, const long stride, const nfloat temp,  nfloat* output){

  nfloat largest = maxv(n, input, stride);
  nfloat sum = 0;
  #pragma unroll 8
  for (long i=0;i<n;i++) {
      nfloat e = expf((input[i*stride] - largest)/temp);
      sum += e;
      output[i*stride] = e;
  }

  #pragma unroll 8
  for (long i=0; i<n; i++)
      output[i*stride]/=sum;
}

extern "C" __global__ void softmaxBatch(const long batch, const long groups, nfloat* input, const long iOffset, const long n
  , const long batch_size, const long group_size, const long stride
  , const nfloat temp,  nfloat* output, const long oOffset){

  const long b = blockDim.x * blockIdx.x + threadIdx.x;
  const long g = blockDim.y * blockIdx.y + threadIdx.y;
  if (b>=batch || g>=groups) return;
  softmax(n
  , input + iOffset + b*batch_size + g*group_size
  , stride
  , temp
  , output + oOffset + b*batch_size + g*group_size);

}

__device__ void move(const nfloat* src,  nfloat* dst , const long count){
  #pragma unroll 8
  for (long i=0 ; i<count; i++) dst[i] = src[i];
}

extern "C" __global__ void crossEntropySoftmax(
  const long N,
  const nfloat* pred,
  const nfloat* truth,
  nfloat* delta,
  nfloat* error){

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=N) return;
  nfloat t = truth[i];
  nfloat p = pred[i];
  if (t!=0)
      error[i] = -log(max(p, sEPSILON));
      //error[i] = -log(p);
  else
      error[i] = 0;
  delta[i] = t - p;

}

extern "C" __global__ void im2col(const long aHeight, const long aWidth
  , const long kernelHeight, const long kernelWidth, const long padHeight, const long padWidth
  , const long strideY, const long strideX, const long dilationY, const long dilationX
  ,  nfloat* im , const long imOffset
  ,  nfloat* col, const long colOffset, const long batch){

  long aChannels = blockDim.x * gridDim.x;
  long chan = blockDim.x * blockIdx.x + threadIdx.x;
  long k = blockDim.y * blockIdx.y + threadIdx.y;
  const long kernelSize = kernelHeight*kernelWidth;

  const long outWidth = (aWidth + 2 * padWidth - (dilationX * (kernelWidth - 1) + 1)) / strideX + 1;
  const long outHeight = (aHeight + 2 * padHeight - (dilationY * (kernelHeight - 1) + 1)) / strideY + 1;
  const long outSize = outWidth * outHeight;
  const long inSize = aWidth * aHeight;
  const long sizeX = outWidth - 2 * padWidth;

    //for (long k=0 ; k<kernelWidth*kernelHeight; k++)
    {
      long kernelRow = k / kernelWidth;
      long kernelCol = k % kernelWidth;
      //const long kernelRow = blockDim.y * blockIdx.y + threadIdx.y;
      //const long kernelCol = blockDim.z * blockIdx.z + threadIdx.z;

      #pragma unroll 8
      for (long b=0 ; b<batch; b++)
      {
        long i = (b*aChannels + chan)*inSize + aWidth*(kernelRow*dilationY - padHeight) + kernelCol*dilationX - padWidth;
         nfloat* im1 = imOffset + im + i;
         nfloat* col1 = colOffset + col + padWidth * outWidth + outSize*kernelSize*(chan + b*aChannels) + outSize*(kernelRow * kernelWidth + kernelCol) ;
        #pragma unroll 8
        for (long outRow=padHeight ; outRow<outHeight - padHeight ; outRow++)
        {
          long j = outRow * aWidth * strideY + padWidth * strideX ;
          if (strideX == 1)
            move(im1 + j, col1 + padWidth, sizeX);
          else
            #pragma unroll 8
            for (long outCol=padWidth ;  outCol<outWidth - padWidth; outCol++)
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

__device__ void fills( nfloat* dst, const long N, const nfloat val){
  #pragma unroll 8
  for (long i=0 ; i<N; i++)
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

__device__ long grid_ceil(const long x, const long step) {
  return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;
}

__device__ void xim2col(const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const bool kernel_flip,
                         const  nfloat* __restrict__ im_buffer, const long im_offset,
                          nfloat* col_buffer, const long col_offset) {

  // Thread IDs
  const long w_id = blockDim.x * blockIdx.x + threadIdx.x; // image width, max 'output_w'
  const long h_id = ((long)blockDim.y * blockIdx.y + threadIdx.y) % output_h; // image height, max 'output_h'
  const long c_id = ((long)blockDim.y * blockIdx.y + threadIdx.y) / output_h; // input channels
  if (h_id < output_h && w_id < output_w && c_id < channels) {
    #pragma unroll 8
    for (long kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height
      #pragma unroll 8
      for (long kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width

        // Retrieves the input value
        const long h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
        const long w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;
        nfloat val;
        if (h_index >= 0 && h_index < input_h &&
            w_index >= 0 && w_index < input_w) {
          const long input_index = w_index + input_w * (h_index + input_h * c_id);
          val = im_buffer[input_index + im_offset];
        }
        else {
          val = 0;
        }

        // Sets the output value
        const long kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const long patch_index = w_id + output_w * h_id;
        const long output_index = patch_index + kernel_index * output_w * output_h +
                                  c_id * output_w * output_h * kernel_h * kernel_w;
        col_buffer[output_index + col_offset] = val;
      }
    }
  }
}

// =================================================================================================

// Kernel flip version of the Xim2col kernel (for convolution)
//#if RELAX_WORKGROUP_SIZE == 1
  extern "C" __global__
//#else
//  extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xim2colKernelFlip(const long M, const long N,
                       const long input_h, const long input_w, const long channels,
                       const long output_h, const long output_w,
                       const long kernel_h, const long kernel_w,
                       const long pad_h, const long pad_w,
                       const long stride_h, const long stride_w,
                       const long dilation_h, const long dilation_w,
                       const  nfloat* __restrict__ im_buffer, const long im_offset,
                        nfloat* col_buffer, const long col_offset) {
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i>=M || j>=N) return;

  const bool kernel_flip = true;
  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          kernel_flip,
          im_buffer, im_offset, col_buffer, col_offset);
}

// Normal version of the Xim2col kernel (for cross-correlation)
//#if RELAX_WORKGROUP_SIZE == 1
  extern "C" __global__
//#else
//  extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xim2colKernelNormal(const long M, const long N,
                         const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const  nfloat* __restrict__ im_buffer, const long im_offset,
                          nfloat* col_buffer, const long col_offset) {
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i>=M || j>=N) return;

  const bool kernel_flip = false;
  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          kernel_flip,
          im_buffer, im_offset, col_buffer, col_offset);
}

__device__ void xcol2im(const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const long stride_bez_h, const long stride_bez_w,
                         const long dilation_bez_h, const long dilation_bez_w,
                         const long gcd_h, const long gcd_w,
                         const bool kernel_flip,
                         const  nfloat* __restrict__ col_buffer, const long col_offset,
                          nfloat* im_buffer, const long im_offset) {

  const long input_h_scaled = (input_h - 1) / gcd_h + 1;

  // Thread IDs
  const long gcd_scale_w = blockDim.x * blockIdx.x + threadIdx.x + (pad_w - 1) / gcd_w + 1;
  const long gcd_scale_h = ((long) blockDim.y * blockIdx.y + threadIdx.y) % input_h_scaled + (pad_h - 1) / gcd_h + 1;
  const long c_id = ((long) blockDim.y * blockIdx.y + threadIdx.y) / input_h_scaled;

  const long w_index = gcd_scale_w * gcd_w - pad_w;
  const long h_index = gcd_scale_h * gcd_h - pad_h;
  const long th_step = stride_h * dilation_h / gcd_h;
  const long th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,
                                     (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),
                                 th_step);
  const long th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,
                         (dilation_bez_h * gcd_scale_h + 1) * dilation_h);
  const long tw_step = stride_w * dilation_w / gcd_w;
  const long tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,
                                     (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),
                                 tw_step);
  const long tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,
                         (dilation_bez_w * gcd_scale_w + 1) * dilation_w);
  if (w_index < input_w && c_id < channels) {
    nfloat val = 0;
    #pragma unroll 8
    for (long th = th_begin; th < th_end; th += th_step) {
      #pragma unroll 8
      for (long tw = tw_begin; tw < tw_end; tw += tw_step) {
        const long kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;
        const long kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;
        const long h_id = th / stride_h + stride_bez_h * gcd_scale_h;
        const long w_id = tw / stride_w + stride_bez_w * gcd_scale_w;
        const long kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const long patch_index = w_id + output_w * h_id;
        const long output_index = patch_index + kernel_index * output_w * output_h +
                                 c_id * output_w * output_h * kernel_h * kernel_w;
        val += col_buffer[output_index + col_offset];
      }
    }

    // Accumulates the resulting value with the existing im-buffer (+= val)
    const long input_index = w_index + input_w * (h_index + input_h * c_id);
    nfloat im_buffer_value = im_buffer[input_index + im_offset];
    im_buffer[input_index + im_offset] = im_buffer_value + val;
  }
}

// =================================================================================================

// Kernel flip version of the Xcol2im kernel (for convolution)
//#if RELAX_WORKGROUP_SIZE == 1
  extern "C" __global__
//#else
  //extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xcol2imKernelFlip(const long M, const long N,
                       const long input_h, const long input_w, const long channels,
                       const long output_h, const long output_w,
                       const long kernel_h, const long kernel_w,
                       const long pad_h, const long pad_w,
                       const long stride_h, const long stride_w,
                       const long dilation_h, const long dilation_w,
                       const long stride_bez_h, const long stride_bez_w,
                       const long dilation_bez_h, const long dilation_bez_w,
                       const long gcd_h, const long gcd_w,
                       const  nfloat* __restrict__ col_buffer, const long col_offset,
                        nfloat* im_buffer, const long im_offset) {

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i>=M || j>=N) return;

  const bool kernel_flip = true;
  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}

// Normal version of the Xcol2im kernel (for cross-correlation)
//#if RELAX_WORKGROUP_SIZE == 1
  extern "C" __global__
//#else
  //extern "C" __global__ __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xcol2imKernelNormal(const long M, const long N,
                         const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const long stride_bez_h, const long stride_bez_w,
                         const long dilation_bez_h, const long dilation_bez_w,
                         const long gcd_h, const long gcd_w,
                         const  nfloat* __restrict__ col_buffer, const long col_offset,
                          nfloat* im_buffer, const long im_offset) {

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i>=M || j>=N) return;

  const bool kernel_flip = false;
  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}


extern "C" __global__ void upsample(const long N, const long M, const long K,
                                    nfloat* in, const long stride, const int isForward, const nfloat scale,  nfloat* out, const int zero){

   const long c = blockDim.x * blockIdx.x + threadIdx.x;
   if (c>=N) return;
   const long y = blockDim.y * blockIdx.y + threadIdx.y;
   if (y>=M) return;
   const long x = blockDim.z * blockIdx.z + threadIdx.z;
   if (x>=K) return;

   const long h = M/stride; //blockDim.y * gridDim.y/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??
   const long w = K/stride; //blockDim.z * gridDim.z/stride; // but why multiplying by stride (look at setWorkgroupSizes) and then dividing by it in the loop??

   const long in_index   = (c*h + (y / stride))*w + x / stride;
   const long out_index  = (c*h*stride + y)*stride*w + x;   // <-- why having to adjust by stride instead of remving it !!!
   if (isForward){
     out[out_index] = scale*in[in_index];
     return;
   }
   if (zero) in[in_index] = 0;
   in[in_index] += scale*out[out_index]; // ok seems like, it's because of trying to add adjust all input pixels in the stride a in here

}

extern "C" __global__ void fmavss(const long N, nfloat* src, const long offset, const nfloat scalar, const nfloat bias,  nfloat* dst){

  //const long w = blockDim.y * gridDim.y;
  const long y = blockDim.x * blockIdx.x + threadIdx.x;
  //const long x = blockDim.y * blockIdx.y + threadIdx.y;
  if (y>=N) return;

  //const long idx = y*w + x;
  src += offset;
  dst += offset;
  //dst[idx] = mad(src[idx], scalar, bias);
  //dst[idx] = src[idx]*scalar + bias;
  dst[y] = src[y]*scalar + bias;

}

extern "C" __global__ void means_vars(const long N, const long blocksize, const long groups,  nfloat* src, const long offset,  nfloat* means,  nfloat* vars){

    nfloat m = 0;
    nfloat v = 0;
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=N) return;

    //const long N = blockDim.x * gridDim.x;

    const long S = groups*blocksize;
    src += offset;
    // take a shortcut
    if(blocksize==1){
      m = sumv(groups, src+i, N);
      m /= S;
      v = rssv(groups, m, src+i, N);
      means[i] = m;
      vars[i]  = v / (S-1);
      return;
    }

    #pragma unroll
    for (long b=0; b<groups; b++){
        const long idx = (i + b*N)*blocksize;
        m += sumv(blocksize, src + idx, 1);
    }
    m /= S;
    #pragma unroll
    for (long b=0; b<groups; b++){
        const long idx = (i + b*N)*blocksize;
        v += rssv(blocksize, m, src + idx, 1);
    }
    means[i] = m;
    vars[i]  = v / (S-1);
}

extern "C" __global__ void normvv( nfloat* mean, const long mean_stride,  nfloat* variance, const long variance_stride,  nfloat* dst, const long dst_stride)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  dst[i * dst_stride] = (dst[i*dst_stride] - mean[i*mean_stride])/sqrt(max(variance[i*variance_stride], sEPSILON));
}

extern "C" __global__ void normvs( nfloat* src, const nfloat mean, const nfloat variance)
{
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  src[i] = (src[i] - mean)/sqrt(max(variance, sEPSILON));
}

extern "C" __global__ void normblkvv( const long M, const long N, const long K,
                                      nfloat* means, const long means_stride,  nfloat* vars, const long vars_stride,  nfloat* dst, const long offset)
{
  const long blocksize = N;                      //blockDim.y * gridDim.y;
  const long batchsize = M*blocksize;            //blockDim.x * gridDim.x*blocksize;
  const long i = blockDim.x * blockIdx.x + threadIdx.x; // means vars pos
  const long b = blockDim.z * blockIdx.z + threadIdx.z; // batch batch pos
  long j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i>=M || j>=N || b>=K) return;
  if (i==0 && j==0 && b==0)
    printf("[GPU] Normalize...");

  j += b*batchsize + i*blocksize + offset; // block pos
  const nfloat v = sqrt(max(vars[i*vars_stride], sEPSILON));
  const nfloat m = means[i*means_stride];
  //dst += b*batchsize + i*blocksize + offset;
  dst[j] = (dst[j] - m)/v;
}

extern "C" __global__ void means_vars_delta(
          const long N, const long groups, const long blocksize,
          nfloat* delta,  nfloat* x, const long offset,
          nfloat* means,  nfloat* vars,
          nfloat* means_delta,  nfloat* vars_delta){

  nfloat m = 0;
  nfloat v = 0;
  const long i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i>=N) return;

  //const long ndst = blockDim.x * gridDim.x;
  x     += offset;
  delta += offset;

  // take a shortcut
  if(blocksize==1){
    #pragma unroll
    for (long j=0 ;j<groups; j++){
      const long index = i+j*N;
      m += delta[index];
      v += delta[index] * (x[index] - means[i]);
    }
    means_delta[i] = m * (-1.0f / sqrt(max(vars[i], sEPSILON)));
    vars_delta[i]  = v * -0.5f * pow(max(vars[i], sEPSILON), -1.5f);
    return;
  }
  #pragma unroll
  for (long j=0 ;j<groups; j++)
    #pragma unroll
    for (long k=0; k<blocksize; k++){
      const long index = (i + j*N)*blocksize + k;
      m += delta[index];
      v += delta[index] * (x[index] - means[i]);
    }
  means_delta[i] = m * (-1.0f / sqrt(max(vars[i], sEPSILON)));
  vars_delta[i]  = v * -0.5f * pow(max(vars[i], sEPSILON), -1.5f);
  //means_delta[i] = m * (-1.0f / sqrt(vars[i]));
  //vars_delta[i]  = v * -0.5f / (vars[i]*sqrt(vars[i]));

}

extern "C" __global__ void norm_delta(const long N, const long blockSize, const long groups,
                                      nfloat* x, const long offset,  nfloat* means,  nfloat* vars,  nfloat* means_delta,  nfloat* vars_delta,  nfloat* delta){
  const long j = blockDim.z * blockIdx.z + threadIdx.z;
  if (j>=groups) return;
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=N) return;
  const long k = blockDim.y * blockIdx.y + threadIdx.y;
  if (k>=blockSize) return;

  //const long groups    = blockDim.z * gridDim.z;
  //const long N         = blockDim.x * gridDim.x;
  //const long blocksize = blockDim.y * gridDim.y;

  const long batchsize = blockSize * groups;
  const long index = (i + j*N) * blockSize +k;
  delta += offset;
  x     += offset;
  delta[index] =
    delta[index] / (sqrt(fmax(vars[i], sEPSILON))) +
    //delta[index] / (sqrt(vars[i])) +
    (2.0f * vars_delta[i] * (x[index] - means[i]) + means_delta[i]) / batchsize;

}

extern "C" __global__ void add_dots(
       const long N
       , const long groups
       , const long blockSize
       , nfloat* src1
       , nfloat* src2
       , const long srcOffset
       , nfloat* dst){

    const long filter = blockIdx.x;
    const long p = threadIdx.x;

    //if (blockDim.x*filter + p>=N) return;

    __shared__ float part[BLOCK];
    src1 += srcOffset;
    src2 += srcOffset;

    long i,b;
    float sum = 0;
    for(b = 0; b < groups; ++b){
        for(i = 0; i < blockSize; i += BLOCK){
            int index = p + i + blockSize*(filter + N*b);
            sum += (p+i < blockSize) ? src1[index]*src2[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) dst[filter] += part[i];
    }

    //const long i = blockDim.x * blockIdx.x + threadIdx.x;
    //if (i>=N) return;
    //
    ////const long ndst = blockDim.x * gridDim.x;
    //nfloat sum = 0;
    //src1 += srcOffset;
    //src2 += srcOffset;
    //// take a shortcut
    //if (blockSize==1){
    //  sum = dotv(groups, src1 + i, N, src2 + i, N);
    //  dst[i] += sum;
    //  return;
    //}
    //#pragma unroll
    //for (long b=0; b<groups; b++){
    //  const long idx = (i + b * N) * blockSize;
    //  sum += dotv(blockSize, src1 + idx, 1, src2 + idx, 1);
    //}
    //dst[i] += sum;
}

extern "C" __global__ void forward_scale(
       const long N,
       const long scaleSize,
       const long blockSize,
       nfloat* output,
       const long outputOffset,
       nfloat* scale,
       const long scaleOffset,
       const long incb)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int f = (i / blockSize) % scaleSize;
    output[i + outputOffset] *= scale[f];
}

extern "C" __global__ void forward_scale_add(
       const long N,
       const long scaleSize,
       const long blockSize,
       nfloat* output,
       const long outputOffset,
       nfloat* scale,
       nfloat* bias,
       const long sOffset,
       const long incb)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int f = (i / blockSize) % scaleSize;
    output[i + outputOffset] = output[i + outputOffset]*scale[f] + bias[f];
}


//  Pseudorandom number generators
// https://en.wikipedia.org/wiki/Mersenne_Twister

#define NR 624
#define MR 397
#define wR 32
#define RR 31
#define UMASK (0xffffffffUL << RR)
#define LMASK (0xffffffffUL >> (wR-RR))
#define AR 0x9908b0dfUL
#define UR 11
#define SR 7
#define TR 15
#define LR 18
#define BR 0x9d2c5680UL
#define CR 0xefc60000UL
#define FR 1812433253UL
//typedef unsigned long long uint64_t;
//typedef unsigned long uint32_t;

typedef struct
{
    unsigned int state_array[NR];         // the array for the state vector
    int state_index;                 // index into state vector array, 0 <= state_index <= NR-1   always
} mt_state;


__device__ void init_mt(mt_state* state, unsigned int seed)
{
    unsigned int* state_array = &(state->state_array[0]);

    state_array[0] = seed;                          // suggested initial seed = 19650218UL

    for (int i=1; i<NR; i++)
    {
        seed = FR * (seed ^ (seed >> (wR-2))) + i;    // Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
        state_array[i] = seed;
    }

    state->state_index = 0;
}


__device__ unsigned int rand_mt(mt_state* state)
{
    unsigned int* state_array = &(state->state_array[0]);

    int k = state->state_index;      // point to current state location
                                     // 0 <= state_index <= NR-1   always

//  int k = k - NR;                   // point to state NR iterations before
//  if (k < 0) k += NR;               // modulo NR circular indexing
                                     // the previous 2 lines actually do nothing
                                     //  for illustration only

    int j = k - (NR-1);               // point to state NR-1 iterations before
    if (j < 0) j += NR;               // modulo NR circular indexing

    unsigned int x = (state_array[k] & UMASK) | (state_array[j] & LMASK);

    unsigned int xA = x >> 1;
    if (x & 0x00000001UL) xA ^= AR;

    j = k - (NR-MR);                   // point to state NR-MR iterations before
    if (j < 0) j += NR;               // modulo NR circular indexing

    x = state_array[j] ^ xA;         // compute next value in the state
    state_array[k++] = x;            // update new state value

    if (k >= NR) k = 0;               // modulo NR circular indexing
    state->state_index = k;

    unsigned int y = x ^ (x >> UR);       // tempering
             y = y ^ ((y << SR) & BR);
             y = y ^ ((y << TR) & CR);
    unsigned int z = y ^ (y >> LR);

    return z;
}



#define RANDOM_MAX 0xffffffffull

// https://en.wikipedia.org/wiki/Xorshift
__device__ unsigned long long rand_xorshift(const unsigned long long seed){
  //ulong res = ((seed + get_global_linear_id()) * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  //return (res >> 16) % RANDOM_MAX;
  //      uint x = seed + get_global_linear_id();
    unsigned long long x = seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x *= 0x2545F4914F6CDD1DULL;
    //seed = x;
    return x%RANDOM_MAX;
}


extern "C" __global__ void forward_dropout(const long N, const unsigned int seed, const nfloat probability, const nfloat scale, const nfloat* src, nfloat* rnd, nfloat* dst)
{
  long i  = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=N) return;
  //mt_state state;
  //init_mt(&state, seed + i);
  //rnd[i]  = rand_mt(&state);
  rnd[i]  = rand_xorshift(seed+i);
  rnd[i] /= RANDOM_MAX;
  dst[i]  = rnd[i] < probability? 0: src[i]*scale;

}

extern "C" __global__ void backward_dropout(const long N, const nfloat probability, const nfloat scale, const nfloat* src, const nfloat* rnd, nfloat* dst)
{
  long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=N) return;
  dst[i] = rnd[i] < probability? 0: src[i]*scale;

}

extern "C" __global__ void cost_l2(const long N, const nfloat* pred, const nfloat* truth, nfloat* delta, nfloat* error){
  long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=N) return;
  nfloat r = truth[i] - pred[i];
  delta[i] = r ;
  error[i] = r*r;

}

extern "C" __global__ void mulv(const long N, nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* dst, const long dstOffset, const long incc){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i>=N) return;
   dst[i*incc + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset];
}

extern "C" __global__ void fmav(  nfloat* src1, const long src1Offset, const long inca,  nfloat* src2, const long src2Offset, const long incb,  nfloat* src3, const long src3Offset, const long incc,  nfloat* dst, const long dstOffset, const long incd){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;

   //dst[i*incd + dstOffset] = mad(src1[i*inca + src1Offset], src2[i*incb + src2Offset], src3[i*incc + src3Offset]);
   dst[i*incd + dstOffset] = src1[i*inca + src1Offset] * src2[i*incb + src2Offset] + src3[i*incc + src3Offset];
}

extern "C" __global__ void power(const long N, nfloat* base, const long srcOffset, const long srcStride, const nfloat expo,  nfloat* dst, const long dstOffset, const long dstStride){

   const long i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i>=N) return;
   dst[i*dstStride + dstOffset] = pow(base[i*srcStride + srcOffset], expo);
}
//extern "C" __global__ void halftest( half* a,  half* b, __global half* c){
//   const long i = blockDim.x * blockIdx.x + threadIdx.x;
//  c[i] = a[i]+b[i];
//}

extern "C" __global__ void means(const long N, const long blocksize, const long groups,  nfloat* src, const long offset,  nfloat* means){

    __shared__ float local[BLOCK];

    const long id = threadIdx.x;
    local[id] = 0;

    long filter = blockIdx.x;
    src += offset;
    long i, j;
    for(j = 0; j < groups; ++j){
        for(i = 0; i < blocksize; i += BLOCK){
            long index = j*blocksize*N + filter*blocksize + i + id;
            local[id] += (i+id < blocksize) ? src[index] : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        float mean_tmp = 0;
        for(i = 0; i < BLOCK; ++i){
            mean_tmp += local[i];
        }
        mean_tmp /= blocksize * groups;
        means[filter] = mean_tmp;
    }

}

extern "C" __global__ void vars(const long N, const long blocksize, const long groups,  nfloat* src, const long offset,  nfloat* means,  nfloat* vars){

    __shared__ float local[BLOCK];

    const long id = threadIdx.x;
    local[id] = 0;

    long filter = blockIdx.x;
    src += offset;
    long i, j;
    for(j = 0; j < groups; ++j){
        for(i = 0; i < blocksize; i += BLOCK){
            long index = j*blocksize*N + filter*blocksize + i + id;
            //local[id] += (i+id < blocksize) ? pow((src[index] - means[filter]), 2.0f) : 0; // pow will not work always when -use_fast_math compiler switch
            local[id] += (i+id < blocksize) ? sqr(src[index] - means[filter]) : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        float variance_tmp = 0;
        for(i = 0; i < BLOCK; ++i){
            variance_tmp += local[i];
        }
        variance_tmp /= (blocksize * groups);
        vars[filter] = variance_tmp;
    }
}

extern "C" __global__ void means_vars_delta_fast(
          const long N, const long groups, const long blocksize,
          nfloat* delta,  nfloat* x, const long offset,
          nfloat* means,  nfloat* vars,
          nfloat* means_delta,  nfloat* vars_delta){

    __shared__ float local[BLOCK];

    long id = threadIdx.x;
    local[id] = 0;

    long filter = blockIdx.x;

    x     += offset;
    delta += offset;

    long i, j;
    for(j = 0; j < groups; ++j){
        for(i = 0; i < blocksize; i += BLOCK){
            long index = j*blocksize*N + filter*blocksize + i + id;
            local[id] += (i+id < blocksize) ? delta[index] : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        means_delta[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            means_delta[filter] += local[i];
        }
        means_delta[filter] *= (-1.0f/sqrt(max(vars[filter] , sEPSILON)));
    }

    __syncthreads();

    local[id] = 0;

    for(j = 0; j < groups; ++j){
        for(i = 0; i < blocksize; i += BLOCK){
            long index = j*blocksize*N + filter*blocksize + i + id;

            local[id] += (i+id < blocksize) ? delta[index]*(x[index] - means[filter]) : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        vars_delta[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            vars_delta[filter] += local[i];
        }
        vars_delta[filter] *= -0.5f * pow(max(vars[filter] , sEPSILON), -1.5f);
    }
}


