unit nActivation;
{$ifdef fpc}
  {$ifdef CPUX86_64}
    {$asmmode intel}
  {$endif}
  {$mode Delphi}{$H+}
  {$modeSwitch nestedprocvars}
  {$modeSwitch typehelpers}

{$endif}
{$pointermath on}
interface

uses
  SysUtils, math, typinfo
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  {$if defined(USE_ONEDNN)}
  , dnnl_api
  {$endif}

  ;

type
  PActivationType = ^TActivationType;
  TActivationType = (
    acLINEAR, acLOGISTIC, acRELU, acRELU6, acRELIE, acRAMP, acTANH, acPLSE,
    acREVLEAKY, acLEAKY, acELU, acLOGGY, acSTAIR, acHARDTAN, acLHTAN, acSELU, acSOFTMAX,
    acGELU, acSWISH, acMISH, acHARD_MISH, acNORM_CHAN, acNORM_CHAN_SOFTMAX,
    acNORM_CHAN_SOFTMAX_MAXVAL
  );

  { TActivationTypeHelper }

  TActivationTypeHelper=record helper for TActivationType
    class function fromString(s:string):TActivationType; static;
    function toString():string;
    {$if defined(USE_ONEDNN)}
    function toDNNLAlgKind():dnnl_alg_kind_t;
    {$endif}
  end;

const
  acSIGMOID = acLOGISTIC;
  acSiLU    = acSWISH;

procedure activate_array(const N: NativeInt; const x: PSingle; const srcStride:NativeInt; output:PSingle; const dstStride:NativeInt=1; const a: TActivationType=acLiNEAR; output2:PSingle =nil);overload;
procedure activate_array(const N: NativeInt; x: PSingle; const a:TActivationType; output:PSingle=nil); overload;
procedure gradient_array(const N: NativeInt; const x: PSingle; const srcStride:NativeInt; delta:PSingle ; const dstStride:NativeInt; const a: TActivationType; output:PSingle = nil);overload;
procedure gradient_array(const N: NativeInt; const x: PSingle; delta:PSingle ; const a: TActivationType; output:PSingle = nil);overload;

//procedure activate_array_swish(const x: Psingle; const N: NativeInt; output_sigmoid, output: Psingle);
procedure activate_array_mish(const x: Psingle; const N: NativeInt; const activation_input, output: Psingle);
procedure activate_array_hard_mish(const x: Psingle; const N: NativeInt; const activation_input, output: PSingle);
procedure activate_array_normalize_channels(const x: Psingle; const N, batch, channels, wh_step: NativeInt; const output: Psingle);
procedure activate_array_normalize_channels_softmax(const x: Psingle; const N, batch, channels, wh_step: NativeInt; const output: PSingle; const use_max_val: boolean);

//procedure gradient_array_swish(const x: Psingle; const N: NativeInt; const sigmoid: Psingle; delta: Psingle);
procedure gradient_array_mish(const N: NativeInt; const activation_input: Psingle; delta: Psingle);
procedure gradient_array_hard_mish(const N: NativeInt; const activation_input, delta: Psingle);
procedure gradient_array_normalize_channels(const x: Psingle; const N, batch, channels, wh_step: NativeInt; const delta: Psingle);
procedure gradient_array_normalize_channels_softmax(const x: PSingle; const N, batch, channels, wh_step: NativeInt; const delta: Psingle);

implementation
uses nTensors, nBaseLayer;


{ TActivationTypeHelper }

class function TActivationTypeHelper.fromString(s: string): TActivationType;
begin
    s := LowerCase(s);
    if s = 'logistic' then  exit(acLOGISTIC);
    if s = 'swish' then exit(acSWISH);
    if s = 'mish' then exit(acMISH);
    if s = 'hard_mish' then exit(acHARD_MISH);
    if s = 'normalize_channels' then exit(acNORM_CHAN);
    if s = 'normalize_channels_softmax' then exit(acNORM_CHAN_SOFTMAX);
    if s = 'normalize_channels_softmax_maxval' then exit(acNORM_CHAN_SOFTMAX_MAXVAL);
    if s = 'loggy' then  exit(acLOGGY);
    if s = 'relu' then  exit(acRELU);
    if s = 'relu6' then  exit(acRELU6);
    if s = 'elu' then  exit(acELU);
    if s = 'selu' then  exit(acSELU);
    if s = 'gelu' then  exit(acGELU);
    if s = 'relie' then  exit(acRELIE);
    if s = 'plse' then  exit(acPLSE);
    if s = 'hardtan' then  exit(acHARDTAN);
    if s = 'lhtan' then  exit(acLHTAN);
    if s = 'linear' then  exit(acLINEAR);
    if s = 'ramp' then  exit(acRAMP);
    if s = 'revleaky' then  exit(acREVLEAKY);
    if s = 'leaky' then  exit(acLEAKY);
    if s = 'tanh' then  exit(acTANH);
    if s = 'stair' then  exit(acSTAIR);
    //writeln('Couldn''t find activation function ',s,' going with ReLU');
    exit(acRELU)
end;

function TActivationTypeHelper.toString(): string;
begin
  result := copy(GetEnumName(TypeInfo(TActivationType), ord(Self)), 3, 20)
end;

{$if defined(USE_ONEDNN)}
function TActivationTypeHelper.toDNNLAlgKind: dnnl_alg_kind_t;
begin
  case self of
    acLINEAR: result := dnnl_eltwise_linear;
    acRELU, acLEAKY, acRELIE:
      result := dnnl_eltwise_relu;
    acLOGISTIC:
      result := dnnl_eltwise_logistic;
    acTANH: result := dnnl_eltwise_tanh;
    acELU: result := dnnl_eltwise_elu;
    acGELU: result := dnnl_eltwise_gelu_tanh;
    acSWISH:
      result := dnnl_eltwise_swish;
    acMISH: result := dnnl_eltwise_mish;
    acSTAIR: result:=dnnl_eltwise_round
  else
    result := dnnl_alg_kind_undef;
  end
end;
{$endif}

{$if defined(CPUX64) and defined(USE_AVX2)}
procedure logistic_array_avx2(const N:NativeInt; const src:PSingle; dst:PSingle);
const
  l2e :single = 1.442695041;// log2(e);
  c0  :single = 1.00172476;
  c1  :single = 0.657636276;
  c2  :single = 0.3371894346;
  //MAX_EXP =  8.8722839052068352E+001;
  //MIN_EXP = -8.7336544750553102E+001;

  MAX_EXP =  8.872283E+001;
  MIN_EXP = -8.733654E+001;

  one :array[0..7] of single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  zero:array[0..7] of single = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  mx  :array[0..7] of single = (MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP);
  mn  :array[0..7] of single = (MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP );
asm
  sub                  rsp      , 16*2                     // making stack space to save one xmm size register
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7

  vpbroadcastd  ymm3  , [rip + l2e]
  vpbroadcastd  ymm4  , [rip + c1]
  vpbroadcastd  ymm5  , [rip + c0]

  mov           r11   , N
  shr           r11   , 3
  jz            @rem

@while:
  vxorps        ymm0  , ymm0        , ymm0              // zero
  vsubps        ymm1  , ymm0        , [src]             // -src
  vcmpgeps      ymm6  , ymm1        , [rip + mx]
  vcmpleps      ymm7  , ymm1        , [rip + mn]
  vblendvps     ymm1  , ymm1 , [rip + mx], ymm6
  vblendvps     ymm1  , ymm1 , [rip + mn], ymm7
  vmulps        ymm1  , ymm3        , ymm1
  vroundps      ymm2  , ymm1        , 1
  vsubps        ymm1  , ymm1        , ymm2
  vcvtps2dq     ymm0  , ymm2
  vpbroadcastd  ymm2  , [rip + c2]
  vfmadd213ps   ymm2  , ymm1        , ymm4
  vpslld        ymm0  , ymm0        , 23
  vfmadd213ps   ymm1  , ymm2        , ymm5
  vpaddd        ymm0  , ymm0        , ymm1
  vaddps        ymm1  , ymm0        , [rip + one]       // 1 +exp(-src)
  vrcpps        ymm1  , ymm1                            // 1/(1+exp(-src))
  //vpcmpeqd      ymm0  , ymm0        , ymm0              // set ymm0 to to 0xffffffff
  //vpandn        ymm6  , ymm6        , ymm0              // ymm6 := not ymm6
  //vpandn        ymm7  , ymm7        , ymm0              // ymm6 := not ymm6
  //vblendvps     ymm1  , ymm1  , [rip+one] , ymm6
  //vblendvps     ymm1  , ymm1  , [rip+zero] , ymm7
  vmovups       [dst] , ymm1
  add           src   , 32
  add           dst   , 32
  dec           r11
  jnz           @while

  and           N   , 7
  jz            @done
@rem:

  vpxor         xmm0  , xmm0        , xmm0
  vsubss        xmm1  , xmm0        , dword [src]              //-src
  vcmpgeps      xmm6  , xmm1        , [rip + mx]
  vcmpleps      xmm7  , xmm1        , [rip + mn]
  vblendvps     xmm1  , xmm1 , [rip + mx], xmm6
  vblendvps     xmm1  , xmm1 , [rip + mn], xmm7
  vmulss        xmm1  , xmm3        , xmm1
  roundss       xmm2  , xmm1        , 1
  vsubss        xmm1  , xmm1        , xmm2
  vcvtps2dq     xmm0  , xmm2
  vmovss        xmm2  , [rip + c2]
  vfmadd213ss   xmm2  , xmm1        , xmm4
  vpslld        xmm0  , xmm0        , 23
  vfmadd213ss   xmm1  , xmm2        , xmm5
  vpaddd        xmm0  , xmm0        , xmm1
  vaddss        xmm1  , xmm0        , [rip + one]       // 1 +exp(-src)
  vrcpss        xmm1  , xmm1        , xmm1              // 1/(1+exp(-src))
  //vpcmpeqd      xmm0  , xmm0        , xmm0              // set ymm0 to to 0xffffffff
  //vpandn        xmm6  , xmm6        , xmm0              // ymm6 := not ymm6
  //vpandn        xmm7  , xmm7        , xmm0              // ymm6 := not ymm6
  //vblendvps     xmm1  , xmm1  , [rip+one] , xmm6
  //vblendvps     xmm1  , xmm1  , [rip+zero]  , xmm7
  vmovss        dword [dst] , xmm1
  add           src   , 4
  add           dst   , 4
  dec           N
  jnz           @rem
@done:
  vmovdqu              xmm6     , [rsp+$00]
  vmovdqu              xmm7     , [rsp+$10]
  add                  rsp      , 16*2                     // restoring stack
end;

// AKA SWISH
procedure SiLU_array(const N:NativeInt; const src, sigmoid, dst:PSingle);assembler;
const
  L2E :single = 1.442695041;// log2(e);
  C0  :single = 1.00172476;
  C1  :single = 0.657636276;
  C2  :single = 0.3371894346;
  //MAX_EXP =  8.8722839052068352E+001;
  //MIN_EXP = -8.7336544750553102E+001;

  MAX_EXP =  8.87E+001;
  MIN_EXP = -8.73E+001;

  one :array[0..7] of single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  zero:array[0..7] of single = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  MX  :array[0..7] of single = (MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP);
  MN  :array[0..7] of single = (MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP );
asm
  sub                  rsp      , 16*2                     // making stack space to save one xmm size register
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7

  vpbroadcastd  ymm3  , [rip + L2E]
  vpbroadcastd  ymm4  , [rip + C1]
  vpbroadcastd  ymm5  , [rip + C0]

  mov           r11   , N
  shr           r11   , 3
  jz            @rem

@while:
  vxorps        ymm0  , ymm0        , ymm0              // zero
  vsubps        ymm1  , ymm0        , [src]             // -src
  vcmpgeps      ymm6  , ymm1        , [rip + MX]
  vcmpleps      ymm7  , ymm1        , [rip + MN]
  vblendvps     ymm1  , ymm1 , [rip + MX], ymm6
  vblendvps     ymm1  , ymm1 , [rip + MN], ymm7
  vmulps        ymm1  , ymm3        , ymm1
  vroundps      ymm2  , ymm1        , 1
  vsubps        ymm1  , ymm1        , ymm2
  vcvtps2dq     ymm0  , ymm2
  vpbroadcastd  ymm2  , [rip + C2]
  vfmadd213ps   ymm2  , ymm1        , ymm4
  vpslld        ymm0  , ymm0        , 23
  vfmadd213ps   ymm1  , ymm2        , ymm5
  vpaddd        ymm0  , ymm0        , ymm1
  vaddps        ymm1  , ymm0        , [rip + one]       // 1 +exp(-src)
  vrcpps        ymm1  , ymm1                            // 1/(1+exp(-src))
  //vpcmpeqd      ymm0  , ymm0        , ymm0              // set ymm0 to to 0xffffffff
  //vpandn        ymm6  , ymm6        , ymm0              // ymm6 := not ymm6
  //vpandn        ymm7  , ymm7        , ymm0              // ymm6 := not ymm6
  //vblendvps     ymm1  , ymm1  , [rip+one] , ymm6
  //vblendvps     ymm1  , ymm1  , [rip+zero] , ymm7
  cmp           sigmoid   , 0
  je            @skip1
  vmovups       yword [sigmoid] , ymm1
  add           sigmoid   , 32
@skip1:
  vmulps        ymm1      , ymm1    , yword [src]
  vmovups      yword [dst], ymm1
  add           src       , 32
  add           dst       , 32
  dec           r11
  jnz           @while

  and           N   , 7
  jz            @done
@rem:

  vpxor         xmm0  , xmm0        , xmm0
  vsubss        xmm1  , xmm0        , dword [src]              //-src
  vcmpgeps      xmm6  , xmm1        , [rip + MX]
  vcmpleps      xmm7  , xmm1        , [rip + MN]
  vblendvps     xmm1  , xmm1 , [rip + MX], xmm6
  vblendvps     xmm1  , xmm1 , [rip + MN], xmm7
  vmulss        xmm1  , xmm3        , xmm1
  roundss       xmm2  , xmm1        , 1
  vsubss        xmm1  , xmm1        , xmm2
  vcvtps2dq     xmm0  , xmm2
  vmovss        xmm2  , [rip + C2]
  vfmadd213ss   xmm2  , xmm1        , xmm4
  vpslld        xmm0  , xmm0        , 23
  vfmadd213ss   xmm1  , xmm2        , xmm5
  vpaddd        xmm0  , xmm0        , xmm1
  vaddss        xmm1  , xmm0        , [rip + one]       // 1 +exp(-src)
  vrcpss        xmm1  , xmm1        , xmm1              // 1/(1+exp(-src))
  //vpcmpeqd      xmm0  , xmm0        , xmm0              // set ymm0 to to 0xffffffff
  //vpandn        xmm6  , xmm6        , xmm0              // ymm6 := not ymm6
  //vpandn        xmm7  , xmm7        , xmm0              // ymm6 := not ymm6
  //vblendvps     xmm1  , xmm1  , [rip+one] , xmm6
  //vblendvps     xmm1  , xmm1  , [rip+zero]  , xmm7
  cmp           sigmoid   , 0   // sigmoid is NULL
  je            @skip2
  vmovss        dword [sigmoid] , xmm1
  add           sigmoid         , 4
@skip2:
  vmulss        xmm1            , xmm1 ,   dword [src]
  vmovss        [dst]           , xmm1
  add           src             , 4
  add           dst             , 4
  dec           N
  jnz           @rem
@done:
  vmovdqu       xmm6            , [rsp+$00]
  vmovdqu       xmm7            , [rsp+$10]
  add           rsp             , 16*2                     // restoring stack
end;

procedure leaky_array(const N:NativeInt; const src:PSingle; dst:PSingle; const f:single=0.1);assembler;{$ifdef FPC}nostackframe;{$endif}
asm
  vxorps            ymm0  , ymm0  ,  ymm0
  movss             xmm1  , f//[rip+f]
  vbroadcastss      ymm1  , xmm1

  mov               r11   , N
  shr               r11   , 3                    // N div 8
  jz                @rem                         // goto rem if zero

@while:
  vmovups           ymm4  , yword [src]
  vmovups           yword [dst]   ,  ymm4
  vcmpgtps          ymm2  , ymm0  ,  ymm4         //  is 0 < src
  vmulps            ymm3  , ymm1  ,  ymm4
  vmaskmovps        yword [dst]  , ymm2  ,  ymm3
  add               src     , 8*4                  // next 8 packed singles
  add               dst     , 8*4                  // next 8 packed singles
  dec               r11
  jnz               @while                       // while r11<>0
  vzeroupper
@rem:
  and               N     , 7                    // N mod 8
  jz                @done                        // exit if zero

@while2:
  vmovss            xmm4      , dword [src]
  vmovss            dword[dst], xmm4
  vcomiss           xmm0  , xmm4
  jbe               @skip
  vmulss            xmm3  , xmm1  ,  dword [src]
  vmovss            dword [dst]   , xmm3
@skip:
  add               src     , 4
  add               dst     , 4
  dec               N
  jnz               @while2

@done:
end;

{$endif}


function stair_activate(const x:single) :single;inline;
var n :NativeInt;
begin
  n := floor(x);
  if n mod 2 = 0 then
    exit(floor(x/ 2));
  exit((x - n) + floor(x/2));
end;

function hardtan_activate(const x:single):single;inline;
begin
    if x < -1 then exit(-1);
    if x > 1 then exit(1);
    exit(x);
end;

function linear_activate(const x:single):single;inline;
begin
  result:=x;
end;

function logistic_activate(const x:single):single;inline;
begin
  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))
  result := 1/(1 + exp(-x))
end;

function loggy_activate(const x:single):single;inline;
begin
  //result := 2/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp))) - 1;
  result := 2/(1 + exp(-x)) - 1;
end;

function relu_activate(const x:single):single;inline;
begin
  result := x*NativeInt(x>0);
  //if x<0 then exit(0);
  //result:=x
end;

function relu6_activate(const x:single):single;inline;
begin
  //min_val_cmp(max_val_cmp(x, 0), 6)
  result := EnsureRange(x,0,6);
  //result:= x*NativeInt(x>0) * NativeInt(x<=6)
end;

function elu_activate(const x:single):single;inline;
begin
  result:=NativeInt(x >= 0)*x + NativeInt(x < 0)*(exp(x)-1);
end;

function selu_activate(const x:single):single;inline;
begin
  result := ord(x >= 0)*1.0507*x + ord(x < 0)*1.0507*1.6732*(exp(x) - 1);
  //result:= NativeInt(x >= 0)*1.0507*x + NativeInt(x < 0)*1.0507*1.6732*(exp(x)-1);
end;

function gelu_activate(const x:single):single;inline;
begin
  result:= 0.5*x*(1 + tanh(0.797885*x + 0.035677*power(x, 3)));
end;

function relie_activate(const x:single):single;inline;
begin
  if x>0 then result:=x
  else result:= 0.01*x;
end;

function ramp_activate(const x:single):single;inline;
begin
  result:= x*NativeInt(x>0)+0.1*x;
end;

function leaky_activate(const x:single):single;inline;
begin
  result := ord(x<0)*x*0.1 + ord(x>0)*x;
  //if x>0 then result:= x
  //else result := 0.1*x
end;

function tanh_activate(const x:single):single;inline;
var px, nx:Single;
begin
  px := exp(x);
  nx := exp(-x);
  result := (px - nx)/(px + nx)
  //result := 2 / (1 + exp(ensureRange(-2 * x, minSingleExp, maxSingleExp))) - 1
  //result := 2 / (1 + exp(-2 * x)) - 1
  //result:= (exp(2*x)-1)/(exp(2*x)+1);
end;

function softplus_activate(const x, threshold : single):single;inline;
begin
    if x > threshold then
      exit(x)                // too large
    else if x < -threshold then
      exit(exp(x));    // too small
//    exit(ln(exp(x) + 1));
    exit(LnXP1(exp(x)));
end;

function plse_activate(const x:single):single;inline;
begin
    if x < -4 then exit( 0.01 * (x + 4));
    if x > 4 then exit( 0.01 * (x - 4) + 1);
    result := 0.125*x + 0.5
end;

function lhtan_activate(const x:single):single;inline;
begin
    if(x < 0) then exit(0.001*x);
    if(x > 1) then exit(0.001*(x-1) + 1);
    result := x
end;

procedure softmax_activate(const N:NativeInt; const x: PSingle; const dst:PSingle);inline;
var
  i:NativeInt;
  mx:single;
begin
  mx := TSingleTensor.maxv(N, Pointer(x), 1);//MaxValue(x, N);
  for i:=0 to N-1 do
    //x[i] := Exp(EnsureRange(x[i]-mx, minSingleExp, maxSingleExp));
    dst[i] := Exp(x[i]-mx);

  mx := TSingleTensor.Sumv(N, pointer(dst), 1);
  //r:=copy(x);
  //r.Exp();
  for i :=0 to N-1 do
    dst[i] := dst[i] / mx
end;

function lhtan_gradient(const x:single):single;inline;
begin
    if (x > 0) and  (x < 1) then
      exit(1);
    exit(0.001)
end;

function hardtan_gradient(const x:single):single;inline;
begin
    if (x > -1) and (x < 1) then
      exit(1);
    exit(0);
end;

function linear_gradient(const x:single):single;inline;
begin
  result:= 1;
end;

function logistic_gradient(const x:single):single;inline;
begin
  result := (1-x)*x;
end;

function loggy_gradient(const x:single):single;inline;
var y:single;
begin
    y := (x+1.0)/2.0;
    result:= 2*(1-y)*y;
end;

function stair_gradient(const x:single):single;inline;
begin
    if floor(x) = x then exit( 0);
    result := 1;
end;

function relu_gradient(const x:single):single;inline;
begin
  result := ord(x>0);
end;

function relu6_gradient(const x:single):single;inline;
begin
  result := NativeInt((x>0) and (x<6));
end;

function elu_gradient(const x:single):single;inline;
begin
  result := NativeInt(x >= 0) + NativeInt(x < 0)*(x + 1);
end;

function selu_gradient(const x:single):single;inline;
begin
  result := NativeInt(x >= 0)*1.0507 + NativeInt(x < 0)*(x + 1.0507*1.6732);
end;

function relie_gradient(const x:single):single;inline;
begin
  result := ord(x>0) + 0.01*ord(x<0);
  //if x>0 then result := 1
  //else result := 0.01
end;

function ramp_gradient(const x:single):single;inline;
begin
  result := NativeInt(x>0) + 0.1;
end;

function leaky_gradient(const x:single):single;inline;
begin
  result := ord(x>0) + 0.1*ord(x<0);
  //if x>0 then result := 1
  //else result := 0.1;
end;

function tanh_gradient(const x:single):single;inline;
begin
  result := 1-x*x;
end;

function sech(const x:single):single;inline;
begin
    result := 2 / (exp(x) + exp(-x))
end;

function gelu_gradient(const x:single):single;inline;
var x3 : single;
begin
    x3 := power(x,3);
    result := 0.5*tanh(0.0356774*x3 + 0.797885*x) + (0.0535161*x3 + 0.398942*x) * power(sech(0.0356774*x3 + 0.797885*x), 2) + 0.5
end;

function plse_gradient(const x:single):single;inline;
begin
  if (x < 0) or (x > 1) then
    result :=  0.01
  else
    result := 0.125;
end;

procedure softmax_gradient(const N:NativeInt; const x: PSingle);inline;
begin

end;
const MISH_THRESHOLD : single=20;

procedure activate_array(const N: NativeInt; const x: PSingle;
  const srcStride: NativeInt; output: PSingle; const dstStride: NativeInt;
  const a: TActivationType; output2: PSingle);

  {$ifdef FPC}
  procedure actMP(const start, finish:IntPtr; const args:pointer);
  {$else}
  var
    actMP : TGroupProcNested;
begin
  actMP := procedure (const start, finish:IntPtr; const args:pointer)
  {$endif}
  var i:NativeInt;
    tmp : single;
  begin
    case a of
        acLOGISTIC:
    {$if defined(CPUX64) and defined(USE_AVX2)}
          if AVX2Support then
            logistic_array_avx2(finish - start+1, x+start, output+start)
          else
    {$endif}
          for i := start to finish do
              output[i*dstStride] := logistic_activate(x[i*srcStride]);
        acRELU:
          for i := start to finish do
              output[i*dstStride] := relu_activate(x[i*srcStride]);

        acRELU6:
          for i := start to finish do
              output[i*dstStride] := relu6_activate(x[i*srcStride]);

        acRELIE:
          for i := start to finish do
              output[i*dstStride] := relie_activate(x[i*srcStride]);

        acLINEAR:
          if output<>x then
            if (srcStride =1) and (dstStride=1)then
              move(x[start], output[start], (finish-start+1)*sizeOf(single))
            else
              for i := start to finish do
                output[i*dstStride] := linear_activate(x[i*srcStride]);
        acRAMP:
          for i := start to finish do
              output[i*dstStride] := ramp_activate(x[i*srcStride]);

        acTANH:
          for i := start to finish do
              output[i*dstStride] := tanh_activate(x[i*srcStride]);

        acPLSE:
          for i := start to finish do
              output[i*dstStride] := plse_activate(x[i*srcStride]);

        acREVLEAKY, acLEAKY:
        {$if defined(CPUX64) and defined(USE_AVX2) and not defined(DEBUG_GPU)}
        if AVX2Support and (srcStride=1) and (dstStride=1) then
          leaky_array(finish - start+1, x+start, output+start)
        else
        {$endif}
          for i := start to finish do
              output[i*dstStride] := leaky_activate(x[i*srcStride]);

        acELU:
          for i := start to finish do
              output[i*dstStride] := elu_activate(x[i*srcStride]);

        acLOGGY:
          for i := start to finish do
              output[i*dstStride] := loggy_activate(x[i*srcStride]);

        acSTAIR:
          for i := start to finish do
              output[i*dstStride] := stair_activate(x[i*srcStride]);

        acHARDTAN:
          for i := start to finish do
              output[i*dstStride] := hardtan_activate(x[i*srcStride]);

        acLHTAN:
          for i := start to finish do
              output[i*dstStride] := lhtan_activate(x[i*srcStride]);

        acSELU:
          for i := start to finish do
              output[i*dstStride] := selu_activate(x[i*srcStride]);

        acGELU:
          for i := start to finish do
              output[i*dstStride] := gelu_activate(x[i*srcStride]);

        acSWISH:
          {$if defined(CPUX64) and defined(USE_AVX2) and not defined(DEBUG_GPU)}
          if AVX2Support and (srcStride=1) and (dstStride=1) then
            if assigned(output2) then
              SiLU_array(finish-start+1, x+start, output2+start, output+start)
            else
              SiLU_array(finish-start+1, x+start, nil          , output+start)
          else
          {$endif}
          if assigned(output2) then
            for i := start to finish do begin
              tmp := x[i*srcStride];
              output2[i*dstStride] := logistic_activate(tmp);
              output[i*dstStride] := tmp*output2[i*dstStride];
            end
          else
            for i := start to finish do begin
              tmp := x[i*srcStride];
              output[i*dstStride] := tmp*logistic_activate(tmp);
            end;

        //
        acMISH:
          if Assigned(output2) then
          for i := start to finish do
            begin
                tmp := x[i*srcStride];
                output2[i] := tmp;
                output[i*dstStride] := tmp * tanh_activate(softplus_activate(tmp, MISH_THRESHOLD))
            end
          else
            for i := start to finish do
              begin
                  tmp := x[i*srcStride];
                  output[i*dstStride] := tmp * tanh_activate(softplus_activate(tmp, MISH_THRESHOLD))
              end;

        //
        //acHARD_MISH:
        //           ;
        //
        //acNORM_CHAN:
        //           ;
        //
        //acNORM_CHAN_SOFTMAX:
        //           ;
        //
        //acNORM_CHAN_SOFTMAX_MAXVAL:
        //           ;
        else
          assert(false, '[Activation] :'+GetEnumName(TypeInfo(TActivationType), ord(a))+' not Implemented')
    end;
  end;
{$ifdef FPC}
begin
{$endif}
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(a);{$endif}

  if a <> acSOFTMAX then // todo : [Activation] CAUSION: acSoftmax here does not work with multithreading yet
    {$ifdef USE_MULTITHREADING}
    mp.&for(actMP, 0, N-1)
    {$else}
    actMP(0, N-1, nil)
    {$endif}
  else
    softmax_activate(N, x, output);

  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(a);{$endif}
end;

procedure activate_array(const N: NativeInt; x: PSingle; const a: TActivationType; output:PSingle=nil);
begin
  activate_array(N, x, 1, x, 1, a, output);
end;

procedure gradient_array(const N: NativeInt; const x: PSingle;
  const srcStride: NativeInt; delta: PSingle; const dstStride: NativeInt;
  const a: TActivationType; output: PSingle);

{$ifdef FPC}
procedure gradMP(const start, finish:IntPtr; const args:pointer);
{$else}
var gradMP : TGroupProcNested;
begin
  gradMP := procedure (const start, finish:IntPtr; const args:pointer)
{$endif}
var
  i  : NativeInt;
  tmp, sp, grad, grad_sp, tsp, grad_tsp: single;
begin
    // todo [Gradient array] SIMDfy & GPU
    case a of
        acLOGISTIC:
          for i := start to finish do
              delta[i] := delta[i] * logistic_gradient(x[i]);

        acRELU:
          for i := start to finish do
              delta[i] := delta[i] * Integer(x[i]>0);//relu_gradient(x[i]);

        acRELU6:
          for i := start to finish do
              delta[i] := delta[i] * relu6_gradient(x[i]);

        acRELIE:
          for i := start to finish do
              delta[i] := delta[i] * relie_gradient(x[i]);

        acLINEAR:
          //for i := start to finish- do
          //    delta[i] := delta[i] *linear_gradient(x[i])
          ;

        acRAMP:
          for i := start to finish do
              delta[i] := delta[i] * ramp_gradient(x[i]);

        acTANH:
          for i := start to finish do
              delta[i] := delta[i] * tanh_gradient(x[i]);

        acPLSE:
          for i := start to finish do
              delta[i] := delta[i] * plse_gradient(x[i]);

        acREVLEAKY, acLEAKY:
          for i := start to finish do
              delta[i] := delta[i] * leaky_gradient(x[i]);

        acELU:
          for i := start to finish do
              delta[i] := delta[i] * elu_gradient(x[i]);

        acLOGGY:
          for i := start to finish do
              delta[i] := delta[i] * loggy_gradient(x[i]);

        acSTAIR:
          for i := start to finish do
              delta[i] := delta[i] * stair_gradient(x[i]);

        acHARDTAN:
          for i := start to finish do
              delta[i] := delta[i] * hardtan_gradient(x[i]);

        acLHTAN:
          for i := start to finish do
              delta[i] := delta[i] * lhtan_gradient(x[i]);

        acSELU:
          for i := start to finish do
              delta[i] := delta[i] * selu_gradient(x[i]);

        acGELU:
          for i := start to finish do
              delta[i] := delta[i] * gelu_gradient(x[i]);
        acSWISH:
          if assigned(output) then
          for i := start to finish do
              begin
                  tmp := x[i];
                  delta[i] := delta[i] * (tmp+output[i] * (-tmp))
              end
          else
          for i := start to finish do
              begin
                  tmp := x[i];
                  delta[i] := delta[i] * (tmp+logistic_activate(x[i]) * (-tmp))
              end;
        acMISH:
          if assigned(output) then
          for i := start to finish do
              begin
                  tmp := output[i];
                  sp := softplus_activate(tmp, MISH_THRESHOLD);
                  grad_sp := 1-exp(-sp);
                  tsp := tanh(sp);
                  grad_tsp := (1-tsp * tsp) * grad_sp;
                  grad := tmp * grad_tsp+tsp;
                  delta[i] := delta[i] * grad
              end
          else
          for i := start to finish do
              begin
                  tmp := x[i];
                  sp := softplus_activate(tmp, MISH_THRESHOLD);
                  grad_sp := 1-exp(-sp);
                  tsp := tanh(sp);
                  grad_tsp := (1-tsp * tsp) * grad_sp;
                  grad := tmp * grad_tsp+tsp;
                  delta[i] := delta[i] * grad
              end ;
    //
    //    acHARD_MISH:
    //               ;
    //
    //    acNORM_CHAN:
    //               ;
    //
    //    acNORM_CHAN_SOFTMAX:
    //               ;
    //
    //    acNORM_CHAN_SOFTMAX_MAXVAL:
    //
      else
        assert(false, '[Derivitive] : not implemented!')
    end;

end;

{$ifdef FPC}
begin
{$endif}
  {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(a);{$endif}

  {$ifdef USE_MULTITHREADING}
  mp.&For(gradMP, 0, N-1);
  {$else}
  gradMP(0, N-1, nil);
  {$endif}

  {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(a);{$endif}
end;

procedure gradient_array(const N: NativeInt; const x: PSingle; delta: PSingle; const a: TActivationType; output: PSingle);
begin
  gradient_array(N, x, 1, delta, 1, a, output)
end;

//procedure swishMP(const f, t:IntPtr; const p:pointer);
//var
//  i: NativeInt;
//  x_val, sigmoid: single;
//  a:PMPParams absolute p;
//  x, output_sigmoid, output:PSingle;
//begin
//    x:=a.A;
//    output_sigmoid:=a.B;
//    output := a.C;
//
//    {$if defined(CPUX64) and defined(USE_AVX2)}
//    if AVX2Support then
//      SiLU_array(output+f, output_sigmoid+f, x+f, t-f+1)
//    else
//    {$endif}
//    for i := f to t do
//      begin
//          x_val := x[i];
//          sigmoid := logistic_activate(x_val);
//          output_sigmoid[i] := sigmoid;
//          output[i] := x_val * sigmoid
//      end
//end;

// SWISH aka SiLU
//procedure activate_array_swish(const x: Psingle; const N: NativeInt; output_sigmoid, output: Psingle);
//var
//  p:TMPParams;
//begin
//// todo simdfy
//  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acSWISH);{$endif}
//  p.A:=x;
//  p.B:=output_sigmoid;
//  p.C:=output;
//  {$if defined(_USE_MULTITHREADING)}
//  mp2.&for(swishMP, 0, N-1,@p);
//  {$else}
//  swishMP(0, N-1, @p);
//  {$endif}
//  {$ifdef USE_TELEMETRY}if benchmark then metrics.act.finish(acSWISH);{$endif}
//end;

{$ifdef USE_MULTITHREADING}
procedure mishMP(const f,t :IntPtr;const p:pointer=nil);
var
  i: NativeInt;
  x_val: single;
  a:PMPParams absolute p;
  x, activation_input, output:Psingle;
begin
    x                 :=a.A;
    activation_input  :=a.B;
    output            :=a.C;

  // todo SIMDfy
    for i := f to t do
      begin
          x_val := x[i];
          activation_input[i] := x_val;
          output[i] := x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD))
      end
end;
{$endif}

procedure activate_array_mish(const x: Psingle; const N: NativeInt; const activation_input, output: Psingle);
var
  x_val :single;
  i:NativeInt;
  {$ifdef USE_MULTITHREADING}
  p:TMPParams;
  {$endif}
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acMISH);{$endif}
  {$if defined(USE_MULTITHREADING)}
  p.A:=x;
  p.B:=activation_input;
  p.C:=output;
  mp.&for(mishMP, 0, n-1,@p);
  {$else}
  for i := 0 to N-1 do
    begin
        x_val := x[i];
        activation_input[i] := x_val;
        output[i] := x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD))
    end;
  {$endif}

  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acMISH);{$endif}
end;


function hard_mish_yashas(x: single):single;
begin
  if (x > 0) then
      exit(x);
  if x > -2 then
      exit(x * x / 2+x);
  exit(0)
end;

procedure activate_array_hard_mish(const x: Psingle; const N: NativeInt;
  const activation_input, output: PSingle);
var
  i: NativeInt;
  x_val: single;
begin
// todo SIMDfy
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acHARD_MISH);{$endif}
  for i := 0 to N -1 do
      begin
          x_val := x[i];
          activation_input[i] := x_val;
          output[i] := hard_mish_yashas(x_val)
      end;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acHARD_MISH);{$endif}
end;

procedure activate_array_normalize_channels(const x: Psingle; const N, batch,
  channels, wh_step: NativeInt; const output: Psingle);
var
  size, i, wh_i, b, k: NativeInt;
  sum, val: single;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acNORM_CHAN);{$endif}
  size := N div channels;
  // todo SIMDfy
  for i := 0 to size -1 do
      begin
          wh_i := i mod wh_step;
          b := i div wh_step;
          if i < size then
              begin
                  sum := sEPSILON;
                  for k := 0 to channels -1 do
                      begin
                          val := x[wh_i+k * wh_step+b * wh_step * channels];
                          if val > 0 then
                              sum := sum + val
                      end;
                  for k := 0 to channels -1 do
                      begin
                          val := x[wh_i+k * wh_step+b * wh_step * channels];
                          if val > 0 then
                              val := val / sum
                          else
                              val := 0;
                          output[wh_i+k * wh_step+b * wh_step * channels] := val
                      end
              end
      end;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acNORM_CHAN);{$endif}
end;

procedure activate_array_normalize_channels_softmax(const x: Psingle; const N,
  batch, channels, wh_step: NativeInt; const output: PSingle;
  const use_max_val: boolean);
var
  size, i, wh_i, b, k: NativeInt;
  sum, max_val, val: single;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acNORM_CHAN_SOFTMAX);{$endif}
  size := N div channels;
  // todo SIMDFy
  for i := 0 to size -1 do
      begin
          wh_i := i mod wh_step;
          b := i div wh_step;
          if i < size then
              begin
                  sum := sEPSILON;
                  max_val := -MaxSingle;
                  if use_max_val then
                      for k := 0 to channels -1 do
                          begin
                              val := x[wh_i+k * wh_step+b * wh_step * channels];
                              if (val > max_val) or (k = 0) then
                                  max_val := val
                          end
                  else
                      max_val := 0;
                  for k := 0 to channels -1 do
                      begin
                          val := x[wh_i+k * wh_step+b * wh_step * channels];
                          sum := sum + exp(val-max_val)
                      end;
                  for k := 0 to channels -1 do
                      begin
                          val := x[wh_i+k * wh_step+b * wh_step * channels];
                          val := exp(val-max_val) / sum;
                          output[wh_i+k * wh_step+b * wh_step * channels] := val
                      end
              end
      end;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acNORM_CHAN_SOFTMAX);{$endif}
end;


//procedure gradient_array_swish(const x: Psingle; const N: NativeInt; const sigmoid: Psingle; delta: Psingle);
//var
//    i: NativeInt;
//    swish: single;
//begin
//    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acSWISH);{$endif}
//    // todo SIMDfy
//    for i := 0 to N -1 do
//        begin
//            swish := x[i];
//            delta[i] := delta[i] * (swish+sigmoid[i] * (1-swish))
//        end;
//    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acSWISH);{$endif}
//end;

procedure gradient_array_mish(const N: NativeInt;
  const activation_input: Psingle; delta: Psingle);
const
    MISH_THRESHOLD: single = 20;
var
    i: NativeInt;
    inp, sp, grad_sp, tsp, grad_tsp, grad: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acMISH);{$endif}

    // todo SIMDfy
    for i := 0 to N -1 do
        begin
            inp := activation_input[i];
            sp := softplus_activate(inp, MISH_THRESHOLD);
            grad_sp := 1-exp(-sp);
            tsp := tanh(sp);
            grad_tsp := (1-tsp * tsp) * grad_sp;
            grad := inp * grad_tsp+tsp;
            delta[i] := delta[i] * grad
        end ;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acMISH);{$endif}
end;

function hard_mish_yashas_grad(x: single):single;
begin
    if (x > 0) then
        exit(1);
    if x > -2 then
        exit(x+1);
    exit(0)
end;

procedure gradient_array_hard_mish(const N: NativeInt; const activation_input,
  delta: Psingle);
var
    i: NativeInt;
    inp: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acHARD_MISH);{$endif}
    // todo SIMDfy
    for i := 0 to N -1 do
        begin
            inp := activation_input[i];
            delta[i] := delta[i] * hard_mish_yashas_grad(inp)
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acHARD_MISH);{$endif}
end;

procedure gradient_array_normalize_channels(const x: Psingle; const N, batch,
  channels, wh_step: NativeInt; const delta: Psingle);
var
    size, i, wh_i, b, k, index: NativeInt;
    grad, &out, d: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acNORM_CHAN);{$endif}
    size := N div channels;
    // todo SIMDfy
    for i := 0 to size -1 do
        begin
            wh_i := i mod wh_step;
            b := i div wh_step;
            if i < size then
                begin
                    grad := 0;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            &out := x[index];
                            d := delta[index];
                            grad := grad + (&out * d)
                        end;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            if x[index] > 0 then
                                begin
                                    d := delta[index];
                                    d := d * grad;
                                    delta[index] := d
                                end
                        end
                end
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acNORM_CHAN);{$endif}
end;

procedure gradient_array_normalize_channels_softmax(const x: PSingle; const N,
  batch, channels, wh_step: NativeInt; const delta: Psingle);
var
    size, i, wh_i, b, k, index: NativeInt;
    grad, &out, d: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acNORM_CHAN_SOFTMAX);{$endif}
    size := N div channels;
    // todo SIMDfy
    for i := 0 to size -1 do
        begin
            wh_i := i mod wh_step;
            b := i div wh_step;
            if i < size then
                begin
                    grad := 0;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            &out := x[index];
                            d := delta[index];
                            grad := grad + (&out * d)
                        end;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            d := delta[index];
                            d := d * grad;
                            delta[index] := d
                        end
                end
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acNORM_CHAN_SOFTMAX);{$endif}
end;

initialization


end.

