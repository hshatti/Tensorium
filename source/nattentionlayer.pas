unit nAttentionLayer;
{$ifdef FPC}
  {$mode Delphi}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$SafeFPUExceptions off}
{$PointerMath on}

interface

uses
  SysUtils, nTypes, nBaseLayer, nTensors, nConnectedlayer, termesc
  , math
  , steroids
  ;

type

  { TAttentionLayer }

  TAttentionLayer = class(TBaseLayer)
    NumDimensions, NumHeads, NumHeadsKV: SizeInt;
    wq, wk, wv, wo : TConnectedLayer;
    constructor Create(const aBatch, aNumDimensions, aNumHeads:SizeInt; const aNumHeadsKV:SizeInt =0);
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
    {$if defined(USE_OPENCL) or defined(USE_CUDART)}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}

  end;

implementation

{ TAttentionLayer }

constructor TAttentionLayer.Create(const aBatch, aNumDimensions,
  aNumHeads: SizeInt; const aNumHeadsKV: SizeInt);
begin

end;

procedure TAttentionLayer.setTrain(ATrain: boolean);
begin

end;

procedure TAttentionLayer.setBatch(ABatch: SizeInt);
begin

end;

procedure TAttentionLayer.forward(var state: TNNetState);
begin

end;

procedure TAttentionLayer.backward(var state: TNNetState);
begin

end;

procedure TAttentionLayer.update(const args: TUpdateArgs);
begin
  inherited update(args);
end;

{$if defined(USE_OPENCL) or defined(USE_CUDART)}
procedure TAttentionLayer.forwardGPU(var state: TNNetState);
begin

end;

procedure TAttentionLayer.backwardGPU(var state: TNNetState);
begin

end;

procedure TAttentionLayer.updateGPU(const args: TUpdateArgs);
begin
  inherited updateGPU(args);
end;
{$endif}

function mapX(const x:single):single;inline;
begin
  exit( x*3-2.1);
end;
// Same purpose as mapX
// [0, 1] -> [-1.25, 1.25]
function mapY(const y:single):single; inline;
begin
  exit( y*3 - 1.5);
end;

function mandel(buf :pbyte; const w, h:integer):TSingleTensor;
const
     max_iteration = 10000 ;
     _max          = 4.0  ;

var
  lnxp1_max_iteration: single;

  procedure renderLine(y:IntPtr; data:pointer);
  var d:PByte;
  _x, x, iteration:integer;
  c : longword;
  xx, yy,
  x0 , y0,
  xtemp, oldAbs, coverageNum, currentAbs,
  diffToLast, diffToMax :single ;
  begin
    d := buf + y*w*3;
    for x :=0 to w-1 do
    begin
      xx := mapX(x/w);
      yy := mapY(y/h);

      _x := x * 3;  // x, y of size for rgba (four bytes)

      x0 := 0.0;
      y0 := 0.0;
      iteration := 0;
      oldAbs := 0.0;
      coverageNum := max_iteration;
      while iteration < max_iteration do begin
          xtemp := x0 * x0 - y0 * y0;
          y0 := 2 * x0 * y0;
          x0 := xtemp;
          x0 := x0 + xx;
          y0 := y0 + yy;
          currentAbs := x0*x0 + y0*y0;
          if currentAbs>4.0 then begin
             diffToLast  := currentAbs - oldAbs;
             diffToMax   := _max - oldAbs;
             coverageNum := iteration + diffToMax/diffToLast;
             break;
          end;
          oldAbs := currentAbs;
          inc(iteration);
      end;
      if iteration = max_iteration then
      begin
          d[_x]   := 0;
          d[_x+1] := 0;
          d[_x+2] := 0;
          //buf[_x+3] := $ff;
      end else
      begin
          c         := trunc($ff * LnXP1(coverageNum)/lnxp1_max_iteration);
          d[_x+0] := trunc((0.5+0.5*sin(c/25)) * $FF);          //R
          d[_x+1] := trunc((0.5+0.5*sin(PI/3+c/25)) * $FF);   //G
          d[_x+2] := trunc((0.5+0.5*sin(PI*2/3+c/25)) * $FF);   //B
          //buf[_x+3] := $ff;
      end
    end;
  end;

begin
  lnxp1_max_iteration := LnXP1(max_iteration);
  {$if defined(USE_MULTITHREADING)}
  mp2.&For(renderLine, 0, h);
  {$else}
  for y:=0 to h-1 do begin
    renderLine(y, nil);
  end
  {$endif}
end;


function compare_dw(const a, b: longword): SizeInt;
begin
  result := a - b
end;

type

  TSixel = packed record
    r,g,b:byte
  end;
  PSixel = ^TSixel;

function col2Index(const pxl:TSixel):longword; inline;
begin
  result := trunc(100*pxl.r / 255) + 100*trunc(100*pxl.g / 255) + 10000*trunc(100*pxl.b / 255)
end;

function  index2Col(const pxl:longword):TSixel; inline;
begin
  result.r := pxl mod 100;
  result.g := (pxl div 100) mod 100;
  result.b := pxl div 10000
end;

procedure printSixel(const buf:PByte; const width, height:SizeInt);
type
  TColorPal = array[0..1010100 -1] of longword;
  utils = TTools<longword>;
var
  P : PSixel absolute buf;
  bitPix  : array of byte;
  charCount, idx, i, x, y: SizeInt;
  colorPal : TColorPal;
  c, lastC : longint;
  //s,
  plt : ansistring;
  clrCount, reg : longInt;
  sxl :TSixel;

begin
  for i:=0 to high(colorPal) do
    colorPal[i] := $ffffff;
  clrCount := 0;
  for i:=0 to height*width-1 do begin
    c := col2Index(p[i]);
    //sxl:=index2Col(c);
    if colorPal[c]=$ffffff then inc(clrCount);
    colorPal[c]:= c;
  end;
  utils.QuickSort(@colorPal[0], 0, high(colorPal), compare_dw);
  plt := '';
  for i:=0 to clrCount-1 do begin
    sxl := index2Col(colorPal[i]);
    plt := plt + '#'+intToStr(i) + ';2;'+intToStr(sxl.r) +';'+intToStr(sxl.g)+';'+intToStr(sxl.b);
    //plt := plt + format('#%d;2;%d;%d;%d',[i, sxl.r, sxl.g, sxl.b])
  end;
  setLength(bitPix, width);
  for y := 0 to Math.ceil(height/6) do begin
    for i:= 0 to 5 do begin
      if y*6 + i >= height then break;
      lastC :=-1;
      for x:=0 to width-1 do begin
        idx := y*6*width + i*width + x;
        c := col2Index(p[idx]);
        bitPix[x] := 63 + (1 shl i);
        if c=lastC then begin
          plt := plt + ansichar(bitPix[x])
        end
        else
        begin
          reg := utils.BinSearch(@colorPal[0], c, clrCount, compare_dw);
          assert(reg>=0, '[printSixel] Cannot fine color in palettte!');
          plt := plt + '#'+ intToStr(reg) + ansichar(bitPix[x]);
        end;
        lastC := c;
      end;
      plt := plt +'$'
    end;
    plt := plt + '-'
  end;

  plt := #$1B'Pq'+
         '"1;1;'+intToStr(width)+';'+intTostr(height)+plt +
         #$1B'\' ;
  write(plt)
end;

var t1,t2: TByteTensor;
  t : UInt64;

initialization
  //t1.resize([4, 5, 6]);
  //t1.Fill(0,1);
  //t1.print();
  //t1.resize([1000, 1000, 3]);
  //mandel(t1.data, 1000, 1000);
  //t2 := t1.permute([1, 2, 0]); // HxWxC => CxHxX
  //t2.printStat();
  //t2.print(0.1);
  //t1.toSingles(t3.data);
  //t3.printStat();
  //t3.print(0.1, true);
  //t := GetTickCount64;
  //printSixel(t1.data, 1000, 1000);
  //writeln('done in [', (getTickCount64()-t)/1000:1:3,']ms');
  //readln


  //sDigits:=3; sSeparator:=', ';
  //t1.resize([4*4*4*4]);
  //t1.Fill(0,1/255);
  //t1.reshape([4,4,4,4]);
  //t1.print(psGray, 4);
  //t1.Permute([0,2,1,3]).print(psGray, 4);


end.

