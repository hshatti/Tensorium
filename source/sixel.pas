unit sixel;
{$ifdef FPC}
  {$mode Delphi}
  {$modeswitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}



interface

uses
  Classes, SysUtils, Math, Generics.Collections, windows, SortedMap, termesc, nTensors
  {$ifdef USE_MULTITHREADING}
  , steroids
  {$endif}
  ;



procedure printSixel(const buf:Pointer; const width, height:SizeInt; const Dither:boolean = false);

implementation


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
  {$ifdef FPC}
  procedure renderLine(y:IntPtr; data:pointer);
  {$else}
  renderLine : TThreadProcNested;
begin
  renderLine := procedure(y:IntPtr; data:pointer)
  {$endif}
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
var
  y:SizeInt;
{$ifdef FPC}
begin
{$endif}
  lnxp1_max_iteration := LnXP1(max_iteration);
  {$if defined(USE_MULTITHREADING)}
  mp2.&For(renderLine, 0, h);
  {$else}
  for y:=0 to h-1 do begin
    renderLine(y, nil);
  end
  {$endif}
end;


type

  { TSixel }

  TSixel = packed record
    r,g,b:byte;
    class operator add      (const a, b:TSixel):TSixel;
    class operator subtract (const a, b:TSixel):TSixel;
    class operator multiply (const a, b:TSixel):TSixel;
    class operator IntDivide(const a, b:TSixel):TSixel;
    class operator multiply (const a:TSixel; const b:byte):TSixel;
    class operator IntDivide(const a:TSixel; const b:byte):TSixel;
    class operator multiply (const a:TSixel; const b:single):TSixel;


  end;
  TSixelS = packed record
    r,g,b:single
  end;
  PSixel = ^TSixel;

  TColorMap = TDictionary<TSixel, longint>;

function comp(const a, b:longword):SizeInt; winapi;
begin
  result := a-b
end;

function sxlIsEqual(const a, b: TSixel):boolean;
begin
  result := (a.r=b.r) and (a.g=b.g) and (a.b=b.b)
end;

function intToAnsiStr(const int:integer):ansistring;   inline;
begin
  result := sysutils.intToStr(int)
end;

procedure printSixel(const buf: Pointer; const width, height: SizeInt; const Dither: boolean);
const pixel = 1;
     pixcount = 6 div pixel;
     COLOR_SPACE : array[0..2] of byte = (8, 8, 4);
var
  P                      : PSixel;
//  bitPix  : array of byte;
  pos                    : SizeInt;
  colorPal               : TColorMap;
  plt                    : ansistring;
  sxlMin, sxlMax         : TSixel;
  sxlScaler              : TSixel;

  function getSixel(const y, x :SizeInt)    : TSixel;
  begin
    result := p[y*width + x]
  end;

  procedure encodeNum(const ctl : ansichar; const num:longint);
  var sz : SizeInt;
  begin
    if length(plt)<pos + 6 then
      setLength(plt, 2*(pos + 5));
    plt[pos] := ctl; inc(pos);
    if num>99 then sz := 3 else if num>9 then sz:=2 else sz := 1;
    move(intToAnsistr(num)[1], plt[pos], sz); inc(pos,sz);
  end;
  procedure encodeChar(const ctl:ansichar; const num:longint; const str :ansichar);
  var sz : SizeInt;
  begin
    if length(plt)<pos + 7 then
      setLength(plt, 2*(pos + 6));
    plt[pos] := ctl; inc(pos);
    if num>99 then sz := 3 else if num>9 then sz:=2 else sz := 1;
    move(intToAnsistr(num)[1], plt[pos], sz); inc(pos,sz);
    plt[pos] := str; inc(pos);
  end;

  procedure encodeStr(const str:ansistring);
  var sz :SizeInt;
  begin
    sz := length(str);
    if length(plt) <pos + sz then
      setLength(plt, 2*(pos + sz));
    move(str[1], plt[pos], sz); inc(pos, sz)
  end;

  procedure encodeRepeat(const num:longint; const str:ansichar); inline;
  var i: longint;
  begin
    for i:= 0 to num div 255 -1 do
      encodeChar('!', 255, str);
    if num mod 255>0 then
      encodeChar('!', num mod 255, str)
  end;

  function clr100(const col:TSixel):TSixel; inline;
  begin
    result.r := trunc(100*col.r/255);
    result.g := trunc(100*col.g/255);
    result.b := trunc(100*col.b/255);
  end;

  function clrSpc(const col:TSixel):TSixel;
  begin
    //result.r := round(100*(col.r - sxlMin.r)/255) div sxlScaler.r;
    //result.g := round(100*(col.g - sxlMin.g)/255) div sxlScaler.g;
    //result.b := round(100*(col.b - sxlMin.b)/255) div sxlScaler.b;
    //result.r := sxlScaler.r*(round(100*col.r/255) div sxlScaler.r);
    //result.g := sxlScaler.g*(round(100*col.g/255) div sxlScaler.g);
    //result.b := sxlScaler.b*(round(100*col.b/255) div sxlScaler.b);
    result := sxlScaler * (col div sxlScaler);
  end;

  procedure reCalcPalette(var mi, ma:TSixel; var s:TSixel);
  var i, r, g, b:longint; sx :TSixel;
  begin
    //clrCount := 0;
    colorPal.Clear;
    mi.r := p[0].r;
    mi.g := p[0].g;
    mi.b := p[0].b;
    ma.r := p[0].r;
    ma.g := p[0].g;
    ma.b := p[0].b;
    for i:=1 to height*width-1 do begin
      sx := p[i];
      mi.r := Math.min(sx.r, mi.r);
      mi.g := Math.min(sx.g, mi.g);
      mi.b := Math.min(sx.b, mi.b);
      ma.r := Math.max(sx.r, ma.r);
      ma.g := Math.max(sx.g, ma.g);
      ma.b := Math.max(sx.b, ma.b);
    end;
    //s.r := round(100 *(ma.r-mi.r) / 255) div 5;     // scaler
    //s.g := round(100 *(ma.g-mi.g) / 255) div 5;
    //s.b := round(100 *(ma.b-mi.b) / 255) div 5;
    s.r := 100 div (COLOR_SPACE[0]-1);
    s.g := 100 div (COLOR_SPACE[1]-1);
    s.b := 100 div (COLOR_SPACE[2]-1);
    i:=16;

    for r:=0 to COLOR_SPACE[0]-1 do
      for g :=0 to COLOR_SPACE[1]-1 do
        for b :=0 to COLOR_SPACE[2]-1 do begin
          sx.r := s.r*r;
          sx.g := s.g*g;
          sx.b := s.b*b;
          encodeNum('#', i);
          encodeNum(';', 2);
          //encodeNum(';', mi.r + s.r*r);
          //encodeNum(';', mi.g + s.g*g);
          //encodeNum(';', mi.b + s.b*b); // encodeStr(#13#10);
          encodeNum(';', s.r*r);
          encodeNum(';', s.g*g);
          encodeNum(';', s.b*b); // encodeStr(#13#10);
          colorPal.Add(sx, i);
          inc(i)
        end;
  end;

const           // enable SIXEL + START SIXEL
  START_SIXEL = #$1B']80h'      + #$1B'Pq';
  END_SIXEL   = #$1B'\';  // ST
var
  charCount              : longword;
  bitPix                 : byte;
  sxl, lastSxl, q_err    : TSixel;
  p1, p2, p3, p4         : PSixel;
  dith                   : TArray<TSixel>;
  //clrCount,
  reg                    : longInt;
  i, idx, x, y           : SizeInt;
  t:int64;
label done;
begin

  colorPal := TColormap.Create(width* height);
  setLength(plt, width*height);
  setLength(dith, width*height);
  pos:=1;
  //move(buf^, dith[0], width*height*sizeOf(p[0]));
  p := pointer(buf);
  reCalcPalette(sxlMin, sxlMax, sxlScaler);

  for i:=0 to high(dith) do dith[i] := clr100(p[i]);
  p := pointer(dith);

  if Dither then
    for y:=0 to height -2 do
      for x:= 1 to width -2 do begin
        idx                  := y*width + x;
        lastSxl              := dith[idx];
        sxl                  := clrSpc(lastSxl);
        dith[idx]            := sxl;
        q_err                := lastSxl - sxl;
        p1 := @p[idx         +1];
        p2 := @p[idx + width -1];
        p3 := @p[idx + width   ];
        p4 := @p[idx + width +1];

        p1^ := p1^ + q_err * (7/16);
        p2^ := p2^ + q_err * (3/16);
        p3^ := p3^ + q_err * (5/16);
        p4^ := p4^ + q_err * (1/16);
      end;

//  setLength(bitPix, width);
  //write(copy(plt,1, pos-1));

  charCount:=0;

  //t:=GetTickCount64;
  for y := 0 to Math.ceil(height/pixCount) do begin
    for i:= 0 to pixCount-1 do begin
      if y*pixCount + i >= height then break;
      //lastC := $ffffff;
      lastSxl.r := $ff;
      for x:=0 to width-1 do begin
        //idx := getSixel(y*pixCount + i, x);
        //c := col2Index(p[idx]);
        //sxl := col100(getSixel(y*pixCount + i, x));
        //sxl := col2Six(sxl);
        sxl := clrSpc(getSixel(y*pixCount + i, x));
        bitPix := 63 + ((1 shl pixel)-1) shl (i*pixel);
        if sxlIsEqual(sxl, lastSxl) then begin
        //if c=lastC then begin
          inc(charCount)
        end
        else
        begin
          if charCount>0 then begin
            encodeRepeat(charCount*pixel, ansichar(bitPix));
            //encodeChar('!', charCount*pixel, ansichar(bitPix));
            charCount:=0;
          end;
          //colorPal.TryGetValue(c, reg);
          if not colorPal.TryGetValue(sxl, reg) then
            raise Exception.Create('Cannot locate color in the palette!');
          encodeChar('#', reg, ansichar(bitPix));
          charCount:=0;
          if pixel>1 then
            fillchar(plt[pos], pixel-1, bitPix); inc(pos, pixel-1);
        end;
        lastSxl := sxl;
        //lastC := c
      end;
      if charCount>0 then begin
        encodeRepeat(charCount*pixel, ansichar(bitPix));
        //encodeChar('!', charCount*pixel, ansichar(bitPix));
        charCount:=0;
      end;
      plt[pos]:='$'; inc(pos);   // overlay line
    end;
    plt[pos]:='-'; inc(pos)  // draw in new line
  end;
done:
  freeAndNil(colorPal);
  //writeln((GetTickCount64-t)/1000:1:3,'ms');

(*
     P1          Pixel Aspect Ratio
                 (Vertical:Horizontal)

     Omitted     2:1
     0 or 1      5:1
     2           3:1
     3 or 4      2:1
     5 or 6      2:1
     7,8, or 9   1:1

    P2 selects how the terminal draws the background color.   You can 3$
    use one of three values.

         P2          Meaning

         0 or 2      Pixel positions  specified as  0 are  set to the
                     current background color.

         1           Pixel positions specified as  0 remain  at their
                     current color.

    P3 is  the horizontal  grid size  parameter.  The horizontal grid
    size is the horizontal distance  between  two  pixel  dots.   The
    VT300 ignores  this parameter because the horizontal grid size is
    fixed at 0.0195 cm (0.0075 in).
*)

  setLength(plt, pos-1);
  write(START_SIXEL);
  write('"1;1;'+intToAnsistr(width)+';'+intToAnsistr(height));
  write(plt);
  write(END_SIXEL);
  //write(plt)
end;


var
  bt : TByteTensor;
  t:int64;

{ TSixel }

class operator TSixel.add(const a, b: TSixel): TSixel;
begin
  result.r := a.r + b.r;
  result.g := a.g + b.g;
  result.b := a.b + b.b;
end;

class operator TSixel.subtract(const a, b: TSixel): TSixel;
begin
  result.r := a.r - b.r;
  result.g := a.g - b.g;
  result.b := a.b - b.b;
end;

class operator TSixel.multiply(const a, b: TSixel): TSixel;
begin
  result.r := a.r * b.r;
  result.g := a.g * b.g;
  result.b := a.b * b.b;
end;

class operator TSixel.IntDivide(const a, b: TSixel): TSixel;
begin
  result.r := a.r div b.r;
  result.g := a.g div b.g;
  result.b := a.b div b.b;
end;

class operator TSixel.multiply(const a: TSixel; const b: byte): TSixel;
begin
  result.r := a.r * b;
  result.g := a.g * b;
  result.b := a.b * b;
end;

class operator TSixel.IntDivide(const a: TSixel; const b: byte): TSixel;
begin
  result.r := a.r div b;
  result.g := a.g div b;
  result.b := a.b div b;
end;

class operator TSixel.multiply(const a: TSixel; const b: single): TSixel;
begin
  result.r := trunc(a.r * b);
  result.g := trunc(a.g * b);
  result.b := trunc(a.b * b);
end;

initialization
  //{$ifdef FPC}
  //bt.loadFromImage('dog.jpg');
  //{$else}
  //bt.loadFromImage('../../../../FPC/ConsoleTest/dog.jpg');
  //{$endif}
  ////bt.print(0.2);
  //bt := bt.Permute([1, 2, 0]);
  ////bt.resize([400, 600, 3]);
  ////Mandel(bt.Data, bt.h(), bt.c());
  ////bt := bt.Permute([1, 2 , 0]);
  ////bt.SaveToImage('mandel.jpg');
  //repeat
  //  cursorHome();
  //  t := getTickCount64;
  //  printSixel(bt.data, bt.h, bt.c, true);
  //  write((GetTickCount64-t)/1000:1:3,'s');
  //  readln
  //until false;

end.

