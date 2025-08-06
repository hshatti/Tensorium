{$ifdef FPC}
{$mode delphi}
{$endif}
unit nXML;

interface
uses SysUtils, Generics.Collections, typInfo;

const SPACES:set of char = [' ', #13, #10, #9];
      LT    = '<';
      LT_CL = '</';
      GT    = '>';
      GT_CL = '/>';  // xhtml, and it's not really usesed
type

 { TNXml }

 TNXml = record
 type
   TNXmlArray = TArray<TNXml>;
   TTagCount = TDictionary<string, integer>;
 private
   class function getInnerText(aTag, aText:string):string;      static;
   class function detectArray(const aText:string):TNXMLArray;         static;
   class function getTagName(const aText:string; const offset: integer=1):string;static;
   class function isPossibleTag(const aText:string):boolean;  overload; static;
   class function asVarient(const XML:string):variant;          static;

 public
   valueType : TTypeKind;
   props : string;
   tagName, innerXML : string;
   //asInt    : NativeInt;
   //asFloat  : Double;
   //asString : string;
   innerValue : variant;
   Children : TNXmlArray;
   class function LoadFromFile(const fileName:string):TNXml;    static;
   function querySelector(selector : string):TNXml;
   function querySelectorAll(selector : string):TNXmlArray;
   class function parse(const XML: string):TNXml;               static;
   class operator initialize({$ifdef fpc}var{$else}out{$endif} dst : TNXml);
   class operator finalize(var dst : TNXml);
   property item[key:string]:TNxml read querySelector; default;
   class operator Implicit(const src:TNXML):Int64;
   class operator Implicit(const src:TNXML):Integer;
   class operator Implicit(const src:TNXML):single;
   class operator Implicit(const src:TNXML):double;
   class operator Implicit(const src:TNXML):boolean;
   class operator Implicit(const src:TNXML):string;

 end;


implementation

{ TNXml }

class function TNXml.getInnerText(aTag, aText: string): string;
var i, len, st, fn:integer;
  tag1, proptxt:string;
begin
  result :='';
  if (aText='') or (aTag='') then exit;
  i:=1;
  while aText[i] in SPACES do inc(i);
  if aText[i]<>LT then exit;
  len := length(aTag)+2; // assumably with '<' and ('>' or maybe ' ')
  tag1:=LowerCase(copy(aText, i, len));
  if not ((Pos(LT+lowercase(aTag),tag1)=1) and
    (tag1[len] in [' ', GT]))
    then
      exit;
  while not (aText[i] = GT) do inc(i);
  st :=i+1;

  i:= length(aText);
  while aText[i] in SPACES do dec(i);
  if aText[i]<>GT then exit;
  len := length(aTag)+3; // assumably with '</' and '>'
  fn := i-len+1;
  tag1:=LowerCase(copy(aText, fn, len));
  if (pos(LT_CL+aTag, tag1) = 1) and
    (tag1[len]=GT)
    then
      result := copy(aText, st, fn-st);
end;

class function TNXml.detectArray(const aText: string): TNXMLArray;
var i, j1, j2, len, st : integer;
  tag, tag1 , lText, props, inner: string;
  FTagCount : TTagCount;
begin
  result := nil;
  if (aText='') then exit;
  i:=1;
  //while (i<=length(aText)) and (aText[i] in SPACES) do inc(i);
  //if i=length(aText)then exit;
  //if (aText[i]<>LT) then exit;
  //st:=i+1;
  //while not(aText[i] in [' ', GT]) do inc(i);
  //
  lText:= lowercase(aText);
  //tag := lowerCase(copy(lText, st, i-st));
  //
  //len := length(Tag)+2;
  //i := st-1;
  while i< length(aText) do begin
    while (i<=length(aText)) and (aText[i] in SPACES) do inc(i);
    tag := getTagName(lText, i);
    len := length(tag)+2;
    tag1:= copy(lText, i, len);
    if not ((Pos(LT+tag,tag1)=1) and
      (tag1[len] in [' ', GT]))
      then
        exit;
    st:=i+len;
    while not (aText[i] = GT) do inc(i);
    props := copy(aText, st, i-st);
    st := i+1;
    inc(i);
    j1 := pos(LT+tag+' ', lText, i);
    j2 := pos(LT+tag+GT, lText, i);
    i := pos(LT_CL+tag+GT, lText, i);
    while (j1<>0) or (j2<>0) do begin // skip if asub tagname was found with the same parent tag
      if (j1<>0) and (j1<i) then begin
        i := j1+len;
        i := pos(LT_CL+tag+GT, lText, i);
        j1 := pos(LT+tag+' ', lText, i);
        inc(i, len+1);
        i := pos(LT_CL+tag+GT, lText, i);
      end else
        j1:=0;
      if (j2<>0) and (j2<i) then begin
        i := j2+len;
        i := pos(LT_CL+tag+GT, lText, i);
        j2 := pos(LT+tag+GT, lText, i);
        inc(i, len+1);
        i := pos(LT_CL+tag+GT, lText, i);
      end else
        j2 := 0;
    end;
    if i> 0 then begin
      setLength(result, length(result)+1);
      result[high(result)].tagName  := tag;
      result[high(result)].props    := props;
      inner                         := copy(aText, st, i-st);
      result[high(result)].innerXML := inner;
      inc(i, len+1);
      result[high(result)].children := detectArray(inner);
      result[high(result)].innerValue := asVarient(inner);
    end
  end;
end;

class function TNXml.getTagName(const aText: string; const offset: integer): string;
var i, st : integer;
begin
  result := '';
  if (aText='') then exit;
  i:=offset;
  while (i<=length(aText)) and (aText[i] in SPACES) do inc(i);
  if i>=length(aText)then exit;
  if (aText[i]<>LT) then exit;
  st:=i+1;
  while not(aText[i] in [' ', GT]) do inc(i);
  result := lowercase(copy(aText, st, i-st));
end;

class function TNXml.isPossibleTag(const aText: string): boolean;
var i:integer;
begin
  result := false;
  if (aText='') then exit();
  i:=1;
  while aText[i] in SPACES do inc(i);
  if i>length(aText) then exit();
  result := aText[i]=LT
end;

class function TNXml.LoadFromFile(const fileName: string): TNXml;
var f:textFile; line, XML:string;
begin
  assert(FileExists(fileName), FileName+': does not exist!');
  XML :='';
  assignFile(f, fileName);
  try
    reset(f);
    while not EOF(f) do begin
      readln(f, line);
      XML := XML + line
    end;
  finally
    closeFile(f)
  end;
  result := parse(XML)
end;

class function TNXml.parse(const XML: string): TNXml;
begin
  result.tagName  := getTagName(XML);
  result.innerXML := getInnerText(result.tagName, XML);
  result.Children := detectArray(XML);
  result.innerValue    := asVarient(result.innerXML);
end;

class function TNXml.asVarient(const XML: string): variant;
var
  int: Int64;
  flt : double;
  bool: boolean;
begin
  if TryStrToInt64(XML, int) then
    exit(int);
  if TryStrToFloat(XML, flt) then
    exit(flt);
  if TryStrToBool(XML, bool) then
    exit(bool);
  result:= XML
end;

class operator TNXml.Implicit(const src: TNXML): Int64;
begin
  result := src.innerValue;
end;

class operator TNXml.Implicit(const src: TNXML): Integer;
begin
  result := src.innerValue;
end;

class operator TNXml.Implicit(const src: TNXML): single;
begin
  result := src.innerValue;
end;

class operator TNXml.Implicit(const src: TNXML): double;
begin
  result := src.innerValue;
end;

class operator TNXml.Implicit(const src: TNXML): boolean;
begin
  result := src.innerValue;
end;

class operator TNXml.Implicit(const src: TNXML): string;
begin
  result := src.innerValue;
end;

class operator TNXml.initialize({$ifdef fpc}var{$else}out{$endif} dst: TNXml);
begin
  //FTagCount := TTagCount.Create;
end;

class operator TNXml.finalize(var dst: TNXml);
begin
  //freeAndNil(FTagCount)
end;

function TNXml.querySelector(selector: string): TNXml;
var
  i:integer;
begin
  for i:=0 to high(Children) do
   if children[i].tagName=lowerCase(selector) then
     exit(children[i]);
end;

function TNXml.querySelectorAll(selector: string): TNXmlArray;
var
  i:integer;
begin
  result := nil;
  for i:=0 to high(Children) do
   if children[i].tagName=lowerCase(selector) then
     insert(children[i], result, length(children));
end;

  var xml : TNXml;
    str : string;
initialization

end.
