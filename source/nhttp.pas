unit nHttp;
{$ifdef FPC}
{$mode Delphi}
{$modeswitch advancedrecords}
{$endif}
interface

uses
  Classes, SysUtils
  {$ifdef FPC}
  , opensslsockets
  , fphttpclient
  , zipper
  {$else}
  , Net.httpClient
  , Zip
  {$endif}
  ;

   const
     clearLine = #$1B'[0K';
     zipExt : array of string =['.zip'];
type

  { TNHttp }

  {$ifdef FPC}
  TNHttp = class(TfpHttpClient)
    procedure FReceiveData(Sender: TObject; const AContentLength, AReadCount: Int64);
  {$else}
  TNHttp = class(THttpClient)
    fs : TFileStream;
    procedure FReceiveData(const Sender: TObject; AContentLength, AReadCount: Int64; var AAbort: Boolean);
  {$endif}
    constructor Create;
    procedure Download(const aURL:string; toFile:string ='');

  end;

  procedure unzip(const zipfile:string);
var
  http: TNHttp;
implementation

procedure unzip(const zipfile:string);
begin
  {$ifdef FPC}
  zipper.TUnZipper.UnZip(zipfile);
  {$else}
  zip.TZipFile.ExtractZipFile(zipFile, '.')
  {$endif}

end;


{ TNHttp }

{$ifdef FPC}
procedure TNHttp.FReceiveData(Sender: TObject; const AContentLength, AReadCount: Int64);
{$else}

procedure TNHttp.FReceiveData(const Sender: TObject; AContentLength, AReadCount: Int64; var AAbort: Boolean);
{$endif}
begin
  if not IsConsole then exit;
  if AContentLength>0 then
    write(clearLine, trunc(100*AReadCount/AContentLength), '% ', AReadCount, '/', AContentLength,' bytes Downloaded', #13)
  else
    write(clearLine, AReadCount, ' bytes Downloaded', #13)
end;

function ExtractURLName(const aURL:string):string;
var i:integer;
begin
  I := aURL.LastDelimiter('/');
  result := copy(aURL, I+2)
end;

procedure TNHttp.Download(const aURL:string; toFile: string);
var fs :TFileStream;
  ext : string;
  i:integer;
begin
  if toFile='' then
    toFile := ExtractURLName(aURL);
  fs := TFileStream.Create(toFile, fmCreate);
  try
  get(aURL, fs);
  finally
  freeAndNil(fs);
  end;
  ext :=ExtractFileExt(toFile);
  for i:=0 to high(zipExt) do
    if lowerCase(ext)=zipExt[i] then begin
      unzip(toFile);
      DeleteFile(toFile);
      break
    end;

end;

constructor TNHttp.Create;
begin

  {$ifdef FPC}
  inherited Create(nil);
  OnDataReceived := FReceiveData;
  {$else}
  self := TNHttp(inherited Create);
  OnReceiveData := FReceiveData;
  {$endif}
end;


initialization
  http := TNHttp(TNHttp.Create);

finalization
  http.Free;
end.

