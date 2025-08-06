unit OpenCLHelper;
{$H+}
{$ifdef FPC}
  {$ifopt D+}
  {$define DEBUG}
  {$endif}
  {$mode delphi}
{$endif}
{$pointermath on}
{$IFDEF MSWINDOWS}

{$endif}
{$H+}
{.$define debug}
interface

uses
  Classes, SysUtils,
  {$if defined(DARWIN) or defined(MACOS)}
  CL
  {$else}
  OpenCL
  {$endif}
  ;

const
  cInfoSize=$7fff;
// missing CL constants
  MAX_EVENTS_COUNT = $100;
  CL_QUEUE_SIZE    = $1094;
  CL_KERNEL_PRIVATE_MEM_SIZE                  = $11B4;


type
{$if not declared(size_t)}
  psize_t = ^size_t;
  size_t  = NativeUInt;
{$endif}

  TCLDeviceType=(
    dtNone = 0 ,
    dtDefault = CL_DEVICE_TYPE_DEFAULT,
    dtCPU = CL_DEVICE_TYPE_CPU,
    dtGPU = CL_DEVICE_TYPE_GPU,
    dtACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
    dtALL = CL_DEVICE_TYPE_ALL
  );
  TCLMemAccess=(
    maReadWrite = CL_MEM_READ_WRITE         ,
    maWrite     = CL_MEM_WRITE_ONLY         ,
    maRead      = CL_MEM_READ_ONLY          ,
    maUseHost   = CL_MEM_USE_HOST_PTR       ,
    maAllocHost = CL_MEM_ALLOC_HOST_PTR     ,
    maCopyHost  = CL_MEM_COPY_HOST_PTR
  );

  PWorkSize=^TWorkSizes;
  TWorkSizes=array [0..2] of size_t;

  TCLKernelArgInfo=record
    ArgName:array[0..cInfoSize-1] of ansichar;
    ArgType:array[0..cInfoSize-1] of ansichar;
    ArgSize : integer;
    ArgAccess:cl_uint;
    ArgAddress:cl_uint;
    ArgTypeQualifier:cl_bitfield;
  end;

  PComplexF=^TComplexF;
  TComplexF=record
    re, im:single
  end;

  PComplexD=^TComplexD;
  TComplexD=record
    re, im:Double
  end;

  TCLKernelInfo = record
    KernelName:array[0..cInfoSize] of ansichar;
    KernelGlobalWorkSize:array[0..2] of size_t;
    KernelWorkGroupSize:size_t;
    KernelSIMDSize : size_t;
    KernelLocalMemSize:cl_ulong;
    KernelPrivateMemSize:cl_ulong;
    KernelArgCount:cl_uint;
    KernelArgs:array of TCLKernelArgInfo;
  end;

  { TOpenCl }

  TOpenCL=class
  type
    TDeviceStr =array[0..cInfoSize-1] of ansichar;
  class var
    Platforms:array of cl_platform_id;
    PlatformCount:cl_uint;
    PlatformsNames : array of ansistring;
    ExtensionStrs : array of ansistring;
  private
    FSrc:TStringList;
    FActivePlatformId: integer;
    FActiveDeviceId: integer;
    FActiveKernelId: integer;
    FActivePlatform: cl_platform_id;
    FActiveDevice: cl_device_id;
    FActiveKernel: cl_kernel ;
    FActiveKernelInfo: TCLKernelInfo;
    FDeviceCount:cl_uint;
    FKernelCount:cl_uint;
    FDevices:array of cl_device_id;  // array of pointer to opaque record
    FDevicesType:array of TCLDeviceType;
    FContext:cl_context;
    FKernels:TArray<cl_kernel>;
    FKernelsInfo: TArray<TCLKernelInfo>;
    FBinaries :TArray<RawByteString>;
    FAutoEnumKernels : boolean; // if true, will enumerate all kernels in the program into FKernels upon Build, otherwise, kernels has to be manually obtained
    FDeviceType: TCLDeviceType;
    FProgramSource: ansistring;
    cinfo:TDeviceStr;
    FWorkItemDimensions: integer;
    FGlobalOffsets:TWorkSizes;
    FGlobalWorkGroupSizes:TWorkSizes;
    FLocalWorkGroupSizes:TWorkSizes;
    FLocalWorkGroupSizesPtr  : pointer;

    FGlobalMemSize:cl_ulong;
    FLocalMemSize:cl_ulong;
    FGlobalCacheSize:cl_ulong;
    FExecCaps:cl_device_exec_capabilities;
    FMaxWorkItemDimensions:cl_uint;
    FMaxWorkGroupSize:size_t;
    FMaxWorkItemSizes:TWorkSizes;

    FMaxComputeUnits:cl_uint;
    FMaxMemAllocSize:cl_ulong;
    FMaxFrequency:cl_uint;
    FSIMDWidth : integer;
    FDeviceBuiltInKernels:TDeviceStr;
    FIsBuilt:boolean;
    FProgram:cl_program;
    FBuildStatus:cl_build_status ;
    FBuildLog:ansistring;
    FCallParams:array[0..$ff] of cl_mem;
    FParamSizes:array[0..$ff] of size_t;

    FDevsTypeStr:ansistring;
    FSharedMemory:boolean;
    function GetDevice(index: cl_uint): cl_device_id;
    function getQueueInOrder: boolean;
    procedure SetActiveDeviceId(AValue: integer);
    procedure SetActiveKernelId(AValue: integer);
    procedure SetActivePlatformId(AValue: integer);
    procedure SetDeviceType(AValue: TCLDeviceType);
    function getCL_Device_Type(const dt:TClDeviceType):cl_uint;
    procedure SetGlobalWorkGroupSizes(AValue: TWorkSizes);overload;
    procedure SetProgramSource(AValue: ansistring);
    procedure SetQueueInOrder(AValue: boolean);
    procedure SetWorkItemDimensions(AValue: integer);
  public
    CLDeviceCVersion:TDeviceStr;
    CLDeviceDriver:TDeviceStr;
    FErr:cl_int;
    FQueue:cl_command_queue;
    AutoCalcLocalSizes : boolean;
    ItemSize : size_t;
    //events    : array[0..MAX_EVENTS_COUNT-1] of cl_event;
    //eventsCount : cl_uint;
    class function initPlatforms():cl_int;static;
    class function getPlatforms:cl_uint;static;
    class function PlatformName(Index: integer): ansistring;static;
    class function getDevices(const platformId: integer):TArray<cl_device_id>;static;
    class function getDeviceName(const device:cl_device_id):ansistring; static;
    class function getDeviceExtensionStr(const device:cl_device_id):ansistring; static;
    class function getDeviceTypeName(const device:cl_device_id):ansistring;static;
    class function CalcLocalSize(const aSize: SizeInt):SizeInt; inline;static;
    procedure CheckError(const msg:string=''); inline;
    constructor Create(deviceType:TCLDeviceType=dtGPU); virtual;
    destructor Destroy;override;
    procedure SetGlobalWorkGroupSizes(const x: size_t; const y: size_t=0; const z: size_t=0); overload;
    procedure SetLocalWorkGroupSizes(const x: size_t; const y: size_t=0; const z: size_t=0);
    procedure SetParamElementSizes(paramSizes: array of size_t);
    function DevicesTypeStr:ansistring;
    function getQueueSize():cl_uint;
    function getQueueRefCount():cl_uint;
    procedure SetGlobalOffsets(const x: size_t; y: size_t=0; z: size_t=0);
    function CleanUp(const keepContext: boolean=false): boolean;
    function ProcessorsCount:integer;
    function ProcessorsFrequency:integer;
    property DeviceType:TCLDeviceType read FDeviceType write SetDeviceType;
    property Device[index:cl_uint]:cl_device_id read GetDevice;
    property ActivePlatformId:Integer read FActivePlatformId write SetActivePlatformId;
    property ActiveDeviceId:Integer read FActiveDeviceId write SetActiveDeviceId;
    property ProgramSource:ansistring read FProgramSource write SetProgramSource;

    property LocalMemSize:cl_ulong                 read FLocalMemSize ;
    property ExecCaps:cl_device_exec_capabilities  read FExecCaps;
    property MaxWorkItemDimensions:cl_uint         read FMaxWorkItemDimensions;
    property MaxWorkGroupSize:size_t               read FMaxWorkGroupSize;
    property MaxWorkItemSizes:TWorkSizes           read FMaxWorkItemSizes;
    property MaxComputeUnits:cl_uint               read FMaxComputeUnits;
    property MaxMemAllocSize:cl_ulong              read FMaxMemAllocSize;
    property MaxFrequency:cl_uint                  read FMaxFrequency;
    property ActivePlatform : cl_platform_id read FActivePlatform;
    property ActiveDevice : cl_device_id read FActiveDevice;
    property ActiveContext : cl_context read FContext;
    property ActiveQueue : cl_command_queue read FQueue;
    property ActiveKernel : cl_kernel read FActiveKernel;
    property ActiveKernelInfo : TCLKernelInfo read FActiveKernelInfo;
    property ExecCapabilities : cl_device_exec_capabilities read FExecCaps;

    function DeviceName(Index: integer): ansistring;
    function DeviceCount:integer;
    function Build(const params:ansistring=''; const withKernels:TArray<ansistring> = nil):boolean;
    function loadBinary(const bin:RawByteString; const withKernels:TArray<ansistring> = nil):boolean;
    function readLog:ansistring;
    property BuildLog:ansistring read FBuildLog;
    property LastError:cl_int read FErr;
    property Kernels:TArray<cl_kernel> read FKernels;
    function getKernels(const Names:TArray<ansiString>):TArray<cl_kernel>;
    function KernelCount:integer;
    function KernelInfo(index:integer):TCLKernelInfo;
    property GlobalWorkGroupSizes:TWorkSizes read FGlobalWorkGroupSizes;
    property LocalWorkGroupSizes:TWorkSizes read FLocalWorkGroupSizes;
    property LocalWorkGroupSizesPtr: pointer read FLocalWorkGroupSizesPtr;
    property GlobalCacheSize : cl_ulong read FGlobalCacheSize;
    property GlobalOffsets : TWorkSizes read FGlobalOffsets;
    function CanExecuteNative:boolean;
    procedure LoadFromFile(FileName:ansistring);
    //function KernelArgs(index:integer):TCLKernelArgInfo;
    property DeviceBuiltInKernels : TDeviceStr read FDeviceBuiltInKernels;
    property ActiveKernelId:Integer read FActiveKernelId write SetActiveKernelId;
    property WorkItemDimensions:integer read FWorkItemDimensions write SetWorkItemDimensions;
    property isBuilt:boolean read FIsBuilt;
    property queueInOrder: boolean read getQueueInOrder write SetQueueInOrder;
    function createDeviceBuffer(const aByteSize:size_t; const aAccess:TCLMemAccess = maReadWrite; const fromHostMem:Pointer=nil):cl_mem;
    procedure freeDeviceBuffer(var aMem:cl_mem);
    procedure readBuffer(const clMem:cl_mem; const bufferSize:size_t; const buffer:pointer; const offset:SizeInt = 0; aQueue: cl_command_queue=nil );
    procedure writeBuffer(const clMem: cl_mem; const bufferSize: size_t; const buffer: pointer; const offset:SizeInt = 0; aQueue: cl_command_queue=nil);
    procedure CallKernel(const Index: integer; const GlobalWorkGroups: TArray<size_t> = nil; const params: TArray<Pointer> = nil; const waitEvents:TArray<cl_event> = nil; outEvent:pcl_event = nil);    overload;
    procedure waitForEvents(const N :longword; const events:pcl_event);
    procedure freeEvents(const N :longword; const events:pcl_event);
    procedure finish(); inline;
    procedure DoActiveDeviceChange(const newDevice:cl_device_id; const newQueue:cl_command_queue); virtual;


  end;
{$if not declared(clGetKernelArgInfo)}
    cl_kernel_arg_info                         = cl_uint;

  const
    CL_KERNEL_ARG_ADDRESS_QUALIFIER = $1196;
    CL_KERNEL_ARG_ACCESS_QUALIFIER  = $1197;
    CL_KERNEL_ARG_TYPE_NAME         = $1198;
    CL_KERNEL_ARG_TYPE_QUALIFIER    = $1199;
    CL_KERNEL_ARG_NAME              = $119A;
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = $11B3;
    CL_DEVICE_BUILT_IN_KERNELS      = $103f;
    CL_DEVICE_HOST_UNIFIED_MEMORY   = $1035;


  function clGetKernelArgInfo (kernel:cl_kernel;
                     arg_indx:cl_uint;
                     param_name:cl_kernel_arg_info;
                     param_value_size:size_t;
                     param_value:pointer;
                     param_value_size_ret:PUintPtr):cl_int;winapi;external OpenCLlib;

  function clCreateKernelsInProgram(
    _program      : cl_program;
    num_kernels   : cl_uint;
    kernels       : Pcl_kernel;
    var num_ret   : cl_uint
    ): cl_int; WINAPI; external OpenCLlib name 'clCreateKernelsInProgram';
{$endif}

implementation

(*
// CONSTANTS
// The source code of the kernel is represented as a ansistring
// located inside file: "fft1D_1024_kernel_src.cl". For the details see the next listing.

// Looking up the available GPUs
case ComboBox1.ItemIndex of
  0:deviceType:=CL_DEVICE_TYPE_GPU;
  1:deviceType:=CL_DEVICE_TYPE_CPU;
  2:deviceType:=CL_DEVICE_TYPE_DEFAULT;
end;


ret:=clGetDeviceIDs(nil, deviceType, 0, nil, @num);
if ret<>CL_SUCCESS then raise Exception.create('Cannot list Processors');

setLength(devices,num);
       //cl_device_id devices[1];
ret:=clGetDeviceIDs(nil, deviceType, num, @devices[0], nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot get ALL Device id ');
ListBox1.Items.Clear;
for i:=0 to num -1 do begin
  clGetDeviceInfo(devices[i],CL_DEVICE_NAME,256,@deviceInfo[0],retSize);
  ListBox1.Items.add(deviceInfo);
end;

// create a compute context with GPU device
context := clCreateContextFromType(nil, deviceType, nil, nil, ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot Create context from GPU Type');

// create a command queue
ret:=clGetDeviceIDs(nil, deviceType, 1, @devices[0], nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot get Default Device');

ret:=clGetDeviceInfo(devices[0],CL_DEVICE_MAX_COMPUTE_UNITS,256,@deviceInfo[0],retSize);
ListBox1.Items.Add(IntToStr(PLongWord(@deviceInfo)^)+' Units');
ret:=clGetDeviceInfo(devices[0],CL_DEVICE_MAX_CLOCK_FREQUENCY,256,@deviceInfo[0],retSize);
ListBox1.Items[ListBox1.Items.count-1]:=ListBox1.Items[ListBox1.Items.count-1]+'@'+IntToStr(PLongWord(@deviceInfo)^)+'Mhz';



queue := clCreateCommandQueue(context, devices[0], 0{props}, ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot create command queue');

t:=GetTickCount64;
bmp.BeginUpdate();

// allocate the buffer memory objects
 memobjs[0]:=  clCreateBuffer(context, CL_MEM_WRITE_ONLY , 4*w*h, nil, ret);
 if ret<>CL_SUCCESS then raise Exception.create('Cannot create ReadMem');
 //memobjs[1]:=  clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(single) * 2 * NUM_ENTRIES, nil, ret);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot create WriteMem');
// cl_mem memobjs[0] = // FIXED, SEE ABOVE
// cl_mem memobjs[1] = // FIXED, SEE ABOVE

// create the compute program
// const ansichar* fft1D_1024_kernel_src[1] = {  };
prog := clCreateProgramWithSource(context, 1, PPAnsiCHAR(@src), nil, ret);

if ret<>CL_SUCCESS then raise Exception.create('Cannot create ProgramWithSource');

// build the compute program executable
ret:=clBuildProgram(prog, 0, nil, nil, nil, nil);

if ret<>CL_BUILD_SUCCESS then begin
  clGetProgramBuildInfo(prog,devices[0],CL_PROGRAM_BUILD_LOG,256,@buildLog[0],retSize);
  raise Exception.CreateFmt('Cannot Build executable message:'#13#10'[%s]',[buildLog]);
end;
// create the compute kernel


kernel := clCreateKernel(prog, 'render', ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot create Kernel');
// set the args values

//size_t local_work_size[1] = { 256 };

ret:=clSetKernelArg(kernel, 0, sizeof(cl_mem), @memobjs[0]);
if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[0]');
ret:=clSetKernelArg(kernel, 1, sizeof(max_iteration), @max_iteration);
if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[1]');
//ret:=clSetKernelArg(kernel, 1, sizeof(cl_mem), @memobjs[1]);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[1]');
//ret:=clSetKernelArg(kernel, 2, sizeof(single)*(local_work_size[0] + 1) * 16, nil);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[2]');
//ret:=clSetKernelArg(kernel, 3, sizeof(single)*(local_work_size[0] + 1) * 16, nil);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[3]');
//
// create N-D range object with work-item dimensions and execute kernel
//size_t global_work_size[1] = { 256 };

//global_work_size[0] := NUM_ENTRIES;
//local_work_size[0] := 64; //Nvidia: 192 or 256

ret:=clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset, global_work_size, nil, 0, nil, nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot Enqueue ND Range kernel');

clEnqueueReadBuffer(queue,memobjs[0],cl_false,{offset in byte }0,w*h*4{size in byte},bmp.ScanLine[0],0,nil,nil);
//clFlush(queue);
clFinish(queue);
ListBox1.Items.Add(format(' -Rendering took %d MilliSeconds',[GetTickCount64-t]));
bmp.EndUpdate();
Image1.picture.Graphic:=bmp ;

clReleaseMemObject(memobjs[0]);
clReleaseCommandQueue(queue);
clReleaseContext(context);
clReleaseKernel(kernel);
clReleaseProgram(prog);
*)

{ TOpenCl }

procedure TOpenCL.SetDeviceType(AValue: TCLDeviceType);
var wasBuilt:boolean;
begin
  if FDeviceType=AValue then Exit;
  if FDeviceCount=0 then raise Exception.Create('No Devices found!');
  FDeviceType:=AValue;
  FActiveDeviceId:=-1;
  wasBuilt:=FIsBuilt;
  SetActivePlatformId(FActivePlatformId);
  if wasBuilt then
    Build
end;

procedure TOpenCL.CheckError(const msg: string);
begin
  {$ifdef DEBUG}
  if FErr<>CL_SUCCESS then
    //writeln(msg, clErrorText(FErr));
    raise Exception.Create(clErrorText(FErr)+msg);
  {$endif}
end;

function TOpenCL.getCL_Device_Type(const dt: TClDeviceType): cl_uint;
begin
  result := 0;
  case dt of
    dtDefault :result:= CL_DEVICE_TYPE_DEFAULT;
    dtCPU :result:=CL_DEVICE_TYPE_CPU;
    dtGPU :result:=CL_DEVICE_TYPE_GPU;
    dtACCELERATOR :result:=CL_DEVICE_TYPE_ACCELERATOR;
    dtALL :result:=CL_DEVICE_TYPE_ALL
  end;
end;

procedure TOpenCL.SetGlobalWorkGroupSizes(AValue: TWorkSizes);
begin
//  if FGlobalWorkGroupSize=AValue then Exit;
  FGlobalWorkGroupSizes:=AValue;
end;

procedure TOpenCL.SetProgramSource(AValue: ansistring);
var i:integer;
begin
  if FProgramSource=AValue then Exit;
  if Assigned(FKernels) then
    for i:=0 to FKernelCount-1 do
      clReleaseKernel(FKernels[i]);
  setLength(FKernels,0);
  if Assigned(FProgram) then clReleaseProgram(FProgram);
  FIsBuilt:=false;
  FProgramSource:=AValue;
end;

procedure TOpenCL.SetQueueInOrder(AValue: boolean);
var oldProp : cl_command_queue_properties;
begin
  clSetCommandQueueProperty(ActiveQueue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, cl_bool(not AValue), oldProp);
end;

procedure TOpenCL.SetWorkItemDimensions(AValue: integer);
begin
  if FWorkItemDimensions=AValue then Exit;
  FWorkItemDimensions:=AValue;
end;

class function TOpenCL.initPlatforms: cl_int;
var
  i: Integer;
begin
  result := clGetPlatformIDs(0,nil,@PlatformCount);
  //assert(FPlatformCount>0, 'No OpenCL supported platforms found!');
  if result=CL_SUCCESS then
    if PlatformCount>0 then begin
      SetLength(Platforms, PlatformCount);
      result := clGetPlatformIDs(PlatformCount, @Platforms[0], nil);
      SetLength(PlatformsNames, PlatformCount);
      setLength(ExtensionStrs, PlatformCount);
      for i:=0 to PlatformCount-1 do
        PlatformsNames[i] := PlatformName(i);
    end;
end;

class function TOpenCL.getDevices(const platformId: integer): TArray<cl_device_id>;
var FErr, i : cl_int;
  buffCnt :Size_t;
  deviceCount:cl_uint;
begin
  assert((platformid >=0) and (platformId< PlatformCount));
  FErr := clGetDeviceIDs(Platforms[platformId], CL_DEVICE_TYPE_ALL, 0, nil, @DeviceCount);
  setLength(result, DeviceCount);
  FErr:=clGetDeviceIDs(Platforms[platformId], CL_DEVICE_TYPE_ALL, deviceCount, @result[0], nil);
end;

class function TOpenCL.getDeviceName(const device: cl_device_id): ansistring;
var cname : array[0..cInfoSize-1] of ansichar;
   buffCnt :size_t;
begin
  clGetDeviceInfo(device, CL_DEVICE_NAME, cInfoSize, @cname[0], buffCnt);
  result := ansistring(cname);
end;

class function TOpenCL.getDeviceExtensionStr(const device: cl_device_id): ansistring;
var cname : array[0..cInfoSize-1] of ansichar;
   buffCnt :size_t;
begin
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, cInfoSize, @cname, buffCnt);
  result := ansistring(cname);
end;

class function TOpenCL.getDeviceTypeName(const device: cl_device_id): ansistring;
var
    FErr:cl_int;
    dt :cl_uint;
    buffCnt : size_t;
begin
  FErr:=clGetDeviceInfo(device ,CL_DEVICE_TYPE_INFO ,SizeOf(size_t),@dt, buffCnt);
  case dt of
    CL_DEVICE_TYPE_DEFAULT:
      result := 'DEFAULT' ;
    CL_DEVICE_TYPE_CPU:
      result := 'CPU' ;
    CL_DEVICE_TYPE_GPU:
      result := 'GPU' ;
    CL_DEVICE_TYPE_ACCELERATOR:
      result := 'ACCELERATOR' ;
    else
      result := format('unknown [%d]', [dt])
    end;
end;

class function TOpenCL.CalcLocalSize(const aSize: SizeInt): SizeInt;
begin
  if ASize=0 then exit(0);
  if aSize mod 13 = 0 then exit(13);
  if aSize mod 11 = 0 then exit(11);
  if aSize mod 8  = 0 then exit(8); // Optimal?
  if aSize mod 7  = 0 then exit(7);
  if aSize mod 6  = 0 then exit(6);
  if aSize mod 5  = 0 then exit(5);
  if aSize mod 4  = 0 then exit(4);
  if aSize mod 3  = 0 then exit(3);
  if aSize mod 2  = 0 then exit(2);
  result := 1;
end;

constructor TOpenCL.Create(deviceType: TCLDeviceType);
var i:integer;
begin
  FDeviceType:=deviceType;
  FProgram:=nil;
  FAutoEnumKernels := true;
  AutoCalcLocalSizes := true;
  itemSize := 4;
  PlatformCount:=0;
  FillChar(FParamSizes,sizeof(FParamSizes),0);
  for i:=0 to high(FGlobalOffsets) do FGlobalOffsets[i]:=0;
  FErr:=clGetPlatformIDs(0,nil,@PlatformCount); checkError();
  assert(PlatformCount>0, 'No OpenCL supported platforms found!');

  if FErr=CL_SUCCESS then
    if PlatformCount>0 then begin
      SetLength(Platforms,PlatformCount);
      FErr:=clGetPlatformIDs(PlatformCount,@Platforms[0],nil);CheckError();
      FSrc:=TStringList.Create;
      FActivePlatformId:=$7fffffff;
      FActiveDeviceId:=$7fffffff;
      FWorkItemDimensions:=1;
      SetActivePlatformId(0);
    end;
end;

destructor TOpenCL.Destroy;
begin
  CleanUp(false);
  FSrc.Free;
  inherited Destroy;
end;

procedure TOpenCL.SetGlobalWorkGroupSizes(const x: size_t; const y: size_t; const z: size_t);
var i, workItemSize:integer;
begin
  FGlobalWorkGroupSizes[0]:=x;
  FGlobalWorkGroupSizes[1]:=y;
  FGlobalWorkGroupSizes[2]:=z;
  if AutoCalcLocalSizes then begin
    FLocalWorkGroupSizes[0]  :=  CalcLocalSize(x);
    FLocalWorkGroupSizes[1]  :=  CalcLocalSize(y);
    FLocalWorkGroupSizes[2]  :=  CalcLocalSize(z);
    // todo setGlobalWorksize : 4 is the size of single make it variable
  end else
    FLocalWorkGroupSizesPtr := nil;
  FillChar(FGlobalOffsets,sizeof(FGlobalOffsets),0);
  if z>0 then begin
    FWorkItemDimensions:=3;
  end else
  if y>0 then begin
    FWorkItemDimensions:=2;
  end else
    FWorkItemDimensions:=1;
  workItemSize := FLocalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    workItemSize:=workItemSize*FLocalWorkGroupSizes[i];
  if workItemSize*itemsize>FMaxWorkGroupSize then
    FLocalWorkGroupSizesPtr:=nil
end;

procedure TOpenCL.SetLocalWorkGroupSizes(const x: size_t; const y: size_t;
  const z: size_t);
begin
  FLocalWorkGroupSizes[0]:=x;
  FLocalWorkGroupSizes[1]:=y;
  FLocalWorkGroupSizes[2]:=z;
  FLocalWorkGroupSizesPtr := @FLocalWorkGroupSizes[0];
end;

procedure TOpenCL.SetParamElementSizes(paramSizes: array of size_t);
var i:integer;
begin
  for i:=0 to High(paramSizes) do
    FParamSizes[i]:=paramSizes[i]
end;

function TOpenCL.DevicesTypeStr: ansistring;
begin
  result:=FDevsTypeStr;
end;

function TOpenCL.getQueueSize(): cl_uint;
var r: size_t;
begin
  FErr := clGetCommandQueueInfo(ActiveQueue, CL_QUEUE_REFERENCE_COUNT, sizeOf(cl_uint), @result, r) ; CheckError();
end;

function TOpenCL.getQueueRefCount(): cl_uint;
var r:size_t;
begin
  FErr := clGetCommandQueueInfo(ActiveQueue, CL_QUEUE_SIZE, sizeOf(cl_uint), @result, r) ; CheckError();
end;

procedure TOpenCL.SetGlobalOffsets(const x: size_t; y: size_t; z: size_t);
begin
  FGlobalOffsets[0]:=x;
  FGlobalOffsets[1]:=y;
  FGlobalOffsets[2]:=z;
end;

function TOpenCL.CleanUp(const keepContext:boolean): boolean;
var i:integer;
begin
  try
    for i:=0 to High(FKernels) do begin
      clReleaseKernel(FKernels[i]);  CheckError();
      FKernels[i] := nil
    end;

    if FProgram<>nil then begin
      clReleaseProgram(FProgram);CheckError();
      FProgram := nil
    end;

    if FQueue<>nil then begin
      clReleaseCommandQueue(FQueue);CheckError();
      FQueue := nil
    end;

    if not keepContext then if
      FContext<>nil then begin
        clReleaseContext(FContext);CheckError();
        FContext:=nil;
      end;

    FIsBuilt:=false;
    result:=true
  except on E:Exception do
    begin
      result:=false
    end;

  end;
end;

function TOpenCL.ProcessorsCount: integer;
begin
  result:=FMaxComputeUnits;
end;

function TOpenCL.ProcessorsFrequency: integer;
begin
  result:=FMaxFrequency;
end;

class function TOpenCL.PlatformName(Index: integer): ansistring;
var cname : array [0..cInfoSize-1] of ansichar;
    buffCount:size_t;
begin
  clGetPlatformInfo(Platforms[Index], CL_PLATFORM_NAME, cInfoSize, @cname[0], buffCount);
  result:=cname;
end;

function TOpenCL.DeviceName(Index: integer): ansistring;
var sz:size_t;
begin
  clGetDeviceInfo(FDevices[Index],CL_DEVICE_NAME,cInfoSize,@cinfo[0], sz);
  result:=cinfo;
end;

function TOpenCL.DeviceCount: integer;
begin
  result:=FDeviceCount
end;

function TOpenCL.Build(const params: ansistring; const withKernels: TArray<ansistring>): boolean;
const
  {$if sizeOf(SizeInt)=8}
  nInt='long';
  {$else}
  nInt='int';
  {$endif}
var
  src, par : PAnsiChar;
  sz       : size_t;
  szui     : cl_uint;
  binSizes : TArray<size_t>;
begin
  result:=False;
  src:=PAnsiChar(FProgramSource);
{$ifdef _DEBUG}
  par:=PAnsiCHar('-cl-kernel-arg-info -cl-std=CL3.0 -cl-opt-disable -Werror -Dn_int='+nInt+' -g '+params);
{$else}
  par:=PAnsiCHar('-cl-kernel-arg-info -cl-std=CL2.0 -cl-fast-relaxed-math -Werror -Dn_int='+nInt+' -cl-mad-enable '+params);
{$endif}
  FProgram:=clCreateProgramWithSource(FContext, 1, @src, nil, FErr);CheckError();
  FErr:=clBuildProgram(Fprogram, FDeviceCount, @FDevices[0], par, nil, nil);
  FErr:=clGetProgramBuildInfo(FProgram, FActiveDevice,CL_PROGRAM_BUILD_STATUS,cInfoSize,@FBuildStatus, sz);CheckError();
  //assert(FBuildStatus = CL_BUILD_SUCCESS, 'Error building program '+intToStr(FBuildStatus));
  FErr:=clGetProgramBuildInfo(FProgram, FActiveDevice,CL_PROGRAM_BUILD_LOG,cInfoSize,@cinfo[0], sz);CheckError();
  FBuildLog:=trim(system.copy(cinfo,0, sz));
  assert(FBuildStatus=CL_BUILD_SUCCESS, '[OpenCL] : cannot compile tensor kernels :' + sLineBreak+ FBuildlog);
  if FBuildStatus= CL_BUILD_SUCCESS then begin
{$ifdef DEBUG}
   if FBuildlog<>'' then begin
      writeln(ErrOutput, FBuildLog);
      readln
    end;
{$endif}
    setLength(FBinaries, FDeviceCount);
    setLength(binSizes, FDeviceCount);
    FErr := clGetProgramInfo(Fprogram, CL_PROGRAM_BINARY_SIZES, FDeviceCount*sizeOf(pointer), pointer(binSizes), sz); checkError();
    for sz:=0 to FDeviceCount-1 do
      setLength(FBinaries[sz], binSizes[sz]);
    FErr := clGetProgramInfo(Fprogram, CL_PROGRAM_BINARIES, FDeviceCount*sizeOf(pointer), pointer(FBinaries), sz); CheckError();
    if not assigned(withKernels) and FAutoEnumKernels then begin
      FErr:=clCreateKernelsInProgram(FProgram,0, nil, FKernelCount);CheckError();
      setLength(FKernels,FKernelCount);
      FErr:=clCreateKernelsInProgram(FProgram,FKernelCount, @FKernels[0], szui);CheckError();
      FActiveKernelId:=-1;
      if FKernelCount>0 then
        SetActiveKernelId(0);
    end else if assigned(withKernels) then
      FKernels := getKernels(withKernels);
    FIsBuilt:=True;
    Result:=True
  end;
//  if cinfo='' then cinfo:='Success';
end;

function TOpenCL.loadBinary(const bin: RawByteString; const withKernels: TArray<
  ansistring>): boolean;
var
  sz:size_t;
  szui : cl_uint;
  binPtr:pointer;
  binStatus: cl_int;
begin
  result := false;
  sz := length(bin);
  binPtr := @bin[1];
  FProgram := clCreateProgramWithBinary(FContext, 1, @FActiveDevice, @sz, @binPtr, binStatus, FErr); CheckError();
  result := binStatus=CL_SUCCESS;
  if not assigned(withKernels) and FAutoEnumKernels then begin
    FErr:=clCreateKernelsInProgram(FProgram,0, nil, FKernelCount);CheckError();
    setLength(FKernels, FKernelCount);
    FErr:=clCreateKernelsInProgram(FProgram,FKernelCount, @FKernels[0], szui);CheckError();
    FActiveKernelId:=-1;
    if FKernelCount>0 then
      SetActiveKernelId(0);
  end else if assigned(withKernels) then
    FKernels := getKernels(withKernels);
  FIsBuilt := true;
  result := true
end;

function TOpenCL.readLog: ansistring;
var
  sz : size_t;
begin
  FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_LOG,cInfoSize,@cinfo[0], sz);CheckError();
  result := cinfo
end;

function TOpenCL.getKernels(const Names: TArray<ansiString>): TArray<cl_kernel>;
var i:SizeInt;res:cl_int;
begin
  assert(assigned(FProgram), 'No program was found, build a program from source 1st!.');
  setLength(result, length(Names));
  for i:=0 to high(Names) do begin
    result[i] := clCreateKernel(FProgram, PAnsiChar(names[i]), FErr);
    CheckError('Kernel ['+Names[i]+'] not found!');
  end;
  FKernelCount:=length(result);
  FKernels:=result;
  setLength(FKernelsInfo, length(names));
  for i:=0 to high(names) do
    FKernelsInfo[i] := KernelInfo(i);
  FSIMDWidth := FKernelsInfo[0].KernelSIMDSize;
end;

function TOpenCL.KernelCount: integer;
begin
  result:=FKernelCount;
end;

function getTypeSize(const typeStr:string):integer;
var vectorSize:integer;
begin
  vectorSize := 1;
  if pos('*', typeStr)>0  then exit(sizeOf(cl_mem));
  if pos('2', typeStr)>0  then vectorSize := 2;
  if pos('4', typeStr)>0  then vectorSize := 4;
  if pos('8', typeStr)>0  then vectorSize := 8;
  if pos('16', typeStr)>0 then vectorSize := 16;

  if pos('int', typeStr)= 1 then exit(sizeOf(cl_int)*vectorSize);
  if pos('long', typeStr)= 1 then exit(sizeOf(cl_long)*vectorSize);
  if pos('half', typeStr)= 1 then exit(sizeOf(cl_half)*vectorSize);
  if pos('float', typeStr)= 1 then exit(sizeOf(cl_float)*vectorSize);
  if pos('double', typeStr)= 1 then exit(sizeOf(cl_double)*vectorSize);
  if pos('char', typeStr)= 1 then exit(sizeOf(cl_char)*vectorSize);
  if pos('short', typeStr)= 1 then exit(sizeOf(cl_short)*vectorSize);

  if pos('uint', typeStr) = 1 then exit(sizeOf(cl_int)*vectorSize);
  if pos('ulong', typeStr) = 1 then exit(sizeOf(cl_long)*vectorSize);
  if pos('uchar', typeStr) = 1 then exit(sizeOf(cl_char)*vectorSize);
  if pos('ushort', typeStr) = 1 then exit(sizeOf(cl_short)*vectorSize);
end;

function TOpenCL.KernelInfo(index: integer): TCLKernelInfo;
var sz:size_t; i:integer;
begin
    FErr:=clGetKernelInfo(FKernels[Index],CL_KERNEL_FUNCTION_NAME, cInfoSize, @result.KernelName[0], sz);                                      CheckError();
    FErr:=clGetKernelInfo(FKernels[Index],CL_KERNEL_NUM_ARGS, cInfoSize, @result.KernelArgCount, sz);                                          CheckError();
    //FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_GLOBAL_WORK_SIZE,cInfoSize,@result.KernelGlobalWorkSize[0],@sz);  CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_WORK_GROUP_SIZE,sizeOf(result.KernelWorkGroupSize), @result.KernelWorkGroupSize, @sz);     CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_LOCAL_MEM_SIZE,sizeOf(result.KernelLocalMemSize), @result.KernelLocalMemSize, @sz);       CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_PRIVATE_MEM_SIZE,sizeOf(result.KernelPrivateMemSize), @result.KernelPrivateMemSize, @sz);     CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeOf(result.KernelSIMDSize), @result.KernelSIMDSize, @sz);     CheckError();

    setLength(result.KernelArgs,result.KernelArgCount);
    for i:=0 to result.KernelArgCount-1 do begin
        FErr:=clGetKernelArgInfo(FKernels[Index], i, CL_KERNEL_ARG_NAME, cInfoSize, @result.KernelArgs[i].ArgName[0], @sz);                         CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index], i, CL_KERNEL_ARG_TYPE_NAME, cInfoSize, @result.KernelArgs[i].ArgType[0], @sz);                    CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index], i, CL_KERNEL_ARG_TYPE_QUALIFIER, sizeof(result.KernelArgs[i].ArgTypeQualifier), @result.KernelArgs[i].ArgTypeQualifier, @sz);   CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index], i, CL_KERNEL_ARG_ACCESS_QUALIFIER,  sizeof(result.KernelArgs[i].ArgAccess), @result.KernelArgs[i].ArgAccess, @sz);              CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index], i, CL_KERNEL_ARG_ADDRESS_QUALIFIER,  sizeof(result.KernelArgs[i].ArgAddress), @result.KernelArgs[i].ArgAddress, @sz);           CheckError();
        result.KernelArgs[i].ArgSize := getTypeSize(result.KernelArgs[i].ArgName);
    end;

end;

function TOpenCL.CanExecuteNative: boolean;
begin
  result:=FExecCaps and CL_EXEC_NATIVE_KERNEL>0;
end;

procedure TOpenCL.LoadFromFile(FileName: ansistring);
begin
  FSrc.LoadFromFile(FileName);
  ProgramSource:=FSrc.Text;
end;

function TOpenCL.createDeviceBuffer(const aByteSize: size_t;
  const aAccess: TCLMemAccess; const fromHostMem: Pointer): cl_mem;
begin
  result := clCreateBuffer(FContext, cl_mem_flags(aAccess), aByteSize, fromHostMem, FErr);CheckError();
end;

procedure TOpenCL.freeDeviceBuffer(var aMem: cl_mem);
var memobj :cl_mem;
begin
  memobj := aMem;
  aMem := nil;
  FErr := clReleaseMemObject(memobj);
  //FErr := clReleaseMemObject(aMem);
  CheckError();
end;

procedure TOpenCL.readBuffer(const clMem: cl_mem; const bufferSize: size_t;
  const buffer: pointer; const offset: SizeInt; aQueue: cl_command_queue);
begin
  if not assigned(aQueue) then aQueue:=FQueue;
  FErr := clEnqueueReadBuffer(aQueue, clMem, CL_TRUE, offset, bufferSize, buffer, 0, nil, nil); checkError();
end;

procedure TOpenCL.writeBuffer(const clMem: cl_mem; const bufferSize: size_t;
  const buffer: pointer; const offset: SizeInt; aQueue: cl_command_queue);
begin
  if not assigned(aQueue) then aQueue:=FQueue;
  FErr := clEnqueueWriteBuffer(aQueue, clMem, CL_TRUE, offset, bufferSize, buffer, 0, nil, nil); checkError();
end;

(*
procedure TOpenCL.CallKernel(const Index: integer; const dst:PLongWord;const c: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];
  FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE,sz*SizeOf(LongWord),nil,FErr);CheckError();
  //FCallParams[1]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY, c*4,nil,FErr);
  //FCallParams[2]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY, c*8,nil,FErr);
  FErr:=clSetKernelArg(FActiveKernel,0,sizeOf(@FCallParams[0]),@FCallParams[0]);CheckError();
  //FErr:=clSetKernelArg(FActiveKernel,1,sizeOf(cl_mem),FCallParams[1]);
  //FErr:=clSetKernelArg(FActiveKernel,2,sizeOf(cl_mem),FCallParams[2]);
  FErr:=clSetKernelArg(FActiveKernel,1,SizeOf(c),@c);CheckError();
  FErr:=clEnqueueNDRangeKernel(FQueue,FActiveKernel,FWorkItemDimensions,FGlobalOffsets,FGlobalWorkGroupSizes,FLocalWorkGroupSizesPtr,0,nil,nil);CheckError();
  FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_True,0,sz*SizeOf(LongWord),dst,0,nil,nil);CheckError();
  //FErr:=clFlush(FQueue);
  //FErr:=clFinish(FQueue);

end;
*)

procedure TOpenCL.CallKernel(const Index: integer;
  const GlobalWorkGroups: TArray<size_t>; const params: TArray<Pointer>;
  const waitEvents: TArray<cl_event>; outEvent: pcl_event);
var
  ki:TCLKernelInfo;
  //sz:size_t;
  i,j:integer; s:ansistring;
begin
  ki := KernelInfo(Index);
  assert(ki.KernelArgCount=length(params), '[CallKernel] Kernel arguments and parameters must be equal!');
  for i:=0 to ki.KernelArgCount-1 do
    if Pos('*', ki.KernelArgs[i].ArgType)>0 then begin
      FErr:=clSetKernelArg(FKernels[Index], i, ki.KernelArgs[i].ArgSize, params[i]); CheckError();
    end;
  if assigned(GlobalWorkGroups) then
    FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index], length(GlobalWorkGroups), @FGlobalOffsets[0], pointer(GlobalWorkGroups), @FLocalWorkGroupSizes[0] , length(waitEvents), pointer(waitEvents), outEvent)
  else
    FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index], FWorkItemDimensions, @FGlobalOffsets[0], @FGlobalWorkGroupSizes[0], @FLocalWorkGroupSizes[0] , length(waitEvents), pointer(waitEvents), outEvent);
  CheckError();
end;

procedure TOpenCL.finish();
begin
  FErr := clFinish(ActiveQueue); CheckError();
end;

procedure TOpenCL.DoActiveDeviceChange(const newDevice: cl_device_id;
  const newQueue: cl_command_queue);
begin
  //
end;

procedure TOpenCL.waitForEvents(const N: longword; const events: pcl_event);
begin
  if N= 0 then exit;
  FErr := clWaitForEvents(N, pointer(events)); CheckError();
end;

procedure TOpenCL.freeEvents(const N: longword; const events: pcl_event);
var i:SizeInt;
begin
  for i:=0 to N-1 do begin
      FErr := clReleaseEvent(events[i]); CheckError();
      events[i] := nil;
  end;
end;

class function TOpenCL.getPlatforms: cl_uint;
begin
    clGetPlatformIDs(0,nil,@result);
end;

//function TOpenCL.KernelArgs(index: integer): TCLKernelArgInfo;
//begin
//  clGetKernelInfo(FKernels[Index],CL_KERNEL_NUM_ARGS,SizeOf(Result),@result,N);
//end;

function TOpenCL.GetDevice(index: cl_uint): cl_device_id;
begin
  result:=FDevices[index];
end;

function TOpenCL.getQueueInOrder: boolean;
var props:cl_bitfield; sz: size_t;
begin
  FErr := clGetCommandQueueInfo(ActiveQueue, CL_QUEUE_PROPERTIES, SizeOf(props), @props, sz);CheckError();
  result := (props and CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) = 0;
end;

procedure TOpenCL.SetActiveDeviceId(AValue: integer);
var
  wasBuilt:boolean;
  isShared:cl_bool;
  sz : size_t;
begin
  if FActiveDevice=FDevices[AValue] then Exit;
  if AValue>High(FDevices) then
    raise Exception.Create('Device index out of bounds!');
  wasBuilt:=FIsBuilt;
  CleanUp(true);
  FQueue:=clCreateCommandQueue(FContext,FDevices[AValue], 0, 0 (* QWord(@FErr) *) ); CheckError();
  FActiveDevice:=FDevices[AValue];
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_EXECUTION_CAPABILITIES,SizeOf(cl_device_exec_capabilities), @FExecCaps, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE,SizeOf(FMaxWorkGroupSize), @FMaxWorkGroupSize, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,SizeOf(FMaxWorkItemDimensions), @FMaxWorkItemDimensions, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_MAX_MEM_ALLOC_SIZE,SizeOf(FMaxMemAllocSize), @FMaxMemAllocSize, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES,SizeOf(size_t)*3, @FMaxWorkItemSizes[0], sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_MAX_COMPUTE_UNITS,SizeOf(FMaxComputeUnits), @FMaxComputeUnits, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY,SizeOf(FMaxFrequency), @FMaxFrequency, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_GLOBAL_MEM_SIZE,SizeOf(FGlobalMemSize), @FGlobalMemSize, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_LOCAL_MEM_SIZE,SizeOf(FLocalMemSize), @FLocalMemSize, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,SizeOf(FGlobalCacheSize), @FGlobalCacheSize, sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_BUILT_IN_KERNELS, cInfoSize, @FDeviceBuiltInKernels[0], sz);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_OPENCL_C_VERSION, cInfoSize, @CLDeviceCVersion[0], sz); CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_VENDOR,cInfoSize, @CLDeviceDriver, sz);            CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice, CL_DEVICE_HOST_UNIFIED_MEMORY,SizeOf(isShared), @isShared, sz);CheckError();
  FSharedMemory:=isShared=CL_TRUE;
  if wasBuilt then
    Build;
  FActiveDeviceId:=AValue;
  DoActiveDeviceChange(FActiveDevice, FQueue);

end;

procedure TOpenCL.SetActiveKernelId(AValue: integer);
begin
  if FActiveKernelId=AValue then exit;
  if not AValue<length(FKernels) then
    Exception.create('Kernel index out of bounds!');
  FActiveKernel:=FKernels[AValue];
  FActiveKernelId:=AValue;
  FActiveKernelInfo := KernelInfo(AValue)
end;

procedure TOpenCL.SetActivePlatformId(AValue: integer);
var
  i  :integer;
  dt :cl_device_type;
  sz : size_t;
begin
  assert(AValue<length(Platforms), format('Invalid platform id [%d]',[AValue]));
  if (FActivePlatform=Platforms[AValue]) then Exit;
  if AValue>High(Platforms) then raise Exception.Create('Platform index out of bounds!');
  FActivePlatform:=Platforms[AValue];
  FErr:=clGetDeviceIDs(FActivePlatform,getCL_Device_Type(FDeviceType),0,nil,@FDeviceCount);  CheckError();
  if FDeviceCount=0 then raise Exception.Create('No Devices found!');
  setLength(FDevices,FDeviceCount);
  setLength(FDevicesType,FDeviceCount);
  FErr:=clGetDeviceIDs(FActivePlatform,getCL_Device_Type(FDeviceType),FDeviceCount,@FDevices[0],nil);  CheckError();
  FDevsTypeStr:='';
  for i:=0 to FDeviceCount-1 do
    begin
      FErr:=clGetDeviceInfo(FDevices[i],CL_DEVICE_TYPE_INFO, SizeOf(dt), @dt, sz);
      CheckError();
      case dt of
        CL_DEVICE_TYPE_DEFAULT:begin FDevicesType[i]:=dtDefault;FDevsTypeStr:=FDevsTypeStr+#13#10'DEFAULT' end;
        CL_DEVICE_TYPE_CPU:begin FDevicesType[i]:=dtCPU;FDevsTypeStr:=FDevsTypeStr+#13#10'CPU' end;
        CL_DEVICE_TYPE_GPU:begin FDevicesType[i]:=dtGPU;FDevsTypeStr:=FDevsTypeStr+#13#10'GPU' end;
        CL_DEVICE_TYPE_ACCELERATOR:begin FDevicesType[i]:=dtACCELERATOR;FDevsTypeStr:=FDevsTypeStr+#13#10'ACCELERATOR' end;
      end;
    end;
  delete(FDevsTypeStr,1,2);
  if FContext<>nil then begin
    clReleaseContext(FContext);CheckError();
    FContext:=nil
  end;
  FContext:=clCreateContext(nil, FDeviceCount, @FDevices[0], nil, nil,FErr);CheckError();
  FActiveDeviceId:=-1;
  SetActiveDeviceId(0);
  FActivePlatformId:=AValue;
end;

initialization
  TOpenCL.initPlatforms();

end.

