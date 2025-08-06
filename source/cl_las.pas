
 (*
 This is a Pascal/Delphi implementation/port of the GEMM part from
 CLBlast library by Cedric Nugteren ( https://github.com/cnugteren/CLBlast )
 which is licensed under Apache Version 2.0.
 *)

{$ifdef FPC}
   {$Mode delphi}
   {$ModeSwitch advancedrecords} 
{$endif}
{$PointerMath On}
unit cl_las;

interface
uses SysUtils, generics.Collections, generics.Defaults
  , FP16
  {$if defined(DARWIN) or defined(MACOS)}
  , CL
  {$else}
  , OpenCL
  {$endif}
  //, OpenCLHelper
  ;

const
  kDeviceTypeAll = 'default';
  kDeviceTypeGPU = 'GPU';
  kDeviceTypeCPU = 'CPU';
  kDeviceTypeAccelerator = 'accelerator';
  kDeviceNameDefault = 'default';

  // Khronos OpenCL extensions
  kKhronosAttributesAMD = 'cl_amd_device_attribute_query';
  kKhronosAttributesNVIDIA = 'cl_nv_device_attribute_query';
  kKhronosIntelSubgroups = 'cl_intel_subgroups';
type
  {$if not declared(Half)}
  PHalf = ^Half;
  Half = record
    v:smallInt;
  end;
  {$endif}

  TComplexHalf = record
    real, imag:Half;
  end;

  TComplexSingle = record
    real, imag:single;
  end;

  TComplexDouble = record
    real, imag:Double;
  end;

{$if not declared(size_t)}
  size_t         = IntPtr;
{$endif}
  TQueue         = cl_command_queue;
  TEvent         = cl_event;
  TContext       = cl_context;
  TProgram       = cl_program;
  TEventPointer  = Pcl_event;

  { TBuffer }

  TBuffer<T>     = record
    buf   : cl_mem;
    function GetSize():size_t;
    constructor create(const aContext:TContext; const size:size_t; const access: cl_mem_flags= CL_MEM_READ_WRITE; const host:pointer =nil);
    function Context():TContext;
    procedure Alloc(const aContext:TContext; const size:size_t; const access: cl_mem_flags= CL_MEM_READ_WRITE; const host:pointer =nil);
    procedure free;
    class operator implicit(const buf:cl_mem):TBuffer<T>;
    class operator implicit(const buf:TBuffer<T>):cl_mem;
  end;

  { TKernel }

  TKernel = record
    kernel : cl_kernel;
    constructor Create(const aProgram:TProgram; const aName:ansistring);
    function Name:ansistring;
    procedure setArgument<T>(const index: cl_uint; const value:T);           overload;
    procedure setArgument<T>(const index: cl_uint; const value:TBuffer<T>);  overload;

    class operator implicit(const src:TKernel):cl_kernel;
    class operator implicit(const src:cl_kernel):TKernel;
    class operator initialize({$ifdef fpc}var{$else}out{$endif} dst:TKernel);
  end;

  { TDevice }

  TDevice        = record
    devId   : cl_device_id;
    function getInfo<T>(const info:cl_device_info):T;
    function getInfoString(const info:cl_device_info):ansistring;
    function getInfoVector<T>(const info:cl_device_info):TArray<T>;
    function &Type() : cl_device_type;
    function TypeStr() : ansistring;
    function Name()  : ansistring;
    function Vendor(): ansistring;
    function MaxWorkGroupSize(): size_t;
    function MaxWorkItemDimensions(): size_t;
    function MaxWorkItemSizes():TArray<size_t>;
    function LocalMemSize():cl_ulong;
    function Capabilities():ansistring;
    function HasExtension(const ext:ansistring):boolean;
    function CoreClock(): size_t ;
    function ComputeUnits(): size_t ;
    function MemoryClock(): size_t ;
    function MemoryBusWidth(): size_t ;
    function MemorySize(): cl_ulong;
    function MaxAllocSize(): cl_ulong;

    // Configuration-validity checks
    function IsLocalMemoryValid(const local_mem_usage :cl_ulong ):boolean;
    function IsThreadConfigValid(const local : TArray<size_t> ):boolean;
    function IsPostNVIDIAVolta(): boolean;

    // Query for a specific type of device or brand
    function SupportsFP64(): boolean;
    function SupportsFP16(): boolean;
    function IsCPU():      boolean ;
    function IsGPU():      boolean ;
    function IsAMD():      boolean ;
    function IsNVIDIA():   boolean ;
    function IsIntel():    boolean ;
    function IsARM():      boolean ;
    function IsQualcomm(): boolean ;

    // Platform specific extensions
    function AMDBoardName(): ansistring;
    function NVIDIAComputeCapability(): ansistring;
    // Returns if the Nvidia chip is a Volta or later archicture (sm_70 or higher)

    // Returns the Qualcomm Adreno GPU version (i.e. a650, a730, a740, etc.)
    function AdrenoVersion(): ansistring;
    // Retrieves the above extra information (if present)
    function GetExtraInfo(): ansistring;
    class operator Implicit(const dev : cl_device_id):TDevice;
    class operator Implicit(const dev : TDevice): cl_device_id;
  end;

(*
  TStatusCode = (

    // Status codes in common with the OpenCL standard
    kSuccess                   =   0, // CL_SUCCESS
    kOpenCLCompilerNotAvailable=  -3, // CL_COMPILER_NOT_AVAILABLE
    kTempBufferAllocFailure    =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
    kOpenCLOutOfResources      =  -5, // CL_OUT_OF_RESOURCES
    kOpenCLOutOfHostMemory     =  -6, // CL_OUT_OF_HOST_MEMORY
    kOpenCLBuildProgramFailure = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
    kInvalidValue              = -30, // CL_INVALID_VALUE
    kInvalidCommandQueue       = -36, // CL_INVALID_COMMAND_QUEUE
    kInvalidMemObject          = -38, // CL_INVALID_MEM_OBJECT
    kInvalidBinary             = -42, // CL_INVALID_BINARY
    kInvalidBuildOptions       = -43, // CL_INVALID_BUILD_OPTIONS
    kInvalidProgram            = -44, // CL_INVALID_PROGRAM
    kInvalidProgramExecutable  = -45, // CL_INVALID_PROGRAM_EXECUTABLE
    kInvalidKernelName         = -46, // CL_INVALID_KERNEL_NAME
    kInvalidKernelDefinition   = -47, // CL_INVALID_KERNEL_DEFINITION
    kInvalidKernel             = -48, // CL_INVALID_KERNEL
    kInvalidArgIndex           = -49, // CL_INVALID_ARG_INDEX
    kInvalidArgValue           = -50, // CL_INVALID_ARG_VALUE
    kInvalidArgSize            = -51, // CL_INVALID_ARG_SIZE
    kInvalidKernelArgs         = -52, // CL_INVALID_KERNEL_ARGS
    kInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
    kInvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
    kInvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
    kInvalidGlobalOffset       = -56, // CL_INVALID_GLOBAL_OFFSET
    kInvalidEventWaitList      = -57, // CL_INVALID_EVENT_WAIT_LIST
    kInvalidEvent              = -58, // CL_INVALID_EVENT
    kInvalidOperation          = -59, // CL_INVALID_OPERATION
    kInvalidBufferSize         = -61, // CL_INVALID_BUFFER_SIZE
    kInvalidGlobalWorkSize     = -63, // CL_INVALID_GLOBAL_WORK_SIZE

    // Status codes in common with the clBLAS library
    kNotImplemented            = -1024, // Routine or functionality not implemented yet
    kInvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
    kInvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
    kInvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
    kInvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
    kInvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
    kInvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
    kInvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
    kInvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
    kInvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
    kInvalidIncrementX         = -1013, // Increment of vector X cannot be zero
    kInvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
    kInsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
    kInsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
    kInsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
    kInsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
    kInsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

    // Custom additional status codes for CLBlast
    kInsufficientMemoryTemp    = -2050, // Temporary buffer provided to GEMM routine is too small
    kInvalidBatchCount         = -2049, // The batch count needs to be positive
    kInvalidOverrideKernel     = -2048, // Trying to override parameters for an invalid kernel
    kMissingOverrideParameter  = -2047, // Missing override parameter(s) for the target kernel
    kInvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
    kNoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
    kNoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
    kInvalidVectorScalar       = -2043, // The unit-sized vector is not a valid OpenCL buffer
    kInsufficientMemoryScalar  = -2042, // The unit-sized vector's OpenCL buffer is too small
    kDatabaseError             = -2041, // Entry for the device was not found in the database
    kUnknownError              = -2040, // A catch-all error code representing an unspecified error
    kUnexpectedError           = -2039 // A catch-all error code representing an unexpected exception
  );
*)

  TPrecision = (
    kAny           = -1,
    kHalf          = 16,
    kSingle        = 32,
    kDouble        = 64,
    kComplexSingle = 3232,
    kComplexDouble = 6464
  );
  TName   = array[0..50] of ansichar;
  TParams = TArray<size_t>;//array[0..15] of byte;

  TParameters = TDictionary<ansistring, size_t>;

  TInitList = array of ansistring;

  PDatabaseDevice = ^TDatabaseDevice;
  TDatabaseDevice = record
    name          : ansistring;
    parameters    : TParams; // parameter values
  end;

  PDatabaseArchitecture = ^TDatabaseArchitecture;

  { TDatabaseArchitecture }

  TDatabaseArchitecture = record
    name      : ansistring;
    devices   : TArray<TDatabaseDevice> ;
    function lastDev():PDatabaseDevice;
    function addDevice(const aDevName: ansistring; const aParams:TParams):PDatabaseDevice;
  end;

  PDatabaseVendor = ^TDatabaseVendor;

  { TDatabaseVendor }

  TDatabaseVendor = record
    devType   : ansistring;
    name      : ansistring;
    architectures    : TArray<TDatabaseArchitecture> ;
    function lastArch():PDatabaseArchitecture;
    function addArch(const aName: ansistring):PDatabaseArchitecture;
  end;

  { TDatabaseEntry }
  PDatabaseEntry = ^TDatabaseEntry;
  TDatabaseEntry = record
    kernel          : ansistring;
    precision       : TPrecision;
    parameter_names : TArray<ansistring>;
    vendors         : TArray<TDatabaseVendor>;
    function init(
      const aKernel: ansistring;
      const aPrecision: TPrecision;
      const aDeviceType: ansistring;
      const aVendorName: ansistring;
      const aArchName : ansistring;
      const aDeviceName : ansistring;
      const aParamNames: TArray<ansistring>;
      const aDeviceParams: TParams):PDatabaseEntry;   overload;
    function init(const aKernel:ansistring; const aPrecition:TPrecision; const aParamNames : TArray<ansistring>): PDatabaseEntry; overload;
    function lastVendor():PDatabaseVendor;
    function addVendor(const aDeviceType, aVendorName: ansistring):PDatabaseVendor ;
  end;


type
  { TDatabase }

  TDatabase = record
    // The OpenCL device vendors
    const kDeviceVendorAll   : ansistring = 'default';
  class var
    // The database consists of separate database entries, stored together in a vector
    database           :  TArray<TDatabaseEntry>;

    // Database for a special case: Apple CPUs support limited number of threads

    apple_cpu_fallback :  TArray<TDatabaseEntry>;

  private
    procedure setItem(key : ansistring; AValue: size_t);
  public
    // The constructor with a user-provided database overlay (potentially an empty vector)
    constructor Create(
      const device      : TDevice;
      const kernel_name : ansistring;
      const aPrecision   : TPrecision;
      const overlay     : TArray<TDatabaseEntry>);
    procedure free();
    // Accessor of values by key
    function getItem(key : ansistring): size_t; { return parameters_->find(key)->second; }
    function exists(const key : ansistring): boolean; { return (parameters_->count(key) == 1); }

    // Obtain a list of OpenCL pre-processor defines based on the parameters
    function GetDefines(): ansistring;

    // Retrieves the values or names of all the parameters
    function GetValuesString(): ansistring;
    function GetParameterNames():TArray<ansistring>;
    function GetParameters():TParameters; { return *parameters_; }
    property item[key : ansistring]: size_t read getItem write setItem; default;

  private
    // Found parameters suitable for this device/kernel
    parameters_ : TParameters;

    // Search method functions, returning a set of parameters (possibly empty)
    function Search(
           const this_kernel,
                 this_vendor, this_type ,
                 this_device, this_architecture : ansistring;
           const this_precision : TPrecision;
           const dbs : TArray<TDatabaseEntry>) : TParameters;

    function SearchDevice(const target_device : ansistring;
                          const devices : TArray<TDatabaseDevice>;
                          const parameter_names : TArray<ansistring>): TParameters;

    function SearchArchitecture(const target_architecture, this_device: ansistring;
                          const architectures : TArray<TDatabaseArchitecture>;
                          const parameter_names : TArray<ansistring>):TParameters;

    function SearchVendorAndType(
                          const target_vendor, target_type, this_device, this_architecture : ansistring;
                          const vendors : TArray<TDatabaseVendor>;
                          const parameter_names : TArray<ansistring>):TParameters;

  end;

  { TDatabases }

  TDatabases = record

  private
    function getDatabases(key : ansistring): TDatabase;
    procedure setDatabases(key : ansistring; AValue: TDatabase);
  public
    constructor create(const kernel_names:TArray<ansistring>);
    procedure free();
  // Database accessor
    function get(const kernel_name: ansistring):TDatabase;

  // Retrieves a parameter from the database
    function getParams(key: ansistring): size_t;
    function setDatabase(const key:ansistring; const db:TDatabase):boolean;
    property params[key : ansistring]:size_t read getParams;
    property Databases[key : ansistring] : TDatabase read getDatabases write setDatabases; default;

  private
    kernel_names_ : TArray<ansistring>;
    databases_    :TDictionary<ansistring, TDatabase>;

  end;

  { TRoutine }

  TRoutine = class
  type

    { TRoutineDesc }

    TRoutineDesc = record
      //platform : TPlatform;
      device   : TDevice;
      precision : TPrecision;
      kernel   : ansistring;
      constructor create(
                  //const aPlatform: TPlatform;
                  const aDevice: TDevice;
                  const aPrecision : TPrecision;
                  const aKernel:ansistring);
    end;

    { TProgramDesc }

    TProgramDesc = record
      context  : cl_context;
      deviceId : cl_device_id;
      Precision : TPrecision;
      routineName : ansistring;
      constructor create(const aContext:cl_context; const aDeviceId:cl_device_id; const aPrecision:TPrecision; const aRoutineName:ansistring);

    end;

    { TBinDesc }

    TBinDesc = record
      Precision   : TPrecision;
      routineName : ansistring;
      deviceName  : ansistring;
      constructor create(const aPrecision : TPrecision; const aRoutineName : ansistring; const aDeviceName : ansistring);
    end;

  const
    routines_axpy      : TArray<ansistring> = ['AXPY', 'COPY', 'SCAL', 'SWAP'];
    routines_dot       : TArray<ansistring> = ['AMAX', 'ASUM', 'DOT', 'DOTC', 'DOTU', 'MAX', 'MIN', 'NRM2', 'SUM'];
    routines_ger       : TArray<ansistring> = ['GER', 'GERC', 'GERU', 'HER', 'HER2', 'HPR', 'HPR2', 'SPR', 'SPR2', 'SYR', 'SYR2'];
    routines_gemv      : TArray<ansistring> = ['GBMV', 'GEMV', 'HBMV', 'HEMV', 'HPMV', 'SBMV', 'SPMV', 'SYMV', 'TMBV', 'TPMV', 'TRMV', 'TRSV'];
    routines_gemm      : TArray<ansistring> = ['GEMM', 'HEMM', 'SYMM', 'TRMM'];
    routines_gemm_syrk : TArray<ansistring> = ['GEMM', 'HEMM', 'HER2K', 'HERK', 'SYMM', 'SYR2K', 'SYRK', 'TRMM', 'TRSM'];
    routines_trsm      : TArray<ansistring> = ['TRSM'];
  class var
    routines_by_kernel : TDictionary<ansistring, TArray<ansistring>>;
    databaseCache : TDictionary<TRoutineDesc, TDatabase>;
    programCache : TDictionary<TProgramDesc, cl_program>;
    binaryCache   : TDictionary<TBinDesc, RawByteString>;

  public
    class function PrecisionValue<T>:TPrecision;static;
    // Initializes db_, fetching cached database or building one
    class procedure InitDatabase(
          const device :TDevice;
          const kernel_names : TArray<ansistring>;
          const precision : TPrecision;
          const userDatabase : TArray<TDatabaseEntry>;
          var dbs : TDatabases); static;

    // Base class constructor. The user database is an optional extra database to override the
    // built-in database.
    // All heavy preparation work is done inside this constructor.
    // NOTE: the caller must provide the same userDatabase for each combination of device, precision
    // and routine list, otherwise the caching logic will break.
    constructor Create(
          const queue: TQueue;
          const event: TEventPointer;
          const name : ansistring;
          const routines   : TArray<ansistring>;
          const precision : TPrecision;
          const userDatabase : TArray<TDatabaseEntry>;
          const source : TInitList);


    // List of kernel-routine look-ups

  private
    // Initializes program_, fetching cached program or building one
    procedure InitProgram(const source:TInitList);

  protected

    // Non-static variable for the precision
    precision_ : TPrecision;

    // The routine's name and the corresponding kernels
    routine_name_ : ansistring;
    kernel_names_ : TArray<ansistring>;

    // The OpenCL objects, accessible only from derived classes
    queue_   : TQueue;
    event_   : TEventPointer;
    context_ : TContext;
    device_  : TDevice;

    // Compiled program (either retrieved from cache or compiled in slow path)
    program_ : TArray<TProgram> ;

    // Connection to the database for all the device-specific parameters
    db_ : TDatabases;
    //kernels_ : TArray<TKernel>;
    //kernelNames_ : TArray<ansistring>;
  end;

  TLayout = (loRowMajor, loColMajor);
  TTranspose = (trNo, trYes, trConjugate);

  { TXgemm }

  TXgemm<T> = class(TRoutine)
      var
        XGEMM_MIN_INDIRECT_SIZE : size_t;
        MWG                     : size_t;
        NWG                     : size_t;
        KWG                     : size_t;
        VWM                     : size_t;
        VWN                     : size_t;
        GEMMK                   : size_t;
        MDIMC                   : size_t;
        NDIMC                   : size_t;
        MDIMCD                  : size_t;
        NDIMCD                  : size_t;
        WGD                     : size_t;
        KREG                    : size_t;
        TRA_WPT                 : size_t;
        TRA_DIM                 : size_t;
        COPY_VW                 : size_t;
        COPY_WPT                : size_t;
        COPY_DIMX               : size_t;
        COPY_DIMY               : size_t;
        PADTRA_WPT              : size_t;
        PADTRA_TILE             : size_t;
        PAD_WPTX                : size_t;
        PAD_WPTY                : size_t;
        PAD_DIMX                : size_t;
        PAD_DIMY                : size_t;


      class var ConstantOne : T ;

      class function a_want_rotated_(const gemm_kernel_id:size_t):boolean;static;
      class function b_want_rotated_(const gemm_kernel_id:size_t):boolean;static;
      class function c_want_rotated_(const gemm_kernel_id:size_t):boolean;static;

      // Computes the size of the temporary GEMM buffer based on user-arguments
      class function GetTempSize(
          const layout :TLayout ;
          const a_transpose, b_transpose: TTranspose;
          const m, n, k
          , a_offset, a_ld
          , b_offset, b_ld
          , c_offset, c_ld
          , mwg, nwg, kwg
          , gemm_kernel_id:size_t):size_t; static;

      // Selects which version of GEMM to run
      class function UseDirectKernel(const m, n, k,
                                  min_indirect_size:size_t):boolean;static;

      // Process the user-arguments, computes secondary parameters
      class procedure ProcessArguments(
            const layout:TLayout; const a_transpose, b_transpose : TTranspose;
            const m, n, k:size_t;
            var a_one, a_two, b_one, b_two, c_one, c_two:size_t;
            var a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
            const gemm_kernel_id:size_t);static;

      // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
      class function ComputeTempSize(
            const a_no_temp, b_no_temp, c_no_temp:boolean;
            const a_size, b_size, c_size: size_t;
            var b_temp_offset, c_temp_offset:size_t):size_t ; static;

      // Determines whether or not temporary matrices are needed
      class function NoTempBuffer(const one, one_i, two, two_i, ld, offset:size_t; const do_transpose, conjugate:boolean):boolean;static;


      // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
      // whether the matrices need to be rotated or not for the kernel.
      class procedure CalculateInternalDimensions(
            const m, n, k, mwg, nwg, kwg : size_t;
            var a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i:size_t;
            const gemm_kernel_id:size_t); static;

      // Constructor
      constructor Create(const queue:TQueue; const event: TEventPointer; const name :ansistring = 'GEMM');

      // Templated-precision implementation of the routine
      procedure DoGemm(const layout :TLayout; const a_transpose, b_transpose: TTranspose;
                  const m, n, k: size_t;
                  const alpha : T;
                  const a_buffer    :TBuffer<T>; const a_offset, a_ld:size_t;
                  const b_buffer    :TBuffer<T>; const b_offset, b_ld:size_t;
                  const beta:T;
                  const c_buffer    :TBuffer<T>; const c_offset, c_ld:size_t;
                  const temp_buffer :cl_mem = nil; const temp_buffer_provided:boolean = false);

      procedure PadCopyTransposeMatrix(
                            const event :TEventPointer; const waitForEvents:TArray<TEvent>;
                            const src_one, src_two, src_ld, src_offset:size_t;
                            const src : TBuffer<T> ;
                            const dest_one, dest_two, dest_ld, dest_offset:size_t;
                            const dest :TBuffer<T> ;
                            const alpha:T;
                            const do_pad, do_transpose, do_conjugate:boolean;
                            const upper : boolean = false;
                            const lower : boolean = false;
                            const diagonal_imag_zero : boolean = false);

      procedure PadCopyTransposeMatrixStridedBatched(
                            const event :TEventPointer; const waitForEvents:TArray<TEvent>;
                            const src_one, src_two, src_ld, src_offset:size_t;
                            const src_stride :size_t; const src : TBuffer<T> ;
                            const dest_one, dest_two, dest_ld, dest_offset:size_t;
                            const dest_stride :size_t; const dest :TBuffer<T> ;
                            const do_pad, do_transpose, do_conjugate:boolean;
                            const batch_count : size_t);
      // Indirect version of GEMM (with pre and post-processing kernels)
      procedure GemmIndirect(
                  const m, n, k: size_t;
                  const alpha : T;
                  const a_buffer    :TBuffer<T>; const a_offset, a_ld:size_t;
                  const b_buffer    :TBuffer<T>; const b_offset, b_ld:size_t;
                  const beta:T;
                  const c_buffer    :TBuffer<T>; const c_offset, c_ld:size_t;
                  const a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
                  const a_one, a_two, b_one, b_two, c_one, c_two : size_t;
                  const temp_buffer :cl_mem = nil; const temp_buffer_provided:boolean = false);

      // Direct version of GEMM (no pre and post-processing kernels)
      procedure GemmDirect(
                  const m, n, k: size_t;
                  const alpha : T;
                  const a_buffer    :TBuffer<T>; const a_offset, a_ld:size_t;
                  const b_buffer    :TBuffer<T>; const b_offset, b_ld:size_t;
                  const beta:T;
                  const c_buffer    :TBuffer<T>; const c_offset, c_ld:size_t;
                  const a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean);

      procedure DoGemmStridedBatch(
                  const layout :TLayout; const a_transpose, b_transpose: TTranspose;
                  const m, n, k: size_t;
                  const alpha : T;
                  const a_buffer    :TBuffer<T>; const a_offset, a_ld, a_stride:size_t;
                  const b_buffer    :TBuffer<T>; const b_offset, b_ld, b_stride:size_t;
                  const beta:T;
                  const c_buffer    :TBuffer<T>; const c_offset, c_ld, c_stride:size_t;
                  const batch_count : size_t);

      procedure BatchedGemmIndirect(
                  const m, n, k: size_t;
                  const alpha : T;
                  const a_buffer    :TBuffer<T>; const a_offset, a_ld, a_stride:size_t;
                  const b_buffer    :TBuffer<T>; const b_offset, b_ld, b_stride:size_t;
                  const beta:T;
                  const c_buffer    :TBuffer<T>; const c_offset, c_ld, c_stride:size_t;
                  const a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
                  const a_one, a_two, b_one, b_two, c_one, c_two,
                  batch_count : size_t);

      procedure BatchedGemmDirect(
                  const m, n, k: size_t;
                  const alpha : T;
                  const a_buffer    :TBuffer<T>; const a_offset, a_ld, a_stride:size_t;
                  const b_buffer    :TBuffer<T>; const b_offset, b_ld, b_stride:size_t;
                  const beta:T;
                  const c_buffer    :TBuffer<T>; const c_offset, c_ld, c_stride:size_t;
                  const a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
                  const batch_count :size_t);

  end;

  THGemm = TxGemm<Half>;
  TSGemm = Txgemm<single>;
  TDGemm = TxGemm<double>;

var
    XaxpyHalf, XaxpySingle, XaxpyDouble, XaxpyComplexSingle, XaxpyComplexDouble,
    XdotHalf, XdotSingle, XdotDouble, XdotComplexSingle, XdotComplexDouble,
    XgemvHalf, XgemvSingle, XgemvDouble, XgemvComplexSingle, XgemvComplexDouble,
    XgemvFastHalf, XgemvFastSingle, XgemvFastDouble, XgemvFastComplexSingle, XgemvFastComplexDouble,
    XgemvFastRotHalf, XgemvFastRotSingle, XgemvFastRotDouble, XgemvFastRotComplexSingle, XgemvFastRotComplexDouble,
    XgerHalf, XgerSingle, XgerDouble, XgerComplexSingle, XgerComplexDouble,
    XgemmHalf, XgemmSingle, XgemmDouble, XgemmComplexSingle, XgemmComplexDouble,
    XgemmDirectHalf, XgemmDirectSingle, XgemmDirectDouble, XgemmDirectComplexSingle, XgemmDirectComplexDouble,
    XconvgemmHalf, XconvgemmSingle, XconvgemmDouble, XconvgemmComplexSingle, XconvgemmComplexDouble,
    CopyHalf, CopySingle, CopyDouble, CopyComplexSingle, CopyComplexDouble,
    PadHalf, PadSingle, PadDouble, PadComplexSingle, PadComplexDouble,
    TransposeHalf, TransposeSingle, TransposeDouble, TransposeComplexSingle, TransposeComplexDouble,
    PadtransposeHalf, PadtransposeSingle, PadtransposeDouble, PadtransposeComplexSingle, PadtransposeComplexDouble,
    InvertHalf, InvertSingle, InvertDouble, InvertComplexSingle, InvertComplexDouble,
    GemmRoutineHalf, GemmRoutineSingle, GemmRoutineDouble, GemmRoutineComplexSingle, GemmRoutineComplexDouble,
    TrsvRoutineHalf, TrsvRoutineSingle, TrsvRoutineDouble, TrsvRoutineComplexSingle, TrsvRoutineComplexDouble
      : TDatabaseEntry;


    hgemm : THGemm;
    sgemm : TSGemm;
    dgemm : TDGemm;

procedure SAFE_CALL(const err:cl_int);inline;
function ifthen(const cond:boolean; const ifTrue:size_t; const ifFalse:size_t):size_t;inline;
function CeilDiv(const x, y:size_t):size_t;inline;
function Ceil(const x, y:size_t):size_t;inline;
// Helper function to determine whether or not 'a' is a multiple of 'b'
function IsMultiple(const a, b:size_t):boolean;inline;
function clErrorText(err:cl_int):ansistring; inline;

implementation
//uses SortedMap;

function clErrorText(err:cl_int):ansistring;
begin
  case err of
    CL_DEVICE_NOT_FOUND : clErrorText:='CL_DEVICE_NOT_FOUND';
    CL_DEVICE_NOT_AVAILABLE : clErrorText:='CL_DEVICE_NOT_AVAILABLE';
    CL_DEVICE_COMPILER_NOT_AVAILABLE : clErrorText:='CL_DEVICE_COMPILER_NOT_AVAILABLE';
    CL_MEM_OBJECT_ALLOCATION_FAILURE : clErrorText:='CL_MEM_OBJECT_ALLOCATION_FAILURE';
    CL_OUT_OF_RESOURCES : clErrorText:='CL_OUT_OF_RESOURCES';
    CL_OUT_OF_HOST_MEMORY : clErrorText:='CL_OUT_OF_HOST_MEMORY';
    CL_PROFILING_INFO_NOT_AVAILABLE : clErrorText:='CL_PROFILING_INFO_NOT_AVAILABLE';
    CL_MEM_COPY_OVERLAP : clErrorText:='CL_MEM_COPY_OVERLAP';
    CL_IMAGE_FORMAT_MISMATCH : clErrorText:='CL_IMAGE_FORMAT_MISMATCH';
    CL_IMAGE_FORMAT_NOT_SUPPORTED : clErrorText:='CL_IMAGE_FORMAT_NOT_SUPPORTED';
    CL_BUILD_PROGRAM_FAILURE : clErrorText:='CL_BUILD_PROGRAM_FAILURE';
    CL_MAP_FAILURE : clErrorText:='CL_MAP_FAILURE';

    CL_INVALID_VALUE : clErrorText:='CL_INVALID_VALUE';
    CL_INVALID_DEVICE_TYPE : clErrorText:='CL_INVALID_DEVICE_TYPE';
    CL_INVALID_PLATFORM : clErrorText:='CL_INVALID_PLATFORM';
    CL_INVALID_DEVICE : clErrorText:='CL_INVALID_DEVICE';
    CL_INVALID_CONTEXT : clErrorText:='CL_INVALID_CONTEXT';
    CL_INVALID_QUEUE_PROPERTIES : clErrorText:='CL_INVALID_QUEUE_PROPERTIES';
    CL_INVALID_COMMAND_QUEUE : clErrorText:='CL_INVALID_COMMAND_QUEUE';
    CL_INVALID_HOST_PTR : clErrorText:='CL_INVALID_HOST_PTR';
    CL_INVALID_MEM_OBJECT : clErrorText:='CL_INVALID_MEM_OBJECT';
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR : clErrorText:='CL_INVALID_IMAGE_FORMAT_DESCRIPTOR';
    CL_INVALID_IMAGE_SIZE : clErrorText:='CL_INVALID_IMAGE_SIZE';
    CL_INVALID_SAMPLER : clErrorText:='CL_INVALID_SAMPLER';
    CL_INVALID_BINARY : clErrorText:='CL_INVALID_BINARY';
    CL_INVALID_BUILD_OPTIONS : clErrorText:='CL_INVALID_BUILD_OPTIONS';
    CL_INVALID_PROGRAM : clErrorText:='CL_INVALID_PROGRAM';
    CL_INVALID_PROGRAM_EXECUTABLE : clErrorText:='CL_INVALID_PROGRAM_EXECUTABLE';
    CL_INVALID_KERNEL_NAME : clErrorText:='CL_INVALID_KERNEL_NAME';
    CL_INVALID_KERNEL_DEFINITION : clErrorText:='CL_INVALID_KERNEL_DEFINITION';
    CL_INVALID_KERNEL : clErrorText:='CL_INVALID_KERNEL';
    CL_INVALID_ARG_INDEX : clErrorText:='CL_INVALID_ARG_INDEX';
    CL_INVALID_ARG_VALUE : clErrorText:='CL_INVALID_ARG_VALUE';
    CL_INVALID_ARG_SIZE : clErrorText:='CL_INVALID_ARG_SIZE';
    CL_INVALID_KERNEL_ARGS : clErrorText:='CL_INVALID_KERNEL_ARGS';
    CL_INVALID_WORK_DIMENSION : clErrorText:='CL_INVALID_WORK_DIMENSION';
    CL_INVALID_WORK_GROUP_SIZE : clErrorText:='CL_INVALID_WORK_GROUP_SIZE';
    CL_INVALID_WORK_ITEM_SIZE : clErrorText:='CL_INVALID_WORK_ITEM_SIZE';
    CL_INVALID_GLOBAL_OFFSET : clErrorText:='CL_INVALID_GLOBAL_OFFSET';
    CL_INVALID_EVENT_WAIT_LIST : clErrorText:='CL_INVALID_EVENT_WAIT_LIST';
    CL_INVALID_EVENT : clErrorText:='CL_INVALID_EVENT';
    CL_INVALID_OPERATION : clErrorText:='CL_INVALID_OPERATION';
    CL_INVALID_GL_OBJECT : clErrorText:='CL_INVALID_GL_OBJECT';
    CL_INVALID_BUFFER_SIZE : clErrorText:='CL_INVALID_BUFFER_SIZE';
    CL_INVALID_MIP_LEVEL : clErrorText:='CL_INVALID_MIP_LEVEL';
  else
     clErrorText:='Unknown OpenCL error ['+intToStr(err)+']';
  end;
end;

type
  TPairedList<K, V> = record
    keys   : TArray<K>;
    values : TArray<V>
  end;

const
  kVendorNames : TPairedList<ansistring, ansistring> = (
    keys : ['Advanced Micro Devices, Inc.', 'GenuineIntel', 'Intel(R) Corporation', 'NVIDIA Corporation'];
    Values : ['AMD', 'Intel', 'Intel', 'NVIDIA']
  );

// Alternative names for some architectures (mid-level)
  kArchitectureNames : TPairedList<ansistring, ansistring> =(
    keys : ['gfx803', 'gfx900'];
    values : ['Fiji', 'Vega']
  );

// Alternative names for some devices (low-level)
  kDeviceNames : TPairedList<ansistring, ansistring> =();

// Things to remove from device names (low-level)
  kDeviceRemovals: TArray<ansistring> = ['pthread-'];

function GetDeviceName(const device:TDevice): ansistring;
var i:integer;
begin
  if device.HasExtension(kKhronosAttributesAMD) then
    result := device.AMDBoardName()
  else
    result := device.Name();

  for i:=0 to length(kDeviceNames.keys)-1 do  // replacing to common names
    if result = kDeviceNames.keys[i] then
      result := kDeviceNames.Values[i];

  for i:=0 to high(kDeviceRemovals) do begin // removing certain things
    if pos(kDeviceRemovals[i], result) >0 then
      result := StringReplace(result, kDeviceRemovals[i], '',[]);
  end;
end;

function GetDeviceVendor(const device:TDevice):ansistring;
var
  i: Integer;
begin
  result := device.Vendor();

  for i:=0 to high(kVendorNames.keys) do  // replacing to common names
    if result = kVendorNames.keys[i] then
      result := kVendorNames.values[i];
end;

function GetDeviceArchitecture(const device:TDevice):ansistring;
var
  i: Integer;
begin
  result := '';
  if device.HasExtension(kKhronosAttributesNVIDIA)then
    result := device.NVIDIAComputeCapability()
  else if device.HasExtension(kKhronosAttributesAMD) then
    result := device.Name() // Name is architecture for AMD APP and AMD ROCm
  else if device.IsQualcomm() and device.IsGPU() then // queries the Adreno GPU architecture version
    result := device.AdrenoVersion();
  // Note: no else - 'device_architecture' might be the empty string

  for i:=0 to high(kArchitectureNames.keys) do  // replacing to common names
    if result = kArchitectureNames.keys[i] then
      result := kArchitectureNames.values[i]
end;


procedure SAFE_CALL(const err:cl_int);
begin
  assert(err=CL_SUCCESS, string(clErrorText(err)));
end;

function getContext(const quoue: TQueue):TContext;
var sz:size_t;
begin
  result := nil;
  SAFE_CALL(clGetCommandQueueInfo(quoue, CL_QUEUE_CONTEXT, sizeOf(result), @result, sz));
end;

function getDevice(const quoue: TQueue):TDevice;
var sz:size_t;
begin
  result := nil;
  SAFE_CALL(clGetCommandQueueInfo(quoue, CL_QUEUE_DEVICE, sizeOf(result), @result, sz));
end;

function LoadFromBinary(const bin:RawByteString; const context:TContext; const device:TDevice):TProgram;
var sz : size_t;
  binPtr:PAnsiChar;
  binStatus, err:cl_int;
begin
  binPtr := PAnsiChar(bin);
  sz := length(bin);
  result := clCreateProgramWithBinary(context, 1, @device, @sz, @binPtr, binStatus, err);
  SAFE_CALL(err)
end;

function getProgramBinary(const prog:TProgram):RawByteString;
var sz:size_t;
  ret:size_t;
  bin : TArray<RawByteString>;
begin
  bin :=[''];                                                         //^ size of one pointer because we are recieving a reference to variables not the variable
  SAFE_CALL(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeOf(size_t), @sz, ret)) ;
  setLength(bin[0], sz);
  SAFE_CALL(clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sz, pointer(bin), ret));
  result := bin[0]
end;

function CompileFromSource(
                          const source_string:ansistring;
                          const precision:TPrecision;
                          const routine_name:ansistring;
                          const device:TDevice;
                          const context:TContext;
                          const options:TArray<ansistring>;
                          const run_preprocessor:size_t; // 0: platform dependent, 1: always, 2: never
                          const silent:boolean = true):TProgram;
var header_string:ansistring;
  do_run_preprocessor: Boolean;
  kernel_string: ansistring;
  kernel_string_ptr :PAnsiChar;
  err, i: cl_int;
  buildStatus : cl_build_status;
  buildLog    : array[0..$7fff] of ansichar;
  buildLogstr, optionsStr  : ansistring;
  sz:size_t;
begin

  header_string := {header_string +} '#define PRECISION ' + ansistring(intToStr(ord(precision))) + sLineBreak;

  // Adds the name of the routine as a define
  header_string := header_string + '#define ROUTINE_' + routine_name + sLineBreak;

  // Not all OpenCL compilers support the 'inline' keyword. The keyword is only used for devices on
  // which it is known to work with all OpenCL platforms.
  if device.IsNVIDIA() or device.IsARM() or device.IsQualcomm() then
    header_string := header_string + '#define USE_INLINE_KEYWORD 1' + sLineBreak;

  // For specific devices, use the non-IEE754 compliant OpenCL mad() instruction. This can improve
  // performance, but might result in a reduced accuracy.
  if (device.IsAMD() and device.IsGPU()) or (device.IsQualcomm() and device.IsGPU()) then
    header_string := header_string + '#define USE_CL_MAD 1'+sLineBreak;

  // For specific devices, use staggered/shuffled workgroup indices.
  if (device.IsAMD() and device.IsGPU()) then
    header_string := header_string +'#define USE_STAGGERED_INDICES 1'+sLineBreak;

  // For specific devices add a global synchronisation barrier to the GEMM kernel to optimize
  // performance through better cache behaviour
  if (device.IsARM() and device.IsGPU()) or (device.IsQualcomm() and device.IsGPU())  then
    header_string := header_string+'#define GLOBAL_MEM_FENCE 1'+sLineBreak;

  // For Intel GPUs with subgroup support, use subgroup shuffling.
  if (device.IsGPU() and device.HasExtension(kKhronosIntelSubgroups)) and
      ((precision = kSingle) or (precision = kHalf)) then begin
    header_string := header_string +'#define USE_SUBGROUP_SHUFFLING 1'+sLineBreak;
    header_string := header_string +'#define SUBGROUP_SHUFFLING_INTEL 1'+sLineBreak;
  end;

  // For NVIDIA GPUs, inline PTX can provide subgroup support
  if device.IsGPU() and device.IsNVIDIA() and (precision = kSingle) then begin
    header_string := header_string+'#define USE_SUBGROUP_SHUFFLING 1'+sLineBreak;

    // Nvidia needs to check pre or post volta due to new shuffle commands
    if device.IsPostNVIDIAVolta() then
      header_string := header_string+'#define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 1'+sLineBreak
    else
      header_string := header_string +'#define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 1'+sLineBreak;
  end;

  // For Qualcomm devices, specifying the OpenCL kernel attribute reqd_work_group_size reduces performance.
  // This option compiles without the workgroup size requirement and does not affect correctness.
  if device.IsQualcomm() then
    header_string := header_string+'#define RELAX_WORKGROUP_SIZE 1'+sLineBreak;

  // Optionally adds a translation header from OpenCL kernels to CUDA kernels
  //#ifdef CUDA_API
  //  header_string +=
  //    #include "kernels/opencl_to_cuda.h"
  //  ;
  //end;
  //#endif

  // Loads the common header (typedefs and defines and such)
  header_string :=  header_string+ {$include inc/common.opencl.inc} ;

  // Prints details of the routine to compile in case of debugging in verbose mode
  //#ifdef VERBOSE
  //  printf("[DEBUG] Compiling routine '%s-%s'\n",
  //         routine_name.c_str(), ToString(precision).c_str());
  //  const auto start_time = std::chrono::steady_clock::now();
  //#endif

  // Runs a pre-processor to unroll loops and perform array-to-register promotion. Most OpenCL
  // compilers do this, but some don't.
  do_run_preprocessor := false;
  if run_preprocessor = 0 then  do_run_preprocessor := device.IsARM() and device.IsGPU();
  if run_preprocessor = 1 then  do_run_preprocessor := true;
  kernel_string := header_string + source_string;
  if do_run_preprocessor then begin
    //log_debug("Running built-in pre-processor");
    assert(false, 'Requesting un-implemented Preprocessor!');
    //kernel_string := PreprocessKernelSource(kernel_string);
  end;
  optionsStr := '';
  for i:=0 to High(options) do
    optionsStr := optionsStr+' '+options[i];
  kernel_string_ptr := PAnsiChar(kernel_string);
  // Compiles the kernel
  sz := length(kernel_string);
  result := clCreateProgramWithSource(context, 1, @kernel_string_ptr, @sz, err); SAFE_CALL(err);
  clBuildProgram(result, 1, @device, PAnsiChar('-cl-kernel-arg-info -cl-std=CL2.0 -cl-fast-relaxed-math -Werror -cl-mad-enable'+optionsStr), nil, nil);
  SAFE_CALL(clGetProgramBuildInfo(result, device, CL_PROGRAM_BUILD_STATUS, sizeOf(buildStatus), @BuildStatus, sz));
  SAFE_CALL(clGetProgramBuildInfo(result, device, CL_PROGRAM_BUILD_LOG, sizeOf(buildLog), @buildLog[0], sz));
  buildLogstr := trim(system.copy(buildLog,0, sz));
  assert(buildStatus=CL_BUILD_SUCCESS, '[OpenCL] : cannot compile kernels :' + sLineBreak+ buildlogStr);

  // Prints the elapsed compilation time in case of debugging in verbose mode

end;

function ifthen(const cond: boolean; const ifTrue: size_t; const ifFalse: size_t): size_t;
begin
  if cond then result :=ifTrue else result := ifFalse
end;

function CeilDiv(const x, y: size_t): size_t;
begin
   result := 1 + ((x - 1) div y);
end;

function Ceil(const x, y: size_t): size_t;
begin
   result := CeilDiv(x,y)*y;
end;

function IsMultiple(const a, b: size_t): boolean;
begin
   result :=(a div b)*b = a;
end;

{ TKernel }

constructor TKernel.Create(const aProgram: TProgram; const aName: ansistring);
var err: cl_int;
begin
  kernel := clCreateKernel(aProgram, PAnsiChar(aName), err);
  SAFE_CALL(err)
end;

function TKernel.Name: ansistring;
var a: array[0..$ff] of ansichar;
  sz:size_t;
begin
  SAFE_CALL(clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeOf(a), @a[0], sz));
  result:= a
end;

procedure TKernel.setArgument<T>(const index: cl_uint; const value: T);
begin
  SAFE_CALL(clSetKernelArg(kernel, index, SizeOf(T), @value))
end;

procedure TKernel.setArgument<T>(const index: cl_uint; const value: TBuffer<T>);
begin
   SAFE_CALL(clSetKernelArg(kernel, index, SizeOf(cl_mem), @value.buf))
end;

class operator TKernel.implicit(const src: TKernel): cl_kernel;
begin
  result := src.kernel
end;

class operator TKernel.implicit(const src: cl_kernel): TKernel;
begin
  result.kernel := src
end;

class operator TKernel.initialize({$ifdef fpc}var{$else}out{$endif} dst: TKernel);
begin
  dst.kernel:=nil;
end;

{ TDevice }

function TDevice.getInfo<T>(const info: cl_device_info): T;
var
  sz: size_t;
begin
  clGetDeviceInfo(devId, info, SizeOf(result), @result, sz);
end;

function TDevice.getInfoString(const info: cl_device_info): ansistring;
var r : array[0..$7fff] of AnsiChar;
  sz: size_t;
begin
  clGetDeviceInfo(devId, info, SizeOf(r), @r, sz);
  result := r
end;

function TDevice.getInfoVector<T>(const info: cl_device_info): TArray<T>;
var sz: size_t;
begin
  clGetDeviceInfo(devId, info, 0, nil, sz);
  setLength(result, sz div sizeOf(T));
  clGetDeviceInfo(devId, info, sz, @result[0], sz);
end;

function TDevice.&Type: cl_device_type;
begin
  result := getInfo<cl_device_type>(CL_DEVICE_TYPE_INFO)
end;

function TDevice.TypeStr: ansistring;
var
  buffCnt : size_t;
  dt :cl_device_type;
begin
  SAFE_CALL(clGetDeviceInfo(devId ,CL_DEVICE_TYPE_INFO , SizeOf(cl_device_type), @dt, buffCnt));
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

function TDevice.Name(): ansistring;
begin
  result := getInfoString(CL_DEVICE_NAME)
end;

function TDevice.Vendor: ansistring;
begin
  result := getInfoString(CL_DEVICE_VENDOR)
end;

function TDevice.MaxWorkGroupSize(): size_t;
begin
  result := getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE)
end;

function TDevice.MaxWorkItemDimensions: size_t;
begin
  result := getInfo<size_t>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
end;

function TDevice.MaxWorkItemSizes(): TArray<size_t>;
begin
  result := getInfoVector<size_t>(CL_DEVICE_VENDOR)
end;

function TDevice.LocalMemSize(): cl_ulong;
begin
  result := getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE)
end;

function TDevice.Capabilities(): ansistring;
begin
  result := getInfoString(CL_DEVICE_EXTENSIONS)
end;

function TDevice.HasExtension(const ext: ansistring): boolean;
begin
  result := pos(ext, Capabilities)>0
end;

function TDevice.CoreClock(): size_t;
begin
  result := getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY)
end;

function TDevice.ComputeUnits(): size_t;
begin
  result := getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS)
end;

function TDevice.MemoryClock(): size_t;
begin
  result := 0; //no CL support
end;

function TDevice.MemoryBusWidth(): size_t;
begin
  result := 0 // no CL support
end;

function TDevice.MemorySize(): cl_ulong;
begin
  result := getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE)
end;

function TDevice.MaxAllocSize(): cl_ulong;
begin
  result := getInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE)
end;

function TDevice.IsLocalMemoryValid(const local_mem_usage: cl_ulong): boolean;
begin
  result := local_mem_usage <= LocalMemSize();
end;

function TDevice.IsThreadConfigValid(const local: TArray<size_t>): boolean;
var local_size, item, i:size_t;
begin
  result := false;
  local_size := 1;
  for item in local do
    local_size := local_size * item;
  for i:=0 to high(local) do
    if local[i] > MaxWorkItemSizes()[i] then exit ;
  if local_size > MaxWorkGroupSize() then exit;
  if length(local) > MaxWorkItemDimensions() then exit;
  result := true
end;

function TDevice.IsPostNVIDIAVolta(): boolean;
begin
  result := false;
  if HasExtension('cl_nv_device_attribute_query') then
    result := GetInfo<cl_uint>($4000{CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV}) >= 7;
end;

function TDevice.SupportsFP64(): boolean;
begin
  result := HasExtension('cl_khr_fp64')
end;

function TDevice.SupportsFP16(): boolean;
begin
  if Name() = 'Mali-T628' then exit(true);
  result := HasExtension('cl_khr_fp16');
end;

function TDevice.IsCPU(): boolean;
begin
  result := &type=CL_DEVICE_TYPE_CPU;
end;

function TDevice.IsGPU(): boolean;
begin
  result := &type=CL_DEVICE_TYPE_GPU;
end;

function TDevice.IsAMD(): boolean;
var ven : ansistring;
begin
  ven := vendor;
  result := (ven='AMD') or (pos('Advanced Micro Devices', ven)>0);
end;

function TDevice.IsNVIDIA(): boolean;
var
  ven: ansistring;
begin
  ven := vendor();
  result := pos('NVIDIA', ven)>0 ;
end;

function TDevice.IsIntel(): boolean;
var
  ven: ansistring;
begin
  ven := vendor();
  result := pos('intel', AnsiLowerCase(ven))>0 ;
end;

function TDevice.IsARM(): boolean;
var
  ven: ansistring;
begin
  ven := vendor();
  result := ven='ARM' ;
end;

function TDevice.IsQualcomm(): boolean;
var
  ven: ansistring;
begin
  ven := vendor();
  result := ven='QUALCOMM' ;
end;

function TDevice.AMDBoardName(): ansistring;
begin
  result := getInfoString($4038 {CL_DEVICE_BOARD_NAME_AMD});
end;

function TDevice.NVIDIAComputeCapability(): ansistring;
begin
  result := 'SM' + ansistring(IntTostr(getInfo<cl_uint>($4000 {CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV})))
           +'.'  + ansistring(IntTostr(getInfo<cl_uint>($4001 {CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV})))
end;

function TDevice.AdrenoVersion(): ansistring;
begin
  if IsQualcomm() then
    result := GetInfoString(CL_DEVICE_OPENCL_C_VERSION)
  else
    result :=''
end;

function TDevice.GetExtraInfo(): ansistring;
begin
  result := '';
  if HasExtension('cl_amd_device_attribute_query') then
    exit(AMDBoardName());
  if HasExtension('cl_nv_device_attribute_query') then
    exit(NVIDIAComputeCapability());
end;

class operator TDevice.Implicit(const dev: cl_device_id): TDevice;
begin
  result.devId := dev;
end;

class operator TDevice.Implicit(const dev: TDevice): cl_device_id;
begin
  result := dev.devId;
end;

{ TBuffer }

function TBuffer<T>.GetSize(): size_t;
var sz:size_t;
begin
  SAFE_CALL(clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeOf(size_t), @result, sz))
end;

constructor TBuffer<T>.create(const aContext: TContext; const size: size_t; const access: cl_mem_flags; const host: pointer);
var err:cl_int;
begin
  buf := clCreateBuffer(aContext, access, size*sizeOf(T), host, err);
  SAFE_CALL(err)
end;

function TBuffer<T>.Context(): TContext;
var sz:size_t;
begin
  SAFE_CALL(clGetMemObjectInfo(buf, CL_MEM_CONTEXT, sizeOf(TContext), @result, sz));
end;

procedure TBuffer<T>.Alloc(const aContext: TContext; const size: size_t; const access: cl_mem_flags; const host: pointer);
var err:cl_int;
begin
  buf := clCreateBuffer(aContext, access, size*sizeOf(T), host, err);
  SAFE_CALL(err)
end;

procedure TBuffer<T>.free;
begin
  SAFE_CALL(clReleaseMemObject(buf));
end;

class operator TBuffer<T>.implicit(const buf: cl_mem): TBuffer<T>;
begin
  result.buf:=buf
end;

class operator TBuffer<T>.implicit(const buf: TBuffer<T>): cl_mem;
begin
  result := buf.buf
end;

{ TDatabaseArchitecture }

function TDatabaseArchitecture.lastDev: PDatabaseDevice;
begin
  result := nil;
  if length(devices)>0 then
    result := @devices[high(devices)]
end;

function TDatabaseArchitecture.addDevice(const aDevName: ansistring;
  const aParams: TParams): PDatabaseDevice;
begin
  setLength(devices, length(devices)+1);
  result := lastDev();
  result.name:=aDevName;
  result.parameters:=aParams;
end;

{ TDatabaseVendor }

function TDatabaseVendor.lastArch(): PDatabaseArchitecture;
begin
  result := nil;
  if length(architectures)>0 then
    result := @architectures[high(architectures)]
end;

function TDatabaseVendor.addArch(const aName: ansistring): PDatabaseArchitecture;
begin
  setLength(architectures, length(architectures)+1);
  result := lastArch();
  result.name:=aName;
end;

{ TDatabaseEntry }

function TDatabaseEntry.init(const aKernel: ansistring;
  const aPrecision: TPrecision; const aDeviceType: ansistring;
  const aVendorName: ansistring; const aArchName: ansistring;
  const aDeviceName: ansistring; const aParamNames: TArray<ansistring>;
  const aDeviceParams: TParams): PDatabaseEntry;
begin
  self.kernel:=aKernel;
  self.precision:=TPrecision.kAny;
  self.parameter_names := aParamNames;
  setLength(self.vendors, 1);
  vendors[0].devType:= aDeviceType;
  vendors[0].name := aVendorName;
  setLength(vendors[0].architectures, 1);
  vendors[0].architectures[0].name:=aArchName;
  setLength(vendors[0].architectures[0].devices, 1);
  vendors[0].architectures[0].devices[0].name:= aDeviceName;
  vendors[0].architectures[0].devices[0].parameters := aDeviceParams;
  result := @self
end;

function TDatabaseEntry.init(const aKernel: ansistring;
  const aPrecition: TPrecision; const aParamNames: TArray<ansistring>
  ): PDatabaseEntry;
begin
  kernel := aKernel;
  precision := aPrecition;
  parameter_names := aParamNames;
  result := @self
end;

function TDatabaseEntry.lastVendor(): PDatabaseVendor;
begin
  result := nil;
  if length(vendors)>0 then
    result:= @vendors[high(vendors)];
end;

function TDatabaseEntry.addVendor(const aDeviceType, aVendorName: ansistring): PDatabaseVendor;
begin
  setLength(vendors, length(vendors)+1);
  result := lastVendor();
  result.devType:=aDeviceType;
  result.name:=aVendorName;
end;



{ TDatabase }

procedure TDatabase.setItem(key : ansistring; AValue: size_t);
begin
  parameters_.Add(key, AValue);
end;

constructor TDatabase.Create(const device: TDevice;
  const kernel_name: ansistring; const aPrecision: TPrecision;
  const overlay: TArray<TDatabaseEntry>);
type TDatabaseList = TArray<TArray<TDatabaseEntry>>;
var
  device_type, device_vendor, device_architecture, device_name,
    extensions: ansistring;
  is_apple, is_likely_apple: Boolean;
  databases : TArray<TArray<TDatabaseEntry>>;
  search_result: TParameters;
  search_pair : TPair<ansistring, size_t>;
  i: Integer;
begin
  parameters_ := TParameters.Create();
  if not assigned(database) then
    database := [
      XaxpyHalf,        XaxpySingle,        XaxpyDouble,        XaxpyComplexSingle,            XaxpyComplexDouble,
      XdotHalf,         XdotSingle,         XdotDouble,         XdotComplexSingle,             XdotComplexDouble,
      XgemvHalf,        XgemvSingle,        XgemvDouble,        XgemvComplexSingle,            XgemvComplexDouble,
      XgemvFastHalf,    XgemvFastSingle,    XgemvFastDouble,    XgemvFastComplexSingle,        XgemvFastComplexDouble,
      XgemvFastRotHalf, XgemvFastRotSingle, XgemvFastRotDouble, XgemvFastRotComplexSingle,     XgemvFastRotComplexDouble,
      XgerHalf,         XgerSingle,         XgerDouble,         XgerComplexSingle,             XgerComplexDouble,
      XgemmHalf,        XgemmSingle,        XgemmDouble,        XgemmComplexSingle,            XgemmComplexDouble,
      XgemmDirectHalf,  XgemmDirectSingle,  XgemmDirectDouble,  XgemmDirectComplexSingle,      XgemmDirectComplexDouble,
      XconvgemmHalf,    XconvgemmSingle,    XconvgemmDouble,    XconvgemmComplexSingle,        XconvgemmComplexDouble,
      CopyHalf,         CopySingle,         CopyDouble,         CopyComplexSingle,             CopyComplexDouble,
      PadHalf,          PadSingle,          PadDouble,          PadComplexSingle,              PadComplexDouble,
      TransposeHalf,    TransposeSingle,    TransposeDouble,    TransposeComplexSingle,        TransposeComplexDouble,
      PadtransposeHalf, PadtransposeSingle, PadtransposeDouble, PadtransposeComplexSingle,     PadtransposeComplexDouble,
      InvertHalf,       InvertSingle,       InvertDouble,        InvertComplexSingle,           InvertComplexDouble,
      GemmRoutineHalf,  GemmRoutineSingle,  GemmRoutineDouble,  GemmRoutineComplexSingle,      GemmRoutineComplexDouble,
      TrsvRoutineHalf,  TrsvRoutineSingle,  TrsvRoutineDouble,  TrsvRoutineComplexSingle,      TrsvRoutineComplexDouble
    ] ;

  device_type := device.TypeStr();
  device_vendor := GetDeviceVendor(device);
  device_architecture := GetDeviceArchitecture(device);
  device_name := GetDeviceName(device);
  databases := [overlay, database];

  {$if defined(DARWIN) or defined(MACOSX)}
    if device.&Type() = CL_DEVICE_TYPE_CPU then begin
      extensions := device.Capabilities();
      is_apple := Pos('cl_APPLE_SetMemObjectDestructor', extensions)>0;
      is_likely_apple := device.MaxWorkGroupSize() <= 32;
      if is_apple or is_likely_apple then
        insert(apple_cpu_fallback, databases, length(databases);
    end;
  {$endif}
  for i:=0 to high(databases) do begin
    search_result := Search(kernel_name, device_vendor, device_type, device_name, device_architecture, aPrecision, databases[i]);
    if assigned(search_result) and (search_result.count>0) then begin
      for search_pair in search_result do
        parameters_.Add(search_pair.Key, search_pair.Value);
      break;
    end;
  end;
  //if not assigned(search_result) then
  //  raise Exception.create('Database Error!')
end;

procedure TDatabase.free;
begin
  freeAndNil(self.parameters_);
end;

function TDatabase.getItem(key: ansistring): size_t;
begin
  parameters_.TryGetValue(key, result);
end;

function TDatabase.exists(const key: ansistring): boolean;
begin
  result := parameters_.ContainsKey(key);
end;

function TDatabase.GetDefines(): ansistring;
var defines : TPair<ansistring, size_t>;
begin
  result :='';
  for defines in parameters_ do
    result  := result + '#define ' + defines.Key+' '+ansistring(intToStr(defines.Value)) + sLineBreak;
end;

function TDatabase.GetValuesString(): ansistring;
var defines : TPair<ansistring, size_t>;
begin
  result := '';
  for defines in parameters_ do
    result  := result + '_' + ansistring(intToStr(defines.Value));
end;

function TDatabase.GetParameterNames(): TArray<ansistring>;
var defines : TPair<ansistring, size_t>;
begin
  result := nil;
  for defines in parameters_ do
    insert(defines.Key, result, length(result))
end;

function TDatabase.GetParameters(): TParameters;
begin
  result := parameters_;
end;

function TDatabase.Search(const this_kernel, this_vendor, this_type,
  this_device, this_architecture: ansistring; const this_precision: TPrecision;
  const dbs: TArray<TDatabaseEntry>): TParameters;
var i:integer;
begin
  // Selects the right kernel
  for i:=0 to high(dbs) do begin
    if (dbs[i].kernel = this_kernel) and ((dbs[i].precision = this_precision) or (dbs[i].precision = kAny)) then begin

      // Searches for the right vendor and device type, or selects the default if unavailable
      result := SearchVendorAndType(this_vendor, this_type, this_device, this_architecture, dbs[i].vendors, dbs[i].parameter_names);
      if assigned(result) and (result.Count <> 0) then exit;
      result := SearchVendorAndType(kDeviceVendorAll, kDeviceTypeAll, this_device, this_architecture, dbs[i].vendors, dbs[i].parameter_names);
      exit
    end
  end;

  result := nil
end;

function TDatabase.SearchDevice(const target_device: ansistring;
  const devices: TArray<TDatabaseDevice>; const parameter_names: TArray<ansistring>): TParameters;
var i, j:integer;
  target_device_cut_off: ansistring;
begin
  target_device_cut_off := target_device;
  for i:=0 to high(devices) do begin
    // Cuts off 'target_device' string at 50 since the database cuts off as well
    //if length(target_device) > 50 then
    //  target_device_cut_off := copy(target_device, 1, 50)
    //else
    if trim(devices[i].name) = target_device_cut_off then begin
      //log_debug("Found parameters for device type '" + target_device_cut_off + "'");

      // Sets the parameters accordingly
      if length(parameter_names) > length(devices[i].parameters) then
        exit(nil); // ERROR
      result := TParameters.create;
      for j := 0 to high(parameter_names) do
        result.AddOrSetValue(parameter_names[j], devices[i].parameters[j]);
      exit
    end
  end;
  result := nil
end;

function TDatabase.SearchArchitecture(const target_architecture,
  this_device: ansistring; const architectures: TArray<TDatabaseArchitecture>;
  const parameter_names: TArray<ansistring>): TParameters;
var i:integer;
begin
  for i:=0 to high(architectures) do begin
    if architectures[i].name = target_architecture then begin
      //log_debug("Found devices of architecture type '" + target_architecture + "'");

      // Searches the device; if unavailable returns the architecture's default parameters
      result := SearchDevice(this_device, architectures[i].devices, parameter_names);
      if assigned(result) and (result.count <> 0) then exit;
      result := SearchDevice('default', architectures[i].devices, parameter_names);
      exit
    end
  end;
  result := nil
end;

function TDatabase.SearchVendorAndType(const target_vendor, target_type,
  this_device, this_architecture: ansistring; const vendors: TArray<TDatabaseVendor>; const parameter_names: TArray<ansistring>): TParameters;
var i: integer;
begin
  for i:=0 to high(vendors) do begin
    if (vendors[i].name = target_vendor) and (vendors[i].devType = target_type) then begin
      //log_debug("Found architectures of vendor '" + target_vendor + "' and type '" + target_type + "'");

      // Searches the architecture; if unavailable returns the vendor's default parameters
      result := SearchArchitecture(this_architecture, this_device, vendors[i].architectures, parameter_names);
      if assigned(result) and (result.count <> 0) then exit;
      result := SearchArchitecture('default', this_device, vendors[i].architectures, parameter_names);
      exit
    end;
  end;
  result := nil;
end;

{ TDatabases }

function TDatabases.getDatabases(key : ansistring): TDatabase;
begin
  databases_.TryGetValue(key, result)
end;

procedure TDatabases.setDatabases(key : ansistring; AValue: TDatabase);
begin
  databases_.AddOrSetValue(key, AValue);
end;

constructor TDatabases.create(const kernel_names: TArray<ansistring>);
begin
  kernel_names_ := kernel_names;
  databases_ := TDictionary<ansistring, TDatabase>.create();
end;

procedure TDatabases.free();
var pair : TPair<ansistring, TDatabase>;
begin
  for pair in databases_ do
    pair.Value.free();
  freeAndNil(databases_)
end;

function TDatabases.get(const kernel_name: ansistring): TDatabase;
begin
  databases_.TryGetValue(kernel_name, result);
end;

function TDatabases.getParams(key: ansistring): size_t;
var
  i: integer;
  kernel_db : TDatabase;
begin
  for i:=0 to high(kernel_names_) do begin
    databases_.tryGetValue(kernel_names_[i], kernel_db);
    if (kernel_db.exists(key)) then
      exit(kernel_db[key])
  end;
  raise Exception.create('Entry for the device was not found in the database');
end;

function TDatabases.setDatabase(const key: ansistring; const db: TDatabase): boolean;
begin
  result := databases_.tryAdd(key, db);
end;


{ TRoutine }

class function TRoutine.PrecisionValue<T>: TPrecision;
begin
  if TypeInfo(Half)           = TypeInfo(T) then exit(kHalf);
  if TypeInfo(Single)         = TypeInfo(T) then exit(kSingle);
  if TypeInfo(Double)         = TypeInfo(T) then exit(kDouble);
  if TypeInfo(TComplexSingle) = TypeInfo(T) then exit(kComplexSingle);
  if TypeInfo(TComplexDouble) = TypeInfo(T) then exit(kComplexDouble);
end;

class procedure TRoutine.InitDatabase(const device: TDevice;
  const kernel_names: TArray<ansistring>; const precision: TPrecision;
  const userDatabase: TArray<TDatabaseEntry>; var dbs: TDatabases);
var
  i:integer;
  db : TDatabase;
begin

  for i:=0 to high(kernel_names) do begin
    // Builds the parameter database for this device and routine set and stores it in the cache
    if not dbs.databases_.ContainsKey(kernel_names[i]) then
      db := TDatabase.Create(device, kernel_names[i], precision, userDatabase);
      dbs.setDatabase(kernel_names[i], db);
    // Queries the cache to see whether or not the kernel parameter database is already there
      if databaseCache.TryAdd(TRoutineDesc.create({ActivePlatform, } device, precision, kernel_names[i]), db ) then

      //log_debug("Searching database for kernel '" + kernel_name + "'");
  end
end;

constructor TRoutine.Create(const queue: TQueue; const event: TEventPointer;
  const name: ansistring; const routines: TArray<ansistring>;
  const precision: TPrecision; const userDatabase: TArray<TDatabaseEntry>;
  const source: TInitList);
begin
  precision_    := precision          ;
  routine_name_ := name               ;
  kernel_names_ := routines;
  queue_        := queue              ;
  event_        := event              ;
  context_      := getContext(queue);
  device_       := getDevice(queue) ;
  db_           := TDatabases.create(routines);

  InitDatabase(device_, routines, precision, userDatabase, db_);
  InitProgram(source);
end;

procedure TRoutine.InitProgram(const source: TInitList);
var
  routine_info, kernel_name, environment_variable, device_name, source_string,s : ansiString;
  options : TArray<ansistring>;
  has_program, has_binary:boolean; bin:RawByteString; prog:cl_program;
  //kernelCount, i: cl_uint;
  //name_fixed : array[0..127] of ansichar;
  //sz : size_t;
begin
  // Determines the identifier for this particular routine call
   routine_info := routine_name_;
   for kernel_name in kernel_names_ do
     routine_info := routine_info +'_' + kernel_name + db_[kernel_name].GetValuesString();

   //log_debug(routine_info);
   //
   //// Queries the cache to see whether or not the program (context-specific) is already there
   //bool has_program;

   has_program := programCache.TryGetValue(TProgramDesc.create(context_, device_, precision_, routine_info), prog);
   if has_program then begin
     program_ := [prog];
     exit;
   end;
   //
   //// Sets the build options from an environmental variable (if set)
   environment_variable := GetEnvironmentVariable('CLBLAST_BUILD_OPTIONS');
   options := nil;
   if environment_variable <> '' then
     insert(environment_variable, options, length(options));
   //
   //// Queries the cache to see whether or not the binary (device-specific) is already there. If it
   //// is, a program is created and stored in the cache
   device_name := GetDeviceName(TDevice(device_));
   //const auto platform_id = device_.PlatformID();
   //bool has_binary;
   has_binary := binaryCache.TryGetValue(TBinDesc.create(precision_, routine_info, device_name), bin );

   if has_binary then begin

     prog := LoadFromBinary(bin, context_, device_);
     program_ := [prog];
     programCache.Add(TProgramDesc.Create(context_, device_, precision_, routine_info), prog);
     exit;
   end;

   // Otherwise, the kernel will be compiled and program will be built. Both the binary and the
   // program will be added to the cache.

   // Inspects whether or not FP64 is supported in case of double precision
   if (precision_ = kDouble) and not device_.SupportsFP64() then
     raise Exception.create('Device does not support Double');

   //As above, but for FP16 (half precision)
   if (precision_ = kHalf) and not device_.SupportsFP16() then
     raise Exception.Create('Device does not support FP16');

   // Collects the parameters for this device in the form of defines
   source_string := '';
   for kernel_name in kernel_names_ do
     source_string := source_string + db_.getDatabases(kernel_name).GetDefines();


   // Adds routine-specific code to the constructed source string
   for s in source do
     source_string := source_string + s;


   // Completes the source and compiles the kernel
   prog := CompileFromSource(source_string, precision_, routine_name_, device_, context_, options, 0);
   program_ := [prog];

   // Store the compiled binary and program in the cache
   programCache.add(TProgramDesc.create(context_, device_, precision_, routine_info), prog);

   //SAFE_CALL(clCreateKernelsInProgram(prog, 0, nil, kernelCount));
   //setLength(kernels_, kernelCount);
   //setLength(kernelNames_, kernelCount);
   //SAFE_CALL(clCreateKernelsInProgram(prog, kernelCount, Pointer(kernels_), kernelCount));
   //binaryCache.add(TBinDesc.create({platform_id, }precision_, routine_info, device_name), getProgramBinary(prog));
   //for i:= 0 to kernelCount-1 do begin
   //  clGetKernelInfo(kernels_[i], CL_KERNEL_FUNCTION_NAME, sizeOf(name_fixed), @name_fixed[0], sz);
   //  kernelNames_[i] := name_fixed
   //end;

end;

{ TRoutine.TRoutineDesc }

constructor TRoutine.TRoutineDesc.create(const aDevice: TDevice;
  const aPrecision: TPrecision; const aKernel: ansistring);
begin
  //platform := aPlatform;
  device:= adevice;
  precision := aPrecision;
  kernel := aKernel
end;

{ TRoutine.TProgramDesc }

constructor TRoutine.TProgramDesc.create(const aContext: cl_context;
  const aDeviceId: cl_device_id; const aPrecision: TPrecision;
  const aRoutineName: ansistring);
begin
  context     := aContext     ;
  deviceId    := aDeviceId    ;
  Precision   := aPrecision   ;
  routineName := aRoutineName
end;

{ TRoutine.TBinDesc }

constructor TRoutine.TBinDesc.create(const aPrecision: TPrecision;
  const aRoutineName: ansistring; const aDeviceName: ansistring);
begin
  Precision   := aPrecision   ;
  routineName := aRoutineName ;
  deviceName  := aDeviceName  ;
end;

{ TXgemm }

class function TXgemm<T>.a_want_rotated_(const gemm_kernel_id: size_t): boolean;
begin
  result := gemm_kernel_id=1;
end;

class function TXgemm<T>.b_want_rotated_(const gemm_kernel_id: size_t): boolean;
begin
  result := true
end;

class function TXgemm<T>.c_want_rotated_(const gemm_kernel_id: size_t): boolean;
begin
  result := gemm_kernel_id=1;
end;

class function TXgemm<T>.GetTempSize(const layout: TLayout; const a_transpose,
  b_transpose: TTranspose; const m, n, k, a_offset, a_ld, b_offset, b_ld,
  c_offset, c_ld, mwg, nwg, kwg, gemm_kernel_id: size_t): size_t;
var
  a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
  a_one, a_two, b_one, b_two, c_one, c_two,
  a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
  b_temp_offset, c_temp_offset:size_t;
  a_no_temp, b_no_temp, c_no_temp : boolean;

begin
// Computes the transpose/conjugate options and sets the a/b/c sizes based on that
   ProcessArguments(layout, a_transpose, b_transpose, m, n, k,
                    a_one, a_two, b_one, b_two, c_one, c_two,
                    a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                    gemm_kernel_id);

   // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
   // whether the matrices need to be rotated or not for the kernel.
   CalculateInternalDimensions(m, n, k, mwg, nwg, kwg,
                               a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                               gemm_kernel_id);

   // Determines whether or not temporary matrices are needed
   a_no_temp := NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
   b_no_temp := NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
   c_no_temp := NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

   // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
   b_temp_offset := 0;
   c_temp_offset := 0;
   result := ComputeTempSize(a_no_temp, b_no_temp, c_no_temp,
                          a_one_i*a_two_i, b_one_i*b_two_i, c_one_i*c_two_i,
                          b_temp_offset, c_temp_offset);
end;

class function TXgemm<T>.UseDirectKernel(const m, n, k, min_indirect_size: size_t): boolean;
var
  m_n_k, min_indirect_size_ll, min_indirect_size_e3: uint64;
begin
   m_n_k := m * n * k;
   min_indirect_size_ll := min_indirect_size;
   min_indirect_size_e3 := min_indirect_size_ll * min_indirect_size_ll * min_indirect_size_ll;
   result := m_n_k < min_indirect_size_e3;
end;

class procedure TXgemm<T>.ProcessArguments(const layout: TLayout;
  const a_transpose, b_transpose: TTranspose; const m, n, k: size_t; var a_one,
  a_two, b_one, b_two, c_one, c_two: size_t; var a_do_transpose,
  b_do_transpose, c_do_transpose, a_conjugate, b_conjugate: boolean;
  const gemm_kernel_id: size_t);
var
  a_rotated, b_rotated, c_rotated :boolean;
begin
   // Makes sure all dimensions are larger than zero
   if (m = 0) and (n = 0) and (k = 0) then
     raise Exception.create('Invalid dimensions');

   // Computes whether or not the matrices are transposed in memory. This is based on their layout
   // (row or column-major) and whether or not they are requested to be pre-transposed. Note
   // that the Xgemm kernel expects either matrices A and C (in case of row-major) or B (in case of
   // col-major) to be transformed, so transposing requirements are not the same as whether or not
   // the matrix is actually transposed in memory.
   a_rotated := ((layout = loColMajor) and (a_transpose <> trNo)) or ((layout = loRowMajor) and (a_transpose = trNo));
   b_rotated := ((layout = loColMajor) and (b_transpose <> trNo)) or ((layout = loRowMajor) and (b_transpose = trNo));
   c_rotated := layout = loRowMajor;
   a_do_transpose := a_rotated <> a_want_rotated_(gemm_kernel_id);
   b_do_transpose := b_rotated <> b_want_rotated_(gemm_kernel_id);
   c_do_transpose := c_rotated <> c_want_rotated_(gemm_kernel_id);

   // In case of complex data-types, the transpose can also become a conjugate transpose
   a_conjugate := (a_transpose = trConjugate);
   b_conjugate := (b_transpose = trConjugate);

   // Computes the first and second dimensions of the 3 matrices taking into account whether the
   // matrices are rotated or not
   a_one := ifthen(a_rotated, k, m);
   a_two := ifthen(a_rotated, m, k);
   b_one := ifthen(b_rotated, n, k);
   b_two := ifthen(b_rotated, k, n);
   c_one := ifthen(c_rotated, n, m);
   c_two := ifthen(c_rotated, m, n);

end;

class function TXgemm<T>.ComputeTempSize(const a_no_temp, b_no_temp,
  c_no_temp: boolean; const a_size, b_size, c_size: size_t; var b_temp_offset,
  c_temp_offset: size_t): size_t;
begin
   result := 0;
   if not a_no_temp then
      inc(result, a_size);
   if not b_no_temp then
     begin
       b_temp_offset := result;
       inc(result, b_size)
     end;
   if not c_no_temp then
     begin
       c_temp_offset := result;
       inc(result, c_size)
     end;
end;

class function TXgemm<T>.NoTempBuffer(const one, one_i, two, two_i, ld,
  offset: size_t; const do_transpose, conjugate: boolean): boolean;
begin
   result := (one = one_i) and (two = two_i) and (ld = one) and (offset = 0) and (not do_transpose and not conjugate);
end;

class procedure TXgemm<T>.CalculateInternalDimensions(const m, n, k, mwg, nwg,
  kwg: size_t; var a_one_i, a_two_i, b_one_i, b_two_i, c_one_i,
  c_two_i: size_t; const gemm_kernel_id: size_t);
var
  global_divider_one, global_divider_two, m_ceiled, n_ceiled, k_ceiled:size_t;
begin
   global_divider_one := ifthen(c_want_rotated_(gemm_kernel_id), nwg, mwg);
   global_divider_two := ifthen(c_want_rotated_(gemm_kernel_id), mwg, nwg);
   m_ceiled := Ceil(m, global_divider_one);
   n_ceiled := Ceil(n, global_divider_two);
   k_ceiled := Ceil(k, kwg);
   a_one_i := ifthen(a_want_rotated_(gemm_kernel_id), k_ceiled, m_ceiled);
   a_two_i := ifthen(a_want_rotated_(gemm_kernel_id), m_ceiled, k_ceiled);
   b_one_i := ifthen(b_want_rotated_(gemm_kernel_id), n_ceiled, k_ceiled);
   b_two_i := ifthen(b_want_rotated_(gemm_kernel_id), k_ceiled, n_ceiled);
   c_one_i := ifthen(c_want_rotated_(gemm_kernel_id), n_ceiled, m_ceiled);
   c_two_i := ifthen(c_want_rotated_(gemm_kernel_id), m_ceiled, n_ceiled);

end;

constructor TXgemm<T>.Create(const queue: TQueue; const event: TEventPointer; const name: ansistring);
begin
   inherited create(queue, event, name, ['Copy','Pad','Transpose','Padtranspose','Xgemm','XgemmDirect','GemmRoutine'],
           PrecisionValue<T>(), {userDatabase} nil, [
     {$include inc/level3.opencl.inc}              +
     {$include inc/copy_fast.opencl.inc}           +
     {$include inc/copy_pad.opencl.inc}            +
     {$include inc/transpose_fast.opencl.inc}      +
     {$include inc/transpose_pad.opencl.inc}       +
     {$include inc/convert_symmetric.opencl.inc}   +
     {$include inc/convert_triangular.opencl.inc}  +
     {$include inc/convert_hermitian.opencl.inc}
     , // separated in multiple parts to prevent C1091 in MSVC 2013
     {$include inc/xgemm_direct_part1.opencl.inc}  +
     {$include inc/xgemm_direct_part2.opencl.inc}  +
     {$include inc/xgemm_direct_part3.opencl.inc}
     , // separated in multiple parts to prevent C1091 in MSVC 2013
     {$include inc/xgemm_part1.opencl.inc}         +
     {$include inc/xgemm_part2.opencl.inc}
     , // separated in multiple parts to prevent C1091 in MSVC 2013
     {$include inc/xgemm_part3.opencl.inc}         +
     {$include inc/xgemm_part4.opencl.inc}
     ,
     {$include inc/xgemm_batched.opencl.inc}       +
     {$include inc/xgemm_direct_batched.opencl.inc}
   ]);

   XGEMM_MIN_INDIRECT_SIZE := db_.params['XGEMM_MIN_INDIRECT_SIZE'];
   MWG                     := db_.params['MWG'];
   NWG                     := db_.params['NWG'];
   KWG                     := db_.params['KWG'];
   VWM                     := db_.params['VWM'];
   VWN                     := db_.params['VWN'];
   GEMMK                   := db_.params['GEMMK'];
   MDIMC                   := db_.params['MDIMC'];
   NDIMC                   := db_.params['NDIMC'];
   MDIMCD                  := db_.params['MDIMCD'];
   NDIMCD                  := db_.params['NDIMCD'];
   WGD                     := db_.params['WGD'];
   KREG                    := db_.params['KREG'];
   TRA_WPT                 := db_.params['TRA_WPT'];
   TRA_DIM                 := db_.params['TRA_DIM'];
   COPY_VW                 := db_.params['COPY_VW'];
   COPY_WPT                := db_.params['COPY_WPT'];
   COPY_DIMX               := db_.params['COPY_DIMX'];
   COPY_DIMY               := db_.params['COPY_DIMY'];
   PADTRA_WPT              := db_.params['PADTRA_WPT'];
   PADTRA_TILE             := db_.params['PADTRA_TILE'];
   PAD_WPTX                := db_.params['PAD_WPTX'];
   PAD_WPTY                := db_.params['PAD_WPTY'];
   PAD_DIMX                := db_.params['PAD_DIMX'];
   PAD_DIMY                := db_.params['PAD_DIMY'];

end;

procedure TXgemm<T>.DoGemm(const layout: TLayout; const a_transpose,
  b_transpose: TTranspose; const m, n, k: size_t; const alpha: T;
  const a_buffer: TBuffer<T>; const a_offset, a_ld: size_t;
  const b_buffer: TBuffer<T>; const b_offset, b_ld: size_t; const beta: T;
  const c_buffer: TBuffer<T>; const c_offset, c_ld: size_t;
  const temp_buffer: cl_mem; const temp_buffer_provided: boolean);
var
  do_gemm_direct, a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
  gemm_kernel_id, a_one, a_two, b_one, b_two, c_one, c_two:size_t;
begin
   // Two methods to choose from, select which one to run
     do_gemm_direct := UseDirectKernel(m, n, k, XGEMM_MIN_INDIRECT_SIZE);
     gemm_kernel_id := ifthen(do_gemm_direct, 0, GEMMK);

     // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
     ProcessArguments(layout, a_transpose, b_transpose, m, n, k,
                      a_one, a_two, b_one, b_two, c_one, c_two,
                      a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                      gemm_kernel_id);

     // Tests three matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
     // their sizes, and then from a perspective of parameter values (e.g. m, n, k). Tests whether the
     // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
     // space. Also tests that the leading dimensions of:
     //    matrix A cannot be less than K when rotated, or less than M when not-rotated
     //    matrix B cannot be less than N when rotated, or less than K when not-rotated
     //    matrix C cannot be less than N when rotated, or less than M when not-rotated
     //TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
     //TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
     //TestMatrixC(c_one, c_two, c_buffer, c_offset, c_ld);

     // Selects which version of GEMM to run
     if do_gemm_direct then // for small sizes (single kernel)
       GemmDirect(m, n, k, alpha,
                  a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta,
                  c_buffer, c_offset, c_ld,
                  a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate)
     else  // for larger sizes (pre/post-processing plus a very fast kernel)
       GemmIndirect(m, n, k, alpha,
                    a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta,
                    c_buffer, c_offset, c_ld,
                    a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                    a_one, a_two, b_one, b_two, c_one, c_two,
                    temp_buffer, temp_buffer_provided);
end;

procedure TXgemm<T>.PadCopyTransposeMatrix(const event: TEventPointer;
  const waitForEvents: TArray<TEvent>; const src_one, src_two, src_ld,
  src_offset: size_t; const src: TBuffer<T>; const dest_one, dest_two, dest_ld,
  dest_offset: size_t; const dest: TBuffer<T>; const alpha: T; const do_pad,
  do_transpose, do_conjugate: boolean; const upper: boolean;
  const lower: boolean; const diagonal_imag_zero: boolean);
var
  use_fast_kernel, pad_kernel:boolean;
  kernel_name: ansistring;
  local, global, dummy : TArray<size_t>;
  kernel :TKernel;
begin
  // Determines whether or not the fast-version could potentially be used
  use_fast_kernel := (src_offset = 0) and (dest_offset = 0) and (do_conjugate = false) and
                         (src_one = dest_one) and (src_two = dest_two) and (src_ld = dest_ld) and
                         (upper = false) and (lower = false) and (diagonal_imag_zero = false);

  // Determines the right kernel
  kernel_name := '';
  pad_kernel := false;
  if do_transpose then begin
    if use_fast_kernel and
        IsMultiple(src_ld, TRA_WPT) and
        IsMultiple(src_one, TRA_WPT*TRA_DIM) and
        IsMultiple(src_two, TRA_WPT*TRA_DIM) then begin
      kernel_name := 'TransposeMatrixFast';
    end
    else begin
      use_fast_kernel := false;
      pad_kernel := do_pad or do_conjugate;
      if pad_kernel then
        kernel_name := 'TransposePadMatrix'
      else
        kernel_name := 'TransposeMatrix';
    end
  end
  else begin
    if use_fast_kernel and
        IsMultiple(src_ld, COPY_VW) and
        IsMultiple(src_one, COPY_VW*COPY_DIMX) and
        IsMultiple(src_two, COPY_WPT*COPY_DIMY) then begin
      kernel_name := 'CopyMatrixFast';
    end
    else begin
      use_fast_kernel := false;
      pad_kernel := do_pad;
      if pad_kernel then
        kernel_name := 'CopyPadMatrix'
      else
        kernel_name := 'CopyMatrix';
    end
  end;

  // Retrieves the kernel from the compiled binary
  kernel := TKernel.create(program_[0], kernel_name);

  // Sets the kernel arguments
  if (use_fast_kernel) then begin
    kernel.SetArgument<integer>(0, src_ld);
    kernel.SetArgument<T>(1, src);
    kernel.SetArgument<T>(2, dest);
    kernel.SetArgument<T>(3, alpha);
  end
  else begin
    kernel.SetArgument<integer>(0, src_one);
    kernel.SetArgument<integer>(1, src_two);
    kernel.SetArgument<integer>(2, src_ld);
    kernel.SetArgument<integer>(3, src_offset);
    kernel.SetArgument<T>(4, src);
    kernel.SetArgument<integer>(5, dest_one);
    kernel.SetArgument<integer>(6, dest_two);
    kernel.SetArgument<integer>(7, dest_ld);
    kernel.SetArgument<integer>(8, dest_offset);
    kernel.SetArgument<T>(9, dest);
    kernel.SetArgument<T>(10, alpha);
    if (pad_kernel) then begin
      kernel.SetArgument<integer>(11, ord(do_conjugate));
    end
    else begin
      kernel.SetArgument<integer>(11, ord(upper));
      kernel.SetArgument<integer>(12, ord(lower));
      kernel.SetArgument<integer>(13, ord(diagonal_imag_zero));
    end
  end;

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  dummy := [0, 0];
  if do_transpose then begin
    if use_fast_kernel then begin
      global := [
        dest_one div TRA_WPT,
        dest_two div TRA_WPT
      ];
      local := [TRA_DIM, TRA_DIM];
    end
    else begin
      global := [
        Ceil(CeilDiv(dest_one, PADTRA_WPT), PADTRA_TILE),
        Ceil(CeilDiv(dest_two, PADTRA_WPT), PADTRA_TILE)
      ];
      local := [PADTRA_TILE, PADTRA_TILE];
    end
  end
  else begin
    if (use_fast_kernel) then begin
      global := [
        dest_one div COPY_VW,
        dest_two div COPY_WPT
      ];
      local := [COPY_DIMX, COPY_DIMY];
    end
    else begin
      global := [
        Ceil(CeilDiv(dest_one, PAD_WPTX), PAD_DIMX),
        Ceil(CeilDiv(dest_two, PAD_WPTY), PAD_DIMY)
      ];
      local := [PAD_DIMX, PAD_DIMY];
    end
  end;
  SAFE_CALL(clEnqueueNDRangeKernel(queue_, kernel, length(global), pointer(dummy), pointer(global), pointer(local), length(waitForEvents), pointer(waitForEvents), event));
end;

procedure TXgemm<T>.PadCopyTransposeMatrixStridedBatched(
  const event: TEventPointer; const waitForEvents: TArray<TEvent>;
  const src_one, src_two, src_ld, src_offset: size_t; const src_stride: size_t;
  const src: TBuffer<T>; const dest_one, dest_two, dest_ld,
  dest_offset: size_t; const dest_stride: size_t; const dest: TBuffer<T>;
  const do_pad, do_transpose, do_conjugate: boolean; const batch_count: size_t);
var kernel_name : ansistring;
  kernel : TKernel;
  local, global, dummy : TArray<size_t>;
begin
  // Determines the right kernel

  if do_transpose then
    if do_pad then
      kernel_name := 'TransposePadMatrixStridedBatched' else kernel_name := 'TransposeMatrixStridedBatched'
    else
      if do_pad then kernel_name := 'CopyPadMatrixStridedBatched' else kernel_name := 'CopyMatrixStridedBatched';


  // Retrieves the kernel from the compiled binary
  kernel := TKernel.Create(program_[0], kernel_name);

  // Sets the kernel arguments
  kernel.SetArgument<integer>(0, src_one);
  kernel.SetArgument<integer>(1, src_two);
  kernel.SetArgument<integer>(2, src_ld);
  kernel.SetArgument<integer>(3, src_offset);
  kernel.SetArgument<integer>(4, src_stride);
  kernel.SetArgument<T>(5, src);
  kernel.SetArgument<integer>(6, dest_one);
  kernel.SetArgument<integer>(7, dest_two);
  kernel.SetArgument<integer>(8, dest_ld);
  kernel.SetArgument<integer>(9, dest_offset);
  kernel.SetArgument<integer>(10, dest_stride);
  kernel.SetArgument<T>(11, dest);
  if do_pad then
    kernel.SetArgument<integer>(12, ord(do_conjugate));

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  dummy := [0, 0, 0];
  if do_transpose then begin
    global := [
        Ceil(CeilDiv(dest_one, PADTRA_WPT), PADTRA_TILE),
        Ceil(CeilDiv(dest_two, PADTRA_WPT), PADTRA_TILE),
        batch_count
    ];
   local := [PADTRA_TILE, PADTRA_TILE, 1];
  end
  else begin
    global := [
        Ceil(CeilDiv(dest_one, PAD_WPTX), PAD_DIMX),
        Ceil(CeilDiv(dest_two, PAD_WPTY), PAD_DIMY),
        batch_count
    ];
    local := [PAD_DIMX, PAD_DIMY, 1];
  end;
  SAFE_CALL(clEnqueueNDRangeKernel(queue_, kernel, length(global), pointer(dummy), pointer(global), pointer(local), length(waitForEvents), pointer(waitForEvents), event));

end;

procedure TXgemm<T>.GemmIndirect(const m, n, k: size_t; const alpha: T;
  const a_buffer: TBuffer<T>; const a_offset, a_ld: size_t;
  const b_buffer: TBuffer<T>; const b_offset, b_ld: size_t; const beta: T;
  const c_buffer: TBuffer<T>; const c_offset, c_ld: size_t;
  const a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate,
  b_conjugate: boolean; const a_one, a_two, b_one, b_two, c_one, c_two: size_t;
  const temp_buffer: cl_mem; const temp_buffer_provided: boolean);
var
  a_no_temp, b_no_temp, c_no_temp : boolean;
  global_divider_one, global_divider_two, m_ceiled, n_ceiled,
  k_ceiled, a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
  b_temp_offset, c_temp_offset, temp_size, required_size: size_t;
  temp_buffer_all, a_temp, b_temp, c_temp : TBuffer<T>;
  err : cl_int;
  eventWaitList :TArray<TEvent>;
  emptyEventList:TArray<TEvent>;
  eventProcessA, eventProcessB, eventProcessC, eventKernel:TEvent;
  kernel : TKernel;
  global, local :TArray<size_t>;
  eventPointer : Pcl_event;
  dummy : TArray<size_t>;
begin
   // Calculates the ceiled versions of m, n, and k
   global_divider_one := ifthen(c_want_rotated_(GEMMK), NWG, MWG);
   global_divider_two := ifthen(c_want_rotated_(GEMMK), MWG, NWG);
   m_ceiled := Ceil(m, global_divider_one);
   n_ceiled := Ceil(n, global_divider_two);
   k_ceiled := Ceil(k, KWG * KREG);

   // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
   // whether the matrices need to be rotated or not for the kernel.

   CalculateInternalDimensions(m, n, k, MWG, NWG, KWG * KREG,
                               a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                               GEMMK);

   // Determines whether or not temporary matrices are needed
   a_no_temp := NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
   b_no_temp := NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
   c_no_temp := NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

   // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
   b_temp_offset := 0;
   c_temp_offset := 0;
   temp_size := ComputeTempSize(a_no_temp, b_no_temp, c_no_temp,
                                          a_one_i*a_two_i, b_one_i*b_two_i, c_one_i*c_two_i,
                                          b_temp_offset, c_temp_offset);
   if not IsMultiple(b_temp_offset, VWN) then raise Exception.create('kUnexpectedError');
   if not IsMultiple(c_temp_offset, VWM) then raise Exception.create('kUnexpectedError');

   // Creates the buffer for the (optional) temporary matrices. Note that we use 'a_buffer' in case
   // when no temporary buffer is needed, but that's just to make it compile: it is never used.
   if temp_buffer_provided then
     temp_buffer_all:=temp_buffer
   else
     if temp_size>0 then
       temp_buffer_all.alloc(context_, temp_size)
     else
       temp_buffer_all := a_buffer;

   // Verifies if the provided temporary buffer is large enough
   if temp_buffer_provided then begin
     required_size := temp_size * sizeof(T);
     if temp_buffer_all.GetSize() < required_size then raise Exception.create('kInsufficientMemoryTemp');
   end;

    //Sets the buffer pointers for (temp) matrices A, B, and C
   if a_no_temp then a_temp := a_buffer else a_temp := temp_buffer_all;
   if b_no_temp then b_temp := b_buffer else b_temp := temp_buffer_all;
   if c_no_temp then c_temp := c_buffer else c_temp := temp_buffer_all;

   // Events of all kernels (including pre/post processing kernels)

   // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
   // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
   // case nothing has to be done, these kernels can be skipped.
   eventWaitList := nil;
   emptyEventList := nil;
   if not a_no_temp then begin
     eventProcessA := nil;
     //Assert(false, 'PadCopyTransposeMatrix not implemented');
     PadCopyTransposeMatrix(@eventProcessA, emptyEventList,
                            a_one, a_two, a_ld, a_offset, a_buffer,
                            a_one_i, a_two_i, a_one_i, 0, a_temp,
                            ConstantOne,
                            true, a_do_transpose, a_conjugate);
     insert(eventProcessA, eventWaitList, length(eventWaitList));
   end;

   // As above, but now for matrix B
   if not b_no_temp then begin
     eventProcessB := nil;
     //Assert(false, 'PadCopyTransposeMatrix not implemented');
     PadCopyTransposeMatrix(@eventProcessB, emptyEventList,
                            b_one, b_two, b_ld, b_offset, b_buffer,
                            b_one_i, b_two_i, b_one_i, b_temp_offset, b_temp,
                            ConstantOne,
                            true, b_do_transpose, b_conjugate);
     insert(eventProcessB, eventWaitList, length(eventWaitList));
   end;

   // As above, but now for matrix C. This is only necessary if C is used both as input and output.
   if not c_no_temp and (TComparer<T>.default.compare(beta, default(T))<>0) then begin
     eventProcessC := nil;
     //Assert(false, 'PadCopyTransposeMatrix not implemented');
     PadCopyTransposeMatrix(@eventProcessC, emptyEventList,
                            c_one, c_two, c_ld, c_offset, c_buffer,
                            c_one_i, c_two_i, c_one_i, c_temp_offset, c_temp,
                            ConstantOne,
                            true, c_do_transpose, false);
     insert(eventProcessC, eventWaitList, length(eventWaitList));
   end;

   // Retrieves the Xgemm kernel from the compiled binary
   kernel := TKernel.create(program_[0], 'Xgemm');

   // Sets the kernel arguments
   kernel.SetArgument<integer>(0, m_ceiled);
   kernel.SetArgument<integer>(1, n_ceiled);
   kernel.SetArgument<integer>(2, k_ceiled);
   kernel.SetArgument<T>(3, {GetRealArg(}alpha{)});
   kernel.SetArgument<T>(4, {GetRealArg(}beta{)});
   kernel.SetArgument<T>(5, a_temp);
   kernel.SetArgument<T>(6, b_temp);
   kernel.SetArgument<T>(7, c_temp);
   kernel.SetArgument<integer>(8, b_temp_offset div VWN);
   kernel.SetArgument<integer>(9, c_temp_offset div VWM);

   // Computes the global and local thread sizes
   global := [
     (c_one_i * MDIMC) div MWG,
     (c_two_i * NDIMC) div NWG
   ];
   local := [MDIMC, NDIMC];

   // Launches the kernel
   if not c_no_temp then
     eventPointer := @eventKernel else eventPointer := event_;
   dummy := [0, 0];
   //RunKernel(kernel, queue_, device_, global, local, eventPointer, eventWaitList);
   SAFE_CALL(clEnqueueNDRangeKernel(queue_, kernel, length(global), pointer(dummy),
       pointer(global), pointer(local),
       length(eventWaitList), pointer(eventWaitList), eventPointer));

   // Runs the post-processing kernel if needed
   if (not c_no_temp) then begin
     //Assert(false, 'PadCopyTransposeMatrix not implemented');
     insert(eventKernel, eventWaitList, length(eventWaitList));
     PadCopyTransposeMatrix(event_, eventWaitList,
                            c_one_i, c_two_i, c_one_i, c_temp_offset, c_temp,
                            c_one, c_two, c_ld, c_offset, c_buffer,
                            ConstantOne,
                            false, c_do_transpose, false);

   end
end;

procedure TXgemm<T>.GemmDirect(const m, n, k: size_t; const alpha: T;
  const a_buffer: TBuffer<T>; const a_offset, a_ld: size_t;
  const b_buffer: TBuffer<T>; const b_offset, b_ld: size_t; const beta: T;
  const c_buffer: TBuffer<T>; const c_offset, c_ld: size_t;
  const a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate,
  b_conjugate: boolean);
var
  name : ansistring;
  kernel : TKernel;
  local, global :TArray<size_t>;
  dummy :TArray<size_t>;
  m_ceiled : size_t;
  n_ceiled : size_t;
begin
   // Retrieves the proper XgemmDirect kernel from the compiled binary
   if a_do_transpose then
     if b_do_transpose then
       name := 'XgemmDirectTT'
     else
       name := 'XgemmDirectTN'
   else
     if b_do_transpose then
       name := 'XgemmDirectNT'
     else
       name:= 'XgemmDirectNN';

   kernel := TKernel.create(program_[0], name);

   // Sets the kernel arguments
   kernel.SetArgument<integer>(0, m);
   kernel.SetArgument<integer>(1, n);
   kernel.SetArgument<integer>(2, k);
   kernel.SetArgument<T>(3, {GetRealArg(}alpha{)});
   kernel.SetArgument<T>(4, {GetRealArg(}beta{)});
   kernel.SetArgument<T>(5, a_buffer);
   kernel.SetArgument<integer>(6, a_offset);
   kernel.SetArgument<integer>(7, a_ld);
   kernel.SetArgument<T>(8, b_buffer);
   kernel.SetArgument<integer>(9, b_offset);
   kernel.SetArgument<integer>(10, b_ld);
   kernel.SetArgument<T>(11, c_buffer);
   kernel.SetArgument<integer>(12, c_offset);
   kernel.SetArgument<integer>(13, c_ld);
   kernel.SetArgument<integer>(14, ord(c_do_transpose));
   kernel.SetArgument<integer>(15, ord(a_conjugate));
   kernel.SetArgument<integer>(16, ord(b_conjugate));

   // Computes the global and local thread sizes
  m_ceiled := Ceil(m, WGD);
  n_ceiled := Ceil(n, WGD);
  global := [
   //  CeilDiv(m * db_.params['MDIMCD'], db_.params['WGD']),
   //  CeilDiv(n * db_.params['NDIMCD'], db_.params['WGD'])
       (m_ceiled * MDIMCD) div WGD,
       (n_ceiled * NDIMCD) div WGD
   ];

  local := [MDIMCD, NDIMCD];

   dummy:=[0, 0];
   // Launches the kernel
   //RunKernel(kernel, queue_, device_, global, local, event_);
   SAFE_CALL(clEnqueueNDRangeKernel(queue_, kernel, length(global), pointer(dummy), pointer(global), pointer(local), ord(assigned(event_)), event_, event_));
end;

procedure TXgemm<T>.DoGemmStridedBatch(const layout: TLayout;
  const a_transpose, b_transpose: TTranspose; const m, n, k: size_t;
  const alpha: T; const a_buffer: TBuffer<T>; const a_offset, a_ld,
  a_stride: size_t; const b_buffer: TBuffer<T>; const b_offset, b_ld,
  b_stride: size_t; const beta: T; const c_buffer: TBuffer<T>; const c_offset,
  c_ld, c_stride: size_t; const batch_count: size_t);
var
  do_gemm_direct, a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate:boolean;
  gemm_kernel_id, a_one, a_two, b_one, b_two, c_one, c_two:size_t;
begin
  //if c_stride=0 then
  //  exit;
  // Two methods to choose from, select which one to run
  do_gemm_direct := UseDirectKernel(m, n, k, XGEMM_MIN_INDIRECT_SIZE);
  gemm_kernel_id := ifthen(do_gemm_direct, 0, GEMMK);

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  ProcessArguments(layout, a_transpose, b_transpose, m, n, k,
                  a_one, a_two, b_one, b_two, c_one, c_two,
                  a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                  gemm_kernel_id);

  // Tests three matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. m, n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than K when rotated, or less than M when not-rotated
  //    matrix B cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N when rotated, or less than M when not-rotated
  //TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  //TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
  //TestMatrixC(c_one, c_two, c_buffer, c_offset, c_ld);

  // Selects which version of GEMM to run
  if do_gemm_direct then // for small sizes (single kernel)
    BatchedGemmDirect(m, n, k, alpha,
            a_buffer, a_offset, a_ld, a_stride,
            b_buffer, b_offset, b_ld, b_stride, beta,
            c_buffer, c_offset, c_ld, c_stride,
            a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
            batch_count)
  else  // for larger sizes (pre/post-processing plus a very fast kernel)
    BatchedGemmIndirect(m, n, k, alpha,
            a_buffer, a_offset, a_ld, a_stride,
            b_buffer, b_offset, b_ld, b_stride, beta,
            c_buffer, c_offset, c_ld, c_stride,
            a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
            a_one, a_two, b_one, b_two, c_one, c_two,
            batch_count);
end;

procedure TXgemm<T>.BatchedGemmIndirect(const m, n, k: size_t; const alpha: T;
  const a_buffer: TBuffer<T>; const a_offset, a_ld, a_stride: size_t;
  const b_buffer: TBuffer<T>; const b_offset, b_ld, b_stride: size_t;
  const beta: T; const c_buffer: TBuffer<T>; const c_offset, c_ld,
  c_stride: size_t; const a_do_transpose, b_do_transpose, c_do_transpose,
  a_conjugate, b_conjugate: boolean; const a_one, a_two, b_one, b_two, c_one,
  c_two, batch_count: size_t);
var m_ceiled, n_ceiled, k_ceiled ,
  a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i: size_t;
  a_no_temp, b_no_temp, c_no_temp : boolean;
  a_temp, b_temp, c_temp : TBuffer<T>;
  eventWaitList, emptyEventList : TArray<TEvent>;
  eventProcessA,eventProcessB, eventProcessC, eventKernel :TEvent;
  eventPointer : TEventPointer;
  local, global, dummy : TArray<size_t>;
  kernel: TKernel;

begin
  // Calculates the ceiled versions of m, n, and k
  m_ceiled := Ceil(Ceil(m, MWG), VWM);
  n_ceiled := Ceil(Ceil(n, NWG), VWN);
  k_ceiled := Ceil(Ceil(k, KWG), VWM);
  // Computes the first and second 'internal' (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  CalculateInternalDimensions(m, n, k, MWG, NWG, KWG,
                                        a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                                        GEMMK);

  // Determines whether or not temporary matrices are needed
  a_no_temp := (a_one = a_one_i ) and ( a_two = a_two_i ) and ( a_ld = a_one ) and not a_do_transpose  and not a_conjugate;
  b_no_temp := (b_one = b_one_i ) and ( b_two = b_two_i ) and ( b_ld = b_one ) and not b_do_transpose  and not b_conjugate;
  c_no_temp := (c_one = c_one_i ) and ( c_two = c_two_i ) and ( c_ld = c_one ) and not c_do_transpose;

  // Creates the temporary matrices
  if a_no_temp then a_temp := a_buffer else a_temp := TBuffer<T>.Create(context_, batch_count * a_one_i * a_two_i);
  if b_no_temp then b_temp := b_buffer else b_temp := TBuffer<T>.Create(context_, batch_count * b_one_i * b_two_i);
  if c_no_temp then c_temp := c_buffer else c_temp := TBuffer<T>.Create(context_, batch_count * c_one_i * c_two_i);

  // Events of all kernels (including pre/post processing kernels)

  // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
  // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
  // case nothing has to be done, these kernels can be skipped.
  emptyEventList := nil;
  eventWaitList := nil;
  if not a_no_temp then begin
    PadCopyTransposeMatrixStridedBatched(@eventProcessA, emptyEventList,
                                         a_one, a_two, a_ld, a_offset, a_stride, a_buffer,
                                         a_one_i, a_two_i, a_one_i, 0, a_one_i * a_two_i, a_temp,
                                         true, a_do_transpose, a_conjugate, batch_count);
    insert(eventProcessA, eventWaitList, length(eventWaitList));
  end;

  // As above, but now for matrix B
  if not b_no_temp then begin
    PadCopyTransposeMatrixStridedBatched(@eventProcessB, emptyEventList,
                                         b_one, b_two, b_ld, b_offset, b_stride, b_buffer,
                                         b_one_i, b_two_i, b_one_i, 0, b_one_i * b_two_i, b_temp,
                                         true, b_do_transpose, b_conjugate, batch_count);
    insert(eventProcessB, eventWaitList, length(eventWaitList));
  end;

  // As above, but now for matrix C
  if not c_no_temp then begin
    PadCopyTransposeMatrixStridedBatched(@eventProcessC, emptyEventList,
                                         c_one, c_two, c_ld, c_offset, c_stride, c_buffer,
                                         c_one_i, c_two_i, c_one_i, 0, c_one_i * c_two_i, c_temp,
                                         true, c_do_transpose, false, batch_count);
    insert(eventProcessC, eventWaitList, length(eventWaitList));
  end;

  // Retrieves the Xgemm kernel from the compiled binary
  kernel := TKernel.create(program_[0], 'XgemmStridedBatched');

  // Sets the kernel arguments
  kernel.SetArgument<integer>(0, m_ceiled);
  kernel.SetArgument<integer>(1, n_ceiled);
  kernel.SetArgument<integer>(2, k_ceiled);
  kernel.SetArgument<T>(3, alpha);
  kernel.SetArgument<T>(4, beta);
  kernel.SetArgument<T>(5, a_temp);
  kernel.SetArgument<integer>(6, a_one_i);
  kernel.SetArgument<integer>(7, a_two_i);
  kernel.SetArgument<T>(8, b_temp);
  kernel.SetArgument<integer>(9, b_one_i);
  kernel.SetArgument<integer>(10, b_two_i);
  kernel.SetArgument<T>(11, c_temp);
  kernel.SetArgument<integer>(12, c_one_i);
  kernel.SetArgument<integer>(13, c_two_i);

  dummy := [0, 0, 0];
  // Computes the global and local thread sizes
  global := [
      c_one_i * MDIMC div MWG,
      c_two_i * NDIMC div NWG,
      batch_count
  ];
  local := [MDIMC, NDIMC, 1];

  // Launches the kernel
  if not c_no_temp then eventPointer := @eventKernel else eventPointer := event_;
  clEnqueueNDRangeKernel(queue_, kernel, length(global), pointer(dummy), pointer(global), pointer(local), length(eventWaitList), pointer(eventWaitList), eventPointer);

  // Runs the post-processing kernel if needed
  if not c_no_temp then begin
    insert(eventKernel, eventWaitList, length(eventWaitList));
    PadCopyTransposeMatrixStridedBatched(event_, eventWaitList,
                                         c_one_i, c_two_i, c_one_i, 0, c_one_i * c_two_i, c_temp,
                                         c_one, c_two, c_ld, c_offset, c_stride, c_buffer,
                                         false, c_do_transpose, false, batch_count);
  end;

end;

procedure TXgemm<T>.BatchedGemmDirect(const m, n, k: size_t; const alpha: T;
  const a_buffer: TBuffer<T>; const a_offset, a_ld, a_stride: size_t;
  const b_buffer: TBuffer<T>; const b_offset, b_ld, b_stride: size_t;
  const beta: T; const c_buffer: TBuffer<T>; const c_offset, c_ld,
  c_stride: size_t; const a_do_transpose, b_do_transpose, c_do_transpose,
  a_conjugate, b_conjugate: boolean; const batch_count: size_t);
var
  name : ansistring;
  kernel : TKernel;
  local, global :TArray<size_t>;
  dummy :TArray<size_t>;
  m_ceiled : size_t;
  n_ceiled : size_t;
begin
   // Retrieves the proper XgemmDirect kernel from the compiled binary
   if a_do_transpose then
     if b_do_transpose then
       name := 'XgemmDirectStridedBatchedTT'
     else
       name := 'XgemmDirectStridedBatchedTN'
   else
     if b_do_transpose then
       name := 'XgemmDirectStridedBatchedNT'
     else
       name:= 'XgemmDirectStridedBatchedNN';

   kernel := TKernel.create(program_[0], name);

   // Sets the kernel arguments
  kernel.SetArgument<integer>(0, m);
  kernel.SetArgument<integer>(1, n);
  kernel.SetArgument<integer>(2, k);
  kernel.SetArgument<T>(3, {GetRealArg(}alpha{)});
  kernel.SetArgument<T>(4, {GetRealArg(}beta{)});
  kernel.SetArgument<T>(5, a_buffer);
  kernel.SetArgument<integer>(6, a_offset);
  kernel.SetArgument<integer>(7, a_ld);
  kernel.SetArgument<integer>(8, a_stride);
  kernel.SetArgument<T>(9, b_buffer);
  kernel.SetArgument<integer>(10, b_offset);
  kernel.SetArgument<integer>(11, b_ld);
  kernel.SetArgument<integer>(12, b_stride);
  kernel.SetArgument<T>(13, c_buffer);
  kernel.SetArgument<integer>(14, c_offset);
  kernel.SetArgument<integer>(15, c_ld);
  kernel.SetArgument<integer>(16, c_stride);
  kernel.SetArgument<integer>(17, ord(c_do_transpose));
  kernel.SetArgument<integer>(18, ord(a_conjugate));
  kernel.SetArgument<integer>(19, ord(b_conjugate));

  // Computes the global and local thread sizes
  m_ceiled := Ceil(m, WGD);
  n_ceiled := Ceil(n, WGD);
  global := [
      (m_ceiled * MDIMCD) div WGD,
      (n_ceiled * NDIMCD) div WGD,
      batch_count
  ];
  local := [MDIMCD, NDIMCD, 1];

  dummy:=[0, 0, 0];
  // Launches the kernel
  SAFE_CALL(clEnqueueNDRangeKernel(queue_, kernel, length(global), pointer(dummy), pointer(global), pointer(local), ord(assigned(event_)), event_, event_));
end;


initialization

  setLength(TDatabase.apple_cpu_fallback, 16);
  TDatabase.apple_cpu_fallback[0].init('Xaxpy', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['VW', 'WGS', 'WPT'], [ 8, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[1].init('Xdot', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['WGS1', 'WGS2'], [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[2].init('Xgemv', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['WGS1', 'WPT1', 'UNROLL1'], [ 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[3].init('XgemvFast', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['VW2', 'WGS2', 'WPT2'], [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[4].init('XgemvFastRot', TPrecision.kAny, kDeviceTypeAll, 'default','default',  kDeviceNameDefault, ['VW3', 'WGS3', 'WPT3'], [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[4].init('XgemvFastRot', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['WGS1', 'WGS2', 'WPT'], [64, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[6].init('Xtrsv', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['TRSV_BLOCK_SIZE'], [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[7].init('Xgemm', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['GEMMK', 'KREG', 'KWG', 'KWI', 'MDIMA', 'MDIMC', 'MWG', 'NDIMB', 'NDIMC', 'NWG', 'SA', 'SB', 'STRM', 'STRN', 'VWM', 'VWN'], [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 ]);
  TDatabase.apple_cpu_fallback[8].init('XgemmDirect', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['KWID', 'MDIMAD', 'MDIMCD', 'NDIMBD', 'NDIMCD', 'PADA', 'PADB', 'VWMD', 'VWND', 'WGD'], [ 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[9].init('Xconvgemm', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['KWID', 'MDIMAD', 'MDIMCD', 'NDIMBD', 'NDIMCD', 'PADA', 'PADB', 'VWMD', 'VWND', 'WGD'], [ 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[10].init('Copy', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['COPY_DIMX', 'COPY_DIMY', 'COPY_VW', 'COPY_WPT'], [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[11].init('Pad', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['PAD_DIMX', 'PAD_DIMY', 'PAD_WPTX', 'PAD_WPTY'], [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[12].init('Transpose', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['TRA_DIM', 'TRA_PAD', 'TRA_SHUFFLE', 'TRA_WPT'], [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[13].init('Padtranspose', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['PADTRA_PAD', 'PADTRA_TILE', 'PADTRA_WPT'], [ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[14].init('Invert', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['INTERNAL_BLOCK_SIZE'], [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);
  TDatabase.apple_cpu_fallback[15].init('TrsvRoutine', TPrecision.kAny, kDeviceTypeAll, 'default', 'default', kDeviceNameDefault, ['TRSV_BLOCK_SIZE'], [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]);

  TRoutine.routines_by_kernel := TDictionary<ansistring, TArray<ansistring>>.create();
  TRoutine.routines_by_kernel.add('Xaxpy'       , TRoutine.routines_axpy);
  TRoutine.routines_by_kernel.add('Xdot'        , TRoutine.routines_dot);
  TRoutine.routines_by_kernel.add('Xgemv'       , TRoutine.routines_gemv);
  TRoutine.routines_by_kernel.add('XgemvFast'   , TRoutine.routines_gemv);
  TRoutine.routines_by_kernel.add('XgemvFastRot', TRoutine.routines_gemv);
  TRoutine.routines_by_kernel.add('Xtrsv'       , TRoutine.routines_gemv);
  TRoutine.routines_by_kernel.add('Xger'        , TRoutine.routines_ger);
  TRoutine.routines_by_kernel.add('Copy'        , TRoutine.routines_gemm_syrk);
  TRoutine.routines_by_kernel.add('Pad'         , TRoutine.routines_gemm_syrk);
  TRoutine.routines_by_kernel.add('Transpose'   , TRoutine.routines_gemm_syrk);
  TRoutine.routines_by_kernel.add('Padtranspose', TRoutine.routines_gemm_syrk);
  TRoutine.routines_by_kernel.add('Xgemm'       , TRoutine.routines_gemm_syrk);
  TRoutine.routines_by_kernel.add('XgemmDirect' , TRoutine.routines_gemm);
  TRoutine.routines_by_kernel.add('GemmRoutine' , TRoutine.routines_gemm);
  TRoutine.routines_by_kernel.add('Invert'      , TRoutine.routines_trsm);

  TRoutine.databaseCache := TDictionary<TRoutine.TRoutineDesc, TDatabase>.create();
  TRoutine.programCache := TDictionary<TRoutine.TProgramDesc, cl_program>.create();
  TRoutine.binaryCache   := TDictionary<TRoutine.TBinDesc, RawByteString>.create();


  TXgemm<half>.constantOne := 1;
  TXgemm<single>.constantOne := 1;
  TXgemm<double>.constantOne := 1;

  {$I inc/cl_las.XgemmSingle.inc}
  {$I inc/cl_las.XgemmHalf.inc}
  {$I inc/cl_las.XgemmDirectSingle.inc}
  {$I inc/cl_las.XgemmDirectHalf.inc}
  {$I inc/cl_las.GemmRoutineSingle.inc}
  {$I inc/cl_las.transposeSingle.inc}
  {$I inc/cl_las.padTransposeSingle.inc}
  {$I inc/cl_las.copySingle.inc}
  {$I inc/cl_las.padSingle.inc}




finalization
  if assigned(sgemm) then freeAndNil(sgemm);
  freeAndNil(TRoutine.databaseCache);
  freeAndNil(TRoutine.programCache);
  freeAndNil(TRoutine.binaryCache);
  freeAndNil(TRoutine.routines_by_kernel);
end.
