(*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 *)
 (* NVIDIA driver API functions *)
{$ifdef FPC}
{$packrecords C}
{$mode delphi}
{$endif}
unit cuda;
interface
{$if defined(MSWINDOWS)}
   //const libcuda='cudart64_12.dll';
   const libcuda='nvcuda.dll';
{$elseif defined(MACOS) or defined(DARWIN)}
   const libcuda= 'libcuda.dylib';
{$elseif defined(LINUX)}
   const libcuda= 'libcuda.so'   ;
{$endif}


type
  Pcuuint32_t=^cuuint32_t;
  cuuint32_t = longword;

  Pcuuint64_t=^cuuint64_t;
  cuuint64_t = uint64;
{$if not declared(psize_t)}
  Psize_t = ^size_t;
{$endif}
{$if not declared(size_t)}
  size_t = NativeUInt;
{$endif}

(*
#define cuDeviceTotalMem                    cuDeviceTotalMem_v2
#define cuCtxCreate                         cuCtxCreate_v2
#define cuCtxCreate_v3                      cuCtxCreate_v3
#define cuModuleGetGlobal                   cuModuleGetGlobal_v2
#define cuMemGetInfo                        cuMemGetInfo_v2
#define cuMemAlloc                          cuMemAlloc_v2
#define cuMemAllocPitch                     cuMemAllocPitch_v2
#define cuMemFree                           cuMemFree_v2
#define cuMemGetAddressRange                cuMemGetAddressRange_v2
#define cuMemAllocHost                      cuMemAllocHost_v2
#define cuMemHostGetDevicePointer           cuMemHostGetDevicePointer_v2
#define cuMemcpyHtoD                        __CUDA_API_PTDS(cuMemcpyHtoD_v2)
#define cuMemcpyDtoH                        __CUDA_API_PTDS(cuMemcpyDtoH_v2)
#define cuMemcpyDtoD                        __CUDA_API_PTDS(cuMemcpyDtoD_v2)
#define cuMemcpyDtoA                        __CUDA_API_PTDS(cuMemcpyDtoA_v2)
#define cuMemcpyAtoD                        __CUDA_API_PTDS(cuMemcpyAtoD_v2)
#define cuMemcpyHtoA                        __CUDA_API_PTDS(cuMemcpyHtoA_v2)
#define cuMemcpyAtoH                        __CUDA_API_PTDS(cuMemcpyAtoH_v2)
#define cuMemcpyAtoA                        __CUDA_API_PTDS(cuMemcpyAtoA_v2)
#define cuMemcpyHtoAAsync                   __CUDA_API_PTSZ(cuMemcpyHtoAAsync_v2)
#define cuMemcpyAtoHAsync                   __CUDA_API_PTSZ(cuMemcpyAtoHAsync_v2)
#define cuMemcpy2D                          __CUDA_API_PTDS(cuMemcpy2D_v2)
#define cuMemcpy2DUnaligned                 __CUDA_API_PTDS(cuMemcpy2DUnaligned_v2)
#define cuMemcpy3D                          __CUDA_API_PTDS(cuMemcpy3D_v2)
#define cuMemcpyHtoDAsync                   __CUDA_API_PTSZ(cuMemcpyHtoDAsync_v2)
#define cuMemcpyDtoHAsync                   __CUDA_API_PTSZ(cuMemcpyDtoHAsync_v2)
#define cuMemcpyDtoDAsync                   __CUDA_API_PTSZ(cuMemcpyDtoDAsync_v2)
#define cuMemcpy2DAsync                     __CUDA_API_PTSZ(cuMemcpy2DAsync_v2)
#define cuMemcpy3DAsync                     __CUDA_API_PTSZ(cuMemcpy3DAsync_v2)
#define cuMemsetD8                          __CUDA_API_PTDS(cuMemsetD8_v2)
#define cuMemsetD16                         __CUDA_API_PTDS(cuMemsetD16_v2)
#define cuMemsetD32                         __CUDA_API_PTDS(cuMemsetD32_v2)
#define cuMemsetD2D8                        __CUDA_API_PTDS(cuMemsetD2D8_v2)
#define cuMemsetD2D16                       __CUDA_API_PTDS(cuMemsetD2D16_v2)
#define cuMemsetD2D32                       __CUDA_API_PTDS(cuMemsetD2D32_v2)
#define cuArrayCreate                       cuArrayCreate_v2
#define cuArrayGetDescriptor                cuArrayGetDescriptor_v2
#define cuArray3DCreate                     cuArray3DCreate_v2
#define cuArray3DGetDescriptor              cuArray3DGetDescriptor_v2
#define cuTexRefSetAddress                  cuTexRefSetAddress_v2
#define cuTexRefGetAddress                  cuTexRefGetAddress_v2
#define cuGraphicsResourceGetMappedPointer  cuGraphicsResourceGetMappedPointer_v2
#define cuCtxDestroy                        cuCtxDestroy_v2
#define cuCtxPopCurrent                     cuCtxPopCurrent_v2
#define cuCtxPushCurrent                    cuCtxPushCurrent_v2
#define cuStreamDestroy                     cuStreamDestroy_v2
#define cuEventDestroy                      cuEventDestroy_v2
#define cuTexRefSetAddress2D                cuTexRefSetAddress2D_v3
#define cuLinkCreate                        cuLinkCreate_v2
#define cuLinkAddData                       cuLinkAddData_v2
#define cuLinkAddFile                       cuLinkAddFile_v2
#define cuMemHostRegister                   cuMemHostRegister_v2
#define cuGraphicsResourceSetMapFlags       cuGraphicsResourceSetMapFlags_v2
#define cuStreamBeginCapture                __CUDA_API_PTSZ(cuStreamBeginCapture_v2)
#define cuDevicePrimaryCtxRelease           cuDevicePrimaryCtxRelease_v2
#define cuDevicePrimaryCtxReset             cuDevicePrimaryCtxReset_v2
#define cuDevicePrimaryCtxSetFlags          cuDevicePrimaryCtxSetFlags_v2
#define cuDeviceGetUuid_v2                  cuDeviceGetUuid_v2
#define cuIpcOpenMemHandle                  cuIpcOpenMemHandle_v2

#define cuGraphInstantiate                  cuGraphInstantiateWithFlags

#define cuGraphExecUpdate                   cuGraphExecUpdate_v2 
#define cuGetProcAddress                    cuGetProcAddress_v2
#define cuGraphAddKernelNode                cuGraphAddKernelNode_v2
#define cuGraphKernelNodeGetParams          cuGraphKernelNodeGetParams_v2
#define cuGraphKernelNodeSetParams          cuGraphKernelNodeSetParams_v2
#define cuGraphExecKernelNodeSetParams      cuGraphExecKernelNodeSetParams_v2

#define cuStreamWriteValue32                __CUDA_API_PTSZ(cuStreamWriteValue32_v2)
#define cuStreamWaitValue32                 __CUDA_API_PTSZ(cuStreamWaitValue32_v2)
#define cuStreamWriteValue64                __CUDA_API_PTSZ(cuStreamWriteValue64_v2)
#define cuStreamWaitValue64                 __CUDA_API_PTSZ(cuStreamWaitValue64_v2)
#define cuStreamBatchMemOp                  __CUDA_API_PTSZ(cuStreamBatchMemOp_v2)
#define cuStreamGetCaptureInfo              __CUDA_API_PTSZ(cuStreamGetCaptureInfo_v2)
#define cuStreamGetCaptureInfo_v2           __CUDA_API_PTSZ(cuStreamGetCaptureInfo_v2)

#if defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define cuMemcpy                            __CUDA_API_PTDS(cuMemcpy)
    #define cuMemcpyAsync                       __CUDA_API_PTSZ(cuMemcpyAsync)
    #define cuMemcpyPeer                        __CUDA_API_PTDS(cuMemcpyPeer)
    #define cuMemcpyPeerAsync                   __CUDA_API_PTSZ(cuMemcpyPeerAsync)
    #define cuMemcpy3DPeer                      __CUDA_API_PTDS(cuMemcpy3DPeer)
    #define cuMemcpy3DPeerAsync                 __CUDA_API_PTSZ(cuMemcpy3DPeerAsync)
    #define cuMemPrefetchAsync                  __CUDA_API_PTSZ(cuMemPrefetchAsync)

    #define cuMemsetD8Async                     __CUDA_API_PTSZ(cuMemsetD8Async)
    #define cuMemsetD16Async                    __CUDA_API_PTSZ(cuMemsetD16Async)
    #define cuMemsetD32Async                    __CUDA_API_PTSZ(cuMemsetD32Async)
    #define cuMemsetD2D8Async                   __CUDA_API_PTSZ(cuMemsetD2D8Async)
    #define cuMemsetD2D16Async                  __CUDA_API_PTSZ(cuMemsetD2D16Async)
    #define cuMemsetD2D32Async                  __CUDA_API_PTSZ(cuMemsetD2D32Async)

    #define cuStreamGetPriority                 __CUDA_API_PTSZ(cuStreamGetPriority)
    #define cuStreamGetId                       __CUDA_API_PTSZ(cuStreamGetId)
    #define cuStreamGetFlags                    __CUDA_API_PTSZ(cuStreamGetFlags)
    #define cuStreamGetCtx                      __CUDA_API_PTSZ(cuStreamGetCtx)
    #define cuStreamWaitEvent                   __CUDA_API_PTSZ(cuStreamWaitEvent)
    #define cuStreamEndCapture                  __CUDA_API_PTSZ(cuStreamEndCapture)
    #define cuStreamIsCapturing                 __CUDA_API_PTSZ(cuStreamIsCapturing)
    #define cuStreamUpdateCaptureDependencies   __CUDA_API_PTSZ(cuStreamUpdateCaptureDependencies)
    #define cuStreamAddCallback                 __CUDA_API_PTSZ(cuStreamAddCallback)
    #define cuStreamAttachMemAsync              __CUDA_API_PTSZ(cuStreamAttachMemAsync)
    #define cuStreamQuery                       __CUDA_API_PTSZ(cuStreamQuery)
    #define cuStreamSynchronize                 __CUDA_API_PTSZ(cuStreamSynchronize)
    #define cuEventRecord                       __CUDA_API_PTSZ(cuEventRecord)
    #define cuEventRecordWithFlags              __CUDA_API_PTSZ(cuEventRecordWithFlags)
    #define cuLaunchKernel                      __CUDA_API_PTSZ(cuLaunchKernel)
    #define cuLaunchKernelEx                    __CUDA_API_PTSZ(cuLaunchKernelEx)
    #define cuLaunchHostFunc                    __CUDA_API_PTSZ(cuLaunchHostFunc)
    #define cuGraphicsMapResources              __CUDA_API_PTSZ(cuGraphicsMapResources)
    #define cuGraphicsUnmapResources            __CUDA_API_PTSZ(cuGraphicsUnmapResources)

    #define cuLaunchCooperativeKernel           __CUDA_API_PTSZ(cuLaunchCooperativeKernel)

    #define cuSignalExternalSemaphoresAsync     __CUDA_API_PTSZ(cuSignalExternalSemaphoresAsync)
    #define cuWaitExternalSemaphoresAsync       __CUDA_API_PTSZ(cuWaitExternalSemaphoresAsync)

    #define cuGraphInstantiateWithParams        __CUDA_API_PTSZ(cuGraphInstantiateWithParams)
    #define cuGraphUpload                       __CUDA_API_PTSZ(cuGraphUpload)
    #define cuGraphLaunch                       __CUDA_API_PTSZ(cuGraphLaunch)
    #define cuStreamCopyAttributes              __CUDA_API_PTSZ(cuStreamCopyAttributes)
    #define cuStreamGetAttribute                __CUDA_API_PTSZ(cuStreamGetAttribute)
    #define cuStreamSetAttribute                __CUDA_API_PTSZ(cuStreamSetAttribute)
    #define cuMemMapArrayAsync                  __CUDA_API_PTSZ(cuMemMapArrayAsync)

    #define cuMemFreeAsync                      __CUDA_API_PTSZ(cuMemFreeAsync)
    #define cuMemAllocAsync                     __CUDA_API_PTSZ(cuMemAllocAsync)
    #define cuMemAllocFromPoolAsync             __CUDA_API_PTSZ(cuMemAllocFromPoolAsync)
#endif
*)
(**
 * \file cuda.h
 * \brief Header file for the CUDA Toolkit application programming interface.
 *
 * \file cudaGL.h
 * \brief Header file for the OpenGL interoperability functions of the
 * low-level CUDA driver application programming interface.
 *
 * \file cudaD3D9.h
 * \brief Header file for the Direct3D 9 interoperability functions of the
 * low-level CUDA driver application programming interface.
 *)

(**
 * \defgroup CUDA_TYPES Data types used by CUDA driver
 * @{
 *)

(**
 * CUDA API version number
 *)
const
  CUDA_VERSION = 12010;
(**
 * If set, each kernel launched as part of ::cuLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins execution.
 *)
  CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC =    $01;

(**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::cuLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins execution.
 *)
  CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC =   $02;

(**
 * If set, the CUDA array is a collection of layers, where each layer is either a 1D
 * or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number
 * of layers, not the depth of a 3D array.
 *)
  CUDA_ARRAY3D_LAYERED =         $01;

(**
 * Deprecated, use CUDA_ARRAY3D_LAYERED
 *)
  CUDA_ARRAY3D_2DARRAY =         $01;

(**
 * This flag must be set in order to bind a surface reference
 * to the CUDA array
 *)
  CUDA_ARRAY3D_SURFACE_LDST =    $02;

(**
 * If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The
 * width of such a CUDA array must be equal to its height, and Depth must be six.
 * If ::CUDA_ARRAY3D_LAYERED flag is also set, then the CUDA array is a collection of cubemaps
 * and Depth must be a multiple of six.
 *)
  CUDA_ARRAY3D_CUBEMAP =         $04;

(**
 * This flag must be set in order to perform texture gather operations
 * on a CUDA array.
 *)
  CUDA_ARRAY3D_TEXTURE_GATHER =  $08;

(**
 * This flag if set indicates that the CUDA
 * array is a DEPTH_TEXTURE.
 *)
  CUDA_ARRAY3D_DEPTH_TEXTURE =  $10;

(**
 * This flag indicates that the CUDA array may be bound as a color target
 * in an external graphics API
 *)
  CUDA_ARRAY3D_COLOR_ATTACHMENT =  $20;

(**
 * This flag if set indicates that the CUDA array or CUDA mipmapped array
 * is a sparse CUDA array or CUDA mipmapped array respectively
 *)
  CUDA_ARRAY3D_SPARSE =  $40;

(**
 * This flag if set indicates that the CUDA array or CUDA mipmapped array
 * will allow deferred memory mapping
 *)
  CUDA_ARRAY3D_DEFERRED_MAPPING =  $80;

(**
 * Override the texref format with a format inferred from the array.
 * Flag for ::cuTexRefSetArray()
 *)
  CU_TRSA_OVERRIDE_FORMAT =  $01;

(**
 * Read the texture as integers rather than promoting the values to floats
 * in the range [0,1].
 * Flag for ::cuTexRefSetFlags() and ::cuTexObjectCreate()
 *)
  CU_TRSF_READ_AS_INTEGER =          $01;

(**
 * Use normalized texture coordinates in the range [0,1) instead of [0,dim).
 * Flag for ::cuTexRefSetFlags() and ::cuTexObjectCreate()
 *)
  CU_TRSF_NORMALIZED_COORDINATES =   $02;

(**
 * Perform sRGB->linear conversion during texture read.
 * Flag for ::cuTexRefSetFlags() and ::cuTexObjectCreate()
 *)
  CU_TRSF_SRGB =   $10;

 (**
  * Disable any trilinear filtering optimizations.
  * Flag for ::cuTexRefSetFlags() and ::cuTexObjectCreate()
  *)
  CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION =   $20;

(**
 * Enable seamless cube map filtering.
 * Flag for ::cuTexObjectCreate()
 *)
  CU_TRSF_SEAMLESS_CUBEMAP =   $40;

(**
 * C++ compile time constant for CU_LAUNCH_PARAM_END
 *)
  CU_LAUNCH_PARAM_END_AS_INT =      $00;

(**
 * End of array terminator for the \p extra parameter to
 * ::cuLaunchKernel
 *)
  CU_LAUNCH_PARAM_END =             pointer(CU_LAUNCH_PARAM_END_AS_INT);

(**
 * C++ compile time constant for CU_LAUNCH_PARAM_BUFFER_POINTER
 *)
  CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT =  $01;

(**
 * Indicator that the next value in the \p extra parameter to
 * ::cuLaunchKernel will be a pointer to a buffer containing all kernel
 * parameters used for launching kernel \p f.  This buffer needs to
 * honor all alignment/padding requirements of the individual parameters.
 * If ::CU_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the
 * \p extra array, then ::CU_LAUNCH_PARAM_BUFFER_POINTER will have no
 * effect.
 *)
  CU_LAUNCH_PARAM_BUFFER_POINTER =         pointer(CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT);

(**
 * C++ compile time constant for CU_LAUNCH_PARAM_BUFFER_SIZE
 *)
  CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT =  $02;

(**
 * Indicator that the next value in the \p extra parameter to
 * ::cuLaunchKernel will be a pointer to a size_t which contains the
 * size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER.
 * It is required that ::CU_LAUNCH_PARAM_BUFFER_POINTER also be specified
 * in the \p extra array if the value associated with
 * ::CU_LAUNCH_PARAM_BUFFER_SIZE is not zero.
 *)
  CU_LAUNCH_PARAM_BUFFER_SIZE =         pointer(CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT);

(**
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 *)
  CU_PARAM_TR_DEFAULT =  -1;

(**
 * Device that represents the CPU
 *)
  CU_DEVICE_CPU =                -1;

(**
 * Device that represents an invalid device
 *)
  CU_DEVICE_INVALID =            -2;
  (**
   * Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers
   *)
  CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL =  $1;

type
  CUstream_st=record end;
  PCUStream =^CUStream ;
  CUStream = ^CUStream_st;
                        (**< CUDA stream *)
const

  (**
   * Legacy stream handle
   *
   * Stream handle that can be passed as a CUstream to use an implicit stream
   * with legacy synchronization behavior.
   *
   * See details of the \link_sync_behavior
   *)
  CU_STREAM_LEGACY =      CUstream($1);

  (**
  * Per-thread stream handle
  *
  * Stream handle that can be passed as a CUstream to use an implicit stream
  * with per-thread synchronization behavior.
  *
  * See details of the \link_sync_behavior
  *)
  CU_STREAM_PER_THREAD =  CUstream($2);

  (*
  * Indicates that compute device class supports accelerated features.
  *)
  CU_COMPUTE_ACCELERATED_TARGET_BASE =    $10000;

  (**
  * CUDA IPC handle size
  *)
  CU_IPC_HANDLE_SIZE =  64;


(**
 * CUDA device pointer
 * CUdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.
 *)
type
  CUdeviceptr_v2 = UIntPTR;

  PCUdeviceptr = ^CUdeviceptr ;
  CUdeviceptr = CUdeviceptr_v2;                          (**< CUDA device pointer *)

  CUdevice_v1 = longint;                                     (**< CUDA device *)
  PCUDevice = ^CUDevice;
  CUdevice = CUdevice_v1;                                (**< CUDA device *)

  CUctx_st=record end;
  PCUContext = ^CUContext;
  CUContext = ^CUctx_st;                          (**< CUDA context *)

  CUmod_st=record end;
  PCUmodule = ^CUmodule;
  CUmodule = ^CUmod_st;
                           (**< CUDA module *)
  CUfunc_st=record end;
  PCUFunction = ^CUfunction ;
  CUFunction = ^CUfunc_st;
                        (**< CUDA function *)
  CUlib_st=record end;
  PCUlibrary = ^CUlibrary ;
  CUlibrary = ^CUlib_st;
                          (**< CUDA library *)
  CUkern_st=record end;
  PCUkernel = ^CUkernel ;
  CUkernel = ^CUkern_st;
                          (**< CUDA kernel *)
  CUarray_st=record end;
  PCUarray = ^CUarray ;
  CUarray = ^CUarray_st;
                          (**< CUDA array *)
  CUmipmappedArray_st=record end;
  PCUmipmappedArray = ^CUmipmappedArray ;
  CUmipmappedArray = ^CUmipmappedArray_st;
        (**< CUDA mipmapped array *)
  CUtexref_st=record end;
  PCUtexref = ^CUtexref ;
  CUtexref = ^CUtexref_st;
                        (**< CUDA texture reference *)
  CUsurfref_st=record end;
  PCUsurfref = ^CUsurfref ;
  CUsurfref = ^CUsurfref_st;
                      (**< CUDA surface reference *)
  CUevent_st=record end;
  PCUevent = ^CUevent ;
  CUevent = ^CUevent_st;
                          (**< CUDA event *)
  CUgraphicsResource_st=record end;
  PCUgraphicsResource =^CUgraphicsResource ;
  CUgraphicsResource = ^CUgraphicsResource_st;
    (**< CUDA graphics interop resource *)
  CUtexObject_v1 = uint64;                   (**< An opaque value that represents a CUDA texture object *)
  PCUtexObject =^CUtexObject ;
  CUtexObject = CUtexObject_v1;                          (**< An opaque value that represents a CUDA texture object *)
  CUsurfObject_v1 = uint64;                  (**< An opaque value that represents a CUDA surface object *)
  PCUsurfObject =^CUsurfObject ;
  CUsurfObject = CUsurfObject_v1;                        (**< An opaque value that represents a CUDA surface object *)
  CUextMemory_st=record end;
  PCUexternalMemory =^CUexternalMemory ;
  CUexternalMemory = ^CUextMemory_st;
             (**< CUDA external memory *)
  CUextSemaphore_st=record end;
  PCUexternalSemaphore =^CUexternalSemaphore ;
  CUexternalSemaphore = ^CUextSemaphore_st;
       (**< CUDA external semaphore *)
  CUgraph_st=record end;
  PCUgraph =^CUgraph ;
  CUgraph = ^CUgraph_st;
                          (**< CUDA graph *)
  CUgraphNode_st=record end;
  PPCUgraphNode =^PCUgraphNode;
  PCUgraphNode=^CUgraphNode;
  CUgraphNode = ^CUgraphNode_st;
                  (**< CUDA graph node *)
  CUgraphExec_st=record end;
  PCUgraphExec =^CUgraphExec ;
  CUgraphExec = ^CUgraphExec_st;
                  (**< CUDA executable graph *)
  CUmemPoolHandle_st=record end;
  PCUmemoryPool = ^CUmemoryPool;
  CUmemoryPool= ^CUmemPoolHandle_st;
             (**< CUDA memory pool *)
  CUuserObject_st=record end;
  PCUuserObject =^CUuserObject ;
  CUuserObject = ^CUuserObject_st;
                (**< CUDA user object for graphs *)

  CUuuid_st=packed record                                 (**< CUDA definition of UUID *)
    bytes : array[0..15] of byte;
  end;
  PCUuuid = ^CUuuid;
  CUuuid = CUuuid_st;

(**
 * CUDA IPC event handle
 *)
  CUipcEventHandle_st=packed record
    reserved : array[0..CU_IPC_HANDLE_SIZE-1] of byte;
  end;
  CUipcEventHandle_v1 = CUipcEventHandle_st;
  PCUipcEventHandle = ^CUipcEventHandle ;
  CUipcEventHandle = CUipcEventHandle_v1;

(**
 * CUDA IPC mem handle
 *)
  CUipcMemHandle_st=packed record
    reserved : array[0..CU_IPC_HANDLE_SIZE-1] of byte;
  end;
  CUipcMemHandle_v1 = CUipcMemHandle_st;
  PCUipcMemHandle = ^CUipcMemHandle ;
  CUipcMemHandle = CUipcMemHandle_v1;

(**
 * CUDA Ipc Mem Flags
 *)
  CUipcMem_flags_enum = (
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = $1 (**< Automatically enable peer access between remote devices as needed *)
);
  CUipcMem_flags = CUipcMem_flags_enum;


(**
 * CUDA Mem Attach Flags
 *)
  CUmemAttach_flags_enum = (
    CU_MEM_ATTACH_GLOBAL = $1, (**< Memory can be accessed by any stream on any device *)
    CU_MEM_ATTACH_HOST   = $2, (**< Memory cannot be accessed by any stream on any device *)
    CU_MEM_ATTACH_SINGLE = $4  (**< Memory can only be accessed by a single stream on the associated device *)
);
  CUmemAttach_flags = CUmemAttach_flags_enum;

(**
 * Context creation flags
 *)
  CUctx_flags_enum = (
    CU_CTX_SCHED_AUTO          = $00, (**< Automatic scheduling *)
    CU_CTX_SCHED_SPIN          = $01, (**< Set spin as default scheduling *)
    CU_CTX_SCHED_YIELD         = $02, (**< Set yield as default scheduling *)
    CU_CTX_SCHED_BLOCKING_SYNC = $04, (**< Set blocking synchronization as default scheduling *)
    CU_CTX_BLOCKING_SYNC       = $04, (**< Set blocking synchronization as default scheduling
                                         *  \deprecated This flag was deprecated as of CUDA 4.0
                                         *  and was replaced with ::CU_CTX_SCHED_BLOCKING_SYNC. *)
    CU_CTX_SCHED_MASK          = $07,
    CU_CTX_MAP_HOST            = $08, (**< \deprecated This flag was deprecated as of CUDA 11.0
                                         *  and it no longer has any effect. All contexts
                                         *  as of CUDA 3.2 behave as though the flag is enabled. *)
    CU_CTX_LMEM_RESIZE_TO_MAX  = $10, (**< Keep local memory allocation after launch *)
    CU_CTX_COREDUMP_ENABLE     = $20, (**< Trigger coredumps from exceptions in this context *)
    CU_CTX_USER_COREDUMP_ENABLE= $40, (**< Enable user pipe to trigger coredumps in this context *)
    CU_CTX_SYNC_MEMOPS         = $80, (**< Force synchronous blocking on cudaMemcpy/cudaMemset *)
    CU_CTX_FLAGS_MASK          = $FF
  );
  CUctx_flags = CUctx_flags_enum;

(**
 * Event sched flags
 *)
  CUevent_sched_flags_enum = (
    CU_EVENT_SCHED_AUTO = $00, (**< Automatic scheduling *)
    CU_EVENT_SCHED_SPIN = $01, (**< Set spin as default scheduling *)
    CU_EVENT_SCHED_YIELD = $02, (**< Set yield as default scheduling *)
    CU_EVENT_SCHED_BLOCKING_SYNC = $04 (**< Set blocking synchronization as default scheduling *)
  );
  CUevent_sched_flags = CUevent_sched_flags_enum;

(**
 * NVCL event scheduling flags
 *)
  cl_event_flags_enum = (
    NVCL_EVENT_SCHED_AUTO = $00, (**< Automatic scheduling *)
    NVCL_EVENT_SCHED_SPIN = $01, (**< Set spin as default scheduling *)
    NVCL_EVENT_SCHED_YIELD = $02, (**< Set yield as default scheduling *)
    NVCL_EVENT_SCHED_BLOCKING_SYNC = $04 (**< Set blocking synchronization as default scheduling *)
  );
  cl_event_flags = cl_event_flags_enum;

(**
 * NVCL context scheduling flags
 *)
  cl_context_flags_enum = (
    NVCL_CTX_SCHED_AUTO = $00, (**< Automatic scheduling *)
    NVCL_CTX_SCHED_SPIN = $01, (**< Set spin as default scheduling *)
    NVCL_CTX_SCHED_YIELD = $02, (**< Set yield as default scheduling *)
    NVCL_CTX_SCHED_BLOCKING_SYNC = $04 (**< Set blocking synchronization as default scheduling *)
  );
  cl_context_flags = cl_context_flags_enum;


(**
 * Stream creation flags
 *)
  CUstream_flags_enum = (
    CU_STREAM_DEFAULT             = $0, (**< Default stream flag *)
    CU_STREAM_NON_BLOCKING        = $1  (**< Stream does not synchronize with stream 0 (the NULL stream) *)
  );
  CUstream_flags = CUstream_flags_enum;


(**
 * Event creation flags
 *)
  CUevent_flags_enum = (
    CU_EVENT_DEFAULT        = $0, (**< Default event flag *)
    CU_EVENT_BLOCKING_SYNC  = $1, (**< Event uses blocking synchronization *)
    CU_EVENT_DISABLE_TIMING = $2, (**< Event will not record timing data *)
    CU_EVENT_INTERPROCESS   = $4  (**< Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set *)
  );
  CUevent_flags = CUevent_flags_enum;

(**
 * Event record flags
 *)
  CUevent_record_flags_enum = (
    CU_EVENT_RECORD_DEFAULT  = $0, (**< Default event record flag *)
    CU_EVENT_RECORD_EXTERNAL = $1  (**< When using stream capture, create an event record node
                                      *  instead of the default behavior.  This flag is invalid
                                      *  when used outside of capture. *)
  );
  CUevent_record_flags = CUevent_record_flags_enum;

(**
 * Event wait flags
 *)
  CUevent_wait_flags_enum = (
    CU_EVENT_WAIT_DEFAULT  = $0, (**< Default event wait flag *)
    CU_EVENT_WAIT_EXTERNAL = $1  (**< When using stream capture, create an event wait node
                                    *  instead of the default behavior.  This flag is invalid
                                    *  when used outside of capture.*)
  );
  CUevent_wait_flags = CUevent_wait_flags_enum;

(**
 * Flags for ::cuStreamWaitValue32 and ::cuStreamWaitValue64
 *)
   CUstreamWaitValue_flags_enum = (
    CU_STREAM_WAIT_VALUE_GEQ   = $0,   (**< Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit
                                             values). Note this is a cyclic comparison which ignores wraparound.
                                             (Default behavior.) *)
    CU_STREAM_WAIT_VALUE_EQ    = $1,   (**< Wait until *addr == value. *)
    CU_STREAM_WAIT_VALUE_AND   = $2,   (**< Wait until (*addr & value) != 0. *)
    CU_STREAM_WAIT_VALUE_NOR   = $3,   (**< Wait until ~(*addr | value) != 0. Support for this operation can be
                                             queried with ::cuDeviceGetAttribute() and
                                             ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.*)
    CU_STREAM_WAIT_VALUE_FLUSH = 1 shl 30  (**< Follow the wait operation with a flush of outstanding remote writes. This
                                             means that, if a remote write operation is guaranteed to have reached the
                                             device before the wait can be satisfied, that write is guaranteed to be
                                             visible to downstream device work. The device is permitted to reorder
                                             remote writes internally. For example, this flag would be required if
                                             two remote writes arrive in a defined order, the wait is satisfied by the
                                             second write, and downstream work needs to observe the first write.
                                             Support for this operation is restricted to selected platforms and can be
                                             queried with ::CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.*)
   );
   CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum;

(**
 * Flags for ::cuStreamWriteValue32
 *)
  CUstreamWriteValue_flags_enum = (
    CU_STREAM_WRITE_VALUE_DEFAULT           = $0, (**< Default behavior *)
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = $1  (**< Permits the write to be reordered with writes which were issued
                                                        before it, as a performance optimization. Normally,
                                                        ::cuStreamWriteValue32 will provide a memory fence before the
                                                        write, which has similar semantics to
                                                        __threadfence_system() but is scoped to the stream
                                                        rather than a CUDA thread.
                                                        This flag is not supported in the v2 API. *)
  );
  CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum;

(**
 * Operations for ::cuStreamBatchMemOp
 *)
  CUstreamBatchMemOpType_enum = (
    CU_STREAM_MEM_OP_WAIT_VALUE_32  = 1,     (**< Represents a ::cuStreamWaitValue32 operation *)
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,     (**< Represents a ::cuStreamWriteValue32 operation *)
    CU_STREAM_MEM_OP_WAIT_VALUE_64  = 4,     (**< Represents a ::cuStreamWaitValue64 operation *)
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,     (**< Represents a ::cuStreamWriteValue64 operation *)
    CU_STREAM_MEM_OP_BARRIER = 6,            (**< Insert a memory barrier of the specified type *)
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3 (**< This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a
                                                  standalone operation. *)
  );
  CUstreamBatchMemOpType = CUstreamBatchMemOpType_enum;

(**
 * Flags for ::cuStreamMemoryBarrier
 *)
  CUstreamMemoryBarrier_flags_enum = (
    CU_STREAM_MEMORY_BARRIER_TYPE_SYS = $0, (**< System-wide memory barrier. *)
    CU_STREAM_MEMORY_BARRIER_TYPE_GPU = $1 (**< Limit memory barrier scope to the GPU. *)
  );
  CUstreamMemoryBarrier_flags = CUstreamMemoryBarrier_flags_enum;

(**
 * Per-operation parameters for ::cuStreamBatchMemOp
 *)
  CUstreamBatchMemOpParams_union = record case byte of
    0 : (operation : CUstreamBatchMemOpType);
    1 : (waitValue : record //struct CUstreamMemOpWaitValueParams_st {
          operation : CUstreamBatchMemOpType;
          address : CUdeviceptr ;
          values : record case boolean of
            false : (value : cuuint32_t);
            true : (value64 : cuuint64_t);
          end;
          flags :longword;
          alias: CUdeviceptr; (**< For driver internal use. Initial value is unimportant. *)
        end) ;
    2 : (writeValue : record
          operation : CUstreamBatchMemOpType;
          address : CUdeviceptr ;
          values : record case boolean of
            false : (value : cuuint32_t);
            true : (value64 : cuuint64_t);
          end;
          flags : longword;
          alias : CUdeviceptr; (**< For driver internal use. Initial value is unimportant. *)
        end);
    3 : (flushRemoteWrites : record
          operation : CUstreamBatchMemOpType;
          flags : longword;
        end;);
    4 : (memoryBarrier : record (**< Only supported in the _v2 API *)
            operation : CUstreamBatchMemOpType;
            flags : longword;
        end);
    5 : (pad : array[0..5] of cuuint64_t);
  end;
  CUstreamBatchMemOpParams_v1 = CUstreamBatchMemOpParams_union;
  PCUstreamBatchMemOpParams =^CUstreamBatchMemOpParams ;
  CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_v1;

  CUDA_BATCH_MEM_OP_NODE_PARAMS_st=packed record
    ctx : CUcontext;
    count : longword;
    paramArray : ^CUstreamBatchMemOpParams;
    flags : longword;
  end;
  PCUDA_BATCH_MEM_OP_NODE_PARAMS =^CUDA_BATCH_MEM_OP_NODE_PARAMS ;
  CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_st;

(**
 * Occupancy calculator flag
 *)
  CUoccupancy_flags_enum = (
    CU_OCCUPANCY_DEFAULT                  = $0, (**< Default behavior *)
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = $1  (**< Assume global caching is enabled and cannot be automatically turned off *)
  );
  CUoccupancy_flags = CUoccupancy_flags_enum;

(**
 * Flags for ::cuStreamUpdateCaptureDependencies
 *)
  CUstreamUpdateCaptureDependencies_flags_enum = (
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = $0, (**< Add new nodes to the dependency set *)
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = $1  (**< Replace the dependency set with the new nodes *)
  );
  CUstreamUpdateCaptureDependencies_flags = CUstreamUpdateCaptureDependencies_flags_enum;

(**
 * Array formats
 *)
  CUarray_format_enum = (
    CU_AD_FORMAT_UNSIGNED_INT8  = $01, (**< Unsigned 8-bit integers *)
    CU_AD_FORMAT_UNSIGNED_INT16 = $02, (**< Unsigned 16-bit integers *)
    CU_AD_FORMAT_UNSIGNED_INT32 = $03, (**< Unsigned 32-bit integers *)
    CU_AD_FORMAT_SIGNED_INT8    = $08, (**< Signed 8-bit integers *)
    CU_AD_FORMAT_SIGNED_INT16   = $09, (**< Signed 16-bit integers *)
    CU_AD_FORMAT_SIGNED_INT32   = $0a, (**< Signed 32-bit integers *)
    CU_AD_FORMAT_HALF           = $10, (**< 16-bit floating point *)
    CU_AD_FORMAT_FLOAT          = $20, (**< 32-bit floating point *)
    CU_AD_FORMAT_NV12           = $b0, (**< 8-bit YUV planar format, with 4:2:0 sampling *)
    CU_AD_FORMAT_UNORM_INT8X1   = $c0, (**< 1 channel unsigned 8-bit normalized integer *)
    CU_AD_FORMAT_UNORM_INT8X2   = $c1, (**< 2 channel unsigned 8-bit normalized integer *)
    CU_AD_FORMAT_UNORM_INT8X4   = $c2, (**< 4 channel unsigned 8-bit normalized integer *)
    CU_AD_FORMAT_UNORM_INT16X1  = $c3, (**< 1 channel unsigned 16-bit normalized integer *)
    CU_AD_FORMAT_UNORM_INT16X2  = $c4, (**< 2 channel unsigned 16-bit normalized integer *)
    CU_AD_FORMAT_UNORM_INT16X4  = $c5, (**< 4 channel unsigned 16-bit normalized integer *)
    CU_AD_FORMAT_SNORM_INT8X1   = $c6, (**< 1 channel signed 8-bit normalized integer *)
    CU_AD_FORMAT_SNORM_INT8X2   = $c7, (**< 2 channel signed 8-bit normalized integer *)
    CU_AD_FORMAT_SNORM_INT8X4   = $c8, (**< 4 channel signed 8-bit normalized integer *)
    CU_AD_FORMAT_SNORM_INT16X1  = $c9, (**< 1 channel signed 16-bit normalized integer *)
    CU_AD_FORMAT_SNORM_INT16X2  = $ca, (**< 2 channel signed 16-bit normalized integer *)
    CU_AD_FORMAT_SNORM_INT16X4  = $cb, (**< 4 channel signed 16-bit normalized integer *)
    CU_AD_FORMAT_BC1_UNORM      = $91, (**< 4 channel unsigned normalized block-compressed (BC1 compression) format *)
    CU_AD_FORMAT_BC1_UNORM_SRGB = $92, (**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*)
    CU_AD_FORMAT_BC2_UNORM      = $93, (**< 4 channel unsigned normalized block-compressed (BC2 compression) format *)
    CU_AD_FORMAT_BC2_UNORM_SRGB = $94, (**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding*)
    CU_AD_FORMAT_BC3_UNORM      = $95, (**< 4 channel unsigned normalized block-compressed (BC3 compression) format *)
    CU_AD_FORMAT_BC3_UNORM_SRGB = $96, (**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding*)
    CU_AD_FORMAT_BC4_UNORM      = $97, (**< 1 channel unsigned normalized block-compressed (BC4 compression) format *)
    CU_AD_FORMAT_BC4_SNORM      = $98, (**< 1 channel signed normalized block-compressed (BC4 compression) format *)
    CU_AD_FORMAT_BC5_UNORM      = $99, (**< 2 channel unsigned normalized block-compressed (BC5 compression) format *)
    CU_AD_FORMAT_BC5_SNORM      = $9a, (**< 2 channel signed normalized block-compressed (BC5 compression) format *)
    CU_AD_FORMAT_BC6H_UF16      = $9b, (**< 3 channel unsigned half-float block-compressed (BC6H compression) format *)
    CU_AD_FORMAT_BC6H_SF16      = $9c, (**< 3 channel signed half-float block-compressed (BC6H compression) format *)
    CU_AD_FORMAT_BC7_UNORM      = $9d, (**< 4 channel unsigned normalized block-compressed (BC7 compression) format *)
    CU_AD_FORMAT_BC7_UNORM_SRGB = $9e  (**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding *)
  );
  PCUarray_format =^CUarray_format ;
  CUarray_format = CUarray_format_enum;

(**
 * Texture reference addressing modes
 *)
  CUaddress_mode_enum = (
    CU_TR_ADDRESS_MODE_WRAP   = 0, (**< Wrapping address mode *)
    CU_TR_ADDRESS_MODE_CLAMP  = 1, (**< Clamp to edge address mode *)
    CU_TR_ADDRESS_MODE_MIRROR = 2, (**< Mirror address mode *)
    CU_TR_ADDRESS_MODE_BORDER = 3  (**< Border address mode *)
  );
  PCUaddress_mode =^CUaddress_mode ;
  CUaddress_mode = CUaddress_mode_enum;

(**
 * Texture reference filtering modes
 *)
  CUfilter_mode_enum = (
    CU_TR_FILTER_MODE_POINT  = 0, (**< Point filter mode *)
    CU_TR_FILTER_MODE_LINEAR = 1  (**< Linear filter mode *)
  );
  PCUfilter_mode =^CUfilter_mode ;
  CUfilter_mode = CUfilter_mode_enum;

(**
 * Device properties
 *)
  CUdevice_attribute_enum = (
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,                          (**< Maximum number of threads per block *)
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                                (**< Maximum block dimension X *)
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                                (**< Maximum block dimension Y *)
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                                (**< Maximum block dimension Z *)
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                                 (**< Maximum grid dimension X *)
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                                 (**< Maximum grid dimension Y *)
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                                 (**< Maximum grid dimension Z *)
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,                    (**< Maximum shared memory available per block in bytes *)
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,                        (**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK *)
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,                          (**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes *)
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                                     (**< Warp size in threads *)
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                                     (**< Maximum pitch in bytes allowed by memory copies *)
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,                       (**< Maximum number of 32-bit registers available per block *)
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,                           (**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK *)
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                                    (**< Typical clock frequency in kilohertz *)
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                             (**< Alignment requirement for textures *)
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                                   (**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. *)
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,                          (**< Number of multiprocessors on device *)
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,                           (**< Specifies whether there is a run time limit on kernels *)
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                                    (**< Device is integrated with host memory *)
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,                           (**< Device can map host memory into CUDA address space *)
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                                  (**< Compute mode (See ::CUcomputemode for details) *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,                       (**< Maximum 1D texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,                       (**< Maximum 2D texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,                      (**< Maximum 2D texture height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,                       (**< Maximum 3D texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,                      (**< Maximum 3D texture height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,                       (**< Maximum 3D texture depth *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,               (**< Maximum 2D layered texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,              (**< Maximum 2D layered texture height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,              (**< Maximum layers in a 2D layered texture *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,                 (**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,                (**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,             (**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS *)
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                             (**< Alignment requirement for surfaces *)
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                            (**< Device can possibly execute multiple kernels concurrently *)
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                                   (**< Device has ECC support enabled *)
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                                    (**< PCI bus ID of the device *)
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                                 (**< PCI device ID of the device *)
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                                    (**< Device is using TCC driver model *)
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                             (**< Peak memory clock frequency in kilohertz *)
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,                       (**< Global memory bus width in bits *)
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                                 (**< Size of L2 cache in bytes *)
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,                (**< Maximum resident threads per multiprocessor *)
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                            (**< Number of asynchronous engines *)
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                            (**< Device shares a unified address space with the host *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,               (**< Maximum 1D layered texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,              (**< Maximum layers in a 1D layered texture *)
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                              (**< Deprecated, do not use. *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,                (**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,               (**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,             (**< Alternate maximum 3D texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,            (**< Alternate maximum 3D texture height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,             (**< Alternate maximum 3D texture depth *)
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                                 (**< PCI domain ID of the device *)
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,                       (**< Pitch alignment requirement for textures *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,                  (**< Maximum cubemap texture width/height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,          (**< Maximum cubemap layered texture width/height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,         (**< Maximum layers in a cubemap layered texture *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,                       (**< Maximum 1D surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,                       (**< Maximum 2D surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,                      (**< Maximum 2D surface height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,                       (**< Maximum 3D surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,                      (**< Maximum 3D surface height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,                       (**< Maximum 3D surface depth *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,               (**< Maximum 1D layered surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,              (**< Maximum layers in a 1D layered surface *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,               (**< Maximum 2D layered surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,              (**< Maximum 2D layered surface height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,              (**< Maximum layers in a 2D layered surface *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,                  (**< Maximum cubemap surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,          (**< Maximum cubemap layered surface width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,         (**< Maximum layers in a cubemap layered surface *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,                (**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,                (**< Maximum 2D linear texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,               (**< Maximum 2D linear texture height *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,                (**< Maximum 2D linear texture pitch in bytes *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,             (**< Maximum mipmapped 2D texture width *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,            (**< Maximum mipmapped 2D texture height *)
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,                      (**< Major compute capability version number *)
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,                      (**< Minor compute capability version number *)
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,             (**< Maximum mipmapped 1D texture width *)
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,                   (**< Device supports stream priorities *)
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,                     (**< Device supports caching globals in L1 *)
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,                      (**< Device supports caching locals in L1 *)
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,          (**< Maximum shared memory available per multiprocessor in bytes *)
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,              (**< Maximum number of 32-bit registers available per multiprocessor *)
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                                (**< Device can allocate managed memory on this system *)
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                               (**< Device is on a multi-GPU board *)
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,                      (**< Unique id for a group of devices on the same multi-GPU board *)
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,                  (**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*)
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,         (**< Ratio of single precision performance (in floating-point operations per second) to double precision performance *)
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,                        (**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it *)
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,                     (**< Device can coherently access managed memory concurrently with the CPU *)
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,                  (**< Device supports compute preemption. *)
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,       (**< Device can access host registered memory at the same virtual address as the CPU *)
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,                     (**< Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related APIs are supported. *)
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,              (**< Deprecated, along with v1 MemOps API, 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. *)
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,              (**< Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is supported. *)
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,                            (**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel *)
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,               (**< Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated. *)
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,             (**< Maximum optin shared memory per block *)
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,                       (**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. *)
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,                       (**< Device supports host memory registration via ::cudaHostRegister. *)
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100, (**< Device accesses pageable memory via the host's page tables. *)
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,          (**< The host can directly access managed memory on the device without migration. *)
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,         (**< Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED*)
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,         (**< Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs *)
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,  (**< Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate *)
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,           (**< Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate *)
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,       (**< Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate *)
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,                (**< Maximum number of blocks per multiprocessor *)
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,                (**< Device supports compression of memory *)
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,                 (**< Maximum L2 persisting lines capacity setting in bytes. *)
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,                (**< Maximum value of CUaccessPolicyWindow::num_bytes. *)
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,      (**< Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate *)
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,             (**< Shared memory reserved by CUDA driver per block in bytes *)
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,                  (**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays *)
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,            (**< Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU *)
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,         (**< External timeline semaphore interop is supported on the device *)
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,                       (**< Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs *)
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,                    (**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) *)
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,         (**< The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum *)
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,              (**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here. *)
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,               (**< Handle types supported with mempool based IPC *)
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,                               (**< Indicates device supports cluster launch *)
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,        (**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays *)
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,                (**< 64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp APIs. *)
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,                (**< ::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs. *)
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,                            (**< Device supports buffer sharing with dma_buf mechanism. *) 
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,                          (**< Device supports IPC Events. *) 
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,                        (**< Number of memory domains the device supports. *)
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,                  (**< Device supports accessing memory using Tensor Map. *)
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,                    (**< Device supports unified function pointers. *)
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,                          (**< Device supports switch multicast and reduction operations. *)
    CU_DEVICE_ATTRIBUTE_MAX
  );
  CUdevice_attribute = CUdevice_attribute_enum;

(**
 * Legacy device properties
 *)
  CUdevprop_st=packed record
    maxThreadsPerBlock  : longint     ;     (**< Maximum number of threads per block *)
    maxThreadsDim       : array[0..2] of longint     ;       (**< Maximum size of each dimension of a block *)
    maxGridSize         : array[0..2] of longint     ;         (**< Maximum size of each dimension of a grid *)
    sharedMemPerBlock   : longint     ;      (**< Shared memory available per block in bytes *)
    totalConstantMemory : longint     ;    (**< Constant memory available on device in bytes *)
    SIMDWidth           : longint     ;              (**< Warp size in threads *)
    memPitch            : longint     ;               (**< Maximum pitch in bytes allowed by memory copies *)
    regsPerBlock        : longint     ;           (**< 32-bit registers available per block *)
    clockRate           : longint     ;              (**< Clock frequency in kilohertz *)
    textureAlign        : longint     ;           (**< Alignment requirement for textures *)
  end;
  CUdevprop_v1 = CUdevprop_st;
  PCUdevprop = ^CUdevprop;
  CUdevprop = CUdevprop_v1;

(**
 * Pointer information
 *)
  CUpointer_attribute_enum = (
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,                     (**< The ::CUcontext on which a pointer was allocated or registered *)
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,                 (**< The ::CUmemorytype describing the physical location of a pointer *)
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,              (**< The address at which a pointer's memory may be accessed on the device *)
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,                (**< The address at which a pointer's memory may be accessed on the host *)
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,                  (**< A pair of tokens for use with the nv-p2p.h Linux kernel interface *)
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,                 (**< Synchronize every synchronous memory operation initiated on this region *)
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,                   (**< A process-wide unique ID for an allocated memory region*)
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,                  (**< Indicates if the pointer points to managed memory *)
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,              (**< A device ordinal of a device on which a pointer was allocated or registered *)
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10, (**< 1 if this pointer maps to an allocation that is suitable for ::cudaIpcGetMemHandle, 0 otherwise **)
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,           (**< Starting address for this requested pointer *)
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,                 (**< Size of the address range for this requested pointer *)
    CU_POINTER_ATTRIBUTE_MAPPED = 13,                     (**< 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise **)
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,       (**< Bitmask of allowed ::CUmemAllocationHandleType for this allocation **)
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15, (**< 1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API **)
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,               (**< Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given *)
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17              (**< Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL. **)
    ,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,               (**< Size of the actual underlying mapping that the pointer belongs to **)
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,          (**< The start address of the mapping that the pointer belongs to **)
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20             (**< A process-wide unique id corresponding to the physical allocation the pointer belongs to **)
  );
  PCUpointer_attribute =^CUpointer_attribute ;
  CUpointer_attribute = CUpointer_attribute_enum;

(**
 * Function properties
 *)
  CUfunction_attribute_enum = (
    (**
     * The maximum number of threads per block, beyond which a launch of the
     * function would fail. This number depends on both the function and the
     * device on which the function is currently loaded.
     *)
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

    (**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     *)
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

    (**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     *)
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

    (**
     * The size in bytes of local memory used by each thread of this function.
     *)
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

    (**
     * The number of registers used by each thread of this function.
     *)
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,

    (**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for cubins
     * compiled prior to CUDA 3.0.
     *)
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,

    (**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy cubins that do not have a
     * properly-encoded binary architecture version.
     *)
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

    (**
     * The attribute to indicate whether the function has been compiled with
     * user specified option "-Xptxas --dlcm=ca" set .
     *)
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,

    (**
     * The maximum size in bytes of dynamically-allocated shared memory that can be used by
     * this function. If the user-specified dynamic shared memory size is larger than this
     * value, the launch will fail.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,

    (**
     * On devices where the L1 cache and shared memory use the same hardware resources, 
     * this sets the shared memory carveout preference, in percent of the total shared memory.
     * Refer to ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR.
     * This is only a hint, and the driver can choose a different ratio if required to execute the function.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,

    (**
     * If this attribute is set, the kernel must launch with a valid cluster
     * size specified.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,

    (**
     * The required cluster width in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,

    (**
     * The required cluster height in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,

    (**
     * The required cluster depth in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,

    (**
     * Whether the function can be launched with non-portable cluster size. 1 is
     * allowed, 0 is disallowed. A non-portable cluster size may only function
     * on the specific SKUs the program is tested on. The launch might fail if
     * the program is run on a different hardware platform.
     *
     * CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking
     * whether the desired size can be launched on the current device.
     *
     * Portable Cluster Size
     *
     * A portable cluster size is guaranteed to be functional on all compute
     * capabilities higher than the target compute capability. The portable
     * cluster size for sm_90 is 8 blocks per cluster. This value may increase
     * for future compute capabilities.
     *
     * The specific hardware unit may support higher cluster sizes that’s not
     * guaranteed to be portable.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,

    (**
     * The block scheduling policy of a function. The value type is
     * CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     *)
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,

    CU_FUNC_ATTRIBUTE_MAX
  );
  CUfunction_attribute = CUfunction_attribute_enum;

(**
 * Function cache configurations
 *)
  CUfunc_cache_enum = (
    CU_FUNC_CACHE_PREFER_NONE    = $00, (**< no preference for shared memory or L1 (default) *)
    CU_FUNC_CACHE_PREFER_SHARED  = $01, (**< prefer larger shared memory and smaller L1 cache *)
    CU_FUNC_CACHE_PREFER_L1      = $02, (**< prefer larger L1 cache and smaller shared memory *)
    CU_FUNC_CACHE_PREFER_EQUAL   = $03  (**< prefer equal sized L1 cache and shared memory *)
  );
  PCUfunc_cache = ^CUfunc_cache;
  CUfunc_cache = CUfunc_cache_enum;

(**
 * Shared memory configurations
 *)
  CUsharedconfig_enum = (
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = $00, (**< set default shared memory bank size *)
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = $01, (**< set shared memory bank width to four bytes *)
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = $02  (**< set shared memory bank width to eight bytes *)
  );
  PCUsharedconfig = ^CUsharedconfig ;
  CUsharedconfig = CUsharedconfig_enum;

(**
 * Shared memory carveout configurations. These may be passed to ::cuFuncSetAttribute or ::cuKernelSetAttribute
 *)
  CUshared_carveout_enum = (
    CU_SHAREDMEM_CARVEOUT_DEFAULT       = -1,  (**< No preference for shared memory or L1 (default) *)
    CU_SHAREDMEM_CARVEOUT_MAX_SHARED    = 100, (**< Prefer maximum available shared memory, minimum L1 cache *)
    CU_SHAREDMEM_CARVEOUT_MAX_L1        = 0    (**< Prefer maximum available L1 cache, minimum shared memory *)
  );
  CUshared_carveout = CUshared_carveout_enum;

(**
 * Memory types
 *)
  CUmemorytype_enum = (
    CU_MEMORYTYPE_HOST    = $01,    (**< Host memory *)
    CU_MEMORYTYPE_DEVICE  = $02,    (**< Device memory *)
    CU_MEMORYTYPE_ARRAY   = $03,    (**< Array memory *)
    CU_MEMORYTYPE_UNIFIED = $04     (**< Unified device or host memory *)
  );
  CUmemorytype = CUmemorytype_enum;

(**
 * Compute Modes
 *)
  CUcomputemode_enum = (
    CU_COMPUTEMODE_DEFAULT           = 0, (**< Default compute mode (Multiple contexts allowed per device) *)
    CU_COMPUTEMODE_PROHIBITED        = 2, (**< Compute-prohibited mode (No contexts can be created on this device at this time) *)
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3  (**< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time) *)
  );
  CUcomputemode = CUcomputemode_enum;

(**
 * Memory advise values
 *)
  CUmem_advise_enum = (
    CU_MEM_ADVISE_SET_READ_MOSTLY          = 1, (**< Data will mostly be read and only occasionally be written to *)
    CU_MEM_ADVISE_UNSET_READ_MOSTLY        = 2, (**< Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY *)
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION   = 3, (**< Set the preferred location for the data as the specified device *)
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4, (**< Clear the preferred location for the data *)
    CU_MEM_ADVISE_SET_ACCESSED_BY          = 5, (**< Data will be accessed by the specified device, so prevent page faults as much as possible *)
    CU_MEM_ADVISE_UNSET_ACCESSED_BY        = 6  (**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device *)
  );
  CUmem_advise = CUmem_advise_enum;

  CUmem_range_attribute_enum = (
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY            = 1, (**< Whether the range will mostly be read and only occasionally be written to *)
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION     = 2, (**< The preferred location of the range *)
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY            = 3, (**< Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device *)
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4  (**< The last location to which the range was prefetched *)
  );
  PCUmem_range_attribute = ^CUmem_range_attribute ;
  CUmem_range_attribute = CUmem_range_attribute_enum;

(**
 * Online compiler and linker options
 *)
  CUjit_option_enum = (
    (**
     * Max number of registers that a thread may use.\n
     * Option type: longint\n
     * Applies to: compiler only
     *)
    CU_JIT_MAX_REGISTERS = 0,

    (**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization of the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Cannot be combined with ::CU_JIT_TARGET.\n
     * Option type: longint\n
     * Applies to: compiler only
     *)
    CU_JIT_THREADS_PER_BLOCK = 1,

    (**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker\n
     * Option type: float\n
     * Applies to: compiler and linker
     *)
    CU_JIT_WALL_TIME = 2,

    (**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     *)
    CU_JIT_INFO_LOG_BUFFER = 3,

    (**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: longint\n
     * Applies to: compiler and linker
     *)
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,

    (**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     *)
    CU_JIT_ERROR_LOG_BUFFER = 5,

    (**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: longint\n
     * Applies to: compiler and linker
     *)
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,

    (**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: longint\n
     * Applies to: compiler only
     *)
    CU_JIT_OPTIMIZATION_LEVEL = 7,

    (**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed\n
     * Applies to: compiler and linker
     *)
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,

    (**
     * Target is chosen based on supplied ::CUjit_target.  Cannot be
     * combined with ::CU_JIT_THREADS_PER_BLOCK.\n
     * Option type: longint for enumerated type ::CUjit_target\n
     * Applies to: compiler and linker
     *)
    CU_JIT_TARGET = 9,

    (**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::CUjit_fallback.  This option cannot be
     * used with cuLink* APIs as the linker requires exact matches.\n
     * Option type: longint for enumerated type ::CUjit_fallback\n
     * Applies to: compiler only
     *)
    CU_JIT_FALLBACK_STRATEGY = 10,

    (**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)\n
     * Option type: longint\n
     * Applies to: compiler and linker
     *)
    CU_JIT_GENERATE_DEBUG_INFO = 11,

    (**
     * Generate verbose log messages (0: false, default)\n
     * Option type: longint\n
     * Applies to: compiler and linker
     *)
    CU_JIT_LOG_VERBOSE = 12,

    (**
     * Generate line number information (-lineinfo) (0: false, default)\n
     * Option type: longint\n
     * Applies to: compiler only
     *)
    CU_JIT_GENERATE_LINE_INFO = 13,

    (**
     * Specifies whether to enable caching explicitly (-dlcm) \n
     * Choice is based on supplied ::CUjit_cacheMode_enum.\n
     * Option type: longint for enumerated type ::CUjit_cacheMode_enum\n
     * Applies to: compiler only
     *)
    CU_JIT_CACHE_MODE = 14,

    (**
     * \deprecated
     * This jit option is deprecated and should not be used.
     *)
    CU_JIT_NEW_SM3X_OPT = 15,

    (**
     * This jit option is used for internal purpose only.
     *)
    CU_JIT_FAST_COMPILE = 16,

    (**
     * Array of device symbol names that will be relocated to the corresponding
     * host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.\n
     * Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * When loading a device module, driver will relocate all encountered
     * unresolved symbols to the host addresses.\n
     * It is only allowed to register symbols that correspond to unresolved
     * global variables.\n
     * It is illegal to register the same device symbol at multiple addresses.\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     *)
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,

    (**
     * Array of host addresses that will be used to relocate corresponding
     * device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.\n
     * Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * Option type: void **\n
     * Applies to: dynamic linker only
     *)
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,

    (**
     * Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
     * ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.\n
     * Option type: longint\n
     * Applies to: dynamic linker only
     *)
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,

    (**
     * \deprecated
     * Enable link-time optimization (-dlto) for device code (Disabled by default).\n
     * This option is not supported on 32-bit platforms.\n
     * Option type: longint\n
     * Applies to: compiler and linker
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_LTO = 20,

    (**
     * \deprecated
     * Control single-precision denormals (-ftz) support (0: false, default).
     * 1 : flushes denormal values to zero
     * 0 : preserves denormal values
     * Option type: longint\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_FTZ = 21,

    (**
     * \deprecated
     * Control single-precision floating-point division and reciprocals
     * (-prec-div) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: longint\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_PREC_DIV = 22,

    (**
     * \deprecated
     * Control single-precision floating-point square root
     * (-prec-sqrt) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: longint\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_PREC_SQRT = 23,

    (**
     * \deprecated
     * Enable/Disable the contraction of floating-point multiplies
     * and adds/subtracts into floating-point multiply-add (-fma)
     * operations (1: Enable, default; 0: Disable).
     * Option type: longint\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_FMA = 24,

    (**
     * \deprecated
     * Array of kernel names that should be preserved at link time while others
     * can be removed.\n
     * Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.\n
     * Note that kernel names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all kernels with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,

    (**
     * \deprecated
     * Number of entries in ::CU_JIT_REFERENCED_KERNEL_NAMES array.\n
     * Option type: longint\n
     * Applies to: dynamic linker only
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,

    (**
     * \deprecated
     * Array of variable names (__device__ and/or __constant__) that should be
     * preserved at link time while others can be removed.\n
     * Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.\n
     * Note that variable names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all variables with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,

    (**
     * \deprecated
     * Number of entries in ::CU_JIT_REFERENCED_VARIABLE_NAMES array.\n
     * Option type: longint\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,

    (**
     * \deprecated
     * This option serves as a hint to enable the JIT compiler/linker
     * to remove constant (__constant__) and device (__device__) variables
     * unreferenced in device code (Disabled by default).\n
     * Note that host references to constant and device variables using APIs like
     * ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
     * the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.\n
     * Option type: longint\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,

    (**
     * Generate position independent code (0: false)\n
     * Option type: longint\n
     * Applies to: compiler only
     *)
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,

    CU_JIT_NUM_OPTIONS

  );
  PCUjit_option = ^CUjit_option ;
  CUjit_option = CUjit_option_enum;


(**
 * Online compilation targets
 *)
  CUjit_target_enum = (
    CU_TARGET_COMPUTE_30 = 30,       (**< Compute device class 3.0 *)
    CU_TARGET_COMPUTE_32 = 32,       (**< Compute device class 3.2 *)
    CU_TARGET_COMPUTE_35 = 35,       (**< Compute device class 3.5 *)
    CU_TARGET_COMPUTE_37 = 37,       (**< Compute device class 3.7 *)
    CU_TARGET_COMPUTE_50 = 50,       (**< Compute device class 5.0 *)
    CU_TARGET_COMPUTE_52 = 52,       (**< Compute device class 5.2 *)
    CU_TARGET_COMPUTE_53 = 53,       (**< Compute device class 5.3 *)
    CU_TARGET_COMPUTE_60 = 60,       (**< Compute device class 6.0.*)
    CU_TARGET_COMPUTE_61 = 61,       (**< Compute device class 6.1.*)
    CU_TARGET_COMPUTE_62 = 62,       (**< Compute device class 6.2.*)
    CU_TARGET_COMPUTE_70 = 70,       (**< Compute device class 7.0.*)
    CU_TARGET_COMPUTE_72 = 72,       (**< Compute device class 7.2.*)
    CU_TARGET_COMPUTE_75 = 75,       (**< Compute device class 7.5.*)
    CU_TARGET_COMPUTE_80 = 80,       (**< Compute device class 8.0.*)
    CU_TARGET_COMPUTE_86 = 86,       (**< Compute device class 8.6.*)
    CU_TARGET_COMPUTE_87 = 87,       (**< Compute device class 8.7.*)
    CU_TARGET_COMPUTE_89 = 89,       (**< Compute device class 8.9.*)
    CU_TARGET_COMPUTE_90 = 90         (**< Compute device class 9.0.*)

    (**< Compute device class 9.0. with accelerated features.*)
  );
  CUjit_target = CUjit_target_enum;
const
  CU_TARGET_COMPUTE_90A = longword(CU_COMPUTE_ACCELERATED_TARGET_BASE) + longword(CU_TARGET_COMPUTE_90);

type

(**
 * Cubin matching fallback strategies
 *)
  CUjit_fallback_enum = (
    CU_PREFER_PTX = 0,  (**< Prefer to compile ptx if exact binary match not found *)

    CU_PREFER_BINARY    (**< Prefer to fall back to compatible binary code if exact match not found *)

  );
  CUjit_fallback = CUjit_fallback_enum;

(**
 * Caching modes for dlcm
 *)
  CUjit_cacheMode_enum = (
    CU_JIT_CACHE_OPTION_NONE = 0, (**< Compile with no -dlcm flag specified *)
    CU_JIT_CACHE_OPTION_CG,       (**< Compile with L1 cache disabled *)
    CU_JIT_CACHE_OPTION_CA        (**< Compile with L1 cache enabled *)
  );
  CUjit_cacheMode = CUjit_cacheMode_enum;

(**
 * Device code formats
 *)
  CUjitInputType_enum = (
    (**
     * Compiled device-class-specific device code\n
     * Applicable options: none
     *)
    CU_JIT_INPUT_CUBIN = 0,

    (**
     * PTX source code\n
     * Applicable options: PTX compiler options
     *)
    CU_JIT_INPUT_PTX = 1,

    (**
     * Bundle of multiple cubins and/or PTX of some device code\n
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     *)
    CU_JIT_INPUT_FATBINARY = 2,

    (**
     * Host object with embedded device code\n
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     *)
    CU_JIT_INPUT_OBJECT = 3,

    (**
     * Archive of host objects with embedded device code\n
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     *)
    CU_JIT_INPUT_LIBRARY = 4,

    (**
     * \deprecated
     * High-level intermediate code for link-time optimization\n
     * Applicable options: NVVM compiler options, PTX compiler options
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     *)
    CU_JIT_INPUT_NVVM = 5,

    CU_JIT_NUM_INPUT_TYPES = 6
  );
  CUjitInputType = CUjitInputType_enum;

  CUlinkState_st=record end;
  PCUlinkState = ^CUlinkState ;
  CUlinkState = ^CUlinkState_st;


(**
 * Flags to register a graphics resource
 *)
  CUgraphicsRegisterFlags_enum = (
    CU_GRAPHICS_REGISTER_FLAGS_NONE           = $00,
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY      = $01,
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD  = $02,
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST   = $04,
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = $08
  );
  CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum;

(**
 * Flags for mapping and unmapping interop resources
 *)
  CUgraphicsMapResourceFlags_enum = (
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = $00,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = $01,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = $02
  );
  CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum;

(**
 * Array indices for cube faces
 *)
  CUarray_cubemap_face_enum = (
    CU_CUBEMAP_FACE_POSITIVE_X  = $00, (**< Positive X face of cubemap *)
    CU_CUBEMAP_FACE_NEGATIVE_X  = $01, (**< Negative X face of cubemap *)
    CU_CUBEMAP_FACE_POSITIVE_Y  = $02, (**< Positive Y face of cubemap *)
    CU_CUBEMAP_FACE_NEGATIVE_Y  = $03, (**< Negative Y face of cubemap *)
    CU_CUBEMAP_FACE_POSITIVE_Z  = $04, (**< Positive Z face of cubemap *)
    CU_CUBEMAP_FACE_NEGATIVE_Z  = $05  (**< Negative Z face of cubemap *)
  );
  CUarray_cubemap_face = CUarray_cubemap_face_enum;

(**
 * Limits
 *)
  CUlimit_enum = (
    CU_LIMIT_STACK_SIZE                       = $00, (**< GPU thread stack size *)
    CU_LIMIT_PRINTF_FIFO_SIZE                 = $01, (**< GPU printf FIFO size *)
    CU_LIMIT_MALLOC_HEAP_SIZE                 = $02, (**< GPU malloc heap size *)
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = $03, (**< GPU device runtime launch synchronize depth *)
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = $04, (**< GPU device runtime pending launch count *)
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY         = $05, (**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint *)
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE         = $06, (**< A size in bytes for L2 persisting lines cache size *)
    CU_LIMIT_MAX
  );
  CUlimit = CUlimit_enum;

(**
 * Resource types
 *)
  CUresourcetype_enum = (
    CU_RESOURCE_TYPE_ARRAY           = $00, (**< Array resource *)
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = $01, (**< Mipmapped array resource *)
    CU_RESOURCE_TYPE_LINEAR          = $02, (**< Linear resource *)
    CU_RESOURCE_TYPE_PITCH2D         = $03  (**< Pitch 2D resource *)
  );
  CUresourcetype = CUresourcetype_enum;


(**
 * CUDA host function
 * \param userData Argument value passed to the function
 *)
  CUhostFn = procedure(userData : pointer);WINAPI;

(**
 * Specifies performance hint with ::CUaccessPolicyWindow for hitProp and missProp members.
 *)
  CUaccessProperty_enum = (
    CU_ACCESS_PROPERTY_NORMAL           = 0,    (**< Normal cache persistence. *)
    CU_ACCESS_PROPERTY_STREAMING        = 1,    (**< Streaming access is less likely to persit from cache. *)
    CU_ACCESS_PROPERTY_PERSISTING       = 2     (**< Persisting access is more likely to persist in cache.*)
  );
  CUaccessProperty = CUaccessProperty_enum;

(**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * num_bytes is limited by CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE.
 * Partition into many segments and assign segments such that:
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 *)
  CUaccessPolicyWindow_st=packed record
   base_ptr  : pointer          ;                     (**< Starting address of the access policy window. CUDA driver may align it. *)
   num_bytes : size_t           ;                   (**< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. *)
   hitRatio  : single           ;                     (**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. *)
   hitProp   : CUaccessProperty ;           (**< ::CUaccessProperty set for hit. *)
   missProp  : CUaccessProperty ;          (**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING *)
  end;
  CUaccessPolicyWindow_v1 = CUaccessPolicyWindow_st;
(**
 * Access policy window
 *)
  CUaccessPolicyWindow = CUaccessPolicyWindow_v1;

(**
 * GPU kernel node parameters
 *)
  CUDA_KERNEL_NODE_PARAMS_st=packed record
     func           : CUfunction ;             (**< Kernel to launch *)
     gridDimX       : longword   ;       (**< Width of grid in blocks *)
     gridDimY       : longword   ;       (**< Height of grid in blocks *)
     gridDimZ       : longword   ;       (**< Depth of grid in blocks *)
     blockDimX      : longword   ;      (**< X dimension of each thread block *)
     blockDimY      : longword   ;      (**< Y dimension of each thread block *)
     blockDimZ      : longword   ;      (**< Z dimension of each thread block *)
     sharedMemBytes : longword   ; (**< Dynamic shared-memory size per thread block in bytes *)
     kernelParams   : PPointer   ;         (**< Array of pointers to kernel parameters *)
     extra          : PPointer   ;                (**< Extra options *)
  end;
  CUDA_KERNEL_NODE_PARAMS_v1 = CUDA_KERNEL_NODE_PARAMS_st;

(**
 * GPU kernel node parameters
 *)
  CUDA_KERNEL_NODE_PARAMS_v2_st=packed record
    func           : CUfunction  ;             (**< Kernel to launch *)
    gridDimX       : longword    ;       (**< Width of grid in blocks *)
    gridDimY       : longword    ;       (**< Height of grid in blocks *)
    gridDimZ       : longword    ;       (**< Depth of grid in blocks *)
    blockDimX      : longword    ;      (**< X dimension of each thread block *)
    blockDimY      : longword    ;      (**< Y dimension of each thread block *)
    blockDimZ      : longword    ;      (**< Z dimension of each thread block *)
    sharedMemBytes : longword    ; (**< Dynamic shared-memory size per thread block in bytes *)
    kernelParams   : PPointer    ;         (**< Array of pointers to kernel parameters *)
    extra          : PPointer    ;                (**< Extra options *)
    kern           : CUkernel    ;               (**< Kernel to launch, will only be referenced if func is NULL *)
    ctx            : CUcontext   ;               (**< Context for the kernel task to run in. The value NULL will indicate the current context should be used by the api. This field is ignored if func is set. *)
  end;
  CUDA_KERNEL_NODE_PARAMS_v2 = CUDA_KERNEL_NODE_PARAMS_v2_st;
  PCUDA_KERNEL_NODE_PARAMS =^CUDA_KERNEL_NODE_PARAMS ;
  CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v2;

(**
 * Memset node parameters
 *)
  CUDA_MEMSET_NODE_PARAMS_st=packed record
     dst         : CUdeviceptr;                        (**< Destination device pointer *)
     pitch       : size_t     ;                           (**< Pitch of destination device pointer. Unused if height is 1 *)
     value       : longword    ;                     (**< Value to be set *)
     elementSize : longword    ;               (**< Size of each element in bytes. Must be 1, 2, or 4. *)
     width       : size_t      ;                           (**< Width of the row in elements *)
     height      : size_t      ;                          (**< Number of rows *)
  end;
  CUDA_MEMSET_NODE_PARAMS_v1 = CUDA_MEMSET_NODE_PARAMS_st;
  PCUDA_MEMSET_NODE_PARAMS=^CUDA_MEMSET_NODE_PARAMS;
  CUDA_MEMSET_NODE_PARAMS= CUDA_MEMSET_NODE_PARAMS_v1;

(**
 * Host node parameters
 *)
  CUDA_HOST_NODE_PARAMS_st=packed record
    fn       : CUhostFn    ;    (**< The function to call when the node executes *)
    userData : pointer     ; (**< Argument to pass to the function *)
  end;
  CUDA_HOST_NODE_PARAMS_v1 = CUDA_HOST_NODE_PARAMS_st;
  PCUDA_HOST_NODE_PARAMS =^CUDA_HOST_NODE_PARAMS ;
  CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_v1;

(**
 * Graph node types
 *)
  CUgraphNodeType_enum = (
    CU_GRAPH_NODE_TYPE_KERNEL           = 0, (**< GPU kernel node *)
    CU_GRAPH_NODE_TYPE_MEMCPY           = 1, (**< Memcpy node *)
    CU_GRAPH_NODE_TYPE_MEMSET           = 2, (**< Memset node *)
    CU_GRAPH_NODE_TYPE_HOST             = 3, (**< Host (executable) node *)
    CU_GRAPH_NODE_TYPE_GRAPH            = 4, (**< Node which executes an embedded graph *)
    CU_GRAPH_NODE_TYPE_EMPTY            = 5, (**< Empty (no-op) node *)
    CU_GRAPH_NODE_TYPE_WAIT_EVENT       = 6, (**< External event wait node *)
    CU_GRAPH_NODE_TYPE_EVENT_RECORD     = 7, (**< External event record node *)
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8, (**< External semaphore signal node *)
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT   = 9, (**< External semaphore wait node *)
    CU_GRAPH_NODE_TYPE_MEM_ALLOC        = 10,(**< Memory Allocation Node *)
    CU_GRAPH_NODE_TYPE_MEM_FREE         = 11,(**< Memory Free Node *)
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP     = 12 (**< Batch MemOp Node *)
  );
  PCUgraphNodeType = ^CUgraphNodeType ;
  CUgraphNodeType = CUgraphNodeType_enum;

(**
 * Graph instantiation results
*)
  CUgraphInstantiateResult_enum = (
    CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,                          (**< Instantiation succeeded *)
    CUDA_GRAPH_INSTANTIATE_ERROR = 1,                            (**< Instantiation failed for an unexpected reason which is described in the return value of the function *)
    CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,                (**< Instantiation failed due to invalid structure, such as cycles *)
    CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,     (**< Instantiation for device launch failed because the graph contained an unsupported operation *)
    CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4       (**< Instantiation for device launch failed due to the nodes belonging to different contexts *)
  );
  CUgraphInstantiateResult = CUgraphInstantiateResult_enum;

(**
 * Graph instantiation parameters
 *)
  CUDA_GRAPH_INSTANTIATE_PARAMS_st=packed record
        flags         : cuuint64_t               ;                    (**< Instantiation flags *)
	hUploadStream : CUstream                 ;              (**< Upload stream *)
	hErrNode_out  : CUgraphNode              ;            (**< The node which caused instantiation to fail, if any *)
	result_out    : CUgraphInstantiateResult ; (**< Whether instantiation was successful.  If it failed, the reason why *)
  end;
  PCUDA_GRAPH_INSTANTIATE_PARAMS =^CUDA_GRAPH_INSTANTIATE_PARAMS ;
  CUDA_GRAPH_INSTANTIATE_PARAMS = CUDA_GRAPH_INSTANTIATE_PARAMS_st;

  CUsynchronizationPolicy_enum = (
    CU_SYNC_POLICY_AUTO = 1,
    CU_SYNC_POLICY_SPIN = 2,
    CU_SYNC_POLICY_YIELD = 3,
    CU_SYNC_POLICY_BLOCKING_SYNC = 4
  );
  CUsynchronizationPolicy = CUsynchronizationPolicy_enum;

(**
 * Cluster scheduling policies. These may be passed to ::cuFuncSetAttribute or ::cuKernelSetAttribute
 *)
  CUclusterSchedulingPolicy_enum = (
    CU_CLUSTER_SCHEDULING_POLICY_DEFAULT        = 0, (**< the default policy *)
    CU_CLUSTER_SCHEDULING_POLICY_SPREAD         = 1, (**< spread the blocks within a cluster to the SMs *)
    CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2  (**< allow the hardware to load-balance the blocks in a cluster to the SMs *)
  );
  CUclusterSchedulingPolicy = CUclusterSchedulingPolicy_enum;

  CUlaunchMemSyncDomain_enum = (
    CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = 0,
    CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE  = 1
  );
  CUlaunchMemSyncDomain = CUlaunchMemSyncDomain_enum;

  CUlaunchMemSyncDomainMap_st=packed record
    default_ : byte ;
    remote   : byte ;
  end;
  CUlaunchMemSyncDomainMap = CUlaunchMemSyncDomainMap_st;

  CUlaunchAttributeID_enum = (
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0 (**< Ignored entry, for convenient composition *)
  , CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW   = 1 (**< Valid for streams, graph nodes, launches. *)
  , CU_LAUNCH_ATTRIBUTE_COOPERATIVE            = 2 (**< Valid for graph nodes, launches. *)
  , CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3 (**< Valid for streams. *)
  , CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION                    = 4 (**< Valid for graph nodes, launches. *)
  , CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5 (**< Valid for graph nodes, launches. *)
  , CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION    = 6 (**< Valid for launches. Setting
                                                                      programmaticStreamSerializationAllowed to non-0
                                                                      signals that the kernel will use programmatic
                                                                      means to resolve its stream dependency, so that
                                                                      the CUDA runtime should opportunistically allow
                                                                      the grid's execution to overlap with the previous
                                                                      kernel in the stream, if that kernel requests the
                                                                      overlap. The dependent launches can choose to wait
                                                                      on the dependency using the programmatic sync
                                                                      (cudaGridDependencySynchronize() or equivalent PTX
                                                                      instructions). *)
  , CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT                   = 7 (**< Valid for launches. Event recorded through this
                                                                      launch attribute is guaranteed to only trigger
                                                                      after all block in the associated kernel trigger
                                                                      the event. A block can trigger the event through
                                                                      PTX launchdep.release or CUDA builtin function
                                                                      cudaTriggerProgrammaticLaunchCompletion(). A
                                                                      trigger can also be inserted at the beginning of
                                                                      each block's execution if triggerAtBlockStart is
                                                                      set to non-0. The dependent launches can choose to
                                                                      wait on the dependency using the programmatic sync
                                                                      (cudaGridDependencySynchronize() or equivalent PTX
                                                                      instructions). Note that dependents (including the
                                                                      CPU thread calling cuEventSynchronize()) are not
                                                                      guaranteed to observe the release precisely when
                                                                      it is released.  For example, cuEventSynchronize()
                                                                      may only observe the event trigger long after the
                                                                      associated kernel has completed. This recording
                                                                      type is primarily meant for establishing
                                                                      programmatic dependency between device tasks. The
                                                                      event supplied must not be an interprocess or
                                                                      interop event. The event must disable timing (i.e.
                                                                      created with ::CU_EVENT_DISABLE_TIMING flag set).
                                                                      *)
  , CU_LAUNCH_ATTRIBUTE_PRIORITY               = 8 (**< Valid for streams, graph nodes, launches. *)
  , CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP    = 9
  , CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN        = 10
  , CU_LAUNCH_ATTRIBUTE_MAX
  );
  CUlaunchAttributeID = CUlaunchAttributeID_enum;

const
  CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW =  CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;
  CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE =           CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION =                     CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE =  CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  CU_KERNEL_NODE_ATTRIBUTE_PRIORITY =              CU_LAUNCH_ATTRIBUTE_PRIORITY;
  CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP =   CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP;
  CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN =       CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;

type

{*< Pad to 64 bytes  }
{*< Attribute ::CUaccessPolicyWindow.  }
{*< Nonzero indicates a cooperative kernel (see ::cuLaunchCooperativeKernel).  }
{*< ::CUsynchronizationPolicy for work queued up in this stream  }
{*< Cluster dimensions for the kernel node.  }
{*< Cluster scheduling policy preference for the kernel node.  }
{ Does not accept ::CU_EVENT_RECORD_EXTERNAL  }
{*< Execution priority of the kernel.  }

  CUlaunchAttributeValue_union = record
  case longint of
    0 : ( pad : array[0..63] of char );
    1 : ( accessPolicyWindow : CUaccessPolicyWindow );
    2 : ( cooperative : longint );
    3 : ( syncPolicy : CUsynchronizationPolicy );
    4 : ( clusterDim : record
        x : longword;
        y : longword;
        z : longword;
      end );
    5 : ( clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy );
    6 : ( programmaticStreamSerializationAllowed : longint );
    7 : ( programmaticEvent : record
        event : CUevent;
        flags : longint;
        triggerAtBlockStart : longint;
      end );
    8 : ( priority : longint );
    9 : ( memSyncDomainMap : CUlaunchMemSyncDomainMap );
    10 : ( memSyncDomain : CUlaunchMemSyncDomain );
  end;
  CUlaunchAttributeValue = CUlaunchAttributeValue_union;

  CUlaunchAttribute_st = record
      id     : CUlaunchAttributeID                                 ;
      pad    : array[0..8-sizeof(CUlaunchAttributeID)-1] of char   ;
      value  : CUlaunchAttributeValue
  end;
  CUlaunchAttribute = CUlaunchAttribute_st;

  CUlaunchConfig_st=packed record
    gridDimX       : longword            ;       (**< Width of grid in blocks *)
    gridDimY       : longword            ;       (**< Height of grid in blocks *)
    gridDimZ       : longword            ;       (**< Depth of grid in blocks *)
    blockDimX      : longword            ;      (**< X dimension of each thread block *)
    blockDimY      : longword            ;      (**< Y dimension of each thread block *)
    blockDimZ      : longword            ;      (**< Z dimension of each thread block *)
    sharedMemBytes : longword            ; (**< Dynamic shared-memory size per thread block in bytes *)
    hStream        : CUstream            ;            (**< Stream identifier *)
    attrs          : ^CUlaunchAttribute  ;          (**< nullable if numAttrs == 0 *)
    numAttrs       : longword             ;       (**< number of attributes populated in attrs *)
  end;
  PCUlaunchConfig =^CUlaunchConfig ;
  CUlaunchConfig = CUlaunchConfig_st;

  CUkernelNodeAttrID = CUlaunchAttributeID;

  CUkernelNodeAttrValue_v1 = CUlaunchAttributeValue;
  PCUkernelNodeAttrValue =^CUkernelNodeAttrValue ;
  CUkernelNodeAttrValue = CUkernelNodeAttrValue_v1;

(**
 * Possible stream capture statuses returned by ::cuStreamIsCapturing
 *)
  CUstreamCaptureStatus_enum = (
    CU_STREAM_CAPTURE_STATUS_NONE        = 0, (**< Stream is not capturing *)
    CU_STREAM_CAPTURE_STATUS_ACTIVE      = 1, (**< Stream is actively capturing *)
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2  (**< Stream is part of a capture sequence that
                                                   has been invalidated, but not terminated *)
  );
  PCUstreamCaptureStatus=^CUstreamCaptureStatus;
  CUstreamCaptureStatus = CUstreamCaptureStatus_enum;

(**
 * Possible modes for stream capture thread interactions. For more details see
 * ::cuStreamBeginCapture and ::cuThreadExchangeStreamCaptureMode
 *)
  CUstreamCaptureMode_enum = (
    CU_STREAM_CAPTURE_MODE_GLOBAL       = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED      = 2
  );
  PCUstreamCaptureMode =^CUstreamCaptureMode ;
  CUstreamCaptureMode = CUstreamCaptureMode_enum;

  CUstreamAttrID = CUlaunchAttributeID;

const
     CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW =    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;
      CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY =  CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY;
      CU_STREAM_ATTRIBUTE_PRIORITY =                CU_LAUNCH_ATTRIBUTE_PRIORITY;
      CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP =     CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP;
      CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN =         CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
type
  CUstreamAttrValue_v1 = CUlaunchAttributeValue;
  PCUstreamAttrValue =^CUstreamAttrValue ;
  CUstreamAttrValue = CUstreamAttrValue_v1;

(**
 * Flags to specify search options. For more details see ::cuGetProcAddress
 *)
  CUdriverProcAddress_flags_enum = (
    CU_GET_PROC_ADDRESS_DEFAULT = 0,                        (**< Default search mode for driver symbols. *)
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1 shl 0,             (**< Search for legacy versions of driver symbols. *)
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 1 shl 1  (**< Search for per-thread versions of driver symbols. *)
  );
  CUdriverProcAddress_flags = CUdriverProcAddress_flags_enum;

(**
 * Flags to indicate search status. For more details see ::cuGetProcAddress
 *)
  CUdriverProcAddressQueryResult_enum = (
    CU_GET_PROC_ADDRESS_SUCCESS                = 0,  (**< Symbol was succesfully found *)
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND       = 1,  (**< Symbol was not found in search *)
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2   (**< Symbol was found but version supplied was not sufficient *)
  );
  PCUdriverProcAddressQueryResult =^CUdriverProcAddressQueryResult ;
  CUdriverProcAddressQueryResult = CUdriverProcAddressQueryResult_enum;

(**
 * Execution Affinity Types 
 *)
  CUexecAffinityType_enum = (
    CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0,  (**< Create a context with limited SMs. *)
    CU_EXEC_AFFINITY_TYPE_MAX
  );
  CUexecAffinityType = CUexecAffinityType_enum;

(**
 * Value for ::CU_EXEC_AFFINITY_TYPE_SM_COUNT
 *)
  CUexecAffinitySmCount_st=packed record
    val : longword ;    (**< The number of SMs the context is limited to use. *)
  end;
  CUexecAffinitySmCount_v1 = CUexecAffinitySmCount_st;
  CUexecAffinitySmCount = CUexecAffinitySmCount_v1;

{* Value for ::CU_EXEC_AFFINITY_TYPE_SM_COUNT  }

  CUexecAffinityParam_st = record
      _type : CUexecAffinityType;
      param : record
          case longint of
            0 : ( smCount : CUexecAffinitySmCount );
          end;
  end;
  CUexecAffinityParam_v1 = CUexecAffinityParam_st;
(**
 * Execution Affinity Parameters
 *)
  PCUexecAffinityParam =^CUexecAffinityParam;
  CUexecAffinityParam= CUexecAffinityParam_v1;

(**
 * Library options to be specified with ::cuLibraryLoadData() or ::cuLibraryLoadFromFile()
 *)
  CUlibraryOption_enum = (
    CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0,

    (**
     * Specifes that the argument \p code passed to ::cuLibraryLoadData() will be preserved.
     * Specifying this option will let the driver know that \p code can be accessed at any point
     * until ::cuLibraryUnload(). The default behavior is for the driver to allocate and
     * maintain its own copy of \p code. Note that this is only a memory usage optimization
     * hint and the driver can choose to ignore it if required.
     * Specifying this option with ::cuLibraryLoadFromFile() is invalid and
     * will return ::CUDA_ERROR_INVALID_VALUE.
     *)
    CU_LIBRARY_BINARY_IS_PRESERVED = 1,

    CU_LIBRARY_NUM_OPTIONS
  );
  PCUlibraryOption = ^CUlibraryOption ;
  CUlibraryOption = CUlibraryOption_enum;

  CUlibraryHostUniversalFunctionAndDataTable_st=packed record
      functionTable      : pointer ;
      functionWindowSize : size_t  ;
      dataTable          : pointer ;
      dataWindowSize     : size_t  ;
  end;
  CUlibraryHostUniversalFunctionAndDataTable = CUlibraryHostUniversalFunctionAndDataTable_st;

(**
 * Error codes
 *)
  cudaError_enum = (
    (**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     *)
    CUDA_SUCCESS                              = 0,

    (**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     *)
    CUDA_ERROR_INVALID_VALUE                  = 1,

    (**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     *)
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,

    (**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     *)
    CUDA_ERROR_NOT_INITIALIZED                = 3,

    (**
     * This indicates that the CUDA driver is in the process of shutting down.
     *)
    CUDA_ERROR_DEINITIALIZED                  = 4,

    (**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     *)
    CUDA_ERROR_PROFILER_DISABLED              = 5,

    (**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cuProfilerStart or
     * ::cuProfilerStop without initialization.
     *)
    CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,

    (**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStart() when profiling is already enabled.
     *)
    CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,

    (**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStop() when profiling is already disabled.
     *)
    CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,

    (**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     *)
    CUDA_ERROR_STUB_LIBRARY                   = 34,

    (**
     * This indicates that requested CUDA device is unavailable at the current
     * time. Devices are often unavailable due to use of
     * ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS or ::CU_COMPUTEMODE_PROHIBITED.
     *)
    CUDA_ERROR_DEVICE_UNAVAILABLE            = 46,

    (**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     *)
    CUDA_ERROR_NO_DEVICE                      = 100,

    (**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     *)
    CUDA_ERROR_INVALID_DEVICE                 = 101,

    (**
     * This error indicates that the Grid license is not applied.
     *)
    CUDA_ERROR_DEVICE_NOT_LICENSED            = 102,

    (**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     *)
    CUDA_ERROR_INVALID_IMAGE                  = 200,

    (**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     *)
    CUDA_ERROR_INVALID_CONTEXT                = 201,

    (**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     *)
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    (**
     * This indicates that a map or register operation has failed.
     *)
    CUDA_ERROR_MAP_FAILED                     = 205,

    (**
     * This indicates that an unmap or unregister operation has failed.
     *)
    CUDA_ERROR_UNMAP_FAILED                   = 206,

    (**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     *)
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,

    (**
     * This indicates that the resource is already mapped.
     *)
    CUDA_ERROR_ALREADY_MAPPED                 = 208,

    (**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     *)
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,

    (**
     * This indicates that a resource has already been acquired.
     *)
    CUDA_ERROR_ALREADY_ACQUIRED               = 210,

    (**
     * This indicates that a resource is not mapped.
     *)
    CUDA_ERROR_NOT_MAPPED                     = 211,

    (**
     * This indicates that a mapped resource is not available for access as an
     * array.
     *)
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    (**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     *)
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    (**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     *)
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,

    (**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     *)
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,

    (**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     *)
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    (**
     * This indicates that peer access is not supported across the given
     * devices.
     *)
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,

    (**
     * This indicates that a PTX JIT compilation failed.
     *)
    CUDA_ERROR_INVALID_PTX                    = 218,

    (**
     * This indicates an error with OpenGL or DirectX context.
     *)
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,

    (**
    * This indicates that an uncorrectable NVLink error was detected during the
    * execution.
    *)
    CUDA_ERROR_NVLINK_UNCORRECTABLE           = 220,

    (**
    * This indicates that the PTX JIT compiler library was not found.
    *)
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND         = 221,

    (**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     *)

    CUDA_ERROR_UNSUPPORTED_PTX_VERSION        = 222,

    (**
     * This indicates that the PTX JIT compilation was disabled.
     *)
    CUDA_ERROR_JIT_COMPILATION_DISABLED       = 223,

    (**
     * This indicates that the ::CUexecAffinityType passed to the API call is not
     * supported by the active device.
     *)
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY      = 224,

    (**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to cudaDeviceSynchronize.
     *)
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC       = 225,

    (**
     * This indicates that the device kernel source is invalid. This includes
     * compilation/linker errors encountered in device code or user error.
     *)
    CUDA_ERROR_INVALID_SOURCE                 = 300,

    (**
     * This indicates that the file specified was not found.
     *)
    CUDA_ERROR_FILE_NOT_FOUND                 = 301,

    (**
     * This indicates that a link to a shared object failed to resolve.
     *)
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    (**
     * This indicates that initialization of a shared object failed.
     *)
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    (**
     * This indicates that an OS call failed.
     *)
    CUDA_ERROR_OPERATING_SYSTEM               = 304,

    (**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     *)
    CUDA_ERROR_INVALID_HANDLE                 = 400,

    (**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     *)
    CUDA_ERROR_ILLEGAL_STATE                  = 401,

    (**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     *)
    CUDA_ERROR_NOT_FOUND                      = 500,

    (**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     *)
    CUDA_ERROR_NOT_READY                      = 600,

    (**
     * While executing a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_ILLEGAL_ADDRESS                = 700,

    (**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit longint is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     *)
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    (**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,

    (**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     *)
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    (**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     *)
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,

    (**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     *)
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,

    (**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     *)
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    (**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     *)
    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    (**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     *)
    CUDA_ERROR_ASSERT                         = 710,

    (**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     *)
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,

    (**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     *)
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    (**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     *)
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    (**
     * While executing a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_HARDWARE_STACK_ERROR           = 714,

    (**
     * While executing a kernel, the device encountered an illegal instruction.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715,

    (**
     * While executing a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_MISALIGNED_ADDRESS             = 716,

    (**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717,

    (**
     * While executing a kernel, the device program counter wrapped its address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_INVALID_PC                     = 718,

    (**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     *)
    CUDA_ERROR_LAUNCH_FAILED                  = 719,

    (**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
     *)
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,

    (**
     * This error indicates that the attempted operation is not permitted.
     *)
    CUDA_ERROR_NOT_PERMITTED                  = 800,

    (**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     *)
    CUDA_ERROR_NOT_SUPPORTED                  = 801,

    (**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     *)
    CUDA_ERROR_SYSTEM_NOT_READY               = 802,

    (**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     *)
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH         = 803,

    (**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     *)
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,

    (**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     *)
    CUDA_ERROR_MPS_CONNECTION_FAILED          = 805,

    (**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     *)
    CUDA_ERROR_MPS_RPC_FAILURE                = 806,

    (**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     *)
    CUDA_ERROR_MPS_SERVER_NOT_READY           = 807,

    (**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     *)
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED        = 808,

    (**
     * This error indicates the the hardware resources required to support device connections have been exhausted.
     *)
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED    = 809,

    (**
     * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
     *)
    CUDA_ERROR_MPS_CLIENT_TERMINATED          = 810,

    (**
     * This error indicates that the module is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     *)
    CUDA_ERROR_CDP_NOT_SUPPORTED              = 811,

    (**
     * This error indicates that a module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
     *)
    CUDA_ERROR_CDP_VERSION_MISMATCH           = 812,

    (**
     * This error indicates that the operation is not permitted when
     * the stream is capturing.
     *)
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,

    (**
     * This error indicates that the current capture sequence on the stream
     * has been invalidated due to a previous error.
     *)
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED     = 901,

    (**
     * This error indicates that the operation would have resulted in a merge
     * of two independent capture sequences.
     *)
    CUDA_ERROR_STREAM_CAPTURE_MERGE           = 902,

    (**
     * This error indicates that the capture was not initiated in this stream.
     *)
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED       = 903,

    (**
     * This error indicates that the capture sequence contains a fork that was
     * not joined to the primary stream.
     *)
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED        = 904,

    (**
     * This error indicates that a dependency would have been created which
     * crosses the capture sequence boundary. Only implicit in-stream ordering
     * dependencies are allowed to cross the boundary.
     *)
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION       = 905,

    (**
     * This error indicates a disallowed implicit dependency on a current capture
     * sequence from cudaStreamLegacy.
     *)
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT        = 906,

    (**
     * This error indicates that the operation is not permitted on an event which
     * was last recorded in a capturing stream.
     *)
    CUDA_ERROR_CAPTURED_EVENT                 = 907,

    (**
     * A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED
     * argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
     * different thread.
     *)
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD    = 908,

    (**
     * This error indicates that the timeout specified for the wait operation has lapsed.
     *)
    CUDA_ERROR_TIMEOUT                        = 909,

    (**
     * This error indicates that the graph update was not performed because it included
     * changes which violated constraints specific to instantiated graph update.
     *)
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE      = 910,

    (**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     *)
    CUDA_ERROR_EXTERNAL_DEVICE               = 911,

    (**
     * Indicates a kernel launch error due to cluster misconfiguration.
     *)
    CUDA_ERROR_INVALID_CLUSTER_SIZE           = 912,

    (**
     * This indicates that an unknown internal error has occurred.
     *)
    CUDA_ERROR_UNKNOWN                        = 999
  );
  CUresult = cudaError_enum;

(**
 * P2P Attributes
 *)
  CUdevice_P2PAttribute_enum = (
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK                     = $01,  (**< A relative value indicating the performance of the link between two devices *)
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED                     = $02,  (**< P2P Access is enable *)
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED              = $03,  (**< Atomic operation over the link supported *)
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED              = $04,  (**< \deprecated use CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED instead *)
    CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED          = $04   (**< Accessing CUDA arrays over the link supported *)
  );
  CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum;

(**
 * CUDA stream callback
 * \param hStream The stream the callback was added to, as passed to ::cuStreamAddCallback.  May be NULL.
 * \param status ::CUDA_SUCCESS or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 *)
  CUstreamCallback = procedure (hStream : CUstream; status: CUresult; userData : pointer); WINAPI;

(**
 * Block size to per-block dynamic shared memory mapping for a certain
 * kernel \param blockSize Block size of the kernel.
 *
 * \return The dynamic shared memory needed by a block.
 *)
  CUoccupancyB2DSize = function (blockSize:longint):IntPtr;WINAPI;
(**
 * If set, host memory is portable between CUDA contexts.
 * Flag for ::cuMemHostAlloc()
 *)
const CU_MEMHOSTALLOC_PORTABLE =         $01;

(**
 * If set, host memory is mapped into CUDA address space and
 * ::cuMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::cuMemHostAlloc()
 *)
  CU_MEMHOSTALLOC_DEVICEMAP =        $02;

(**
 * If set, host memory is allocated as write-combined - fast to write,
 * faster to DMA, slow to read except via SSE4 streaming load instruction
 * (MOVNTDQA).
 * Flag for ::cuMemHostAlloc()
 *)
  CU_MEMHOSTALLOC_WRITECOMBINED =    $04;

(**
 * If set, host memory is portable between CUDA contexts.
 * Flag for ::cuMemHostRegister()
 *)
  CU_MEMHOSTREGISTER_PORTABLE =      $01;

(**
 * If set, host memory is mapped into CUDA address space and
 * ::cuMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::cuMemHostRegister()
 *)
  CU_MEMHOSTREGISTER_DEVICEMAP =     $02;

(**
 * If set, the passed memory pointer is treated as pointing to some
 * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
 * On Windows the flag is a no-op.
 * On Linux that memory is marked as non cache-coherent for the GPU and
 * is expected to be physically contiguous. It may return
 * ::CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
 * ::CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
 * On all other platforms, it is not supported and ::CUDA_ERROR_NOT_SUPPORTED
 * is returned.
 * Flag for ::cuMemHostRegister()
 *)
  CU_MEMHOSTREGISTER_IOMEMORY =      $04;

(**
* If set, the passed memory pointer is treated as pointing to memory that is
* considered read-only by the device.  On platforms without
* ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, this flag is
* required in order to register memory mapped to the CPU as read-only.  Support
* for the use of this flag can be queried from the device attribute
* ::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED.  Using this flag with
* a current context associated with a device that does not have this attribute
* set will cause ::cuMemHostRegister to error with ::CUDA_ERROR_NOT_SUPPORTED.
*)
  CU_MEMHOSTREGISTER_READ_ONLY =     $08;

type
(**
 * 2D memory copy parameters
 *)
  CUDA_MEMCPY2D_st=packed record
    srcXInBytes   : size_t           ;         (**< Source X in bytes *)
    srcY          : size_t           ;                (**< Source Y *)

    srcMemoryType : CUmemorytype     ; (**< Source memory type (host, device, array) *)
    srcHost       : pointer          ;        (**< Source host pointer *)
    srcDevice     : CUdeviceptr      ;      (**< Source device pointer *)
    srcArray      : CUarray          ;           (**< Source array reference *)
    srcPitch      : size_t           ;            (**< Source pitch (ignored when src is array) *)

    dstXInBytes   : size_t           ;         (**< Destination X in bytes *)
    dstY          : size_t           ;                (**< Destination Y *)

    dstMemoryType : CUmemorytype     ; (**< Destination memory type (host, device, array) *)
    dstHost       : pointer          ;              (**< Destination host pointer *)
    dstDevice     : CUdeviceptr      ;      (**< Destination device pointer *)
    dstArray      : CUarray          ;           (**< Destination array reference *)
    dstPitch      : size_t           ;            (**< Destination pitch (ignored when dst is array) *)

    WidthInBytes  : size_t           ;        (**< Width of 2D memory copy in bytes *)
    Height        : size_t           ;              (**< Height of 2D memory copy *)
  end;
  CUDA_MEMCPY2D_v2 = CUDA_MEMCPY2D_st;
  PCUDA_MEMCPY2D =^CUDA_MEMCPY2D ;
  CUDA_MEMCPY2D = CUDA_MEMCPY2D_v2;

(**
 * 3D memory copy parameters
 *)
  CUDA_MEMCPY3D_st=packed record
    srcXInBytes : size_t;         (**< Source X in bytes *)
    srcY : size_t;                (**< Source Y *)
    srcZ : size_t;                (**< Source Z *)
    srcLOD : size_t;              (**< Source LOD *)
    srcMemoryType : CUmemorytype; (**< Source memory type (host, device, array) *)
    srcHost : pointer;            (**< Source host pointer *)
    srcDevice : CUdeviceptr;      (**< Source device pointer *)
    srcArray : CUarray;           (**< Source array reference *)
    reserved0 : pointer;            (**< Must be NULL *)
    srcPitch : size_t;            (**< Source pitch (ignored when src is array) *)
    srcHeight : size_t;           (**< Source height (ignored when src is array; may be 0 if Depth==1) *)

    dstXInBytes : size_t;         (**< Destination X in bytes *)
    dstY : size_t;                (**< Destination Y *)
    dstZ : size_t;                (**< Destination Z *)
    dstLOD : size_t;              (**< Destination LOD *)
    dstMemoryType : CUmemorytype; (**< Destination memory type (host, device, array) *)
    dstHost : pointer;              (**< Destination host pointer *)
    dstDevice : CUdeviceptr;      (**< Destination device pointer *)
    dstArray : CUarray;           (**< Destination array reference *)
    reserved1 : pointer;            (**< Must be NULL *)
    dstPitch : size_t;            (**< Destination pitch (ignored when dst is array) *)
    dstHeight : size_t;           (**< Destination height (ignored when dst is array; may be 0 if Depth==1) *)

    WidthInBytes : size_t;        (**< Width of 3D memory copy in bytes *)
    Height : size_t;              (**< Height of 3D memory copy *)
    Depth : size_t;               (**< Depth of 3D memory copy *)
  end;
  CUDA_MEMCPY3D_v2 = CUDA_MEMCPY3D_st;
  PCUDA_MEMCPY3D = ^CUDA_MEMCPY3D ;
  CUDA_MEMCPY3D = CUDA_MEMCPY3D_v2;

(**
 * 3D memory cross-context copy parameters
 *)
  CUDA_MEMCPY3D_PEER_st=packed record
    srcXInBytes : size_t;         (**< Source X in bytes *)
    srcY : size_t;                (**< Source Y *)
    srcZ : size_t;                (**< Source Z *)
    srcLOD : size_t;              (**< Source LOD *)
    srcMemoryType : CUmemorytype; (**< Source memory type (host, device, array) *)
    srcHost : pointer;            (**< Source host pointer *)
    srcDevice : CUdeviceptr;      (**< Source device pointer *)
    srcArray : CUarray;           (**< Source array reference *)
    srcContext : CUcontext;       (**< Source context (ignored with srcMemoryType is ::CU_MEMORYTYPE_ARRAY) *)
    srcPitch : size_t;            (**< Source pitch (ignored when src is array) *)
    srcHeight : size_t;           (**< Source height (ignored when src is array; may be 0 if Depth==1) *)

    dstXInBytes : size_t;         (**< Destination X in bytes *)
    dstY : size_t;                (**< Destination Y *)
    dstZ : size_t;                (**< Destination Z *)
    dstLOD : size_t;              (**< Destination LOD *)
    dstMemoryType : CUmemorytype; (**< Destination memory type (host, device, array) *)
    dstHost : pointer;              (**< Destination host pointer *)
    dstDevice : CUdeviceptr;      (**< Destination device pointer *)
    dstArray : CUarray;           (**< Destination array reference *)
    dstContext : CUcontext;       (**< Destination context (ignored with dstMemoryType is ::CU_MEMORYTYPE_ARRAY) *)
    dstPitch : size_t;            (**< Destination pitch (ignored when dst is array) *)
    dstHeight : size_t;           (**< Destination height (ignored when dst is array; may be 0 if Depth==1) *)

    WidthInBytes : size_t;        (**< Width of 3D memory copy in bytes *)
    Height : size_t;              (**< Height of 3D memory copy *)
    Depth : size_t;               (**< Depth of 3D memory copy *)
  end;
  CUDA_MEMCPY3D_PEER_v1 = CUDA_MEMCPY3D_PEER_st;
  PCUDA_MEMCPY3D_PEER = ^CUDA_MEMCPY3D_PEER ;
  CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_v1;

(**
 * Array descriptor
 *)
  CUDA_ARRAY_DESCRIPTOR_st=packed record
    Width : size_t;             (**< Width of array *)
    Height : size_t;            (**< Height of array *)

    Format : CUarray_format;    (**< Array format *)
    NumChannels : longword; (**< Channels per array element *)
  end;
  CUDA_ARRAY_DESCRIPTOR_v2 = CUDA_ARRAY_DESCRIPTOR_st;
  PCUDA_ARRAY_DESCRIPTOR = ^CUDA_ARRAY_DESCRIPTOR ;
  CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_v2;

(**
 * 3D array descriptor
 *)
  CUDA_ARRAY3D_DESCRIPTOR_st=packed record
    Width : size_t;             (**< Width of 3D array *)
    Height : size_t;            (**< Height of 3D array *)
    Depth : size_t;             (**< Depth of 3D array *)

    Format : CUarray_format;    (**< Array format *)
    NumChannels : longword; (**< Channels per array element *)
    Flags : longword;       (**< Flags *)
  end;
  CUDA_ARRAY3D_DESCRIPTOR_v2 = CUDA_ARRAY3D_DESCRIPTOR_st;
  PCUDA_ARRAY3D_DESCRIPTOR = ^CUDA_ARRAY3D_DESCRIPTOR ;
  CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_v2;


{*< Width of sparse tile in elements  }
{*< Height of sparse tile in elements  }
{*< Depth of sparse tile in elements  }
{*
     * First mip level at which the mip tail begins.
      }
{*
     * Total size of the mip tail.
      }
{*
     * Flags will either be zero or ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
      }

  CUDA_ARRAY_SPARSE_PROPERTIES_st = record
      tileExtent : record
          width : longword;
          height : longword;
          depth : longword;
        end;
      miptailFirstLevel : longword;
      miptailSize : uint64;
      flags : longword;
      reserved : array[0..3] of longword;
  end;
  CUDA_ARRAY_SPARSE_PROPERTIES_v1 = CUDA_ARRAY_SPARSE_PROPERTIES_st;
  PCUDA_ARRAY_SPARSE_PROPERTIES = ^CUDA_ARRAY_SPARSE_PROPERTIES ;
  CUDA_ARRAY_SPARSE_PROPERTIES = CUDA_ARRAY_SPARSE_PROPERTIES_v1;


(**
 * CUDA array memory requirements
 *)
  CUDA_ARRAY_MEMORY_REQUIREMENTS_st=packed record
    size : size_t;                (**< Total required memory size *)
    alignment : size_t;           (**< alignment requirement *)
    reserved : array[0..3] of longword;
  end;
  CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = CUDA_ARRAY_MEMORY_REQUIREMENTS_st;
  PCUDA_ARRAY_MEMORY_REQUIREMENTS = ^CUDA_ARRAY_MEMORY_REQUIREMENTS ;
  CUDA_ARRAY_MEMORY_REQUIREMENTS = CUDA_ARRAY_MEMORY_REQUIREMENTS_v1;

(**
 * CUDA Resource descriptor
 *)
 {*< Resource type  }
 {*< CUDA array  }
 {*< CUDA mipmapped array  }
 {*< Device pointer  }
 {*< Array format  }
 {*< Channels per array element  }
 {*< Size in bytes  }
 {*< Device pointer  }
 {*< Array format  }
 {*< Channels per array element  }
 {*< Width of the array in elements  }
 {*< Height of the array in elements  }
 {*< Pitch between two rows in bytes  }
 {*< Flags (must be zero)  }

  CUDA_RESOURCE_DESC_st = record
    resType : CUresourcetype;
    res : record
       case longint of
         0 : ( &array : record
             hArray : CUarray;
           end );
         1 : ( mipmap : record
             hMipmappedArray : CUmipmappedArray;
           end );
         2 : ( linear : record
             devPtr : CUdeviceptr;
             format : CUarray_format;
             numChannels : longword;
             sizeInBytes : size_t;
           end );
         3 : ( pitch2D : record
             devPtr : CUdeviceptr;
             format : CUarray_format;
             numChannels : longword;
             width : size_t;
             height : size_t;
             pitchInBytes : size_t;
           end );
         4 : ( reserved : record
             reserved : array[0..31] of longint;
           end );
       end;
    flags : longint;
  end;
  CUDA_RESOURCE_DESC_v1 = CUDA_RESOURCE_DESC_st;
  PCUDA_RESOURCE_DESC =^CUDA_RESOURCE_DESC ;
  CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_v1;

(**
 * Texture descriptor
 *)
  CUDA_TEXTURE_DESC_st=packed record
    addressMode : array[0..2] of CUaddress_mode;  (**< Address modes *)
    filterMode : CUfilter_mode;       (**< Filter mode *)
    flags : longword;                 (**< Flags *)
    maxAnisotropy : longword;         (**< Maximum anisotropy ratio *)
    mipmapFilterMode : CUfilter_mode; (**< Mipmap filter mode *)
    mipmapLevelBias : single;          (**< Mipmap level bias *)
    minMipmapLevelClamp : single;      (**< Mipmap minimum level clamp *)
    maxMipmapLevelClamp : single;      (**< Mipmap maximum level clamp *)
    borderColor : array[0..3] of single;           (**< Border Color *)
    reserved : array[0..11] of longint;
  end;
  CUDA_TEXTURE_DESC_v1 = CUDA_TEXTURE_DESC_st;
  PCUDA_TEXTURE_DESC =^CUDA_TEXTURE_DESC ;
  CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_v1;

(**
 * Resource view format
 *)
  CUresourceViewFormat_enum = (
    CU_RES_VIEW_FORMAT_NONE          = $00, (**< No resource view format (use underlying resource format) *)
    CU_RES_VIEW_FORMAT_UINT_1X8      = $01, (**< 1 channel unsigned 8-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_2X8      = $02, (**< 2 channel unsigned 8-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_4X8      = $03, (**< 4 channel unsigned 8-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_1X8      = $04, (**< 1 channel signed 8-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_2X8      = $05, (**< 2 channel signed 8-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_4X8      = $06, (**< 4 channel signed 8-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_1X16     = $07, (**< 1 channel unsigned 16-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_2X16     = $08, (**< 2 channel unsigned 16-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_4X16     = $09, (**< 4 channel unsigned 16-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_1X16     = $0a, (**< 1 channel signed 16-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_2X16     = $0b, (**< 2 channel signed 16-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_4X16     = $0c, (**< 4 channel signed 16-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_1X32     = $0d, (**< 1 channel unsigned 32-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_2X32     = $0e, (**< 2 channel unsigned 32-bit integers *)
    CU_RES_VIEW_FORMAT_UINT_4X32     = $0f, (**< 4 channel unsigned 32-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_1X32     = $10, (**< 1 channel signed 32-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_2X32     = $11, (**< 2 channel signed 32-bit integers *)
    CU_RES_VIEW_FORMAT_SINT_4X32     = $12, (**< 4 channel signed 32-bit integers *)
    CU_RES_VIEW_FORMAT_FLOAT_1X16    = $13, (**< 1 channel 16-bit floating point *)
    CU_RES_VIEW_FORMAT_FLOAT_2X16    = $14, (**< 2 channel 16-bit floating point *)
    CU_RES_VIEW_FORMAT_FLOAT_4X16    = $15, (**< 4 channel 16-bit floating point *)
    CU_RES_VIEW_FORMAT_FLOAT_1X32    = $16, (**< 1 channel 32-bit floating point *)
    CU_RES_VIEW_FORMAT_FLOAT_2X32    = $17, (**< 2 channel 32-bit floating point *)
    CU_RES_VIEW_FORMAT_FLOAT_4X32    = $18, (**< 4 channel 32-bit floating point *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1  = $19, (**< Block compressed 1 *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2  = $1a, (**< Block compressed 2 *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3  = $1b, (**< Block compressed 3 *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4  = $1c, (**< Block compressed 4 unsigned *)
    CU_RES_VIEW_FORMAT_SIGNED_BC4    = $1d, (**< Block compressed 4 signed *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5  = $1e, (**< Block compressed 5 unsigned *)
    CU_RES_VIEW_FORMAT_SIGNED_BC5    = $1f, (**< Block compressed 5 signed *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = $20, (**< Block compressed 6 unsigned half-float *)
    CU_RES_VIEW_FORMAT_SIGNED_BC6H   = $21, (**< Block compressed 6 signed half-float *)
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7  = $22  (**< Block compressed 7 *)
  );
  CUresourceViewFormat = CUresourceViewFormat_enum;

(**
 * Resource view descriptor
 *)
  CUDA_RESOURCE_VIEW_DESC_st=packed record
    format : CUresourceViewFormat;   (**< Resource view format *)
    width : size_t;                  (**< Width of the resource view *)
    height : size_t;                 (**< Height of the resource view *)
    depth : size_t;                  (**< Depth of the resource view *)
    firstMipmapLevel : longword; (**< First defined mipmap level *)
    lastMipmapLevel : longword;  (**< Last defined mipmap level *)
    firstLayer : longword;       (**< First layer index *)
    lastLayer : longword;        (**< Last layer index *)
    reserved : array[0..15] of longword;
  end;
  CUDA_RESOURCE_VIEW_DESC_v1 = CUDA_RESOURCE_VIEW_DESC_st;
  PCUDA_RESOURCE_VIEW_DESC =^CUDA_RESOURCE_VIEW_DESC ;
  CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_v1;

(**
 * Size of tensor map descriptor
 *)
const CU_TENSOR_MAP_NUM_QWORDS =  16;
type
(**
 * Tensor map descriptor. Requires compiler support for aligning to 64 bytes.
 *)

  CUtensorMap_st=record
    opaque : array[0..CU_TENSOR_MAP_NUM_QWORDS-1] of cuuint64_t;
  end;
  PCUtensorMap =^CUtensorMap ;
  CUtensorMap = CUtensorMap_st;

(**
 * Tensor map data type
 *)
  CUtensorMapDataType_enum = (
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16,
    CU_TENSOR_MAP_DATA_TYPE_UINT32,
    CU_TENSOR_MAP_DATA_TYPE_INT32,
    CU_TENSOR_MAP_DATA_TYPE_UINT64,
    CU_TENSOR_MAP_DATA_TYPE_INT64,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ
  );
  CUtensorMapDataType = CUtensorMapDataType_enum;

(**
 * Tensor map interleave layout type
 *)
  CUtensorMapInterleave_enum = (
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    CU_TENSOR_MAP_INTERLEAVE_16B,
    CU_TENSOR_MAP_INTERLEAVE_32B
  );
  CUtensorMapInterleave = CUtensorMapInterleave_enum;

(**
 * Tensor map swizzling mode of shared memory banks
 *)
  CUtensorMapSwizzle_enum = (
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B,
    CU_TENSOR_MAP_SWIZZLE_64B,
    CU_TENSOR_MAP_SWIZZLE_128B
  );
  CUtensorMapSwizzle = CUtensorMapSwizzle_enum;

(**
 * Tensor map L2 promotion type
 *)
  CUtensorMapL2promotion_enum = (
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B
  );
  CUtensorMapL2promotion = CUtensorMapL2promotion_enum;

(**
 * Tensor map out-of-bounds fill type
 *)
  CUtensorMapFloatOOBfill_enum = (
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
  );
  CUtensorMapFloatOOBfill = CUtensorMapFloatOOBfill_enum;

(**
 * GPU Direct v3 tokens
 *)
  CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st=packed record
     p2pToken     : uint64 ;
     vaSpaceToken : longword;
  end;
  CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
  CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1;

(**
* Access flags that specify the level of access the current context's device has
* on the memory referenced.
*)
  CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = (
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE      = $0,   (**< No access, meaning the device cannot access this memory at all, thus must be staged through accessible memory in order to complete certain operations *)
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ      = $1,   (**< Read-only access, meaning writes to this memory are considered invalid accesses and thus return error in that case. *)
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = $3    (**< Read-write access, the device has full read-write access to the memory *)
  );
  CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum;

(**
 * Kernel launch parameters
 *)
  CUDA_LAUNCH_PARAMS_st=packed record
    &function      : CUfunction  ;         (**< Kernel to launch *)
    gridDimX       : longword    ;       (**< Width of grid in blocks *)
    gridDimY       : longword    ;       (**< Height of grid in blocks *)
    gridDimZ       : longword    ;       (**< Depth of grid in blocks *)
    blockDimX      : longword    ;      (**< X dimension of each thread block *)
    blockDimY      : longword    ;      (**< Y dimension of each thread block *)
    blockDimZ      : longword    ;      (**< Z dimension of each thread block *)
    sharedMemBytes : longword    ; (**< Dynamic shared-memory size per thread block in bytes *)
    hStream        : CUstream   ;            (**< Stream identifier *)
    kernelParams   : PPointer          (**< Array of pointers to kernel parameters *)
  end;
  CUDA_LAUNCH_PARAMS_v1 = CUDA_LAUNCH_PARAMS_st;
  PCUDA_LAUNCH_PARAMS =^CUDA_LAUNCH_PARAMS ;
  CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_v1;

(**
 * External memory handle types
 *)
  CUexternalMemoryHandleType_enum = (
    (**
     * Handle is an opaque file descriptor
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD          = 1,
    (**
     * Handle is an opaque shared NT handle
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32       = 2,
    (**
     * Handle is an opaque, globally shared handle
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT   = 3,
    (**
     * Handle is a D3D12 heap object
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP         = 4,
    (**
     * Handle is a D3D12 committed resource
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE     = 5,
    (**
     * Handle is a shared NT handle to a D3D11 resource
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE     = 6,
    (**
     * Handle is a globally shared handle to a D3D11 resource
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
    (**
     * Handle is an NvSciBuf object
     *)
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
  );
  CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum;

(**
 * Indicates that the external memory object is a dedicated resource
 *)
const CUDA_EXTERNAL_MEMORY_DEDICATED =    $1;

(** When the \p flags parameter of ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
 * contains this flag, it indicates that signaling an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 *)
const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC =  $01;

(** When the \p flags parameter of ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
 * contains this flag, it indicates that waiting on an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 *)
const CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC =  $02;

(**
 * When \p flags of ::cuDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application needs signaler specific NvSciSyncAttr
 * to be filled by ::cuDeviceGetNvSciSyncAttributes.
 *)
const CUDA_NVSCISYNC_ATTR_SIGNAL =  $1;

(**
 * When \p flags of ::cuDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application needs waiter specific NvSciSyncAttr
 * to be filled by ::cuDeviceGetNvSciSyncAttributes.
 *)
const CUDA_NVSCISYNC_ATTR_WAIT =  $2;
type
{*
     * Type of the handle
      }
{*
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
          }
{*
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
         * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
         * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
         * then 'name' must be NULL.
          }
{*
             * Valid NT handle. Must be NULL if 'name' is non-NULL
              }
{*
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
              }
{*
         * A handle representing an NvSciBuf Object. Valid when type
         * is ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
          }
{*
     * Size of the memory allocation
      }
{*
     * Flags must either be zero or ::CUDA_EXTERNAL_MEMORY_DEDICATED
      }

  CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st = record
      _type : CUexternalMemoryHandleType;
      handle : record
          case longint of
            0 : ( fd : longint );
            1 : ( win32 : record
                handle : pointer;
                name : pointer;
              end );
            2 : ( nvSciBufObject : pointer );
          end;
      size : uint64;
      flags : longword;
      reserved : array[0..15] of longword;
    end;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st;
  PCUDA_EXTERNAL_MEMORY_HANDLE_DESC =^CUDA_EXTERNAL_MEMORY_HANDLE_DESC ;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1;

  {*
       * Offset into the memory object where the buffer's base is
        }
  {*
       * Size of the buffer
        }
  {*
       * Flags reserved for future use. Must be zero.
        }

  CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st = record
      offset : uint64;
      size : uint64;
      flags : longword;
      reserved : array[0..15] of longword;
    end;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
  PCUDA_EXTERNAL_MEMORY_BUFFER_DESC =^CUDA_EXTERNAL_MEMORY_BUFFER_DESC ;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1;

(**
 * External memory mipmap descriptor
 *)
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st=packed record
    (**
     * Offset into the memory object where the base level of the
     * mipmap chain is.
     *)
    offset : uint64;
    (**
     * Format, dimension and type of base level of the mipmap chain
     *)
    arrayDesc : CUDA_ARRAY3D_DESCRIPTOR;
    (**
     * Total number of levels in the mipmap chain
     *)
    numLevels : longword;
    reserved : array[0..15] of longword;
  end;
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
  PCUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = ^CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC ;
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1;

(**
 * External semaphore handle types
 *)
  CUexternalSemaphoreHandleType_enum = (
    (**
     * Handle is an opaque file descriptor
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD             = 1,
    (**
     * Handle is an opaque shared NT handle
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32          = 2,
    (**
     * Handle is an opaque, globally shared handle
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT      = 3,
    (**
     * Handle is a shared NT handle referencing a D3D12 fence object
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE           = 4,
    (**
     * Handle is a shared NT handle referencing a D3D11 fence object
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE           = 5,
    (**
     * Opaque handle to NvSciSync Object
	 *)
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC             = 6,
    (**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX     = 7,
    (**
     * Handle is a globally shared handle referencing a D3D11 keyed mutex object
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
    (**
     * Handle is an opaque file descriptor referencing a timeline semaphore
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
    (**
     * Handle is an opaque shared NT handle referencing a timeline semaphore
     *)
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10
  );
  CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum;

  {*
       * Type of the handle
        }
  {*
           * File descriptor referencing the semaphore object. Valid
           * when type is one of the following:
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
            }
  {*
           * Win32 handle referencing the semaphore object. Valid when
           * type is one of the following:
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32
           * Exactly one of 'handle' and 'name' must be non-NULL. If
           * type is one of the following:
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
           * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
           * then 'name' must be NULL.
            }
  {*
               * Valid NT handle. Must be NULL if 'name' is non-NULL
                }
  {*
               * Name of a valid synchronization primitive.
               * Must be NULL if 'handle' is non-NULL.
                }
  {*
           * Valid NvSciSyncObj. Must be non NULL
            }
  {*
       * Flags reserved for the future. Must be zero.
        }

  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st = record
      _type : CUexternalSemaphoreHandleType;
      handle : record
          case longint of
            0 : ( fd : longint );
            1 : ( win32 : record
                handle : pointer;
                name : pointer;
              end );
            2 : ( nvSciSyncObj : pointer );
          end;
      flags : longword;
      reserved : array[0..15] of longword;
    end;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st;
  PCUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC =^CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC ;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1;

(**
 * External semaphore signal parameters
 *)
 {*
          * Parameters for fence objects
           }
 {*
              * Value of fence to be signaled
               }
 {*
              * Pointer to NvSciSyncFence. Valid if ::CUexternalSemaphoreHandleType
              * is of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
               }
 {*
          * Parameters for keyed mutex objects
           }
 {*
              * Value of key to release the mutex with
               }
 {*
      * Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to
      * signal a ::CUexternalSemaphore of type
      * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
      * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which indicates
      * that while signaling the ::CUexternalSemaphore, no memory synchronization
      * operations should be performed for any external memory object imported
      * as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
      * For all other types of ::CUexternalSemaphore, flags must be zero.
       }

  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st = record
    params : record
      fence : record
         value : uint64;
       end;
      nvSciSync : record
         case longint of
           0 : ( fence : pointer );
           1 : ( reserved : uint64 );
         end;
      keyedMutex : record
         key : uint64;
      end;
      reserved : array[0..11] of longword;
    end;
    flags : longword;
    reserved : array[0..15] of longword;
  end;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st;
  PCUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS =^CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1;

(**
 * External semaphore wait parameters
 *)

 {*
          * Parameters for fence objects
           }
 {*
              * Value of fence to be waited on
               }
 {*
          * Pointer to NvSciSyncFence. Valid if CUexternalSemaphoreHandleType
          * is of type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
           }
 {*
          * Parameters for keyed mutex objects
           }
 {*
              * Value of key to acquire the mutex with
               }
 {*
              * Timeout in milliseconds to wait to acquire the mutex
               }
 {*
      * Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on
      * a ::CUexternalSemaphore of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC,
      * the valid flag is ::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC
      * which indicates that while waiting for the ::CUexternalSemaphore, no memory
      * synchronization operations should be performed for any external memory
      * object imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
      * For all other types of ::CUexternalSemaphore, flags must be zero.
       }

   CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st = record
   params : record
     fence : record
         value : uint64;
     end;
     nvSciSync : record
         case longint of
           0 : ( fence : pointer );
           1 : ( reserved : uint64 );
         end;
     keyedMutex : record
         key : uint64;
         timeoutMs : longword;
     end;
     reserved : array[0..9] of longword;
end;
    flags : longword;
    reserved : array[0..15] of longword;
  end;
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st;
  PCUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = ^CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS ;
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1;

(**
 * Semaphore signal node parameters
 *)
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st=packed record
    extSemArray : ^CUexternalSemaphore;                         (**< Array of external semaphore handles. *)
    paramsArray : ^CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS; (**< Array of external semaphore signal parameters. *)
    numExtSems : longword ;                                  (**< Number of handles and parameters supplied in extSemArray and paramsArray. *)
  end;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st;
  PCUDA_EXT_SEM_SIGNAL_NODE_PARAMS =^CUDA_EXT_SEM_SIGNAL_NODE_PARAMS ;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1;

(**
 * Semaphore wait node parameters
 *)
  CUDA_EXT_SEM_WAIT_NODE_PARAMS_st=packed record
    extSemArray : ^CUexternalSemaphore;                       (**< Array of external semaphore handles. *)
    paramsArray : ^CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS; (**< Array of external semaphore wait parameters. *)
    numExtSems : longword;                                (**< Number of handles and parameters supplied in extSemArray and paramsArray. *)
  end;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = CUDA_EXT_SEM_WAIT_NODE_PARAMS_st;
  PCUDA_EXT_SEM_WAIT_NODE_PARAMS =^CUDA_EXT_SEM_WAIT_NODE_PARAMS ;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1;

  CUmemGenericAllocationHandle_v1 = uint64;
  PCUmemGenericAllocationHandle =^CUmemGenericAllocationHandle;
  CUmemGenericAllocationHandle = CUmemGenericAllocationHandle_v1;

(**
 * Flags for specifying particular handle types
 *)
  CUmemAllocationHandleType_enum = (
    CU_MEM_HANDLE_TYPE_NONE                  = $0,  (**< Does not allow any export mechanism. > *)
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = $1,  (**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (longint) *)
    CU_MEM_HANDLE_TYPE_WIN32                 = $2,  (**< Allows a Win32 NT handle to be used for exporting. (HANDLE) *)
    CU_MEM_HANDLE_TYPE_WIN32_KMT             = $4,  (**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) *)
    CU_MEM_HANDLE_TYPE_MAX                   = $7FFFFFFF
  );
  CUmemAllocationHandleType = CUmemAllocationHandleType_enum;

(**
 * Specifies the memory protection flags for mapping.
 *)
  CUmemAccess_flags_enum = (
    CU_MEM_ACCESS_FLAGS_PROT_NONE        = $0,  (**< Default, make the address range not accessible *)
    CU_MEM_ACCESS_FLAGS_PROT_READ        = $1,  (**< Make the address range read accessible *)
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE   = $3,  (**< Make the address range read-write accessible *)
    CU_MEM_ACCESS_FLAGS_PROT_MAX         = $7FFFFFFF
  );
  PCUmemAccess_flags = ^CUmemAccess_flags ;
  CUmemAccess_flags = CUmemAccess_flags_enum;

(**
 * Specifies the type of location
 *)
  CUmemLocationType_enum = (
    CU_MEM_LOCATION_TYPE_INVALID = $0,
    CU_MEM_LOCATION_TYPE_DEVICE  = $1,  (**< Location is a device location, thus id is a device ordinal *)
    CU_MEM_LOCATION_TYPE_MAX     = $7FFFFFFF
  );
  CUmemLocationType = CUmemLocationType_enum;

(**
* Defines the allocation types available
*)
  CUmemAllocationType_enum = (
    CU_MEM_ALLOCATION_TYPE_INVALID = $0,

    (** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      *)
    CU_MEM_ALLOCATION_TYPE_PINNED  = $1,
    CU_MEM_ALLOCATION_TYPE_MAX     = $7FFFFFFF
  );
  CUmemAllocationType = CUmemAllocationType_enum;

(**
* Flag for requesting different optimal and required granularities for an allocation.
*)
  CUmemAllocationGranularity_flags_enum = (
    CU_MEM_ALLOC_GRANULARITY_MINIMUM     = $0,     (**< Minimum required granularity for allocation *)
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = $1      (**< Recommended granularity for allocation for best performance *)
  );
  CUmemAllocationGranularity_flags = CUmemAllocationGranularity_flags_enum;

(**
* Specifies the handle type for address range
*)
  CUmemRangeHandleType_enum = (
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = $1,
    CU_MEM_RANGE_HANDLE_TYPE_MAX        = $7FFFFFFF
  );
  CUmemRangeHandleType = CUmemRangeHandleType_enum;

(**
 * Sparse subresource types
 *)
  CUarraySparseSubresourceType_enum = (
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
  );
  CUarraySparseSubresourceType = CUarraySparseSubresourceType_enum;

(**
 * Memory operation types
 *)
  CUmemOperationType_enum = (
    CU_MEM_OPERATION_TYPE_MAP = 1,
    CU_MEM_OPERATION_TYPE_UNMAP = 2
  );
  CUmemOperationType = CUmemOperationType_enum;

(**
 * Memory handle types
 *)
  CUmemHandleType_enum = (
    CU_MEM_HANDLE_TYPE_GENERIC = 0
  );
  CUmemHandleType = CUmemHandleType_enum;

(**
 * Specifies the CUDA array or CUDA mipmapped array memory mapping information
 *)
 {*< Resource type  }
 {*< Sparse subresource type  }
 {*< For CUDA mipmapped arrays must a valid mipmap level. For CUDA arrays must be zero  }  {*< For CUDA layered arrays must be a valid layer index. Otherwise, must be zero  }
 {*< Starting X offset in elements  }
 {*< Starting Y offset in elements  }
 {*< Starting Z offset in elements  }  {*< Width in elements  }
 {*< Height in elements  }
 {*< Depth in elements  }
 {*< For CUDA layered arrays must be a valid layer index. Otherwise, must be zero  }
 {*< Offset within mip tail  }
 {*< Extent in bytes  }
 {*< Memory operation type  }
 {*< Memory handle type  }
 {*< Offset within the memory  }
 {*< Device ordinal bit mask  }
 {*< flags for future use, must be zero now.  }
 {*< Reserved for future use, must be zero now.  }

  CUarrayMapInfo_st = record
     resourceType : CUresourcetype;
     resource : record
         case longint of
           0 : ( mipmap : CUmipmappedArray );
           1 : ( &array : CUarray );
         end;
     subresourceType : CUarraySparseSubresourceType;
     subresource : record
         case longint of
           0 : ( sparseLevel : record
               level : longword;
               layer : longword;
               offsetX : longword;
               offsetY : longword;
               offsetZ : longword;
               extentWidth : longword;
               extentHeight : longword;
               extentDepth : longword;
             end );
           1 : ( miptail : record
               layer : longword;
               offset : uint64;
               size : uint64;
             end );
         end;
     memOperationType : CUmemOperationType;
     memHandleType : CUmemHandleType;
     memHandle : record
         case longint of
           0 : ( memHandle : CUmemGenericAllocationHandle );
         end;
     offset : uint64;
     deviceBitMask : longword;
     flags : longword;
     reserved : array[0..1] of longword;
  end;
  CUarrayMapInfo_v1 = CUarrayMapInfo_st;
  PCUarrayMapInfo = ^CUarrayMapInfo ;
  CUarrayMapInfo = CUarrayMapInfo_v1;

(**
 * Specifies a memory location.
 *)
  CUmemLocation_st = packed record
    &type : CUmemLocationType; (**< Specifies the location type, which modifies the meaning of id. *)
    id : longint ;                 (**< identifier for a given this location's ::CUmemLocationType. *)
  end;
  CUmemLocation_v1 = CUmemLocation_st;
  PCUmemLocation = ^CUmemLocation ;
  CUmemLocation = CUmemLocation_v1;

(**
 * Specifies compression attribute for an allocation.
 *)
  CUmemAllocationCompType_enum = (
    CU_MEM_ALLOCATION_COMP_NONE = $0, (**< Allocating non-compressible memory *)
    CU_MEM_ALLOCATION_COMP_GENERIC = $1 (**< Allocating  compressible memory *)
  );
  CUmemAllocationCompType = CUmemAllocationCompType_enum;

(**
 * This flag if set indicates that the memory will be used as a tile pool.
 *)
const CU_MEM_CREATE_USAGE_TILE_POOL =     $1;

type
(**
* Specifies the allocation properties for a allocation.
*)
{* Allocation type  }
{* requested ::CUmemAllocationHandleType  }
{* Location of allocation  }
{*
     * Windows-specific POBJECT_ATTRIBUTES required when
     * ::CU_MEM_HANDLE_TYPE_WIN32 is specified.  This object attributes structure
     * includes security attributes that define
     * the scope of which exported allocations may be transferred to other
     * processes.  In all other cases, this field is required to be zero.
      }
{*
         * Allocation hint for requesting compressible memory.
         * On devices that support Compute Data Compression, compressible
         * memory can be used to accelerate accesses to data with unstructured
         * sparsity and other compressible data patterns. Applications are
         * expected to query allocation property of the handle obtained with
         * ::cuMemCreate using ::cuMemGetAllocationPropertiesFromHandle to
         * validate if the obtained allocation is compressible or not. Note that
         * compressed memory may not be mappable on all devices.
          }
{* Bitmask indicating intended usage for this allocation  }

  CUmemAllocationProp_st = record
      _type : CUmemAllocationType;
      requestedHandleTypes : CUmemAllocationHandleType;
      location : CUmemLocation;
      win32HandleMetaData : pointer;
      allocFlags : record
          compressionType : byte;
          gpuDirectRDMACapable : byte;
          usage : word;
          reserved : array[0..3] of byte;
        end;
    end;
  CUmemAllocationProp_v1 = CUmemAllocationProp_st;
  PCUmemAllocationProp = ^CUmemAllocationProp ;
  CUmemAllocationProp = CUmemAllocationProp_v1;

(**
* Flags for querying different granularities for a multicast object
*)
  CUmulticastGranularity_flags_enum = (
    CU_MULTICAST_GRANULARITY_MINIMUM     = $0,     (**< Minimum required granularity *)
    CU_MULTICAST_GRANULARITY_RECOMMENDED = $1      (**< Recommended granularity for best performance *)
  );
  CUmulticastGranularity_flags = CUmulticastGranularity_flags_enum;

(**
* Specifies the properties for a multicast object.
*)
  CUmulticastObjectProp_st=packed record
    (**
     * The number of devices in the multicast team that will bind memory to this
     * object
     *)
    numDevices : longword;
    (**
     * The maximum amount of memory that can be bound to this multicast object
     * per device
     *)
    size : size_t;
    (**
     * Bitmask of exportable handle types (see ::CUmemAllocationHandleType) for
     * this object
     *)
    handleTypes : uint64;
    (**
     * Flags for future use, must be zero now
     *)
    flags : uint64;
  end;
  CUmulticastObjectProp_v1 = CUmulticastObjectProp_st;
  PCUmulticastObjectProp =^CUmulticastObjectProp ;
  CUmulticastObjectProp = CUmulticastObjectProp_v1;

(**
 * Memory access descriptor
 *)
  CUmemAccessDesc_st = packed record
    location : CUmemLocation;        (**< Location on which the request is to change it's accessibility *)
    flags : CUmemAccess_flags;       (**< ::CUmemProt accessibility flags to set on the request *)
  end;
  CUmemAccessDesc_v1 = CUmemAccessDesc_st;
  PCUmemAccessDesc = ^CUmemAccessDesc ;
  CUmemAccessDesc = CUmemAccessDesc_v1;

(**
 * CUDA Graph Update error types
 *)
  CUgraphExecUpdateResult_enum = (
    CU_GRAPH_EXEC_UPDATE_SUCCESS                     = $0, (**< The update succeeded *)
    CU_GRAPH_EXEC_UPDATE_ERROR                       = $1, (**< The update failed for an unexpected reason which is described in the return value of the function *)
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED      = $2, (**< The update failed because the topology changed *)
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED     = $3, (**< The update failed because a node type changed *)
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED      = $4, (**< The update failed because the function of a kernel node changed (CUDA driver < 11.2) *)
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED    = $5, (**< The update failed because the parameters changed in a way that is not supported *)
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED         = $6, (**< The update failed because something about the node is not supported *)
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = $7, (**< The update failed because the function of a kernel node changed in an unsupported way *)
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED    = $8  (**< The update failed because the node attributes changed in a way that is not supported *)
  );
  CUgraphExecUpdateResult = CUgraphExecUpdateResult_enum;

(**
 * Result information returned by cuGraphExecUpdate
 *)
  CUgraphExecUpdateResultInfo_st = packed record
    (**
     * Gives more specific detail when a cuda graph update fails.
     *)
    result : CUgraphExecUpdateResult;

    (**
     * The "to node" of the error edge when the topologies do not match.
     * The error node when the error is associated with a specific node.
     * NULL when the error is generic.
     *)
    errorNode : CUgraphNode;

    (**
     * The from node of error edge when the topologies do not match. Otherwise NULL.
     *)
    errorFromNode : CUgraphNode;
  end;
  CUgraphExecUpdateResultInfo_v1 = CUgraphExecUpdateResultInfo_st;
  PCUgraphExecUpdateResultInfo =^CUgraphExecUpdateResultInfo ;
  CUgraphExecUpdateResultInfo = CUgraphExecUpdateResultInfo_v1;

(**
 * CUDA memory pool attributes
 *)
  CUmemPool_attribute_enum = (
    (**
     * (value type = longint)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     *)
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,

    (**
     * (value type = longint)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     *)
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,

    (**
     * (value type = longint)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     *)
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,

    (**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     *)
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,

    (**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     *)
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,

    (**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     *)
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,

    (**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     *)
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT,

    (**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     *)
    CU_MEMPOOL_ATTR_USED_MEM_HIGH
  );
  CUmemPool_attribute = CUmemPool_attribute_enum;

(**
 * Specifies the properties of allocations made from the pool.
 *)
  CUmemPoolProps_st=packed record
    allocType : CUmemAllocationType;         (**< Allocation type. Currently must be specified as CU_MEM_ALLOCATION_TYPE_PINNED *)
    handleTypes : CUmemAllocationHandleType; (**< Handle types that will be supported by allocations from the pool. *)
    location : CUmemLocation;                (**< Location where allocations should reside. *)
    (**
     * Windows-specific LPSECURITYATTRIBUTES required when
     * ::CU_MEM_HANDLE_TYPE_WIN32 is specified.  This security attribute defines
     * the scope of which exported allocations may be transferred to other
     * processes.  In all other cases, this field is required to be zero.
     *)
    win32SecurityAttributes : pointer;
    reserved : array[0..63] of byte; (**< reserved for future use, must be 0 *)
  end;
  CUmemPoolProps_v1 = CUmemPoolProps_st;
  PCUmemPoolProps = ^CUmemPoolProps ;
  CUmemPoolProps = CUmemPoolProps_v1;

(**
 * Opaque data for exporting a pool allocation
 *)
  CUmemPoolPtrExportData_st=packed record
    reserved : array[0..63] of byte;
  end;
  CUmemPoolPtrExportData_v1 = CUmemPoolPtrExportData_st;
  PCUmemPoolPtrExportData = ^CUmemPoolPtrExportData ;
  CUmemPoolPtrExportData = CUmemPoolPtrExportData_v1;

(**
 * Memory allocation node parameters
 *)
  CUDA_MEM_ALLOC_NODE_PARAMS_st=packed record
    (**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::CU_MEM_HANDLE_TYPE_NONE. IPC is not supported.
    *)
    poolProps : CUmemPoolProps ;
    accessDescs : ^CUmemAccessDesc;  (**< in: array of memory access descriptors. Used to describe peer GPU access *)
    accessDescCount : size_t;      (**< in: number of memory access descriptors.  Must not exceed the number of GPUs. *)
    bytesize : size_t;             (**< in: size in bytes of the requested allocation *)
    dptr : CUdeviceptr;            (**< out: address of the allocation returned by CUDA *)
  end;
  PCUDA_MEM_ALLOC_NODE_PARAMS =^CUDA_MEM_ALLOC_NODE_PARAMS ;
  CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_st;

  CUgraphMem_attribute_enum = (
    (**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently associated with graphs
     *)
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,

    (**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, associated with graphs since the
     * last time it was reset.  High watermark can only be reset to zero.
     *)
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,

    (**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     *)
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,

    (**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     *)
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH
  );
  CUgraphMem_attribute = CUgraphMem_attribute_enum;


(**
 * Bitmasks for ::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
 *)
  CUflushGPUDirectRDMAWritesOptions_enum = (
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST   = 1 shl 0, (**< ::cuFlushGPUDirectRDMAWrites() and its CUDA Runtime API counterpart are supported on the device. *)
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 1 shl 1  (**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. *)
  );
  CUflushGPUDirectRDMAWritesOptions = CUflushGPUDirectRDMAWritesOptions_enum;

(**
 * Platform native ordering for GPUDirect RDMA writes
 *)
  CUGPUDirectRDMAWritesOrdering_enum = (
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE        = 0,   (**< The device does not natively support ordering of remote writes. ::cuFlushGPUDirectRDMAWrites() can be leveraged if supported. *)
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER       = 100, (**< Natively, the device can consistently consume remote writes, although other CUDA devices may not. *)
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200  (**< Any CUDA device in the system can consistently consume remote writes to this device. *)
  );
  CUGPUDirectRDMAWritesOrdering = CUGPUDirectRDMAWritesOrdering_enum;

(**
 * The scopes for ::cuFlushGPUDirectRDMAWrites
 *)
  CUflushGPUDirectRDMAWritesScope_enum = (
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER       = 100, (**< Blocks until remote writes are visible to the CUDA device context owning the data. *)
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200  (**< Blocks until remote writes are visible to all CUDA device contexts. *)
  );
  CUflushGPUDirectRDMAWritesScope = CUflushGPUDirectRDMAWritesScope_enum;
 
(**
 * The targets for ::cuFlushGPUDirectRDMAWrites
 *)
  CUflushGPUDirectRDMAWritesTarget_enum = (
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0 (**< Sets the target for ::cuFlushGPUDirectRDMAWrites() to the currently active CUDA device context. *)
  );
  CUflushGPUDirectRDMAWritesTarget = CUflushGPUDirectRDMAWritesTarget_enum;

(**
 * The additional write options for ::cuGraphDebugDotPrint
 *)
  CUgraphDebugDot_flags_enum = (
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE                        = 1 shl 0,  (**< Output all debug data as if every debug flag is enabled *)
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES                  = 1 shl 1,  (**< Use CUDA Runtime structures for output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS             = 1 shl 2,  (**< Adds CUDA_KERNEL_NODE_PARAMS values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS             = 1 shl 3,  (**< Adds CUDA_MEMCPY3D values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS             = 1 shl 4,  (**< Adds CUDA_MEMSET_NODE_PARAMS values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS               = 1 shl 5,  (**< Adds CUDA_HOST_NODE_PARAMS values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS              = 1 shl 6,  (**< Adds CUevent handle from record and wait nodes to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS   = 1 shl 7,  (**< Adds CUDA_EXT_SEM_SIGNAL_NODE_PARAMS values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS     = 1 shl 8,  (**< Adds CUDA_EXT_SEM_WAIT_NODE_PARAMS values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES         = 1 shl 9,  (**< Adds CUkernelNodeAttrValue values to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES                        = 1 shl 10, (**< Adds node handles and every kernel function handle to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS          = 1 shl 11, (**< Adds memory alloc node parameters to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS           = 1 shl 12, (**< Adds memory free node parameters to output *)
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS       = 1 shl 13  (**< Adds batch mem op node parameters to output *)
    , CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO                = 1 shl 14  (**< Adds edge numbering information *)
  );
  CUgraphDebugDot_flags = CUgraphDebugDot_flags_enum;

(**
 * Flags for user objects for graphs
 *)
  CUuserObject_flags_enum = (
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1  (**< Indicates the destructor execution is not synchronized by any CUDA handle. *)
  );
  CUuserObject_flags = CUuserObject_flags_enum;

(**
 * Flags for retaining user object references for graphs
 *)
  CUuserObjectRetain_flags_enum = (
    CU_GRAPH_USER_OBJECT_MOVE = 1  (**< Transfer references from the caller rather than creating new references. *)
  );
  CUuserObjectRetain_flags = CUuserObjectRetain_flags_enum;

(**
 * Flags for instantiating a graph
 *)
  CUgraphInstantiate_flags_enum = (
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH  = 1 (**< Automatically free memory allocated in a graph before relaunching. *)
  , CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD               = 2 (**< Automatically upload the graph after instantiaton. *)
  , CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH        = 4 (**< Instantiate the graph to be launchable from the device. *)
  , CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY    = 8 (**< Run the graph using the per-node priority attributes rather than the
                                                              priority of the stream it is launched into. *)
  );
  CUgraphInstantiate_flags = CUgraphInstantiate_flags_enum;

(** @} *) (* END CUDA_TYPES *)


(**
 * \defgroup CUDA_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Gets the string description of an error code
 *
 * Sets \p *pStr to the address of a NULL-terminated string description
 * of the error code \p error.
 * If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
 * will be returned and \p *pStr will be set to the NULL address.
 *
 * \param error - Error code to convert to string
 * \param pStr - Address of the string pointer.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::CUresult,
 * ::cudaGetErrorString
 *)

function cuGetErrorString(error : CUresult; const pStr : PPAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets the string representation of an error code enum name
 *
 * Sets \p *pStr to the address of a NULL-terminated string representation
 * of the name of the enum error code \p error.
 * If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
 * will be returned and \p *pStr will be set to the NULL address.
 *
 * \param error - Error code to convert to string
 * \param pStr - Address of the string pointer.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::CUresult,
 * ::cudaGetErrorName
 *)
function cuGetErrorName(error : CUresult; const pStr : PPAnsiChar):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_ERROR *)

(**
 * \defgroup CUDA_INITIALIZE Initialization
 *
 * ___MANBRIEF___ initialization functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the initialization functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Initialize the CUDA driver API
 * Initializes the driver API and must be called before any other function from
 * the driver API in the current process. Currently, the \p Flags parameter must be 0. If ::cuInit()
 * has not been called, any function from the driver API will return
 * ::CUDA_ERROR_NOT_INITIALIZED.
 *
 * \param Flags - Initialization flag for CUDA.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
 * ::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
 * \notefnerr
 *)
function cuInit(Flags : longword):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_INITIALIZE *)

(**
 * \defgroup CUDA_VERSION Version Management
 *
 * ___MANBRIEF___ version management functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the version management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns the latest CUDA version supported by driver
 *
 * Returns in \p *driverVersion the version of CUDA supported by
 * the driver.  The version is returned as
 * (1000 &times; major + 10 &times; minor). For example, CUDA 9.2
 * would be represented by 9020.
 *
 * This function automatically returns ::CUDA_ERROR_INVALID_VALUE if
 * \p driverVersion is NULL.
 *
 * \param driverVersion - Returns the CUDA driver version
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cudaDriverGetVersion,
 * ::cudaRuntimeGetVersion
 *)
function cuDriverGetVersion(driverVersion : Plongint):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_VERSION *)

(**
 * \defgroup CUDA_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device handle given an ordinal in the range <b>[0,
 * ::cuDeviceGetCount()-1]</b>.
 *
 * \param device  - Returned device handle
 * \param ordinal - Device number to get handle for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGetLuid,
 * ::cuDeviceTotalMem,
 * ::cuDeviceGetExecAffinitySupport
 *)

function cuDeviceGet(device : PCUdevice; ordinal : longint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the number of compute-capable devices
 *
 * Returns in \p *count the number of devices with compute capability greater
 * than or equal to 2.0 that are available for execution. If there is no such
 * device, ::cuDeviceGetCount() returns 0.
 *
 * \param count - Returned number of compute-capable devices
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGetLuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem,
 * ::cuDeviceGetExecAffinitySupport,
 * ::cudaGetDeviceCount
 *)
function cuDeviceGetCount(count : Plongint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns an identifier string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p name. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param name - Returned identifier string for the device
 * \param len  - Maximum length of string to store in \p name
 * \param dev  - Device to get identifier string for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGetLuid,
 * ::cuDeviceGetCount,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem,
 * ::cuDeviceGetExecAffinitySupport,
 * ::cudaGetDeviceProperties
 *)
function cuDeviceGetName(name : PAnsiChar; len : longint; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Return an UUID for the device
 *
 * Note there is a later version of this API, ::cuDeviceGetUuid_v2. It will
 * supplant this version in 12.0, which is retained for minor version compatibility.
 *
 * Returns 16-octets identifying the device \p dev in the structure
 * pointed by the \p uuid.
 *
 * \param uuid - Returned UUID
 * \param dev  - Device to get identifier string for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetUuid_v2
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetLuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem,
 * ::cuDeviceGetExecAffinitySupport,
 * ::cudaGetDeviceProperties
 *)
function cuDeviceGetUuid(uuid : PCUuuid; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Return an UUID for the device (11.4+)
 *
 * Returns 16-octets identifying the device \p dev in the structure
 * pointed by the \p uuid. If the device is in MIG mode, returns its
 * MIG UUID which uniquely identifies the subscribed MIG compute instance.
 *
 * \param uuid - Returned UUID
 * \param dev  - Device to get identifier string for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetLuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem,
 * ::cudaGetDeviceProperties
 *)
function cuDeviceGetUuid_v2(uuid : PCUuuid; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Return an LUID and device node mask for the device
 *
 * Return identifying information (\p luid and \p deviceNodeMask) to allow
 * matching device with graphics APIs.
 *
 * \param luid - Returned LUID
 * \param deviceNodeMask - Returned device node mask
 * \param dev  - Device to get identifier string for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem,
 * ::cuDeviceGetExecAffinitySupport,
 * ::cudaGetDeviceProperties
 *)
function cuDeviceGetLuid(luid : PAnsiChar; deviceNodeMask : Plongword; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the total amount of memory on the device
 *
 * Returns in \p *bytes the total amount of memory available on the device
 * \p dev in bytes.
 *
 * \param bytes - Returned memory available on device in bytes
 * \param dev   - Device handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGet,
 * ::cuDeviceGetExecAffinitySupport,
 * ::cudaMemGetInfo
 *)
function cuDeviceTotalMem(bytes : Psize_t; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the maximum number of elements allocatable in a 1D linear texture for a given texture element size.
 *
 * Returns in \p maxWidthInElements the maximum number of texture elements allocatable in a 1D linear texture
 * for given \p format and \p numChannels.
 *
 * \param maxWidthInElements    - Returned maximum number of texture elements allocatable for given \p format and \p numChannels.
 * \param format                - Texture format.
 * \param numChannels           - Number of channels per texture element.
 * \param dev                   - Device handle.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGet,
 * ::cudaMemGetInfo,
 * ::cuDeviceTotalMem
 *)
function cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements : Psize_t; format : CUarray_format; numChannels : longword; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns information about the device
 *
 * Returns in \p *pi the integer value of the attribute \p attrib on device
 * \p dev. The supported attributes are:
 * - ::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Maximum number of threads per
 *   block;
 * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: Maximum x-dimension of a block
 * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: Maximum y-dimension of a block
 * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: Maximum z-dimension of a block
 * - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: Maximum x-dimension of a grid
 * - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: Maximum y-dimension of a grid
 * - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: Maximum z-dimension of a grid
 * - ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: Maximum amount of
 *   shared memory available to a thread block in bytes
 * - ::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: Memory available on device for
 *   __constant__ variables in a CUDA C kernel in bytes
 * - ::CU_DEVICE_ATTRIBUTE_WARP_SIZE: Warp size in threads
 * - ::CU_DEVICE_ATTRIBUTE_MAX_PITCH: Maximum pitch in bytes allowed by the
 *   memory copy functions that involve memory regions allocated through
 *   ::cuMemAllocPitch()
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: Maximum 1D
 *  texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: Maximum width
 *  for a 1D texture bound to linear memory
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH: Maximum
 *  mipmapped 1D texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: Maximum 2D
 *  texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: Maximum 2D
 *  texture height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: Maximum width
 *  for a 2D texture bound to linear memory
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: Maximum height
 *  for a 2D texture bound to linear memory
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: Maximum pitch
 *  in bytes for a 2D texture bound to linear memory
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: Maximum
 *  mipmapped 2D texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: Maximum
 *  mipmapped 2D texture height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: Maximum 3D
 *  texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: Maximum 3D
 *  texture height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: Maximum 3D
 *  texture depth
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE:
 *  Alternate maximum 3D texture width, 0 if no alternate
 *  maximum 3D texture size is supported
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE:
 *  Alternate maximum 3D texture height, 0 if no alternate
 *  maximum 3D texture size is supported
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE:
 *  Alternate maximum 3D texture depth, 0 if no alternate
 *  maximum 3D texture size is supported
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH:
 *  Maximum cubemap texture width or height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH:
 *  Maximum 1D layered texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:
 *   Maximum layers in a 1D layered texture
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH:
 *  Maximum 2D layered texture width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:
 *   Maximum 2D layered texture height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:
 *   Maximum layers in a 2D layered texture
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH:
 *   Maximum cubemap layered texture width or height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS:
 *   Maximum layers in a cubemap layered texture
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH:
 *   Maximum 1D surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH:
 *   Maximum 2D surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT:
 *   Maximum 2D surface height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH:
 *   Maximum 3D surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT:
 *   Maximum 3D surface height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH:
 *   Maximum 3D surface depth
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH:
 *   Maximum 1D layered surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS:
 *   Maximum layers in a 1D layered surface
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH:
 *   Maximum 2D layered surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT:
 *   Maximum 2D layered surface height
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS:
 *   Maximum layers in a 2D layered surface
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH:
 *   Maximum cubemap surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH:
 *   Maximum cubemap layered surface width
 * - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS:
 *   Maximum layers in a cubemap layered surface
 * - ::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: Maximum number of 32-bit
 *   registers available to a thread block
 * - ::CU_DEVICE_ATTRIBUTE_CLOCK_RATE: The typical clock frequency in kilohertz
 * - ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: Alignment requirement; texture
 *   base addresses aligned to ::textureAlign bytes do not need an offset
 *   applied to texture fetches
 * - ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: Pitch alignment requirement
 *   for 2D texture references bound to pitched memory
 * - ::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: 1 if the device can concurrently copy
 *   memory between host and device while executing a kernel, or 0 if not
 * - ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Number of multiprocessors on
 *   the device
 * - ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 1 if there is a run time limit
 *   for kernels executed on the device, or 0 if not
 * - ::CU_DEVICE_ATTRIBUTE_INTEGRATED: 1 if the device is integrated with the
 *   memory subsystem, or 0 if not
 * - ::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 1 if the device can map host
 *   memory into the CUDA address space, or 0 if not
 * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: Compute mode that device is currently
 *   in. Available modes are as follows:
 *   - ::CU_COMPUTEMODE_DEFAULT: Default mode - Device is not restricted and
 *     can have multiple CUDA contexts present at a single time.
 *   - ::CU_COMPUTEMODE_PROHIBITED: Compute-prohibited mode - Device is
 *     prohibited from creating new CUDA contexts.
 *   - ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS:  Compute-exclusive-process mode - Device
 *     can have only one context used by a single process at a time.
 * - ::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: 1 if the device supports
 *   executing multiple kernels within the same context simultaneously, or 0 if
 *   not. It is not guaranteed that multiple kernels will be resident
 *   on the device concurrently so this feature should not be relied upon for
 *   correctness.
 * - ::CU_DEVICE_ATTRIBUTE_ECC_ENABLED: 1 if error correction is enabled on the
 *    device, 0 if error correction is disabled or not supported by the device
 * - ::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: PCI bus identifier of the device
 * - ::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: PCI device (also known as slot) identifier
 *   of the device
 * - ::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: PCI domain identifier of the device
 * - ::CU_DEVICE_ATTRIBUTE_TCC_DRIVER: 1 if the device is using a TCC driver. TCC
 *    is only available on Tesla hardware running Windows Vista or later
 * - ::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: Peak memory clock frequency in kilohertz
 * - ::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: Global memory bus width in bits
 * - ::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
 * - ::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: Maximum resident threads per multiprocessor
 * - ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: 1 if the device shares a unified address space with
 *   the host, or 0 if not
 * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Major compute capability version number
 * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Minor compute capability version number
 * - ::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: 1 if device supports caching globals
 *    in L1 cache, 0 if caching globals in L1 cache is not supported by the device
 * - ::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: 1 if device supports caching locals
 *    in L1 cache, 0 if caching locals in L1 cache is not supported by the device
 * - ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: Maximum amount of
 *   shared memory available to a multiprocessor in bytes; this amount is shared
 *   by all thread blocks simultaneously resident on a multiprocessor
 * - ::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: Maximum number of 32-bit
 *   registers available to a multiprocessor; this number is shared by all thread
 *   blocks simultaneously resident on a multiprocessor
 * - ::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: 1 if device supports allocating managed memory
 *   on this system, 0 if allocating managed memory is not supported by the device on this system.
 * - ::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: 1 if device is on a multi-GPU board, 0 if not.
 * - ::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: Unique identifier for a group of devices
 *   associated with the same board. Devices on the same multi-GPU board will share the same identifier.
 * - ::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: 1 if Link between the device and the host
 *   supports native atomic operations.
 * - ::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: Ratio of single precision performance
 *   (in floating-point operations per second) to double precision performance.
 * - ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: Device supports coherently accessing
 *   pageable memory without calling cudaHostRegister on it.
 * - ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: Device can coherently access managed memory
 *   concurrently with the CPU.
 * - ::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: Device supports Compute Preemption.
 * - ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: Device can access host registered
 *   memory at the same virtual address as the CPU.
 * -  ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: The maximum per block shared memory size
 *    supported on this device. This is the maximum value that can be opted into when using the cuFuncSetAttribute() or cuKernelSetAttribute() call.
 *    For more details see ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
 * - ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: Device accesses pageable memory via the host's
 *   page tables.
 * - ::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: The host can directly access managed memory on the device without migration.
 * - ::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED:  Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
 * - ::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED: Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
 * - ::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED:  Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
 * - ::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED: Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
 * - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: Maximum number of thread blocks that can reside on a multiprocessor
 * - ::CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED: Device supports compressible memory allocation via ::cuMemCreate
 * - ::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: Maximum L2 persisting lines capacity setting in bytes
 * - ::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE: Maximum value of CUaccessPolicyWindow::num_bytes 
 * - ::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED: Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate.
 * - ::CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: Amount of shared memory per block reserved by CUDA driver in bytes
 * - ::CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED: Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays. 
 * - ::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED: Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
 * - ::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs
 * - ::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED: Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
 * - ::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS: The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum
 * - ::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING: GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
 * - ::CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES: Bitmask of handle types supported with mempool based IPC
 * - ::CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED: Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.
 *
 * \param pi     - Returned device attribute value
 * \param attrib - Device attribute to query
 * \param dev    - Device handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem,
 * ::cuDeviceGetExecAffinitySupport,
 * ::cudaDeviceGetAttribute,
 * ::cudaGetDeviceProperties
 *)
function cuDeviceGetAttribute(pi : Plongint; attrib : CUdevice_attribute; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Return NvSciSync attributes that this device can support.
 *
 * Returns in \p nvSciSyncAttrList, the properties of NvSciSync that
 * this CUDA device, \p dev can support. The returned \p nvSciSyncAttrList
 * can be used to create an NvSciSync object that matches this device's capabilities.
 * 
 * If NvSciSyncAttrKey_RequiredPerm field in \p nvSciSyncAttrList is
 * already set this API will return ::CUDA_ERROR_INVALID_VALUE.
 * 
 * The applications should set \p nvSciSyncAttrList to a valid 
 * NvSciSyncAttrList failing which this API will return
 * ::CUDA_ERROR_INVALID_HANDLE.
 * 
 * The \p flags controls how applications intends to use
 * the NvSciSync created from the \p nvSciSyncAttrList. The valid flags are:
 * - ::CUDA_NVSCISYNC_ATTR_SIGNAL, specifies that the applications intends to 
 * signal an NvSciSync on this CUDA device.
 * - ::CUDA_NVSCISYNC_ATTR_WAIT, specifies that the applications intends to 
 * wait on an NvSciSync on this CUDA device.
 *
 * At least one of these flags must be set, failing which the API
 * returns ::CUDA_ERROR_INVALID_VALUE. Both the flags are orthogonal
 * to one another: a developer may set both these flags that allows to
 * set both wait and signal specific attributes in the same \p nvSciSyncAttrList.
 *
 * Note that this API updates the input \p nvSciSyncAttrList with values equivalent
 * to the following public attribute key-values:
 * NvSciSyncAttrKey_RequiredPerm is set to
 * - NvSciSyncAccessPerm_SignalOnly if ::CUDA_NVSCISYNC_ATTR_SIGNAL is set in \p flags.
 * - NvSciSyncAccessPerm_WaitOnly if ::CUDA_NVSCISYNC_ATTR_WAIT is set in \p flags.
 * - NvSciSyncAccessPerm_WaitSignal if both ::CUDA_NVSCISYNC_ATTR_WAIT and
 * ::CUDA_NVSCISYNC_ATTR_SIGNAL are set in \p flags.
 * NvSciSyncAttrKey_PrimitiveInfo is set to
 * - NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid \p device.
 * - NvSciSyncAttrValPrimitiveType_Syncpoint if \p device is a Tegra device.
 * - NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if \p device is GA10X+.
 * NvSciSyncAttrKey_GpuId is set to the same UUID that is returned for this 
 * \p device from ::cuDeviceGetUuid.
 *
 * \param nvSciSyncAttrList     - Return NvSciSync attributes supported.
 * \param dev                   - Valid Cuda Device to get NvSciSync attributes for.
 * \param flags                 - flags describing NvSciSync usage.
 *
 * \return
 *
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa
 * ::cuImportExternalSemaphore,
 * ::cuDestroyExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList : Pointer; dev : CUdevice; flags : longint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the current memory pool of a device
 *
 * The memory pool must be local to the specified device.
 * ::cuMemAllocAsync allocates from the current mempool of the provided stream's device.
 * By default, a device's current memory pool is its default memory pool.
 *
 * \note Use ::cuMemAllocFromPoolAsync to specify asynchronous allocations from a device different
 * than the one the stream runs on. 
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuDeviceGetDefaultMemPool, ::cuDeviceGetMemPool, ::cuMemPoolCreate, ::cuMemPoolDestroy, ::cuMemAllocFromPoolAsync
 *)
function cuDeviceSetMemPool(dev : CUdevice; pool : CUmemoryPool):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets the current mempool for a device
 *
 * Returns the last pool provided to ::cuDeviceSetMemPool for this device
 * or the device's default memory pool if ::cuDeviceSetMemPool has never been called.
 * By default the current mempool is the default mempool for a device.
 * Otherwise the returned pool must have been set with ::cuDeviceSetMemPool.
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuDeviceGetDefaultMemPool, ::cuMemPoolCreate, ::cuDeviceSetMemPool
 *)
function cuDeviceGetMemPool(pool : PCUmemoryPool; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the default mempool of a device
 *
 * The default mempool of a device contains device memory from that device.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuMemAllocAsync, ::cuMemPoolTrimTo, ::cuMemPoolGetAttribute, ::cuMemPoolSetAttribute, cuMemPoolSetAccess, ::cuDeviceGetMemPool, ::cuMemPoolCreate
 *)
function cuDeviceGetDefaultMemPool(pool_out : PCUmemoryPool; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns information about the execution affinity support of the device.
 *
 * Returns in \p *pi whether execution affinity type \p type is supported by device \p dev.
 * The supported types are:
 * - ::CU_EXEC_AFFINITY_TYPE_SM_COUNT: 1 if context with limited SMs is supported by the device,
 *   or 0 if not;
 *
 * \param pi   - 1 if the execution affinity type \p type is supported by the device, or 0 if not
 * \param type - Execution affinity type to query
 * \param dev  - Device handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem
 *)
function cuDeviceGetExecAffinitySupport(pi : Plongint; &type : CUexecAffinityType; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Blocks until remote writes are visible to the specified scope
 *
 * Blocks until GPUDirect RDMA writes to the target context via mappings
 * created through APIs like nvidia_p2p_get_pages (see
 * https://docs.nvidia.com/cuda/gpudirect-rdma for more information), are
 * visible to the specified scope.
 *
 * If the scope equals or lies within the scope indicated by
 * ::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, the call
 * will be a no-op and can be safely omitted for performance. This can be
 * determined by comparing the numerical values between the two enums, with
 * smaller scopes having smaller values.
 *
 * Users may query support for this API via
 * ::CU_DEVICE_ATTRIBUTE_FLUSH_FLUSH_GPU_DIRECT_RDMA_OPTIONS.
 *
 * \param target - The target of the operation, see ::CUflushGPUDirectRDMAWritesTarget
 * \param scope  - The scope of the operation, see ::CUflushGPUDirectRDMAWritesScope
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 *)
function cuFlushGPUDirectRDMAWrites(target : CUflushGPUDirectRDMAWritesTarget; scope : CUflushGPUDirectRDMAWritesScope):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_DEVICE *)

(**
 * \defgroup CUDA_DEVICE_DEPRECATED Device Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated device management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns properties for a selected device
 *
 * \deprecated
 *
 * This function was deprecated as of CUDA 5.0 and replaced by ::cuDeviceGetAttribute().
 *
 * Returns in \p *prop the properties of device \p dev. The ::CUdevprop
 * structure is defined as:
 *
 * \code
       CUdevprop_st=packed record
     longint maxThreadsPerBlock;
     longint maxThreadsDim[3];
     longint maxGridSize[3];
     longint sharedMemPerBlock;
     longint totalConstantMemory;
     longint SIMDWidth;
     longint memPitch;
     longint regsPerBlock;
     longint clockRate;
     longint textureAlign
    end;
  CUdevprop = CUdevprop_st;
 * \endcode
 * where:
 *
 * - ::maxThreadsPerBlock is the maximum number of threads per block;
 * - ::maxThreadsDim[3] is the maximum sizes of each dimension of a block;
 * - ::maxGridSize[3] is the maximum sizes of each dimension of a grid;
 * - ::sharedMemPerBlock is the total amount of shared memory available per
 *   block in bytes;
 * - ::totalConstantMemory is the total amount of constant memory available on
 *   the device in bytes;
 * - ::SIMDWidth is the warp size;
 * - ::memPitch is the maximum pitch allowed by the memory copy functions that
 *   involve memory regions allocated through ::cuMemAllocPitch();
 * - ::regsPerBlock is the total number of registers available per block;
 * - ::clockRate is the clock frequency in kilohertz;
 * - ::textureAlign is the alignment requirement; texture base addresses that
 *   are aligned to ::textureAlign bytes do not need an offset applied to
 *   texture fetches.
 *
 * \param prop - Returned properties of device
 * \param dev  - Device to get properties for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem
 *)
function cuDeviceGetProperties(prop : PCUdevprop; dev : CUdevice):CUresult;  WINAPI; external libcuda; deprecated 'CUDA_DEPRECATED';

(**
 * \brief Returns the compute capability of the device
 *
 * \deprecated
 *
 * This function was deprecated as of CUDA 5.0 and its functionality superseded
 * by ::cuDeviceGetAttribute().
 *
 * Returns in \p *major and \p *minor the major and minor revision numbers that
 * define the compute capability of the device \p dev.
 *
 * \param major - Major revision number
 * \param minor - Minor revision number
 * \param dev   - Device handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetCount,
 * ::cuDeviceGetName,
 * ::cuDeviceGetUuid,
 * ::cuDeviceGet,
 * ::cuDeviceTotalMem
 *)
function cuDeviceComputeCapability(major : Plongint; minor : Plongint; dev : CUdevice):CUresult;  WINAPI; external libcuda; deprecated;

(** @} *) (* END CUDA_DEVICE_DEPRECATED *)

(**
 * \defgroup CUDA_PRIMARY_CTX Primary Context Management
 *
 * ___MANBRIEF___ primary context management functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the primary context management functions of the low-level
 * CUDA driver application programming interface.
 *
 * The primary context is unique per device and shared with the CUDA runtime API.
 * These functions allow integration with other libraries using CUDA.
 *
 * @{
 *)

(**
 * \brief Retain the primary context on the GPU
 *
 * Retains the primary context on the device.
 * Once the user successfully retains the primary context, the primary context
 * will be active and available to the user until the user releases it
 * with ::cuDevicePrimaryCtxRelease() or resets it with ::cuDevicePrimaryCtxReset().
 * Unlike ::cuCtxCreate() the newly retained context is not pushed onto the stack.
 *
 * Retaining the primary context for the first time will fail with ::CUDA_ERROR_UNKNOWN
 * if the compute mode of the device is ::CU_COMPUTEMODE_PROHIBITED. The function
 * ::cuDeviceGetAttribute() can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to
 * determine the compute mode  of the device.
 * The <i>nvidia-smi</i> tool can be used to set the compute mode for
 * devices. Documentation for <i>nvidia-smi</i> can be obtained by passing a
 * -h option to it.
 *
 * Please note that the primary context always supports pinned allocations. Other
 * flags can be specified by ::cuDevicePrimaryCtxSetFlags().
 *
 * \param pctx  - Returned context handle of the new context
 * \param dev   - Device for which primary context is requested
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuDevicePrimaryCtxRelease,
 * ::cuDevicePrimaryCtxSetFlags,
 * ::cuCtxCreate,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuDevicePrimaryCtxRetain(pctx : PCUcontext; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Release the primary context on the GPU
 *
 * Releases the primary context interop on the device.
 * A retained context should always be released once the user is done using
 * it. The context is automatically reset once the last reference to it is
 * released. This behavior is different when the primary context was retained
 * by the CUDA runtime from CUDA 4.0 and earlier. In this case, the primary
 * context remains always active.
 *
 * Releasing a primary context that has not been previously retained will
 * fail with ::CUDA_ERROR_INVALID_CONTEXT.
 *
 * Please note that unlike ::cuCtxDestroy() this method does not pop the context
 * from stack in any circumstances.
 *
 * \param dev - Device which primary context is released
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuDevicePrimaryCtxRetain,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuDevicePrimaryCtxRelease(dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Set flags for the primary context
 *
 * Sets the flags for the primary context on the device overwriting perviously
 * set ones.
 *
 * The three LSBs of the \p flags parameter can be used to control how the OS
 * thread, which owns the CUDA context at the time of an API call, interacts
 * with the OS scheduler when waiting for results from the GPU. Only one of
 * the scheduling flags can be set when creating a context.
 *
 * - ::CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
 * results from the GPU. This can decrease latency when waiting for the GPU,
 * but may lower the performance of CPU threads if they are performing work in
 * parallel with the CUDA thread.
 *
 * - ::CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
 * results from the GPU. This can increase latency when waiting for the GPU,
 * but can increase the performance of CPU threads performing work in parallel
 * with the GPU.
 *
 * - ::CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work.
 *
 * - ::CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work. <br>
 * <b>Deprecated:</b> This flag was deprecated as of CUDA 4.0 and was
 * replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
 *
 * - ::CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
 * uses a heuristic based on the number of active CUDA contexts in the
 * process \e C and the number of logical processors in the system \e P. If
 * \e C > \e P, then CUDA will yield to other OS threads when waiting for
 * the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while
 * waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN).
 * Additionally, on Tegra devices, ::CU_CTX_SCHED_AUTO uses a heuristic based on
 * the power profile of the platform and may choose ::CU_CTX_SCHED_BLOCKING_SYNC
 * for low-powered devices.
 *
 * - ::CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage. <br>
 * <b>Deprecated:</b> This flag is deprecated and the behavior enabled
 * by this flag is now the default and cannot be disabled.
 *
 * - ::CU_CTX_COREDUMP_ENABLE: If GPU coredumps have not been enabled globally
 * with ::cuCoredumpSetAttributeGlobal or environment variables, this flag can
 * be set during context creation to instruct CUDA to create a coredump if
 * this context raises an exception during execution. These environment variables
 * are described in the CUDA-GDB user guide under the "GPU core dump support"
 * section.
 * The initial settings will be taken from the global settings at the time of
 * context creation. The other settings that control coredump output can be 
 * modified by calling ::cuCoredumpSetAttribute from the created context after
 * it becomes current.
 *
 * - ::CU_CTX_USER_COREDUMP_ENABLE: If user-triggered GPU coredumps have not
 * been enabled globally with ::cuCoredumpSetAttributeGlobal or environment 
 * variables, this flag can be set during context creation to instruct CUDA to
 * create a coredump if data is written to a certain pipe that is present in the
 * OS space. These environment variables are described in the CUDA-GDB user
 * guide under the "GPU core dump support" section.
 * It is important to note that the pipe name *must* be set with
 * ::cuCoredumpSetAttributeGlobal before creating the context if this flag is
 * used. Setting this flag implies that ::CU_CTX_COREDUMP_ENABLE is set.
 * The initial settings will be taken from the global settings at the time of
 * context creation. The other settings that control coredump output can be 
 * modified by calling ::cuCoredumpSetAttribute from the created context after 
 * it becomes current.
 *
 * \param dev   - Device for which the primary context flags are set
 * \param flags - New flags for the device
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa ::cuDevicePrimaryCtxRetain,
 * ::cuDevicePrimaryCtxGetState,
 * ::cuCtxCreate,
 * ::cuCtxGetFlags,
 * ::cuCtxSetFlags,
 * ::cudaSetDeviceFlags
 *)
function cuDevicePrimaryCtxSetFlags(dev : CUdevice; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get the state of the primary context
 *
 * Returns in \p *flags the flags for the primary context of \p dev, and in
 * \p *active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
 * values.
 *
 * \param dev    - Device to get primary context flags for
 * \param flags  - Pointer to store flags
 * \param active - Pointer to store context state; 0 = inactive, 1 = active
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa
 * ::cuDevicePrimaryCtxSetFlags,
 * ::cuCtxGetFlags,
 * ::cuCtxSetFlags,
 * ::cudaGetDeviceFlags
 *)
function cuDevicePrimaryCtxGetState(dev : CUdevice; flags : Plongword; active : Plongint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroy all allocations and reset all state on the primary context
 *
 * Explicitly destroys and cleans up all resources associated with the current
 * device in the current process.
 *
 * Note that it is responsibility of the calling function to ensure that no
 * other module in the process is using the device any more. For that reason
 * it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
 * However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
 * even after resetting the device.
 * Resetting the primary context does not release it, an application that has
 * retained the primary context should explicitly release its usage.
 *
 * \param dev - Device for which primary context is destroyed
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
 * \notefnerr
 *
 * \sa ::cuDevicePrimaryCtxRetain,
 * ::cuDevicePrimaryCtxRelease,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cudaDeviceReset
 *)
function cuDevicePrimaryCtxReset(dev : CUdevice):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_PRIMARY_CTX *)

(**
 * \defgroup CUDA_CTX Context Management
 *
 * ___MANBRIEF___ context management functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the context management functions of the low-level
 * CUDA driver application programming interface.
 *
 * Please note that some functions are described in
 * \ref CUDA_PRIMARY_CTX "Primary Context Management" section.
 *
 * @{
 *)

(**
 * \brief Create a CUDA context
 *
 * \note In most cases it is recommended to use ::cuDevicePrimaryCtxRetain.
 *
 * Creates a new CUDA context and associates it with the calling thread. The
 * \p flags parameter is described below. The context is created with a usage
 * count of 1 and the caller of ::cuCtxCreate() must call ::cuCtxDestroy()
 * when done using the context. If a context is already current to the thread,
 * it is supplanted by the newly created context and may be restored by a subsequent
 * call to ::cuCtxPopCurrent().
 *
 * The three LSBs of the \p flags parameter can be used to control how the OS
 * thread, which owns the CUDA context at the time of an API call, interacts
 * with the OS scheduler when waiting for results from the GPU. Only one of
 * the scheduling flags can be set when creating a context.
 *
 * - ::CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
 * results from the GPU. This can decrease latency when waiting for the GPU,
 * but may lower the performance of CPU threads if they are performing work in
 * parallel with the CUDA thread.
 *
 * - ::CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
 * results from the GPU. This can increase latency when waiting for the GPU,
 * but can increase the performance of CPU threads performing work in parallel
 * with the GPU.
 *
 * - ::CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work.
 *
 * - ::CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work. <br>
 * <b>Deprecated:</b> This flag was deprecated as of CUDA 4.0 and was
 * replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
 *
 * - ::CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
 * uses a heuristic based on the number of active CUDA contexts in the
 * process \e C and the number of logical processors in the system \e P. If
 * \e C > \e P, then CUDA will yield to other OS threads when waiting for
 * the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while
 * waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN).
 * Additionally, on Tegra devices, ::CU_CTX_SCHED_AUTO uses a heuristic based on
 * the power profile of the platform and may choose ::CU_CTX_SCHED_BLOCKING_SYNC
 * for low-powered devices.
 *
 * - ::CU_CTX_MAP_HOST: Instruct CUDA to support mapped pinned allocations.
 * This flag must be set in order to allocate pinned host memory that is
 * accessible to the GPU.
 *
 * - ::CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage. <br>
 * <b>Deprecated:</b> This flag is deprecated and the behavior enabled
 * by this flag is now the default and cannot be disabled.
 * Instead, the per-thread stack size can be controlled with ::cuCtxSetLimit().
 *
 * - ::CU_CTX_COREDUMP_ENABLE: If GPU coredumps have not been enabled globally
 * with ::cuCoredumpSetAttributeGlobal or environment variables, this flag can
 * be set during context creation to instruct CUDA to create a coredump if
 * this context raises an exception during execution. These environment variables
 * are described in the CUDA-GDB user guide under the "GPU core dump support"
 * section.
 * The initial attributes will be taken from the global attributes at the time of
 * context creation. The other attributes that control coredump output can be 
 * modified by calling ::cuCoredumpSetAttribute from the created context after
 * it becomes current.
 *
 * - ::CU_CTX_USER_COREDUMP_ENABLE: If user-triggered GPU coredumps have not
 * been enabled globally with ::cuCoredumpSetAttributeGlobal or environment 
 * variables, this flag can be set during context creation to instruct CUDA to
 * create a coredump if data is written to a certain pipe that is present in the
 * OS space. These environment variables are described in the CUDA-GDB user
 * guide under the "GPU core dump support" section.
 * It is important to note that the pipe name *must* be set with
 * ::cuCoredumpSetAttributeGlobal before creating the context if this flag is
 * used. Setting this flag implies that ::CU_CTX_COREDUMP_ENABLE is set.
 * The initial attributes will be taken from the global attributes at the time of
 * context creation. The other attributes that control coredump output can be 
 * modified by calling ::cuCoredumpSetAttribute from the created context after 
 * it becomes current.
 * Setting this flag on any context creation is equivalent to setting the 
 * ::CU_COREDUMP_ENABLE_USER_TRIGGER attribute to \p true globally.
 *
 * Context creation will fail with ::CUDA_ERROR_UNKNOWN if the compute mode of
 * the device is ::CU_COMPUTEMODE_PROHIBITED. The function ::cuDeviceGetAttribute()
 * can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the
 * compute mode of the device. The <i>nvidia-smi</i> tool can be used to set
 * the compute mode for * devices.
 * Documentation for <i>nvidia-smi</i> can be obtained by passing a
 * -h option to it.
 *
 * \param pctx  - Returned context handle of the new context
 * \param flags - Context creation flags
 * \param dev   - Device to create context on
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCoredumpSetAttributeGlobal,
 * ::cuCoredumpSetAttribute,
 * ::cuCtxSynchronize
 *)
function cuCtxCreate(pctx : PCUcontext; flags : CUctx_flags; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Create a CUDA context with execution affinity
 *
 * Creates a new CUDA context with execution affinity and associates it with
 * the calling thread. The \p paramsArray and \p flags parameter are described below.
 * The context is created with a usage count of 1 and the caller of ::cuCtxCreate() must
 * call ::cuCtxDestroy() when done using the context. If a context is already
 * current to the thread, it is supplanted by the newly created context and may
 * be restored by a subsequent call to ::cuCtxPopCurrent().
 *
 * The type and the amount of execution resource the context can use is limited by \p paramsArray
 * and \p numParams. The \p paramsArray is an array of \p CUexecAffinityParam and the \p numParams
 * describes the size of the array. If two \p CUexecAffinityParam in the array have the same type,
 * the latter execution affinity parameter overrides the former execution affinity parameter.
 * The supported execution affinity types are:
 * - ::CU_EXEC_AFFINITY_TYPE_SM_COUNT limits the portion of SMs that the context can use. The portion
 *   of SMs is specified as the number of SMs via \p CUexecAffinitySmCount. This limit will be internally
 *   rounded up to the next hardware-supported amount. Hence, it is imperative to query the actual execution
 *   affinity of the context via \p cuCtxGetExecAffinity after context creation. Currently, this attribute
 *   is only supported under Volta+ MPS.
 *
 * The three LSBs of the \p flags parameter can be used to control how the OS
 * thread, which owns the CUDA context at the time of an API call, interacts
 * with the OS scheduler when waiting for results from the GPU. Only one of
 * the scheduling flags can be set when creating a context.
 *
 * - ::CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
 * results from the GPU. This can decrease latency when waiting for the GPU,
 * but may lower the performance of CPU threads if they are performing work in
 * parallel with the CUDA thread.
 *
 * - ::CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
 * results from the GPU. This can increase latency when waiting for the GPU,
 * but can increase the performance of CPU threads performing work in parallel
 * with the GPU.
 *
 * - ::CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work.
 *
 * - ::CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work. <br>
 * <b>Deprecated:</b> This flag was deprecated as of CUDA 4.0 and was
 * replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
 *
 * - ::CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
 * uses a heuristic based on the number of active CUDA contexts in the
 * process \e C and the number of logical processors in the system \e P. If
 * \e C > \e P, then CUDA will yield to other OS threads when waiting for
 * the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while
 * waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN).
 * Additionally, on Tegra devices, ::CU_CTX_SCHED_AUTO uses a heuristic based on
 * the power profile of the platform and may choose ::CU_CTX_SCHED_BLOCKING_SYNC
 * for low-powered devices.
 *
 * - ::CU_CTX_MAP_HOST: Instruct CUDA to support mapped pinned allocations.
 * This flag must be set in order to allocate pinned host memory that is
 * accessible to the GPU.
 *
 * - ::CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage. <br>
 * <b>Deprecated:</b> This flag is deprecated and the behavior enabled
 * by this flag is now the default and cannot be disabled.
 * Instead, the per-thread stack size can be controlled with ::cuCtxSetLimit().
 *
 * - ::CU_CTX_COREDUMP_ENABLE: If GPU coredumps have not been enabled globally
 * with ::cuCoredumpSetAttributeGlobal or environment variables, this flag can
 * be set during context creation to instruct CUDA to create a coredump if
 * this context raises an exception during execution. These environment variables
 * are described in the CUDA-GDB user guide under the "GPU core dump support"
 * section.
 * The initial attributes will be taken from the global attributes at the time of
 * context creation. The other attributes that control coredump output can be 
 * modified by calling ::cuCoredumpSetAttribute from the created context after
 * it becomes current.
 *
 * - ::CU_CTX_USER_COREDUMP_ENABLE: If user-triggered GPU coredumps have not
 * been enabled globally with ::cuCoredumpSetAttributeGlobal or environment 
 * variables, this flag can be set during context creation to instruct CUDA to
 * create a coredump if data is written to a certain pipe that is present in the
 * OS space. These environment variables are described in the CUDA-GDB user
 * guide under the "GPU core dump support" section.
 * It is important to note that the pipe name *must* be set with
 * ::cuCoredumpSetAttributeGlobal before creating the context if this flag is
 * used. Setting this flag implies that ::CU_CTX_COREDUMP_ENABLE is set.
 * The initial attributes will be taken from the global attributes at the time of
 * context creation. The other attributes that control coredump output can be 
 * modified by calling ::cuCoredumpSetAttribute from the created context after 
 * it becomes current.
 * Setting this flag on any context creation is equivalent to setting the 
 * ::CU_COREDUMP_ENABLE_USER_TRIGGER attribute to \p true globally.
 *
 * Context creation will fail with ::CUDA_ERROR_UNKNOWN if the compute mode of
 * the device is ::CU_COMPUTEMODE_PROHIBITED. The function ::cuDeviceGetAttribute()
 * can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the
 * compute mode of the device. The <i>nvidia-smi</i> tool can be used to set
 * the compute mode for * devices.
 * Documentation for <i>nvidia-smi</i> can be obtained by passing a
 * -h option to it.
 *
 * \param pctx        - Returned context handle of the new context
 * \param paramsArray - Execution affinity parameters
 * \param numParams   - Number of execution affinity parameters
 * \param flags       - Context creation flags
 * \param dev         - Device to create context on
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cuCoredumpSetAttributeGlobal,
 * ::cuCoredumpSetAttribute,
 * ::CUexecAffinityParam
 *)
function cuCtxCreate_v3(pctx : PCUcontext; paramsArray : PCUexecAffinityParam; numParams : longint; flags : longword; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroy a CUDA context
 *
 * Destroys the CUDA context specified by \p ctx.  The context \p ctx will be
 * destroyed regardless of how many threads it is current to.
 * It is the responsibility of the calling function to ensure that no API
 * call issues using \p ctx while ::cuCtxDestroy() is executing.
 *
 * Destroys and cleans up all resources associated with the context.
 * It is the caller's responsibility to ensure that the context or its resources
 * are not accessed or passed in subsequent API calls and doing so will result in undefined behavior.
 * These resources include CUDA types such as ::CUmodule, ::CUfunction, ::CUstream, ::CUevent,
 * ::CUarray, ::CUmipmappedArray, ::CUtexObject, ::CUsurfObject, ::CUtexref, ::CUsurfref,
 * ::CUgraphicsResource, ::CUlinkState, ::CUexternalMemory and ::CUexternalSemaphore.
 *
 * If \p ctx is current to the calling thread then \p ctx will also be
 * popped from the current thread's context stack (as though ::cuCtxPopCurrent()
 * were called).  If \p ctx is current to other threads, then \p ctx will
 * remain current to those threads, and attempting to access \p ctx from
 * those threads will result in the error ::CUDA_ERROR_CONTEXT_IS_DESTROYED.
 *
 * \param ctx - Context to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuCtxDestroy(ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Pushes a context on the current CPU thread
 *
 * Pushes the given context \p ctx onto the CPU thread's stack of current
 * contexts. The specified context becomes the CPU thread's current context, so
 * all CUDA functions that operate on the current context are affected.
 *
 * The previous current context may be made current again by calling
 * ::cuCtxDestroy() or ::cuCtxPopCurrent().
 *
 * \param ctx - Context to push
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuCtxPushCurrent(ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Pops the current CUDA context from the current CPU thread.
 *
 * Pops the current CUDA context from the CPU thread and passes back the
 * old context handle in \p *pctx. That context may then be made current
 * to a different CPU thread by calling ::cuCtxPushCurrent().
 *
 * If a context was current to the CPU thread before ::cuCtxCreate() or
 * ::cuCtxPushCurrent() was called, this function makes that context current to
 * the CPU thread again.
 *
 * \param pctx - Returned popped context handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuCtxPopCurrent(pctx : PCUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Binds the specified CUDA context to the calling CPU thread
 *
 * Binds the specified CUDA context to the calling CPU thread.
 * If \p ctx is NULL then the CUDA context previously bound to the
 * calling CPU thread is unbound and ::CUDA_SUCCESS is returned.
 *
 * If there exists a CUDA context stack on the calling CPU thread, this
 * will replace the top of that stack with \p ctx.
 * If \p ctx is NULL then this will be equivalent to popping the top
 * of the calling CPU thread's CUDA context stack (or a no-op if the
 * calling CPU thread's CUDA context stack is empty).
 *
 * \param ctx - Context to bind to the calling CPU thread
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::cuCtxGetCurrent,
 * ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cudaSetDevice
 *)
function cuCtxSetCurrent(ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the CUDA context bound to the calling CPU thread.
 *
 * Returns in \p *pctx the CUDA context bound to the calling CPU thread.
 * If no context is bound to the calling CPU thread then \p *pctx is
 * set to NULL and ::CUDA_SUCCESS is returned.
 *
 * \param pctx - Returned context handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * \notefnerr
 *
 * \sa
 * ::cuCtxSetCurrent,
 * ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cudaGetDevice
 *)
function cuCtxGetCurrent(pctx : PCUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the device ID for the current context
 *
 * Returns in \p *device the ordinal of the current context's device.
 *
 * \param device - Returned device ID for the current context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cudaGetDevice
 *)
function cuCtxGetDevice(device : PCUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the flags for the current context
 *
 * Returns in \p *flags the flags of the current context. See ::cuCtxCreate
 * for flag values.
 *
 * \param flags - Pointer to store flags of current context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetCurrent,
 * ::cuCtxGetDevice,
 * ::cuCtxGetLimit,
 * ::cuCtxGetSharedMemConfig,
 * ::cuCtxGetStreamPriorityRange,
 * ::cuCtxSetFlags,
 * ::cudaGetDeviceFlags
 *)
function cuCtxGetFlags(flags : Plongword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the flags for the current context
 *
 * \param flags - Flags to set on the current context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetCurrent,
 * ::cuCtxGetDevice,
 * ::cuCtxGetLimit,
 * ::cuCtxGetSharedMemConfig,
 * ::cuCtxGetStreamPriorityRange,
 * ::cuCtxGetFlags,
 * ::cudaGetDeviceFlags,
 * ::cuDevicePrimaryCtxSetFlags,
 *)
function cuCtxSetFlags(flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the unique Id associated with the context supplied
 *
 * Returns in \p ctxId the unique Id which is associated with a given context.
 * The Id is unique for the life of the program for this instance of CUDA.
 * If context is supplied as NULL and there is one current, the Id of the
 * current context is returned.
 *
 * \param ctx - Context for which to obtain the Id
 * \param ctxId - Pointer to store the Id of the context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPushCurrent
 *)
function cuCtxGetId(ctx : CUcontext; ctxId : Puint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Block for a context's tasks to complete
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::cuCtxSynchronize() returns an error if one of the preceding tasks failed.
 * If the context was created with the ::CU_CTX_SCHED_BLOCKING_SYNC flag, the
 * CPU thread will block until the GPU context has finished its work.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cudaDeviceSynchronize
 *)
function cuCtxSynchronize():CUresult;  WINAPI; external libcuda;

(**
 * \brief Set resource limits
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the context. The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc). The application can use ::cuCtxGetLimit() to find out exactly
 * what the limit has been set to.
 *
 * Setting each ::CUlimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::CU_LIMIT_STACK_SIZE controls the stack size in bytes of each GPU thread.
 *   The driver automatically increases the per-thread stack size
 *   for each kernel launch as needed. This size isn't reset back to the
 *   original value after each launch. Setting this value will take effect 
 *   immediately, and if necessary, the device will block until all preceding 
 *   requested tasks are complete.
 *
 * - ::CU_LIMIT_PRINTF_FIFO_SIZE controls the size in bytes of the FIFO used
 *   by the ::printf() device system call. Setting ::CU_LIMIT_PRINTF_FIFO_SIZE
 *   must be performed before launching any kernel that uses the ::printf()
 *   device system call, otherwise ::CUDA_ERROR_INVALID_VALUE will be returned.
 *
 * - ::CU_LIMIT_MALLOC_HEAP_SIZE controls the size in bytes of the heap used
 *   by the ::malloc() and ::free() device system calls. Setting
 *   ::CU_LIMIT_MALLOC_HEAP_SIZE must be performed before launching any kernel
 *   that uses the ::malloc() or ::free() device system calls, otherwise
 *   ::CUDA_ERROR_INVALID_VALUE will be returned.
 *
 * - ::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH controls the maximum nesting depth of
 *   a grid at which a thread can safely call ::cudaDeviceSynchronize(). Setting
 *   this limit must be performed before any launch of a kernel that uses the
 *   device runtime and calls ::cudaDeviceSynchronize() above the default sync
 *   depth, two levels of grids. Calls to ::cudaDeviceSynchronize() will fail
 *   with error code ::cudaErrorSyncDepthExceeded if the limitation is
 *   violated. This limit can be set smaller than the default or up the maximum
 *   launch depth of 24. When setting this limit, keep in mind that additional
 *   levels of sync depth require the driver to reserve large amounts of device
 *   memory which can no longer be used for user allocations. If these
 *   reservations of device memory fail, ::cuCtxSetLimit() will return
 *   ::CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability < 9.0.
 *   Attempting to set this limit on devices of other compute capability
 *   versions will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT being
 *   returned.
 *
 * - ::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT controls the maximum number of
 *   outstanding device runtime launches that can be made from the current
 *   context. A grid is outstanding from the point of launch up until the grid
 *   is known to have been completed. Device runtime launches which violate
 *   this limitation fail and return ::cudaErrorLaunchPendingCountExceeded when
 *   ::cudaGetLastError() is called after launch. If more pending launches than
 *   the default (2048 launches) are needed for a module using the device
 *   runtime, this limit can be increased. Keep in mind that being able to
 *   sustain additional pending launches will require the driver to reserve
 *   larger amounts of device memory upfront which can no longer be used for
 *   allocations. If these reservations fail, ::cuCtxSetLimit() will return
 *   ::CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT being
 *   returned.
 *
 * - ::CU_LIMIT_MAX_L2_FETCH_GRANULARITY controls the L2 cache fetch granularity.
 *   Values can range from 0B to 128B. This is purely a performance hint and
 *   it can be ignored or clamped depending on the platform.
 *
 * - ::CU_LIMIT_PERSISTING_L2_CACHE_SIZE controls size in bytes available for
 *   persisting L2 cache. This is purely a performance hint and it can be
 *   ignored or clamped depending on the platform.
 *
 * \param limit - Limit to set
 * \param value - Size of limit
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNSUPPORTED_LIMIT,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSynchronize,
 * ::cudaDeviceSetLimit
 *)
function cuCtxSetLimit(limit : CUlimit; value : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns resource limits
 *
 * Returns in \p *pvalue the current size of \p limit.  The supported
 * ::CUlimit values are:
 * - ::CU_LIMIT_STACK_SIZE: stack size in bytes of each GPU thread.
 * - ::CU_LIMIT_PRINTF_FIFO_SIZE: size in bytes of the FIFO used by the
 *   ::printf() device system call.
 * - ::CU_LIMIT_MALLOC_HEAP_SIZE: size in bytes of the heap used by the
 *   ::malloc() and ::free() device system calls.
 * - ::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: maximum grid depth at which a thread
 *   can issue the device runtime call ::cudaDeviceSynchronize() to wait on
 *   child grid launches to complete.
 * - ::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: maximum number of outstanding
 *   device runtime launches that can be made from this context.
 * - ::CU_LIMIT_MAX_L2_FETCH_GRANULARITY: L2 cache fetch granularity.
 * - ::CU_LIMIT_PERSISTING_L2_CACHE_SIZE: Persisting L2 cache size in bytes
 *
 * \param limit  - Limit to query
 * \param pvalue - Returned size of limit
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNSUPPORTED_LIMIT
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cudaDeviceGetLimit
 *)
function cuCtxGetLimit(pvalue : Psize_t; limit : CUlimit):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the preferred cache configuration for the current context.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this function returns through \p pconfig the preferred cache configuration
 * for the current context. This is only a preference. The driver will use
 * the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute functions.
 *
 * This will return a \p pconfig of ::CU_FUNC_CACHE_PREFER_NONE on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \param pconfig - Returned cache configuration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cuFuncSetCacheConfig,
 * ::cudaDeviceGetCacheConfig
 *)
function cuCtxGetCacheConfig(pconfig : PCUfunc_cache):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the preferred cache configuration for the current context.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p config the preferred cache configuration for
 * the current context. This is only a preference. The driver will use
 * the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute the function. Any function preference
 * set via ::cuFuncSetCacheConfig() or ::cuKernelSetCacheConfig() will be preferred over this context-wide
 * setting. Setting the context-wide cache configuration to
 * ::CU_FUNC_CACHE_PREFER_NONE will cause subsequent kernel launches to prefer
 * to not change the cache configuration unless required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \param config - Requested cache configuration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cuFuncSetCacheConfig,
 * ::cudaDeviceSetCacheConfig,
 * ::cuKernelSetCacheConfig
 *)
function cuCtxSetCacheConfig(config : CUfunc_cache):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the current shared memory configuration for the current context.
 *
 * This function will return in \p pConfig the current size of shared memory banks
 * in the current context. On devices with configurable shared memory banks,
 * ::cuCtxSetSharedMemConfig can be used to change this setting, so that all
 * subsequent kernel launches will by default use the new bank size. When
 * ::cuCtxGetSharedMemConfig is called on devices without configurable shared
 * memory, it will return the fixed bank size of the hardware.
 *
 * The returned bank configurations can be either:
 * - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:  shared memory bank width is
 *   four bytes.
 * - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: shared memory bank width will
 *   eight bytes.
 *
 * \param pConfig - returned shared memory configuration
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cuCtxGetSharedMemConfig,
 * ::cuFuncSetCacheConfig,
 * ::cudaDeviceGetSharedMemConfig
 *)
function cuCtxGetSharedMemConfig(pConfig : PCUsharedconfig):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the shared memory configuration for the current context.
 *
 * On devices with configurable shared memory banks, this function will set
 * the context's shared memory bank size which is used for subsequent kernel
 * launches.
 *
 * Changed the shared memory configuration between launches may insert a device
 * side synchronization point between those launches.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance.
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: set bank width to the default initial
 *   setting (currently, four bytes).
 * - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively four bytes.
 * - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively eight bytes.
 *
 * \param config - requested shared memory configuration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cuCtxGetSharedMemConfig,
 * ::cuFuncSetCacheConfig,
 * ::cudaDeviceSetSharedMemConfig
 *)
function cuCtxSetSharedMemConfig(config : CUsharedconfig):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets the context's API version.
 *
 * Returns a version number in \p version corresponding to the capabilities of
 * the context (e.g. 3010 or 3020), which library developers can use to direct
 * callers to a specific API version. If \p ctx is NULL, returns the API version
 * used to create the currently bound context.
 *
 * Note that new API versions are only introduced when context capabilities are
 * changed that break binary compatibility, so the API version and driver version
 * may be different. For example, it is valid for the API version to be 3020 while
 * the driver version is 4020.
 *
 * \param ctx     - Context to check
 * \param version - Pointer to version
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuCtxGetApiVersion(ctx : CUcontext; version : Plongword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns numerical values that correspond to the least and
 * greatest stream priorities.
 *
 * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
 * to the least and greatest stream priorities respectively. Stream priorities
 * follow a convention where lower numbers imply greater priorities. The range of
 * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
 * If the user attempts to create a stream with a priority value that is
 * outside the meaningful range as specified by this API, the priority is
 * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
 * respectively. See ::cuStreamCreateWithPriority for details on creating a
 * priority stream.
 * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
 * is not desired.
 *
 * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
 * the current context's device does not support stream priorities
 * (see ::cuDeviceGetAttribute).
 *
 * \param leastPriority    - Pointer to an longint in which the numerical value for least
 *                           stream priority is returned
 * \param greatestPriority - Pointer to an longint in which the numerical value for greatest
 *                           stream priority is returned
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa ::cuStreamCreateWithPriority,
 * ::cuStreamGetPriority,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize,
 * ::cudaDeviceGetStreamPriorityRange
 *)
function cuCtxGetStreamPriorityRange(leastPriority : Plongint; greatestPriority : Plongint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Resets all persisting lines in cache to normal status.
 *
 * ::cuCtxResetPersistingL2Cache Resets all persisting lines in cache to normal
 * status. Takes effect on function return.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuCtxResetPersistingL2Cache():CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the execution affinity setting for the current context.
 *
 * Returns in \p *pExecAffinity the current value of \p type. The supported
 * ::CUexecAffinityType values are:
 * - ::CU_EXEC_AFFINITY_TYPE_SM_COUNT: number of SMs the context is limited to use.
 *
 * \param type          - Execution affinity type to query
 * \param pExecAffinity - Returned execution affinity
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY
 * \notefnerr
 *
 * \sa
 * ::CUexecAffinityParam
 *)
function cuCtxGetExecAffinity(pExecAffinity : PCUexecAffinityParam; &type : CUexecAffinityType):CUresult;  WINAPI; external libcuda;


(** @} *) (* END CUDA_CTX *)

(**
 * \defgroup CUDA_CTX_DEPRECATED Context Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated context management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated context management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Increment a context's usage-count
 *
 * \deprecated
 *
 * Note that this function is deprecated and should not be used.
 *
 * Increments the usage count of the context and passes back a context handle
 * in \p *pctx that must be passed to ::cuCtxDetach() when the application is
 * done with the context. ::cuCtxAttach() fails if there is no context current
 * to the thread.
 *
 * Currently, the \p flags parameter must be 0.
 *
 * \param pctx  - Returned context handle of the current context
 * \param flags - Context attach flags (must be 0)
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxDetach,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuCtxAttach(pctx : PCUcontext; flags : longword):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Decrement a context's usage-count
 *
 * \deprecated
 *
 * Note that this function is deprecated and should not be used.
 *
 * Decrements the usage count of the context \p ctx, and destroys the context
 * if the usage count goes to 0. The context must be a handle that was passed
 * back by ::cuCtxCreate() or ::cuCtxAttach(), and must be current to the
 * calling thread.
 *
 * \param ctx - Context to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuCtxCreate,
 * ::cuCtxDestroy,
 * ::cuCtxGetApiVersion,
 * ::cuCtxGetCacheConfig,
 * ::cuCtxGetDevice,
 * ::cuCtxGetFlags,
 * ::cuCtxGetLimit,
 * ::cuCtxPopCurrent,
 * ::cuCtxPushCurrent,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxSetLimit,
 * ::cuCtxSynchronize
 *)
function cuCtxDetach(ctx : CUcontext):CUresult;  WINAPI; external libcuda; deprecated;

(** @} *) (* END CUDA_CTX_DEPRECATED *)


(**
 * \defgroup CUDA_MODULE Module Management
 *
 * ___MANBRIEF___ module management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the module management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Loads a compute module
 *
 * Takes a filename \p fname and loads the corresponding module \p module into
 * the current context. The CUDA driver API does not attempt to lazily
 * allocate the resources needed by a module; if the memory for functions and
 * data (constant and global) needed by the module cannot be allocated,
 * ::cuModuleLoad() fails. The file should be a \e cubin file as output by
 * \b nvcc, or a \e PTX file either as output by \b nvcc or handwritten, or
 * a \e fatbin file as output by \b nvcc from toolchain 4.0 or later.
 *
 * \param module - Returned module
 * \param fname  - Filename of module to load
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_FILE_NOT_FOUND,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
 * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload
 *)
function cuModuleLoad(module : PCUmodule; const fname : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Load a module's data
 *
 * Takes a pointer \p image and loads the corresponding module \p module into
 * the current context. The pointer may be obtained by mapping a \e cubin or
 * \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
 * as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
 * object into the executable resources and using operating system calls such
 * as Windows \c FindResource() to obtain the pointer.
 *
 * \param module - Returned module
 * \param image  - Module data to load
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
 * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload
 *)
function cuModuleLoadData(module : PCUmodule; const image : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Load a module's data with options
 *
 * Takes a pointer \p image and loads the corresponding module \p module into
 * the current context. The pointer may be obtained by mapping a \e cubin or
 * \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
 * as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
 * object into the executable resources and using operating system calls such
 * as Windows \c FindResource() to obtain the pointer. Options are passed as
 * an array via \p options and any corresponding parameters are passed in
 * \p optionValues. The number of total options is supplied via \p numOptions.
 * Any outputs will be returned via \p optionValues.
 *
 * \param module       - Returned module
 * \param image        - Module data to load
 * \param numOptions   - Number of options
 * \param options      - Options for JIT
 * \param optionValues - Option values for JIT
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
 * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload
 *)
function cuModuleLoadDataEx(module : PCUmodule; const image: Pointer; numOptions : longword; options : PCUjit_option; optionValues : Ppointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Load a module's data
 *
 * Takes a pointer \p fatCubin and loads the corresponding module \p module
 * into the current context. The pointer represents a <i>fat binary</i> object,
 * which is a collection of different \e cubin and/or \e PTX files, all
 * representing the same device code, but compiled and optimized for different
 * architectures.
 *
 * Prior to CUDA 4.0, there was no documented API for constructing and using
 * fat binary objects by programmers.  Starting with CUDA 4.0, fat binary
 * objects can be constructed by providing the <i>-fatbin option</i> to \b nvcc.
 * More information can be found in the \b nvcc document.
 *
 * \param module   - Returned module
 * \param fatCubin - Fat binary to load
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
 * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleUnload
 *)
function cuModuleLoadFatBinary(module : PCUmodule; const fatCubin : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Unloads a module
 *
 * Unloads a module \p hmod from the current context.
 *
 * \param hmod - Module to unload
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_destroy_ub
 *
 * \sa ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary
 *)
function cuModuleUnload(hmod : CUmodule):CUresult;  WINAPI; external libcuda;

(**
 * CUDA Lazy Loading status
 *)
type
  CUmoduleLoadingMode_enum = (
    CU_MODULE_EAGER_LOADING = $1, (**< Lazy Kernel Loading is not enabled *)
    CU_MODULE_LAZY_LOADING  = $2 (**< Lazy Kernel Loading is enabled *)
  );
  PCUmoduleLoadingMode = ^CUmoduleLoadingMode ;
  CUmoduleLoadingMode = CUmoduleLoadingMode_enum;

(**
 * \brief Query lazy loading mode
 *
 * Returns lazy loading mode
 * Module loading mode is controlled by CUDA_MODULE_LOADING env variable
 *
 * \param mode      - Returns the lazy loading mode
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \notefnerr
 *
 * \sa
 * ::cuModuleLoad,
 *)
function cuModuleGetLoadingMode(mode : PCUmoduleLoadingMode):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a function handle
 *
 * Returns in \p *hfunc the handle of the function of name \p name located in
 * module \p hmod. If no function of that name exists, ::cuModuleGetFunction()
 * returns ::CUDA_ERROR_NOT_FOUND.
 *
 * \param hfunc - Returned function handle
 * \param hmod  - Module to retrieve function from
 * \param name  - Name of function to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload
 *)
function cuModuleGetFunction(hfunc : PCUfunction;hmod : CUmodule; const name : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a global pointer from a module
 *
 * Returns in \p *dptr and \p *bytes the base pointer and size of the
 * global of name \p name located in module \p hmod. If no variable of that name
 * exists, ::cuModuleGetGlobal() returns ::CUDA_ERROR_NOT_FOUND.
 * One of the parameters \p dptr or \p bytes (not both) can be NULL in which
 * case it is ignored.
 *
 * \param dptr  - Returned global device pointer
 * \param bytes - Returned global size in bytes
 * \param hmod  - Module to retrieve global from
 * \param name  - Name of global to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuModuleGetFunction,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload,
 * ::cudaGetSymbolAddress,
 * ::cudaGetSymbolSize
 *)
function cuModuleGetGlobal(dptr : PCUdeviceptr; bytes : Psize_t;hmod : CUmodule; const name : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a pending JIT linker invocation.
 *
 * If the call is successful, the caller owns the returned CUlinkState, which
 * should eventually be destroyed with ::cuLinkDestroy.  The
 * device code machine size (32 or 64 bit) will match the calling application.
 *
 * Both linker and compiler options may be specified.  Compiler options will
 * be applied to inputs to this linker action which must be compiled from PTX.
 * The options ::CU_JIT_WALL_TIME,
 * ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, and ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
 * will accumulate data until the CUlinkState is destroyed.
 *
 * \p optionValues must remain valid for the life of the CUlinkState if output
 * options are used.  No other references to inputs are maintained after this
 * call returns.
 *
 * \note For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted
 *
 * \param numOptions   Size of options arrays
 * \param options      Array of linker and compiler options
 * \param optionValues Array of option values, each cast to void *
 * \param stateOut     On success, this will contain a CUlinkState to specify
 *                     and complete this action
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::cuLinkAddData,
 * ::cuLinkAddFile,
 * ::cuLinkComplete,
 * ::cuLinkDestroy
 *)
function cuLinkCreate(numOptions : longword; options : PCUjit_option; optionValues : Ppointer; stateOut : PCUlinkState):CUresult;  WINAPI; external libcuda;

(**
 * \brief Add an input to a pending linker invocation
 *
 * Ownership of \p data is retained by the caller.  No reference is retained to any
 * inputs after this call returns.
 *
 * This method accepts only compiler options, which are used if the data must
 * be compiled from PTX, and does not accept any of
 * ::CU_JIT_WALL_TIME, ::CU_JIT_INFO_LOG_BUFFER, ::CU_JIT_ERROR_LOG_BUFFER,
 * ::CU_JIT_TARGET_FROM_CUCONTEXT, or ::CU_JIT_TARGET.
 *
 * \note For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted
 *
 * \param state        A pending linker action.
 * \param type         The type of the input data.
 * \param data         The input data.  PTX must be NULL-terminated.
 * \param size         The length of the input data.
 * \param name         An optional name for this input in log messages.
 * \param numOptions   Size of options.
 * \param options      Options to be applied only for this input (overrides options from ::cuLinkCreate).
 * \param optionValues Array of option values, each cast to void *.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_IMAGE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU
 *
 * \sa ::cuLinkCreate,
 * ::cuLinkAddFile,
 * ::cuLinkComplete,
 * ::cuLinkDestroy
 *)
function cuLinkAddData(state : CUlinkState; &type : CUjitInputType; data : Pointer; size : size_t; const name : PAnsiChar;
    numOptions : longword; options : PCUjit_option; optionValues : Ppointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Add a file input to a pending linker invocation
 *
 * No reference is retained to any inputs after this call returns.
 *
 * This method accepts only compiler options, which are used if the input
 * must be compiled from PTX, and does not accept any of
 * ::CU_JIT_WALL_TIME, ::CU_JIT_INFO_LOG_BUFFER, ::CU_JIT_ERROR_LOG_BUFFER,
 * ::CU_JIT_TARGET_FROM_CUCONTEXT, or ::CU_JIT_TARGET.
 *
 * This method is equivalent to invoking ::cuLinkAddData on the contents
 * of the file.
 *
 * \note For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted
 *
 * \param state        A pending linker action
 * \param type         The type of the input data
 * \param path         Path to the input file
 * \param numOptions   Size of options
 * \param options      Options to be applied only for this input (overrides options from ::cuLinkCreate)
 * \param optionValues Array of option values, each cast to void *
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_FILE_NOT_FOUND
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_IMAGE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU
 *
 * \sa ::cuLinkCreate,
 * ::cuLinkAddData,
 * ::cuLinkComplete,
 * ::cuLinkDestroy
 *)
function cuLinkAddFile(state : CUlinkState; &type : CUjitInputType; const path : PAnsiChar;
    numOptions : longword; options : PCUjit_option; optionValues : Ppointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Complete a pending linker invocation
 *
 * Completes the pending linker action and returns the cubin image for the linked
 * device code, which can be used with ::cuModuleLoadData.  The cubin is owned by
 * \p state, so it should be loaded before \p state is destroyed via ::cuLinkDestroy.
 * This call does not destroy \p state.
 *
 * \param state    A pending linker invocation
 * \param cubinOut On success, this will point to the output image
 * \param sizeOut  Optional parameter to receive the size of the generated image
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuLinkCreate,
 * ::cuLinkAddData,
 * ::cuLinkAddFile,
 * ::cuLinkDestroy,
 * ::cuModuleLoadData
 *)
function cuLinkComplete(state : CUlinkState; cubinOut : Ppointer; sizeOut : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys state for a JIT linker invocation.
 *
 * \param state State object for the linker invocation
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE
 *
 * \sa ::cuLinkCreate
 *)
function cuLinkDestroy(state : CUlinkState):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_MODULE *)

(**
 * \defgroup CUDA_MODULE_DEPRECATED Module Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated module management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated module management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns a handle to a texture reference
 *
 * \deprecated
 *
 * Returns in \p *pTexRef the handle of the texture reference of name \p name
 * in the module \p hmod. If no texture reference of that name exists,
 * ::cuModuleGetTexRef() returns ::CUDA_ERROR_NOT_FOUND. This texture reference
 * handle should not be destroyed, since it will be destroyed when the module
 * is unloaded.
 *
 * \param pTexRef  - Returned texture reference
 * \param hmod     - Module to retrieve texture reference from
 * \param name     - Name of texture reference to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa
 * ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetSurfRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload
 *)
function cuModuleGetTexRef(pTexRef : PCUtexref; hmod : CUmodule; const name : PAnsiChar):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Returns a handle to a surface reference
 *
 * \deprecated
 *
 * Returns in \p *pSurfRef the handle of the surface reference of name \p name
 * in the module \p hmod. If no surface reference of that name exists,
 * ::cuModuleGetSurfRef() returns ::CUDA_ERROR_NOT_FOUND.
 *
 * \param pSurfRef  - Returned surface reference
 * \param hmod     - Module to retrieve surface reference from
 * \param name     - Name of surface reference to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa
 * ::cuModuleGetFunction,
 * ::cuModuleGetGlobal,
 * ::cuModuleGetTexRef,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx,
 * ::cuModuleLoadFatBinary,
 * ::cuModuleUnload
 *)
function cuModuleGetSurfRef(pSurfRef : PCUsurfref; hmod : CUmodule; const name : PAnsiChar):CUresult;  WINAPI; external libcuda; deprecated;

(** @} *) (* END CUDA_MODULE_DEPRECATED *)

(**
 * \defgroup CUDA_LIBRARY Library Management
 *
 * ___MANBRIEF___ library management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the library management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Load a library with specified code and options
 *
 * Takes a pointer \p code and loads the corresponding library \p library into
 * all contexts existent at the time of the call and future contexts at the time
 * of creation until the library is unloaded with ::cuLibraryUnload().
 *
 * The pointer may be obtained by mapping a \e cubin or \e PTX or \e fatbin file,
 * passing a \e cubin or \e PTX or \e fatbin file as a NULL-terminated text string, or
 * incorporating a \e cubin or \e fatbin object into the executable resources and
 * using operating system calls such as Windows \c FindResource() to obtain the pointer.
 *
 * Options are passed as an array via \p jitOptions and any corresponding parameters are passed in
 * \p jitOptionsValues. The number of total JIT options is supplied via \p numJitOptions.
 * Any outputs will be returned via \p jitOptionsValues.
 *
 * Library load options are passed as an array via \p libraryOptions and any corresponding parameters are passed in
 * \p libraryOptionValues. The number of total library load options is supplied via \p numLibraryOptions.
 *
 * \param library             - Returned library
 * \param code                - Code to load
 * \param jitOptions          - Options for JIT
 * \param jitOptionsValues    - Option values for JIT
 * \param numJitOptions       - Number of options
 * \param libraryOptions      - Options for loading
 * \param libraryOptionValues - Option values for loading
 * \param numLibraryOptions   - Number of options for loading
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
 * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 *
 * \sa ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx
 *)
function cuLibraryLoadData(&library : PCUlibrary; const code : Pointer;
                                   jitOptions : PCUjit_option; jitOptionsValues : Ppointer; numJitOptions : longword;
                                   libraryOptions : PCUlibraryOption; libraryOptionValues : Ppointer; numLibraryOptions : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Load a library with specified file and options
 *
 * Takes a filename \p fileName and loads the corresponding library \p library into
 * all contexts existent at the time of the call and future contexts at the time of
 * creation until the library is unloaded with ::cuLibraryUnload().
 *
 * The file should be a \e cubin file as output by \b nvcc, or a \e PTX file either
 * as output by \b nvcc or handwritten, or a \e fatbin file as output by \b nvcc
 * from toolchain 4.0 or later.
 *
 * Options are passed as an array via \p jitOptions and any corresponding parameters are
 * passed in \p jitOptionsValues. The number of total options is supplied via \p numJitOptions.
 * Any outputs will be returned via \p jitOptionsValues.
 *
 * Library load options are passed as an array via \p libraryOptions and any corresponding parameters are passed in
 * \p libraryOptionValues. The number of total library load options is supplied via \p numLibraryOptions.
 *
 * \param library             - Returned library
 * \param fileName            - File to load from
 * \param jitOptions          - Options for JIT
 * \param jitOptionsValues    - Option values for JIT
 * \param numJitOptions       - Number of options
 * \param libraryOptions      - Options for loading
 * \param libraryOptionValues - Option values for loading
 * \param numLibraryOptions   - Number of options for loading
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_PTX,
 * ::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NO_BINARY_FOR_GPU,
 * ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryUnload,
 * ::cuModuleLoad,
 * ::cuModuleLoadData,
 * ::cuModuleLoadDataEx
 *)
function cuLibraryLoadFromFile(&library : PCUlibrary; const fileName : PAnsiChar;
                                       jitOptions : PCUjit_option; jitOptionsValues : Ppointer; numJitOptions : longword;
                                       libraryOptions : PCUlibraryOption; libraryOptionValues : Ppointer; numLibraryOptions : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Unloads a library
 *
 * Unloads the library specified with \p library
 *
 * \param library - Library to unload
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuModuleUnload
 *)
function cuLibraryUnload(&library : CUlibrary):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a kernel handle
 *
 * Returns in \p pKernel the handle of the kernel with name \p name located in library \p library.
 * If kernel handle is not found, the call returns ::CUDA_ERROR_NOT_FOUND.
 *
 * \param pKernel - Returned kernel handle
 * \param library - Library to retrieve kernel from
 * \param name - Name of kernel to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_FOUND,
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuKernelGetFunction,
 * ::cuLibraryGetModule,
 * ::cuModuleGetFunction
 *)
function cuLibraryGetKernel(pKernel : PCUkernel; &library : CUlibrary; const name : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a module handle
 *
 * Returns in \p pMod the module handle associated with the current context located in
 * library \p library. If module handle is not found, the call returns ::CUDA_ERROR_NOT_FOUND.
 *
 * \param pMod - Returned module handle
 * \param library - Library to retrieve module from
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuModuleGetFunction
 *)
function cuLibraryGetModule(pMod : PCUmodule; &library : CUlibrary):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a function handle
 *
 * Returns in \p pFunc the handle of the function for the requested kernel \p kernel and
 * the current context. If function handle is not found, the call returns ::CUDA_ERROR_NOT_FOUND.
 *
 * \param pFunc - Returned function handle
 * \param kernel - Kernel to retrieve function for the requested context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuLibraryGetKernel,
 * ::cuLibraryGetModule,
 * ::cuModuleGetFunction
 *)
function cuKernelGetFunction(pFunc : PCUfunction; kernel : CUkernel):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a global device pointer
 *
 * Returns in \p *dptr and \p *bytes the base pointer and size of the global with
 * name \p name for the requested library \p library and the current context.
 * If no global for the requested name \p name exists, the call returns ::CUDA_ERROR_NOT_FOUND.
 * One of the parameters \p dptr or \p bytes (not both) can be NULL in which
 * case it is ignored.
 *
 * \param dptr - Returned global device pointer for the requested context
 * \param bytes - Returned global size in bytes
 * \param library - Library to retrieve global from
 * \param name - Name of global to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuLibraryGetModule,
 * cuModuleGetGlobal
 *)
function cuLibraryGetGlobal(dptr : PCUdeviceptr; bytes : Psize_t; &library : CUlibrary; const name : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a pointer to managed memory
 *
 * Returns in \p *dptr and \p *bytes the base pointer and size of the managed memory with
 * name \p name for the requested library \p library. If no managed memory with the
 * requested name \p name exists, the call returns ::CUDA_ERROR_NOT_FOUND. One of the parameters
 * \p dptr or \p bytes (not both) can be NULL in which case it is ignored.
 * Note that managed memory for library \p library is shared across devices and is registered
 * when the library is loaded into atleast one context.
 *
 * \note The API requires a CUDA context to be present and initialized on at least one device.
 * If no context is present, the call returns ::CUDA_ERROR_NOT_FOUND.
 *
 * \param dptr - Returned pointer to the managed memory
 * \param bytes - Returned memory size in bytes
 * \param library - Library to retrieve managed memory from
 * \param name - Name of managed memory to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_FOUND,
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 *)
function cuLibraryGetManaged(dptr : PCUdeviceptr; bytes : Psize_t; &library : CUlibrary; const name : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a pointer to a unified function
 *
 * Returns in \p *fptr the function pointer to a unified function denoted by \p symbol.
 * If no unified function with name \p symbol exists, the call returns ::CUDA_ERROR_NOT_FOUND.
 * If there is no device with attribute ::CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS present in the system,
 * the call may return ::CUDA_ERROR_NOT_FOUND.
 *
 * \param fptr - Returned pointer to a unified function
 * \param library - Library to retrieve function pointer memory from
 * \param symbol - Name of function pointer to retrieve
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_FOUND,
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 *)
function cuLibraryGetUnifiedFunction(fptr : Ppointer; &library : CUlibrary; const symbol : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns information about a kernel
 *
 * Returns in \p *pi the integer value of the attribute \p attrib for the kernel
 * \p kernel for the requested device \p dev. The supported attributes are:
 * - ::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: The maximum number of threads
 *   per block, beyond which a launch of the kernel would fail. This number
 *   depends on both the kernel and the requested device.
 * - ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: The size in bytes of
 *   statically-allocated shared memory per block required by this kernel.
 *   This does not include dynamically-allocated shared memory requested by
 *   the user at runtime.
 * - ::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: The size in bytes of user-allocated
 *   constant memory required by this kernel.
 * - ::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: The size in bytes of local memory
 *   used by each thread of this kernel.
 * - ::CU_FUNC_ATTRIBUTE_NUM_REGS: The number of registers used by each thread
 *   of this kernel.
 * - ::CU_FUNC_ATTRIBUTE_PTX_VERSION: The PTX virtual architecture version for
 *   which the kernel was compiled. This value is the major PTX version * 10
 *   + the minor PTX version, so a PTX version 1.3 function would return the
 *   value 13. Note that this may return the undefined value of 0 for cubins
 *   compiled prior to CUDA 3.0.
 * - ::CU_FUNC_ATTRIBUTE_BINARY_VERSION: The binary architecture version for
 *   which the kernel was compiled. This value is the major binary
 *   version * 10 + the minor binary version, so a binary version 1.3 function
 *   would return the value 13. Note that this will return a value of 10 for
 *   legacy cubins that do not have a properly-encoded binary architecture
 *   version.
 * - ::CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the kernel has
 *   been compiled with user specified option "-Xptxas --dlcm=ca" set.
 * - ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: The maximum size in bytes of
 *   dynamically-allocated shared memory.
 * - ::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: Preferred shared memory-L1
 *   cache split ratio in percent of total shared memory.
 * - ::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: If this attribute is set, the
 *   kernel must launch with a valid cluster size specified.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: The required cluster width in
 *   blocks.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: The required cluster height in
 *   blocks.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: The required cluster depth in
 *   blocks.
 * - ::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: Indicates whether
 *   the function can be launched with non-portable cluster size. 1 is allowed,
 *   0 is disallowed. A non-portable cluster size may only function on the
 *   specific SKUs the program is tested on. The launch might fail if the
 *   program is run on a different hardware platform. CUDA API provides
 *   cudaOccupancyMaxActiveClusters to assist with checking whether the desired
 *   size can be launched on the current device. A portable cluster size is
 *   guaranteed to be functional on all compute capabilities higher than the
 *   target compute capability. The portable cluster size for sm_90 is 8 blocks
 *   per cluster. This value may increase for future compute capabilities. The
 *   specific hardware unit may support higher cluster sizes that’s not
 *   guaranteed to be portable.
 * - ::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block
 *   scheduling policy of a function. The value type is CUclusterSchedulingPolicy.
 *
 * \note If another thread is trying to set the same attribute on the same device using
 * ::cuKernelSetAttribute() simultaneously, the attribute query will give the old or new
 * value depending on the interleavings chosen by the OS scheduler and memory consistency.
 *
 * \param pi     - Returned attribute value
 * \param attrib - Attribute requested
 * \param kernel  - Kernel to query attribute of
 * \param dev - Device to query attribute of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuKernelSetAttribute,
 * ::cuLibraryGetKernel,
 * ::cuLaunchKernel,
 * ::cuKernelGetFunction,
 * ::cuLibraryGetModule,
 * ::cuModuleGetFunction,
 * ::cuFuncGetAttribute
 *)
function cuKernelGetAttribute(pi : Plongint; attrib : CUfunction_attribute; kernel : CUkernel; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets information about a kernel
 *
 * This call sets the value of a specified attribute \p attrib on the kernel \p kernel
 * for the requested device \p dev to an integer value specified by \p val.
 * This function returns CUDA_SUCCESS if the new value of the attribute could be
 * successfully set. If the set fails, this call will return an error.
 * Not all attributes can have values set. Attempting to set a value on a read-only
 * attribute will result in an error (CUDA_ERROR_INVALID_VALUE)
 *
 * Note that attributes set using ::cuFuncSetAttribute() will override the attribute
 * set by this API irrespective of whether the call to ::cuFuncSetAttribute() is made
 * before or after this API call. However, ::cuKernelGetAttribute() will always
 * return the attribute value set by this API.
 *
 * Supported attributes are:
 * - ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This is the maximum size in bytes of
 *   dynamically-allocated shared memory. The value should contain the requested
 *   maximum size of dynamically-allocated shared memory. The sum of this value and
 *   the function attribute ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the
 *   device attribute ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
 *   The maximal size of requestable dynamic shared memory may differ by GPU
 *   architecture.
 * - ::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On devices where the L1
 *   cache and shared memory use the same hardware resources, this sets the shared memory
 *   carveout preference, in percent of the total shared memory.
 *   See ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
 *   This is only a hint, and the driver can choose a different ratio if required to execute the function.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: The required cluster width in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: The required cluster height in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: The required cluster depth in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
 * - ::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block
 *   scheduling policy of a function. The value type is CUclusterSchedulingPolicy.
 *
 * \note The API has stricter locking requirements in comparison to its legacy counterpart
 * ::cuFuncSetAttribute() due to device-wide semantics. If multiple threads are trying to
 * set the same attribute on the same device simultaneously, the attribute setting will depend
 * on the interleavings chosen by the OS scheduler and memory consistency.
 *
 * \param attrib - Attribute requested
 * \param val - Value to set
 * \param kernel  - Kernel to set attribute of
 * \param dev - Device to set attribute of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuKernelGetAttribute,
 * ::cuLibraryGetKernel,
 * ::cuLaunchKernel,
 * ::cuKernelGetFunction,
 * ::cuLibraryGetModule,
 * ::cuModuleGetFunction,
 * ::cuFuncSetAttribute
 *)
function cuKernelSetAttribute(attrib : CUfunction_attribute; val : longint; kernel : CUkernel; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the preferred cache configuration for a device kernel.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p config the preferred cache configuration for
 * the device kernel \p kernel on the requested device \p dev. This is only a preference.
 * The driver will use the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute \p kernel.  Any context-wide preference
 * set via ::cuCtxSetCacheConfig() will be overridden by this per-kernel
 * setting.
 *
 * Note that attributes set using ::cuFuncSetCacheConfig() will override the attribute
 * set by this API irrespective of whether the call to ::cuFuncSetCacheConfig() is made
 * before or after this API call.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 *
 * The supported cache configurations are:
 * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \note The API has stricter locking requirements in comparison to its legacy counterpart
 * ::cuFuncSetCacheConfig() due to device-wide semantics. If multiple threads are trying to
 * set a config on the same device simultaneously, the cache config setting will depend
 * on the interleavings chosen by the OS scheduler and memory consistency.
 *
 * \param kernel  - Kernel to configure cache for
 * \param config - Requested cache configuration
 * \param dev - Device to set attribute of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuLibraryLoadData,
 * ::cuLibraryLoadFromFile,
 * ::cuLibraryUnload,
 * ::cuLibraryGetKernel,
 * ::cuKernelGetFunction,
 * ::cuLibraryGetModule,
 * ::cuModuleGetFunction,
 * ::cuFuncSetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuLaunchKernel
 *)
function cuKernelSetCacheConfig(kernel : CUkernel; config : CUfunc_cache; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_LIBRARY *)

(**
 * \defgroup CUDA_MEM Memory Management
 *
 * ___MANBRIEF___ memory management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Gets free and total memory
 *
 * Returns in \p *total the total amount of memory available to the the current context.
 * Returns in \p *free the amount of memory on the device that is free according to the OS.
 * CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free.
 * In a multi-tenet situation, free estimate returned is prone to race condition where
 * a new allocation/free done by a different process or a different thread in the same
 * process between the time when free memory was estimated and reported, will result in
 * deviation in free value reported and actual free memory.
 *
 * The integrated GPU on Tegra shares memory with CPU and other component
 * of the SoC. The free and total values returned by the API excludes
 * the SWAP memory space maintained by the OS on some platforms.
 * The OS may move some of the memory pages into swap area as the GPU or
 * CPU allocate or access memory. See Tegra app note on how to calculate
 * total and free memory on Tegra.
 *
 * \param free  - Returned free memory in bytes
 * \param total - Returned total memory in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemGetInfo
 *)
function cuMemGetInfo(free : Psize_t; total : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates device memory
 *
 * Allocates \p bytesize bytes of linear memory on the device and returns in
 * \p *dptr a pointer to the allocated memory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p bytesize
 * is 0, ::cuMemAlloc() returns ::CUDA_ERROR_INVALID_VALUE.
 *
 * \param dptr     - Returned device pointer
 * \param bytesize - Requested allocation size in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMalloc
 *)
function cuMemAlloc(dptr : PCUdeviceptr; bytesize : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates pitched device memory
 *
 * Allocates at least \p WidthInBytes * \p Height bytes of linear memory on
 * the device and returns in \p *dptr a pointer to the allocated memory. The
 * function may pad the allocation to ensure that corresponding pointers in
 * any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. \p ElementSizeBytes
 * specifies the size of the largest reads and writes that will be performed
 * on the memory range. \p ElementSizeBytes may be 4, 8 or 16 (since coalesced
 * memory transactions are not possible on other data sizes). If
 * \p ElementSizeBytes is smaller than the actual read/write size of a kernel,
 * the kernel will run correctly, but possibly at reduced speed. The pitch
 * returned in \p *pPitch by ::cuMemAllocPitch() is the width in bytes of the
 * allocation. The intended usage of pitch is as a separate parameter of the
 * allocation, used to compute addresses within the 2D array. Given the row
 * and column of an array element of type \b T, the address is computed as:
 * \code
   T* pElement = (T* )((PAnsiChar  )BaseAddress + Row * Pitch) + Column;
 * \endcode
 *
 * The pitch returned by ::cuMemAllocPitch() is guaranteed to work with
 * ::cuMemcpy2D() under all circumstances. For allocations of 2D arrays, it is
 * recommended that programmers consider performing pitch allocations using
 * ::cuMemAllocPitch(). Due to alignment restrictions in the hardware, this is
 * especially true if the application will be performing 2D memory copies
 * between different regions of device memory (whether linear memory or CUDA
 * arrays).
 *
 * The byte alignment of the pitch returned by ::cuMemAllocPitch() is guaranteed
 * to match or exceed the alignment requirement for texture binding with
 * ::cuTexRefSetAddress2D().
 *
 * \param dptr             - Returned device pointer
 * \param pPitch           - Returned pitch of allocation in bytes
 * \param WidthInBytes     - Requested allocation width in bytes
 * \param Height           - Requested allocation height in rows
 * \param ElementSizeBytes - Size of largest reads/writes for range
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMallocPitch
 *)
function cuMemAllocPitch(dptr : PCUdeviceptr; pPitch : Psize_t; WidthInBytes : size_t; Height : size_t; ElementSizeBytes : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Frees device memory
 *
 * Frees the memory space pointed to by \p dptr, which must have been returned
 * by a previous call to one of the following memory allocation APIs - ::cuMemAlloc(), 
 * ::cuMemAllocPitch(), ::cuMemAllocManaged(), ::cuMemAllocAsync(), ::cuMemAllocFromPoolAsync()
 *
 * Note - This API will not perform any implict synchronization when the pointer was allocated with
 * ::cuMemAllocAsync or ::cuMemAllocFromPoolAsync. Callers must ensure that all accesses to the
 * pointer have completed before invoking ::cuMemFree. For best performance and memory reuse, users
 * should use ::cuMemFreeAsync to free memory allocated via the stream ordered memory allocator.
 * 
 * \param dptr - Pointer to memory to free
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemAllocManaged, ::cuMemAllocAsync, ::cuMemAllocFromPoolAsync, 
 * ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned, ::cuMemcpy3D, ::cuMemcpy3DAsync,
 * ::cuMemcpyAtoA, ::cuMemcpyAtoD, ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA,
 * ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync, ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA,
 * ::cuMemcpyHtoAAsync, ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc, ::cuMemFreeAsync,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaFree
 *)
function cuMemFree(dptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get information on memory allocations
 *
 * Returns the base address in \p *pbase and size in \p *psize of the
 * allocation by ::cuMemAlloc() or ::cuMemAllocPitch() that contains the input
 * pointer \p dptr. Both parameters \p pbase and \p psize are optional. If one
 * of them is NULL, it is ignored.
 *
 * \param pbase - Returned base address
 * \param psize - Returned size of device memory allocation
 * \param dptr  - Device pointer to query
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32
 *)
function cuMemGetAddressRange(pbase : PCUdeviceptr; psize : Psize_t; dptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates page-locked host memory
 *
 * Allocates \p bytesize bytes of host memory that is page-locked and
 * accessible to the device. The driver tracks the virtual memory ranges
 * allocated with this function and automatically accelerates calls to
 * functions such as ::cuMemcpy(). Since the memory can be accessed directly by
 * the device, it can be read or written with much higher bandwidth than
 * pageable memory obtained with functions such as ::malloc(). Allocating
 * excessive amounts of memory with ::cuMemAllocHost() may degrade system
 * performance, since it reduces the amount of memory available to the system
 * for paging. As a result, this function is best used sparingly to allocate
 * staging areas for data exchange between host and device.
 *
 * Note all host memory allocated using ::cuMemHostAlloc() will automatically
 * be immediately accessible to all contexts on all devices which support unified
 * addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
 * The device pointer that may be used to access this host memory from those
 * contexts is always equal to the returned host pointer \p *pp.
 * See \ref CUDA_UNIFIED for additional details.
 *
 * \param pp       - Returned host pointer to page-locked memory
 * \param bytesize - Requested allocation size in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMallocHost
 *)
function cuMemAllocHost(pp : Ppointer; bytesize : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Frees page-locked host memory
 *
 * Frees the memory space pointed to by \p p, which must have been returned by
 * a previous call to ::cuMemAllocHost().
 *
 * \param p - Pointer to memory to free
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaFreeHost
 *)
function cuMemFreeHost(p : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates page-locked host memory
 *
 * Allocates \p bytesize bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::cuMemcpyHtoD(). Since the memory can be accessed directly by the device,
 * it can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between
 * host and device.
 *
 * The \p Flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::CU_MEMHOSTALLOC_PORTABLE: The memory returned by this call will be
 *   considered as pinned memory by all CUDA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::CU_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the CUDA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::cuMemHostGetDevicePointer().
 *
 * - ::CU_MEMHOSTALLOC_WRITECOMBINED: Allocates the memory as write-combined
 *   (WC). WC memory can be transferred across the PCI Express bus more
 *   quickly on some system configurations, but cannot be read efficiently by
 *   most CPUs. WC memory is a good option for buffers that will be written by
 *   the CPU and read by the GPU via mapped pinned memory or host->device
 *   transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * The ::CU_MEMHOSTALLOC_DEVICEMAP flag may be specified on CUDA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::cuMemHostGetDevicePointer() because the memory may be mapped into
 * other CUDA contexts via the ::CU_MEMHOSTALLOC_PORTABLE flag.
 *
 * The memory allocated by this function must be freed with ::cuMemFreeHost().
 *
 * Note all host memory allocated using ::cuMemHostAlloc() will automatically
 * be immediately accessible to all contexts on all devices which support unified
 * addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
 * Unless the flag ::CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer
 * that may be used to access this host memory from those contexts is always equal
 * to the returned host pointer \p *pp.  If the flag ::CU_MEMHOSTALLOC_WRITECOMBINED
 * is specified, then the function ::cuMemHostGetDevicePointer() must be used
 * to query the device pointer, even if the context supports unified addressing.
 * See \ref CUDA_UNIFIED for additional details.
 *
 * \param pp       - Returned host pointer to page-locked memory
 * \param bytesize - Requested allocation size in bytes
 * \param Flags    - Flags for allocation request
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaHostAlloc
 *)
function cuMemHostAlloc(pp : Ppointer; bytesize : size_t; Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Passes back device pointer of mapped pinned memory
 *
 * Passes back the device pointer \p pdptr corresponding to the mapped, pinned
 * host buffer \p p allocated by ::cuMemHostAlloc.
 *
 * ::cuMemHostGetDevicePointer() will fail if the ::CU_MEMHOSTALLOC_DEVICEMAP
 * flag was not specified at the time the memory was allocated, or if the
 * function is called on a GPU that does not support mapped pinned memory.
 *
 * For devices that have a non-zero value for the device attribute
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the memory
 * can also be accessed from the device using the host pointer \p p.
 * The device pointer returned by ::cuMemHostGetDevicePointer() may or may not
 * match the original host pointer \p p and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::cuMemHostGetDevicePointer()
 * will match the original pointer \p p. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::cuMemHostGetDevicePointer() will not match the original host pointer \p p,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only one of the two pointers and not both.
 *
 * \p Flags provides for future releases. For now, it must be set to 0.
 *
 * \param pdptr - Returned device pointer
 * \param p     - Host pointer
 * \param Flags - Options (must be 0)
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaHostGetDevicePointer
 *)
function cuMemHostGetDevicePointer(pdptr : PCUdeviceptr; p : Pointer; Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Passes back flags that were used for a pinned allocation
 *
 * Passes back the flags \p pFlags that were specified when allocating
 * the pinned host buffer \p p allocated by ::cuMemHostAlloc.
 *
 * ::cuMemHostGetFlags() will fail if the pointer does not reside in
 * an allocation performed by ::cuMemAllocHost() or ::cuMemHostAlloc().
 *
 * \param pFlags - Returned flags word
 * \param p     - Host pointer
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cuMemAllocHost,
 * ::cuMemHostAlloc,
 * ::cudaHostGetFlags
 *)
function cuMemHostGetFlags(pFlags : Plongword; p : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p bytesize bytes of managed memory on the device and returns in
 * \p *dptr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::CUDA_ERROR_NOT_SUPPORTED is returned. Support
 * for managed memory can be queried using the device attribute
 * ::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p bytesize
 * is 0, ::cuMemAllocManaged returns ::CUDA_ERROR_INVALID_VALUE. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST. If
 * ::CU_MEM_ATTACH_GLOBAL is specified, then this memory is accessible from
 * any stream on any device. If ::CU_MEM_ATTACH_HOST is specified, then the
 * allocation should not be accessed from devices that have a zero value for the
 * device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS; an explicit call to
 * ::cuStreamAttachMemAsync will be required to enable access on such devices.
 *
 * If the association is later changed via ::cuStreamAttachMemAsync to
 * a single stream, the default association as specified during ::cuMemAllocManaged
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::cuMemAllocManaged should be released with ::cuMemFree.
 *
 * Device memory oversubscription is possible for GPUs that have a non-zero value for the
 * device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Managed memory on
 * such GPUs may be evicted from device memory to host memory at any time by the Unified
 * Memory driver in order to make room for other allocations.
 *
 * In a multi-GPU system where all GPUs have a non-zero value for the device attribute
 * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, managed memory may not be populated when this
 * API returns and instead may be populated on access. In such systems, managed memory can
 * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
 * maintain data locality and prevent excessive page faults to the extent possible. The application
 * can also guide the driver about memory usage patterns via ::cuMemAdvise. The application
 * can also explicitly migrate memory to a desired processor's memory via
 * ::cuMemPrefetchAsync.
 *
 * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
 * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS and all the GPUs have peer-to-peer support
 * with each other, the physical storage for managed memory is created on the GPU which is active
 * at the time ::cuMemAllocManaged is called. All other GPUs will reference the data at reduced
 * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
 * memory among such GPUs.
 *
 * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
 * where the value of the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
 * is zero for at least one of those GPUs, the location chosen for physical storage of managed
 * memory is system-dependent.
 * - On Linux, the location chosen will be device memory as long as the current set of active
 * contexts are on devices that either have peer-to-peer support with each other or have a
 * non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
 * If there is an active context on a GPU that does not have a non-zero value for that device
 * attribute and it does not have peer-to-peer support with the other devices that have active
 * contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
 * Note that this means that managed memory that is located in device memory is migrated to
 * host memory if a new context is created on a GPU that doesn't have a non-zero value for
 * the device attribute and does not support peer-to-peer with at least one of the other devices
 * that has an active context. This in turn implies that context creation may fail if there is
 * insufficient host memory to migrate all managed allocations.
 * - On Windows, the physical storage is always created in 'zero-copy' or host memory.
 * All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
 * circumstances, use of the environment variable CUDA_VISIBLE_DEVICES is recommended to
 * restrict CUDA to only use those GPUs that have peer-to-peer support.
 * Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a
 * non-zero value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all contexts created in
 * that process on devices that support managed memory have to be peer-to-peer compatible
 * with each other. Context creation will fail if a context is created on a device that
 * supports managed memory and is not peer-to-peer compatible with any of the other
 * managed memory supporting devices on which contexts were previously created, even if
 * those contexts have been destroyed. These environment variables are described
 * in the CUDA programming guide under the "CUDA environment variables" section.
 * - On ARM, managed memory is not available on discrete gpu with Drive PX-2.
 *
 * \param dptr     - Returned device pointer
 * \param bytesize - Requested allocation size in bytes
 * \param flags    - Must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cuDeviceGetAttribute, ::cuStreamAttachMemAsync,
 * ::cudaMallocManaged
 *)
function cuMemAllocManaged(dptr : PCUdeviceptr; bytesize : size_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device handle given a PCI bus ID string.
 *
 * \param dev      - Returned device handle
 *
 * \param pciBusId - String in one of the following forms:
 * [domain]:[bus]:[device].[function]
 * [domain]:[bus]:[device]
 * [bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGet,
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetPCIBusId,
 * ::cudaDeviceGetByPCIBusId
 *)
function cuDeviceGetByPCIBusId(dev : PCUdevice; const pciBusId : PAnsiChar):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a PCI Bus Id string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p pciBusId. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param pciBusId - Returned identifier string for the device in the following format
 * [domain]:[bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
 * pciBusId should be large enough to store 13 characters including the NULL-terminator.
 *
 * \param len      - Maximum length of string to store in \p name
 *
 * \param dev      - Device to get identifier string for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceGet,
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetByPCIBusId,
 * ::cudaDeviceGetPCIBusId
 *)
function cuDeviceGetPCIBusId(pciBusId : PAnsiChar; len : longint; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets an interprocess handle for a previously allocated event
 *
 * Takes as input a previously allocated event. This event must have been
 * created with the ::CU_EVENT_INTERPROCESS and ::CU_EVENT_DISABLE_TIMING
 * flags set. This opaque handle may be copied into other processes and
 * opened with ::cuIpcOpenEventHandle to allow efficient hardware
 * synchronization between GPU work in different processes.
 *
 * After the event has been opened in the importing process,
 * ::cuEventRecord, ::cuEventSynchronize, ::cuStreamWaitEvent and
 * ::cuEventQuery may be used in either process. Performing operations
 * on the imported event after the exported event has been freed
 * with ::cuEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is restricted to GPUs in TCC mode
 * Users can test their device for IPC functionality by calling
 * ::cuapiDeviceGetAttribute with ::CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
 *
 * \param pHandle - Pointer to a user allocated CUipcEventHandle
 *                    in which to return the opaque event handle
 * \param event   - Event allocated with ::CU_EVENT_INTERPROCESS and
 *                    ::CU_EVENT_DISABLE_TIMING flags.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_MAP_FAILED,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuEventCreate,
 * ::cuEventDestroy,
 * ::cuEventSynchronize,
 * ::cuEventQuery,
 * ::cuStreamWaitEvent,
 * ::cuIpcOpenEventHandle,
 * ::cuIpcGetMemHandle,
 * ::cuIpcOpenMemHandle,
 * ::cuIpcCloseMemHandle,
 * ::cudaIpcGetEventHandle
 *)
function cuIpcGetEventHandle(pHandle : PCUipcEventHandle; event : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Opens an interprocess event handle for use in the current process
 *
 * Opens an interprocess event handle exported from another process with
 * ::cuIpcGetEventHandle. This function returns a ::CUevent that behaves like
 * a locally created event with the ::CU_EVENT_DISABLE_TIMING flag specified.
 * This event must be freed with ::cuEventDestroy.
 *
 * Performing operations on the imported event after the exported event has
 * been freed with ::cuEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is restricted to GPUs in TCC mode
 * Users can test their device for IPC functionality by calling
 * ::cuapiDeviceGetAttribute with ::CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
 *
 * \param phEvent - Returns the imported event
 * \param handle  - Interprocess handle to open
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_MAP_FAILED,
 * ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuEventCreate,
 * ::cuEventDestroy,
 * ::cuEventSynchronize,
 * ::cuEventQuery,
 * ::cuStreamWaitEvent,
 * ::cuIpcGetEventHandle,
 * ::cuIpcGetMemHandle,
 * ::cuIpcOpenMemHandle,
 * ::cuIpcCloseMemHandle,
 * ::cudaIpcOpenEventHandle
 *)
function cuIpcOpenEventHandle(phEvent : PCUevent; handle : CUipcEventHandle):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets an interprocess memory handle for an existing device memory
 * allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created
 * with ::cuMemAlloc and exports it for use in another process. This is a
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects.
 *
 * If a region of memory is freed with ::cuMemFree and a subsequent call
 * to ::cuMemAlloc returns memory with the same device address,
 * ::cuIpcGetMemHandle will return a unique handle for the
 * new memory.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is restricted to GPUs in TCC mode
 * Users can test their device for IPC functionality by calling
 * ::cuapiDeviceGetAttribute with ::CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
 *
 * \param pHandle - Pointer to user allocated ::CUipcMemHandle to return
 *                    the handle in.
 * \param dptr    - Base pointer to previously allocated device memory
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_MAP_FAILED,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuMemAlloc,
 * ::cuMemFree,
 * ::cuIpcGetEventHandle,
 * ::cuIpcOpenEventHandle,
 * ::cuIpcOpenMemHandle,
 * ::cuIpcCloseMemHandle,
 * ::cudaIpcGetMemHandle
 *)
function cuIpcGetMemHandle(pHandle : PCUipcMemHandle; dptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Opens an interprocess memory handle exported from another process
 * and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with ::cuIpcGetMemHandle into
 * the current device address space. For contexts on different devices
 * ::cuIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called ::cuCtxEnablePeerAccess. This behavior is
 * controlled by the ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag.
 * ::cuDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open ::CUipcMemHandles are restricted in the following way.
 * ::CUipcMemHandles from each ::CUdevice in a given process may only be opened
 * by one ::CUcontext per ::CUdevice per other process.
 *
 * If the memory handle has already been opened by the current context, the
 * reference count on the handle is incremented by 1 and the existing device pointer
 * is returned.
 *
 * Memory returned from ::cuIpcOpenMemHandle must be freed with
 * ::cuIpcCloseMemHandle.
 *
 * Calling ::cuMemFree on an exported memory region before calling
 * ::cuIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is restricted to GPUs in TCC mode
 * Users can test their device for IPC functionality by calling
 * ::cuapiDeviceGetAttribute with ::CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
 *
 * \param pdptr  - Returned device pointer
 * \param handle - ::CUipcMemHandle to open
 * \param Flags  - Flags for this operation. Must be specified as ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_MAP_FAILED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_TOO_MANY_PEERS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \note No guarantees are made about the address returned in \p *pdptr.
 * In particular, multiple processes may not receive the same address for the same \p handle.
 *
 * \sa
 * ::cuMemAlloc,
 * ::cuMemFree,
 * ::cuIpcGetEventHandle,
 * ::cuIpcOpenEventHandle,
 * ::cuIpcGetMemHandle,
 * ::cuIpcCloseMemHandle,
 * ::cuCtxEnablePeerAccess,
 * ::cuDeviceCanAccessPeer,
 * ::cudaIpcOpenMemHandle
 *)
function cuIpcOpenMemHandle(pdptr : PCUdeviceptr; handle : CUipcMemHandle; Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Attempts to close memory mapped with ::cuIpcOpenMemHandle
 *
 * Decrements the reference count of the memory returned by ::cuIpcOpenMemHandle by 1.
 * When the reference count reaches 0, this API unmaps the memory. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is restricted to GPUs in TCC mode
 * Users can test their device for IPC functionality by calling
 * ::cuapiDeviceGetAttribute with ::CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
 *
 * \param dptr - Device pointer returned by ::cuIpcOpenMemHandle
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_MAP_FAILED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \sa
 * ::cuMemAlloc,
 * ::cuMemFree,
 * ::cuIpcGetEventHandle,
 * ::cuIpcOpenEventHandle,
 * ::cuIpcGetMemHandle,
 * ::cuIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle
 *)
function cuIpcCloseMemHandle(dptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Registers an existing host memory range for use by CUDA
 *
 * Page-locks the memory range specified by \p p and \p bytesize and maps it
 * for the device(s) as specified by \p Flags. This memory range also is added
 * to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
 * calls to functions such as ::cuMemcpyHtoD(). Since the memory can be accessed
 * directly by the device, it can be read or written with much higher bandwidth
 * than pageable memory that has not been registered.  Page-locking excessive
 * amounts of memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to register staging areas for data exchange between
 * host and device.
 *
 * On systems where ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES 
 * is true, ::cuMemHostRegister will not page-lock the memory range specified 
 * by \p ptr but only populate unpopulated pages.
 *
 * The \p Flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::CU_MEMHOSTREGISTER_PORTABLE: The memory returned by this call will be
 *   considered as pinned memory by all CUDA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::CU_MEMHOSTREGISTER_DEVICEMAP: Maps the allocation into the CUDA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::cuMemHostGetDevicePointer().
 *
 * - ::CU_MEMHOSTREGISTER_IOMEMORY: The pointer is treated as pointing to some
 *   I/O memory space, e.g. the PCI Express resource of a 3rd party device.
 *
 * - ::CU_MEMHOSTREGISTER_READ_ONLY: The pointer is treated as pointing to memory
 *   that is considered read-only by the device.  On platforms without
 *   ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, this flag is
 *   required in order to register memory mapped to the CPU as read-only.  Support
 *   for the use of this flag can be queried from the device attribute
 *   ::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED.  Using this flag with
 *   a current context associated with a device that does not have this attribute
 *   set will cause ::cuMemHostRegister to error with CUDA_ERROR_NOT_SUPPORTED.
 *
 * All of these flags are orthogonal to one another: a developer may page-lock
 * memory that is portable or mapped with no restrictions.
 *
 * The ::CU_MEMHOSTREGISTER_DEVICEMAP flag may be specified on CUDA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::cuMemHostGetDevicePointer() because the memory may be mapped into
 * other CUDA contexts via the ::CU_MEMHOSTREGISTER_PORTABLE flag.
 *
 * For devices that have a non-zero value for the device attribute
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the memory
 * can also be accessed from the device using the host pointer \p p.
 * The device pointer returned by ::cuMemHostGetDevicePointer() may or may not
 * match the original host pointer \p ptr and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::cuMemHostGetDevicePointer()
 * will match the original pointer \p ptr. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::cuMemHostGetDevicePointer() will not match the original host pointer \p ptr,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * The memory page-locked by this function must be unregistered with
 * ::cuMemHostUnregister().
 *
 * \param p        - Host pointer to memory to page-lock
 * \param bytesize - Size in bytes of the address range to page-lock
 * \param Flags    - Flags for allocation request
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa
 * ::cuMemHostUnregister,
 * ::cuMemHostGetFlags,
 * ::cuMemHostGetDevicePointer,
 * ::cudaHostRegister
 *)
function cuMemHostRegister(p : Pointer; bytesize : size_t; Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Unregisters a memory range that was registered with cuMemHostRegister.
 *
 * Unmaps the memory range whose base address is specified by \p p, and makes
 * it pageable again.
 *
 * The base address must be the same one specified to ::cuMemHostRegister().
 *
 * \param p - Host pointer to memory to unregister
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
 * \notefnerr
 *
 * \sa
 * ::cuMemHostRegister,
 * ::cudaHostUnregister
 *)
function cuMemHostUnregister(p : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory
 *
 * Copies data between two pointers.
 * \p dst and \p src are base pointers of the destination and source, respectively.
 * \p ByteCount specifies the number of bytes to copy.
 * Note that this function infers the type of the transfer (host to host, host to
 *   device, device to device, or device to host) from the pointer values.  This
 *   function is only allowed in contexts which support unified addressing.
 *
 * \param dst - Destination unified virtual address space pointer
 * \param src - Source unified virtual address space pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy,
 * ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol
 *)
function cuMemcpy(dst : CUdeviceptr; src : CUdeviceptr; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies device memory between two contexts
 *
 * Copies from device memory in one context to device memory in another
 * context. \p dstDevice is the base device pointer of the destination memory
 * and \p dstContext is the destination context.  \p srcDevice is the base
 * device pointer of the source memory and \p srcContext is the source pointer.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice  - Destination device pointer
 * \param dstContext - Destination context
 * \param srcDevice  - Source device pointer
 * \param srcContext - Source context
 * \param ByteCount  - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuMemcpyDtoD, ::cuMemcpy3DPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
 * ::cuMemcpy3DPeerAsync,
 * ::cudaMemcpyPeer
 *)
function cuMemcpyPeer(dstDevice : CUdeviceptr; dstContext : CUcontext; srcDevice : CUdeviceptr; srcContext : CUcontext; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Host to Device
 *
 * Copies from host memory to device memory. \p dstDevice and \p srcHost are
 * the base addresses of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy,
 * ::cudaMemcpyToSymbol
 *)
function cuMemcpyHtoD(dstDevice : CUdeviceptr; const srcHost : Pointer; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Device to Host
 *
 * Copies from device to host memory. \p dstHost and \p srcDevice specify the
 * base pointers of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination host pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy,
 * ::cudaMemcpyFromSymbol
 *)
function cuMemcpyDtoH(dstHost : Pointer; srcDevice : CUdeviceptr; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Device to Device
 *
 * Copies from device memory to device memory. \p dstDevice and \p srcDevice
 * are the base pointers of the destination and source, respectively.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy,
 * ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol
 *)
function cuMemcpyDtoD(dstDevice : CUdeviceptr; srcDevice : CUdeviceptr; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Device to Array
 *
 * Copies from device memory to a 1D CUDA array. \p dstArray and \p dstOffset
 * specify the CUDA array handle and starting index of the destination data.
 * \p srcDevice specifies the base pointer of the source. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpyToArray
 *)
function cuMemcpyDtoA(dstArray : CUarray; dstOffset : size_t; srcDevice : CUdeviceptr; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Array to Device
 *
 * Copies from one 1D CUDA array to device memory. \p dstDevice specifies the
 * base pointer of the destination and must be naturally aligned with the CUDA
 * array elements. \p srcArray and \p srcOffset specify the CUDA array handle
 * and the offset in bytes into the array where the copy is to begin.
 * \p ByteCount specifies the number of bytes to copy and must be evenly
 * divisible by the array element size.
 *
 * \param dstDevice - Destination device pointer
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpyFromArray
 *)
function cuMemcpyAtoD(dstDevice : CUdeviceptr; srcArray : CUarray; srcOffset : size_t; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Host to Array
 *
 * Copies from host memory to a 1D CUDA array. \p dstArray and \p dstOffset
 * specify the CUDA array handle and starting offset in bytes of the destination
 * data.  \p pSrc specifies the base address of the source. \p ByteCount specifies
 * the number of bytes to copy.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpyToArray
 *)
function cuMemcpyHtoA(dstArray : CUarray; dstOffset : size_t; const srcHost : Pointer; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Array to Host
 *
 * Copies from one 1D CUDA array to host memory. \p dstHost specifies the base
 * pointer of the destination. \p srcArray and \p srcOffset specify the CUDA
 * array handle and starting offset in bytes of the source data.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination device pointer
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpyFromArray
 *)
function cuMemcpyAtoH(dstHost : Pointer; srcArray : CUarray; srcOffset : size_t; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Array to Array
 *
 * Copies from one 1D CUDA array to another. \p dstArray and \p srcArray
 * specify the handles of the destination and source CUDA arrays for the copy,
 * respectively. \p dstOffset and \p srcOffset specify the destination and
 * source offsets in bytes into the CUDA arrays. \p ByteCount is the number of
 * bytes to be copied. The size of the elements in the CUDA arrays need not be
 * the same format, but the elements must be the same size; and count must be
 * evenly divisible by that size.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpyArrayToArray
 *)
function cuMemcpyAtoA(dstArray : CUarray; dstOffset : size_t; srcArray : CUarray; srcOffset : size_t; ByteCount : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory for 2D arrays
 *
 * Perform a 2D memory copy according to the parameters specified in \p pCopy.
 * The ::CUDA_MEMCPY2D structure is defined as:
 *
 * \code
     CUDA_MEMCPY2D_st=packed record
      longint srcXInBytes, srcY;
      CUmemorytype srcMemoryType;
          const Pointer srcHost;
          CUdeviceptr srcDevice;
          CUarray srcArray;
          longint srcPitch;

      longint dstXInBytes, dstY;
      CUmemorytype dstMemoryType;
          Pointer dstHost;
          CUdeviceptr dstDevice;
          CUarray dstArray;
          longint dstPitch;

      longint WidthInBytes;
      longint Height;
     end;
  CUDA_MEMCPY2D = CUDA_MEMCPY2D_st;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::CUmemorytype_enum is defined as:
 *
 * \code
     CUmemorytype_enum = (
      CU_MEMORYTYPE_HOST = $01,
      CU_MEMORYTYPE_DEVICE = $02,
      CU_MEMORYTYPE_ARRAY = $03,
      CU_MEMORYTYPE_UNIFIED = $04
     );
  CUmemorytype = CUmemorytype_enum;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::srcArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
 * specify the (host) base address of the source data and the bytes per row to
 * apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
 * specify the (device) base address of the source data and the bytes per row
 * to apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
 * ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::dstArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
 * ignored.
 *
 * - ::srcXInBytes and ::srcY specify the base address of the source data for
 *   the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  pointer  Start = (pointer  )((PAnsiChar  )srcHost+srcY*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::dstXInBytes and ::dstY specify the base address of the destination data
 *   for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  pointer  dstStart = (pointer  )((PAnsiChar  )dstHost+dstY*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
 *   the 2D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 *
 * \par
 * ::cuMemcpy2D() returns an error if any pitch is greater than the maximum
 * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH). ::cuMemAllocPitch() passes back
 * pitches that always work with ::cuMemcpy2D(). On intra-device memory copies
 * (device to device, CUDA array to device, CUDA array to CUDA array),
 * ::cuMemcpy2D() may fail for pitches not computed by ::cuMemAllocPitch().
 * ::cuMemcpy2DUnaligned() does not have this restriction, but may run
 * significantly slower in the cases where ::cuMemcpy2D() would have returned
 * an error code.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray,
 * ::cudaMemcpy2DFromArray
 *)
function cuMemcpy2D(const pCopy : PCUDA_MEMCPY2D):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory for 2D arrays
 *
 * Perform a 2D memory copy according to the parameters specified in \p pCopy.
 * The ::CUDA_MEMCPY2D structure is defined as:
 *
 * \code
     CUDA_MEMCPY2D_st=packed record
      longint srcXInBytes, srcY;
      CUmemorytype srcMemoryType;
      const Pointer srcHost;
      CUdeviceptr srcDevice;
      CUarray srcArray;
      longint srcPitch;
      longint dstXInBytes, dstY;
      CUmemorytype dstMemoryType;
      Pointer dstHost;
      CUdeviceptr dstDevice;
      CUarray dstArray;
      longint dstPitch;
      longint WidthInBytes;
      longint Height;
     end;
  CUDA_MEMCPY2D = CUDA_MEMCPY2D_st;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::CUmemorytype_enum is defined as:
 *
 * \code
     CUmemorytype_enum = (
      CU_MEMORYTYPE_HOST = $01,
      CU_MEMORYTYPE_DEVICE = $02,
      CU_MEMORYTYPE_ARRAY = $03,
      CU_MEMORYTYPE_UNIFIED = $04
     );
  CUmemorytype = CUmemorytype_enum;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::srcArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
 * specify the (host) base address of the source data and the bytes per row to
 * apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
 * specify the (device) base address of the source data and the bytes per row
 * to apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
 * ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::dstArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
 * ignored.
 *
 * - ::srcXInBytes and ::srcY specify the base address of the source data for
 *   the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  pointer  Start = (pointer  )((PAnsiChar  )srcHost+srcY*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::dstXInBytes and ::dstY specify the base address of the destination data
 *   for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  pointer  dstStart = (pointer  )((PAnsiChar  )dstHost+dstY*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
 *   the 2D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 *
 * \par
 * ::cuMemcpy2D() returns an error if any pitch is greater than the maximum
 * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH). ::cuMemAllocPitch() passes back
 * pitches that always work with ::cuMemcpy2D(). On intra-device memory copies
 * (device to device, CUDA array to device, CUDA array to CUDA array),
 * ::cuMemcpy2D() may fail for pitches not computed by ::cuMemAllocPitch().
 * ::cuMemcpy2DUnaligned() does not have this restriction, but may run
 * significantly slower in the cases where ::cuMemcpy2D() would have returned
 * an error code.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray,
 * ::cudaMemcpy2DFromArray
 *)
function cuMemcpy2DUnaligned(const pCopy : PCUDA_MEMCPY2D):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory for 3D arrays
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy. The ::CUDA_MEMCPY3D structure is defined as:
 *
 * \code
          CUDA_MEMCPY3D_st=packed record

            longint srcXInBytes, srcY, srcZ;
            longint srcLOD;
            CUmemorytype srcMemoryType;
                const Pointer srcHost;
                CUdeviceptr srcDevice;
                CUarray srcArray;
                longint srcPitch;  // ignored when src is array
                longint srcHeight; // ignored when src is array; may be 0 if Depth==1

            longint dstXInBytes, dstY, dstZ;
            longint dstLOD;
            CUmemorytype dstMemoryType;
                Pointer dstHost;
                CUdeviceptr dstDevice;
                CUarray dstArray;
                longint dstPitch;  // ignored when dst is array
                longint dstHeight; // ignored when dst is array; may be 0 if Depth==1

            longint WidthInBytes;
            longint Height;
            longint Depth;
          end;
  CUDA_MEMCPY3D = CUDA_MEMCPY3D_st;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::CUmemorytype_enum is defined as:
 *
 * \code
     CUmemorytype_enum = (
      CU_MEMORYTYPE_HOST = $01,
      CU_MEMORYTYPE_DEVICE = $02,
      CU_MEMORYTYPE_ARRAY = $03,
      CU_MEMORYTYPE_UNIFIED = $04
     );
  CUmemorytype = CUmemorytype_enum;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::srcArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost, ::srcPitch and
 * ::srcHeight specify the (host) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice, ::srcPitch and
 * ::srcHeight specify the (device) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice, ::srcPitch and
 * ::srcHeight are ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::dstArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data, the bytes per row,
 * and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data, the bytes per
 * row, and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice, ::dstPitch and
 * ::dstHeight are ignored.
 *
 * - ::srcXInBytes, ::srcY and ::srcZ specify the base address of the source
 *   data for the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  pointer  Start = (pointer  )((PAnsiChar  )srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - dstXInBytes, ::dstY and ::dstZ specify the base address of the
 *   destination data for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  pointer  dstStart = (pointer  )((PAnsiChar  )dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes, ::Height and ::Depth specify the width (in bytes), height
 *   and depth of the 3D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcHeight must be greater than or equal to ::Height +
 *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
 *
 * \par
 * ::cuMemcpy3D() returns an error if any pitch is greater than the maximum
 * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH).
 *
 * The ::srcLOD and ::dstLOD members of the ::CUDA_MEMCPY3D structure must be
 * set to 0.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMemcpy3D
 *)
function cuMemcpy3D(const pCopy : PCUDA_MEMCPY3D):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory between contexts
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy.  See the definition of the ::CUDA_MEMCPY3D_PEER structure
 * for documentation of its parameters.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
 * ::cuMemcpy3DPeerAsync,
 * ::cudaMemcpy3DPeer
 *)
function cuMemcpy3DPeer(const pCopy : PCUDA_MEMCPY3D_PEER):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory asynchronously
 *
 * Copies data between two pointers.
 * \p dst and \p src are base pointers of the destination and source, respectively.
 * \p ByteCount specifies the number of bytes to copy.
 * Note that this function infers the type of the transfer (host to host, host to
 *   device, device to device, or device to host) from the pointer values.  This
 *   function is only allowed in contexts which support unified addressing.
 *
 * \param dst       - Destination unified virtual address space pointer
 * \param src       - Source unified virtual address space pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpyAsync,
 * ::cudaMemcpyToSymbolAsync,
 * ::cudaMemcpyFromSymbolAsync
 *)
function cuMemcpyAsync(dst : CUdeviceptr; src : CUdeviceptr; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies device memory between two contexts asynchronously.
 *
 * Copies from device memory in one context to device memory in another
 * context. \p dstDevice is the base device pointer of the destination memory
 * and \p dstContext is the destination context.  \p srcDevice is the base
 * device pointer of the source memory and \p srcContext is the source pointer.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice  - Destination device pointer
 * \param dstContext - Destination context
 * \param srcDevice  - Source device pointer
 * \param srcContext - Source context
 * \param ByteCount  - Size of memory copy in bytes
 * \param hStream    - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpy3DPeer, ::cuMemcpyDtoDAsync,
 * ::cuMemcpy3DPeerAsync,
 * ::cudaMemcpyPeerAsync
 *)
function cuMemcpyPeerAsync(dstDevice : CUdeviceptr; dstContext : CUcontext; srcDevice : CUdeviceptr; srcContext : CUcontext; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Host to Device
 *
 * Copies from host memory to device memory. \p dstDevice and \p srcHost are
 * the base addresses of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpyAsync,
 * ::cudaMemcpyToSymbolAsync
 *)
function cuMemcpyHtoDAsync(dstDevice : CUdeviceptr; const srcHost : Pointer; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Device to Host
 *
 * Copies from device to host memory. \p dstHost and \p srcDevice specify the
 * base pointers of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination host pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpyAsync,
 * ::cudaMemcpyFromSymbolAsync
 *)
function cuMemcpyDtoHAsync(dstHost : Pointer; srcDevice : CUdeviceptr; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Device to Device
 *
 * Copies from device memory to device memory. \p dstDevice and \p srcDevice
 * are the base pointers of the destination and source, respectively.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpyAsync,
 * ::cudaMemcpyToSymbolAsync,
 * ::cudaMemcpyFromSymbolAsync
 *)
function cuMemcpyDtoDAsync(dstDevice : CUdeviceptr; srcDevice : CUdeviceptr; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Host to Array
 *
 * Copies from host memory to a 1D CUDA array. \p dstArray and \p dstOffset
 * specify the CUDA array handle and starting offset in bytes of the
 * destination data. \p srcHost specifies the base address of the source.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpyToArrayAsync
 *)
function cuMemcpyHtoAAsync(dstArray : CUarray; dstOffset : size_t; const srcHost : Pointer; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory from Array to Host
 *
 * Copies from one 1D CUDA array to host memory. \p dstHost specifies the base
 * pointer of the destination. \p srcArray and \p srcOffset specify the CUDA
 * array handle and starting offset in bytes of the source data.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination pointer
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_memcpy
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpyFromArrayAsync
 *)
function cuMemcpyAtoHAsync(dstHost : Pointer; srcArray : CUarray; srcOffset : size_t; ByteCount : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory for 2D arrays
 *
 * Perform a 2D memory copy according to the parameters specified in \p pCopy.
 * The ::CUDA_MEMCPY2D structure is defined as:
 *
 * \code
     CUDA_MEMCPY2D_st=packed record
      longint srcXInBytes, srcY;
      CUmemorytype srcMemoryType;
      const Pointer srcHost;
      CUdeviceptr srcDevice;
      CUarray srcArray;
      longint srcPitch;
      longint dstXInBytes, dstY;
      CUmemorytype dstMemoryType;
      Pointer dstHost;
      CUdeviceptr dstDevice;
      CUarray dstArray;
      longint dstPitch;
      longint WidthInBytes;
      longint Height;
     end;
  CUDA_MEMCPY2D = CUDA_MEMCPY2D_st;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::CUmemorytype_enum is defined as:
 *
 * \code
     CUmemorytype_enum = (
      CU_MEMORYTYPE_HOST = $01,
      CU_MEMORYTYPE_DEVICE = $02,
      CU_MEMORYTYPE_ARRAY = $03,
      CU_MEMORYTYPE_UNIFIED = $04
     );
  CUmemorytype = CUmemorytype_enum;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
 * specify the (host) base address of the source data and the bytes per row to
 * apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::srcArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
 * specify the (device) base address of the source data and the bytes per row
 * to apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
 * ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::dstArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
 * ignored.
 *
 * - ::srcXInBytes and ::srcY specify the base address of the source data for
 *   the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  pointer  Start = (pointer  )((PAnsiChar  )srcHost+srcY*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::dstXInBytes and ::dstY specify the base address of the destination data
 *   for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  pointer  dstStart = (pointer  )((PAnsiChar  )dstHost+dstY*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
 *   the 2D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcHeight must be greater than or equal to ::Height +
 *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
 *
 * \par
 * ::cuMemcpy2DAsync() returns an error if any pitch is greater than the maximum
 * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH). ::cuMemAllocPitch() passes back
 * pitches that always work with ::cuMemcpy2D(). On intra-device memory copies
 * (device to device, CUDA array to device, CUDA array to CUDA array),
 * ::cuMemcpy2DAsync() may fail for pitches not computed by ::cuMemAllocPitch().
 *
 * \param pCopy   - Parameters for the memory copy
 * \param hStream - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync
 *)
function cuMemcpy2DAsync(const pCopy : PCUDA_MEMCPY2D; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory for 3D arrays
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy. The ::CUDA_MEMCPY3D structure is defined as:
 *
 * \code
          CUDA_MEMCPY3D_st=packed record

            longint srcXInBytes, srcY, srcZ;
            longint srcLOD;
            CUmemorytype srcMemoryType;
                const Pointer srcHost;
                CUdeviceptr srcDevice;
                CUarray srcArray;
                longint srcPitch;  // ignored when src is array
                longint srcHeight; // ignored when src is array; may be 0 if Depth==1

            longint dstXInBytes, dstY, dstZ;
            longint dstLOD;
            CUmemorytype dstMemoryType;
                Pointer dstHost;
                CUdeviceptr dstDevice;
                CUarray dstArray;
                longint dstPitch;  // ignored when dst is array
                longint dstHeight; // ignored when dst is array; may be 0 if Depth==1

            longint WidthInBytes;
            longint Height;
            longint Depth;
          end;
  CUDA_MEMCPY3D = CUDA_MEMCPY3D_st;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::CUmemorytype_enum is defined as:
 *
 * \code
     CUmemorytype_enum = (
      CU_MEMORYTYPE_HOST = $01,
      CU_MEMORYTYPE_DEVICE = $02,
      CU_MEMORYTYPE_ARRAY = $03,
      CU_MEMORYTYPE_UNIFIED = $04
     );
  CUmemorytype = CUmemorytype_enum;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::srcArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_HOST, ::srcHost, ::srcPitch and
 * ::srcHeight specify the (host) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_DEVICE, ::srcDevice, ::srcPitch and
 * ::srcHeight specify the (device) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::CU_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice, ::srcPitch and
 * ::srcHeight are ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data
 *   and the bytes per row to apply.  ::dstArray is ignored.
 * This value may be used only if unified addressing is supported in the calling
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data, the bytes per row,
 * and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data, the bytes per
 * row, and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::CU_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice, ::dstPitch and
 * ::dstHeight are ignored.
 *
 * - ::srcXInBytes, ::srcY and ::srcZ specify the base address of the source
 *   data for the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  pointer  Start = (pointer  )((PAnsiChar  )srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - dstXInBytes, ::dstY and ::dstZ specify the base address of the
 *   destination data for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  pointer  dstStart = (pointer  )((PAnsiChar  )dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  CUdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For CUDA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes, ::Height and ::Depth specify the width (in bytes), height
 *   and depth of the 3D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcHeight must be greater than or equal to ::Height +
 *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
 *
 * \par
 * ::cuMemcpy3DAsync() returns an error if any pitch is greater than the maximum
 * allowed (::CU_DEVICE_ATTRIBUTE_MAX_PITCH).
 *
 * The ::srcLOD and ::dstLOD members of the ::CUDA_MEMCPY3D structure must be
 * set to 0.
 *
 * \param pCopy - Parameters for the memory copy
 * \param hStream - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemcpy3DAsync
 *)
function cuMemcpy3DAsync(const pCopy : PCUDA_MEMCPY3D; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies memory between contexts asynchronously.
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy.  See the definition of the ::CUDA_MEMCPY3D_PEER structure
 * for documentation of its parameters.
 *
 * \param pCopy - Parameters for the memory copy
 * \param hStream - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
 * ::cuMemcpy3DPeerAsync,
 * ::cudaMemcpy3DPeerAsync
 *)
function cuMemcpy3DPeerAsync(const pCopy : PCUDA_MEMCPY3D_PEER; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Initializes device memory
 *
 * Sets the memory range of \p N 8-bit values to the specified value
 * \p uc.
 *
 * \param dstDevice - Destination device pointer
 * \param uc        - Value to set
 * \param N         - Number of elements
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset
 *)
function cuMemsetD8(dstDevice : CUdeviceptr; uc : byte; N : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Initializes device memory
 *
 * Sets the memory range of \p N 16-bit values to the specified value
 * \p us. The \p dstDevice pointer must be two byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param us        - Value to set
 * \param N         - Number of elements
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset
 *)
function cuMemsetD16(dstDevice : CUdeviceptr; us : word; N : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Initializes device memory
 *
 * Sets the memory range of \p N 32-bit values to the specified value
 * \p ui. The \p dstDevice pointer must be four byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param ui        - Value to set
 * \param N         - Number of elements
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32Async,
 * ::cudaMemset
 *)
function cuMemsetD32(dstDevice : CUdeviceptr; ui : longword; N : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Initializes device memory
 *
 * Sets the 2D memory range of \p Width 8-bit values to the specified value
 * \p uc. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::cuMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer(Unused if \p Height is 1)
 * \param uc        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset2D
 *)
function cuMemsetD2D8(dstDevice : CUdeviceptr; dstPitch : size_t; uc : byte; Width : size_t; Height : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Initializes device memory
 *
 * Sets the 2D memory range of \p Width 16-bit values to the specified value
 * \p us. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be two byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::cuMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer(Unused if \p Height is 1)
 * \param us        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset2D
 *)
function cuMemsetD2D16(dstDevice : CUdeviceptr; dstPitch : size_t; us : word; Width : size_t; Height : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Initializes device memory
 *
 * Sets the 2D memory range of \p Width 32-bit values to the specified value
 * \p ui. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be four byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::cuMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer(Unused if \p Height is 1)
 * \param ui        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset2D
 *)
function cuMemsetD2D32(dstDevice : CUdeviceptr; dstPitch : size_t; ui : longword; Width : size_t; Height : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets device memory
 *
 * Sets the memory range of \p N 8-bit values to the specified value
 * \p uc.
 *
 * \param dstDevice - Destination device pointer
 * \param uc        - Value to set
 * \param N         - Number of elements
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemsetAsync
 *)
function cuMemsetD8Async(dstDevice : CUdeviceptr; uc : byte; N : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets device memory
 *
 * Sets the memory range of \p N 16-bit values to the specified value
 * \p us. The \p dstDevice pointer must be two byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param us        - Value to set
 * \param N         - Number of elements
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemsetAsync
 *)
function cuMemsetD16Async(dstDevice : CUdeviceptr; us : word; N : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets device memory
 *
 * Sets the memory range of \p N 32-bit values to the specified value
 * \p ui. The \p dstDevice pointer must be four byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param ui        - Value to set
 * \param N         - Number of elements
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async, ::cuMemsetD32,
 * ::cudaMemsetAsync
 *)
function cuMemsetD32Async(dstDevice : CUdeviceptr; ui : longword; N : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets device memory
 *
 * Sets the 2D memory range of \p Width 8-bit values to the specified value
 * \p uc. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::cuMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer(Unused if \p Height is 1)
 * \param uc        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset2DAsync
 *)
function cuMemsetD2D8Async(dstDevice : CUdeviceptr; dstPitch : size_t; uc : byte; Width : size_t; Height : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets device memory
 *
 * Sets the 2D memory range of \p Width 16-bit values to the specified value
 * \p us. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be two byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::cuMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer(Unused if \p Height is 1)
 * \param us        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset2DAsync
 *)
function cuMemsetD2D16Async(dstDevice : CUdeviceptr; dstPitch : size_t; us : word; Width : size_t; Height : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets device memory
 *
 * Sets the 2D memory range of \p Width 32-bit values to the specified value
 * \p ui. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be four byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::cuMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer(Unused if \p Height is 1)
 * \param ui        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 * \param hStream   - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32,
 * ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
 * ::cuMemsetD32, ::cuMemsetD32Async,
 * ::cudaMemset2DAsync
 *)
function cuMemsetD2D32Async(dstDevice : CUdeviceptr; dstPitch : size_t; ui : longword; Width : size_t; Height : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a 1D or 2D CUDA array
 *
 * Creates a CUDA array according to the ::CUDA_ARRAY_DESCRIPTOR structure
 * \p pAllocateArray and returns a handle to the new CUDA array in \p *pHandle.
 * The ::CUDA_ARRAY_DESCRIPTOR is defined as:
 *
 * \code
    typedef struct {
        longint Width;
        longint Height;
        CUarray_format Format;
        longint NumChannels;
    } CUDA_ARRAY_DESCRIPTOR;
 * \endcode
 * where:
 *
 * - \p Width, and \p Height are the width, and height of the CUDA array (in
 * elements); the CUDA array is one-dimensional if height is 0, two-dimensional
 * otherwise;
 * - ::Format specifies the format of the elements; ::CUarray_format is
 * defined as:
 * \code
      CUarray_format_enum = (
        CU_AD_FORMAT_UNSIGNED_INT8 = $01,
        CU_AD_FORMAT_UNSIGNED_INT16 = $02,
        CU_AD_FORMAT_UNSIGNED_INT32 = $03,
        CU_AD_FORMAT_SIGNED_INT8 = $08,
        CU_AD_FORMAT_SIGNED_INT16 = $09,
        CU_AD_FORMAT_SIGNED_INT32 = $0a,
        CU_AD_FORMAT_HALF = $10,
        CU_AD_FORMAT_FLOAT = $20
      );
  CUarray_format = CUarray_format_enum;
 *  \endcode
 * - \p NumChannels specifies the number of packed components per CUDA array
 * element; it may be 1, 2, or 4;
 *
 * Here are examples of CUDA array descriptions:
 *
 * Description for a CUDA array of 2048 floats:
 * \code
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 2048;
    desc.Height = 1;
 * \endcode
 *
 * Description for a 64 x 64 CUDA array of floats:
 * \code
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 64;
    desc.Height = 64;
 * \endcode
 *
 * Description for a \p width x \p height CUDA array of 64-bit, 4x16-bit
 * float16's:
 * \code
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_HALF;
    desc.NumChannels = 4;
    desc.Width = width;
    desc.Height = height;
 * \endcode
 *
 * Description for a \p width x \p height CUDA array of 16-bit elements, each
 * of which is two 8-bit unsigned chars:
 * \code
    CUDA_ARRAY_DESCRIPTOR arrayDesc;
    desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.NumChannels = 2;
    desc.Width = width;
    desc.Height = height;
 * \endcode
 *
 * \param pHandle        - Returned array
 * \param pAllocateArray - Array descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMallocArray
 *)
function cuArrayCreate(pHandle : PCUarray;const pAllocateArray : PCUDA_ARRAY_DESCRIPTOR):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get a 1D or 2D CUDA array descriptor
 *
 * Returns in \p *pArrayDescriptor a descriptor containing information on the
 * format and dimensions of the CUDA array \p hArray. It is useful for
 * subroutines that have been passed a CUDA array, but need to know the CUDA
 * array parameters for validation or other purposes.
 *
 * \param pArrayDescriptor - Returned array descriptor
 * \param hArray           - Array to get descriptor of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaArrayGetInfo
 *)
function cuArrayGetDescriptor(pArrayDescriptor : PCUDA_ARRAY_DESCRIPTOR; hArray : CUarray):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the layout properties of a sparse CUDA array
 *
 * Returns the layout properties of a sparse CUDA array in \p sparseProperties
 * If the CUDA array is not allocated with flag ::CUDA_ARRAY3D_SPARSE 
 * ::CUDA_ERROR_INVALID_VALUE will be returned.
 *
 * If the returned value in ::CUDA_ARRAY_SPARSE_PROPERTIES::flags contains ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL,
 * then ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize represents the total size of the array. Otherwise, it will be zero.
 * Also, the returned value in ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is always zero.
 * Note that the \p array must have been allocated using ::cuArrayCreate or ::cuArray3DCreate. For CUDA arrays obtained
 * using ::cuMipmappedArrayGetLevel, ::CUDA_ERROR_INVALID_VALUE will be returned. Instead, ::cuMipmappedArrayGetSparseProperties 
 * must be used to obtain the sparse properties of the entire CUDA mipmapped array to which \p array belongs to.
 *
 * \return
 * ::CUDA_SUCCESS
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \param[out] sparseProperties - Pointer to ::CUDA_ARRAY_SPARSE_PROPERTIES
 * \param[in] array - CUDA array to get the sparse properties of
 * \sa ::cuMipmappedArrayGetSparseProperties, ::cuMemMapArrayAsync
 *)
function cuArrayGetSparseProperties(sparseProperties : PCUDA_ARRAY_SPARSE_PROPERTIES; &array : CUarray):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the layout properties of a sparse CUDA mipmapped array
 *
 * Returns the sparse array layout properties in \p sparseProperties
 * If the CUDA mipmapped array is not allocated with flag ::CUDA_ARRAY3D_SPARSE 
 * ::CUDA_ERROR_INVALID_VALUE will be returned.
 *
 * For non-layered CUDA mipmapped arrays, ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize returns the
 * size of the mip tail region. The mip tail region includes all mip levels whose width, height or depth
 * is less than that of the tile.
 * For layered CUDA mipmapped arrays, if ::CUDA_ARRAY_SPARSE_PROPERTIES::flags contains ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL,
 * then ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize specifies the size of the mip tail of all layers combined. 
 * Otherwise, ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize specifies mip tail size per layer.
 * The returned value of ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is valid only if ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize is non-zero.
 *
 * \return
 * ::CUDA_SUCCESS
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \param[out] sparseProperties - Pointer to ::CUDA_ARRAY_SPARSE_PROPERTIES
 * \param[in] mipmap - CUDA mipmapped array to get the sparse properties of
 * \sa ::cuArrayGetSparseProperties, ::cuMemMapArrayAsync
 *)
function cuMipmappedArrayGetSparseProperties(sparseProperties : PCUDA_ARRAY_SPARSE_PROPERTIES; mipmap : CUmipmappedArray):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the memory requirements of a CUDA array
 *
 * Returns the memory requirements of a CUDA array in \p memoryRequirements
 * If the CUDA array is not allocated with flag ::CUDA_ARRAY3D_DEFERRED_MAPPING
 * ::CUDA_ERROR_INVALID_VALUE will be returned.
 *
 * The returned value in ::CUDA_ARRAY_MEMORY_REQUIREMENTS::size 
 * represents the total size of the CUDA array.
 * The returned value in ::CUDA_ARRAY_MEMORY_REQUIREMENTS::alignment 
 * represents the alignment necessary for mapping the CUDA array.
 *
 * \return
 * ::CUDA_SUCCESS
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \param[out] memoryRequirements - Pointer to ::CUDA_ARRAY_MEMORY_REQUIREMENTS
 * \param[in] array - CUDA array to get the memory requirements of
 * \param[in] device - Device to get the memory requirements for
 * \sa ::cuMipmappedArrayGetMemoryRequirements, ::cuMemMapArrayAsync
 *)
function cuArrayGetMemoryRequirements(memoryRequirements : PCUDA_ARRAY_MEMORY_REQUIREMENTS; &array : CUarray; device : CUdevice):CUresult;  WINAPI; external libcuda;
 
(**
 * \brief Returns the memory requirements of a CUDA mipmapped array
 *
 * Returns the memory requirements of a CUDA mipmapped array in \p memoryRequirements
 * If the CUDA mipmapped array is not allocated with flag ::CUDA_ARRAY3D_DEFERRED_MAPPING
 * ::CUDA_ERROR_INVALID_VALUE will be returned.
 *
 * The returned value in ::CUDA_ARRAY_MEMORY_REQUIREMENTS::size 
 * represents the total size of the CUDA mipmapped array.
 * The returned value in ::CUDA_ARRAY_MEMORY_REQUIREMENTS::alignment 
 * represents the alignment necessary for mapping the CUDA mipmapped  
 * array.
 *
 * \return
 * ::CUDA_SUCCESS
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \param[out] memoryRequirements - Pointer to ::CUDA_ARRAY_MEMORY_REQUIREMENTS
 * \param[in] mipmap - CUDA mipmapped array to get the memory requirements of
 * \param[in] device - Device to get the memory requirements for
 * \sa ::cuArrayGetMemoryRequirements, ::cuMemMapArrayAsync
 *)
function cuMipmappedArrayGetMemoryRequirements(memoryRequirements : PCUDA_ARRAY_MEMORY_REQUIREMENTS; mipmap : CUmipmappedArray; device : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets a CUDA array plane from a CUDA array
 *
 * Returns in \p pPlaneArray a CUDA array that represents a single format plane
 * of the CUDA array \p hArray.
 *
 * If \p planeIdx is greater than the maximum number of planes in this array or if the array does
 * not have a multi-planar format e.g: ::CU_AD_FORMAT_NV12, then ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * Note that if the \p hArray has format ::CU_AD_FORMAT_NV12, then passing in 0 for \p planeIdx returns
 * a CUDA array of the same size as \p hArray but with one channel and ::CU_AD_FORMAT_UNSIGNED_INT8 as its format.
 * If 1 is passed for \p planeIdx, then the returned CUDA array has half the height and width
 * of \p hArray with two channels and ::CU_AD_FORMAT_UNSIGNED_INT8 as its format.
 *
 * \param pPlaneArray   - Returned CUDA array referenced by the \p planeIdx
 * \param hArray        - Multiplanar CUDA array
 * \param planeIdx      - Plane index
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa
 * ::cuArrayCreate,
 * ::cudaArrayGetPlane
 *)
function cuArrayGetPlane(pPlaneArray : PCUarray; hArray : CUarray; planeIdx : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys a CUDA array
 *
 * Destroys the CUDA array \p hArray.
 *
 * \param hArray - Array to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ARRAY_IS_MAPPED,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaFreeArray
 *)
function cuArrayDestroy(hArray : CUarray):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a 3D CUDA array
 *
 * Creates a CUDA array according to the ::CUDA_ARRAY3D_DESCRIPTOR structure
 * \p pAllocateArray and returns a handle to the new CUDA array in \p *pHandle.
 * The ::CUDA_ARRAY3D_DESCRIPTOR is defined as:
 *
 * \code
    typedef struct {
        longint Width;
        longint Height;
        longint Depth;
        CUarray_format Format;
        longint NumChannels;
        longint Flags;
    } CUDA_ARRAY3D_DESCRIPTOR;
 * \endcode
 * where:
 *
 * - \p Width, \p Height, and \p Depth are the width, height, and depth of the
 * CUDA array (in elements); the following types of CUDA arrays can be allocated:
 *     - A 1D array is allocated if \p Height and \p Depth extents are both zero.
 *     - A 2D array is allocated if only \p Depth extent is zero.
 *     - A 3D array is allocated if all three extents are non-zero.
 *     - A 1D layered CUDA array is allocated if only \p Height is zero and the
 *       ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number
 *       of layers is determined by the depth extent.
 *     - A 2D layered CUDA array is allocated if all three extents are non-zero and
 *       the ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number
 *       of layers is determined by the depth extent.
 *     - A cubemap CUDA array is allocated if all three extents are non-zero and the
 *       ::CUDA_ARRAY3D_CUBEMAP flag is set. \p Width must be equal to \p Height, and
 *       \p Depth must be six. A cubemap is a special type of 2D layered CUDA array,
 *       where the six layers represent the six faces of a cube. The order of the six
 *       layers in memory is the same as that listed in ::CUarray_cubemap_face.
 *     - A cubemap layered CUDA array is allocated if all three extents are non-zero,
 *       and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set.
 *       \p Width must be equal to \p Height, and \p Depth must be a multiple of six.
 *       A cubemap layered CUDA array is a special type of 2D layered CUDA array that
 *       consists of a collection of cubemaps. The first six layers represent the first
 *       cubemap, the next six layers form the second cubemap, and so on.
 *
 * - ::Format specifies the format of the elements; ::CUarray_format is
 * defined as:
 * \code
      CUarray_format_enum = (
        CU_AD_FORMAT_UNSIGNED_INT8 = $01,
        CU_AD_FORMAT_UNSIGNED_INT16 = $02,
        CU_AD_FORMAT_UNSIGNED_INT32 = $03,
        CU_AD_FORMAT_SIGNED_INT8 = $08,
        CU_AD_FORMAT_SIGNED_INT16 = $09,
        CU_AD_FORMAT_SIGNED_INT32 = $0a,
        CU_AD_FORMAT_HALF = $10,
        CU_AD_FORMAT_FLOAT = $20
      );
  CUarray_format = CUarray_format_enum;
 *  \endcode
 *
 * - \p NumChannels specifies the number of packed components per CUDA array
 * element; it may be 1, 2, or 4;
 *
 * - ::Flags may be set to
 *   - ::CUDA_ARRAY3D_LAYERED to enable creation of layered CUDA arrays. If this flag is set,
 *     \p Depth specifies the number of layers, not the depth of a 3D array.
 *   - ::CUDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to the CUDA array.
 *     If this flag is not set, ::cuSurfRefSetArray will fail when attempting to bind the CUDA array
 *     to a surface reference.
 *   - ::CUDA_ARRAY3D_CUBEMAP to enable creation of cubemaps. If this flag is set, \p Width must be
 *     equal to \p Height, and \p Depth must be six. If the ::CUDA_ARRAY3D_LAYERED flag is also set,
 *     then \p Depth must be a multiple of six.
 *   - ::CUDA_ARRAY3D_TEXTURE_GATHER to indicate that the CUDA array will be used for texture gather.
 *     Texture gather can only be performed on 2D CUDA arrays.
 *
 * \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table.
 * All values are specified in elements. Note that for brevity's sake, the full name of the device attribute
 * is not specified. For ex., TEXTURE1D_WIDTH refers to the device attribute
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH.
 *
 * Note that 2D CUDA arrays have different size requirements if the ::CUDA_ARRAY3D_TEXTURE_GATHER flag
 * is set. \p Width and \p Height must not be greater than ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH
 * and ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT respectively, in that case.
 *
 * <table>
 * <tr><td><b>CUDA array type</b></td>
 * <td><b>Valid extents that must always be met<br>{(width range in elements), (height range),
 * (depth range)}</b></td>
 * <td><b>Valid extents with CUDA_ARRAY3D_SURFACE_LDST set<br>
 * {(width range in elements), (height range), (depth range)}</b></td></tr>
 * <tr><td>1D</td>
 * <td><small>{ (1,TEXTURE1D_WIDTH), 0, 0 }</small></td>
 * <td><small>{ (1,SURFACE1D_WIDTH), 0, 0 }</small></td></tr>
 * <tr><td>2D</td>
 * <td><small>{ (1,TEXTURE2D_WIDTH), (1,TEXTURE2D_HEIGHT), 0 }</small></td>
 * <td><small>{ (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }</small></td></tr>
 * <tr><td>3D</td>
 * <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
 * <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE),
 * (1,TEXTURE3D_DEPTH_ALTERNATE) }</small></td>
 * <td><small>{ (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT),
 * (1,SURFACE3D_DEPTH) }</small></td></tr>
 * <tr><td>1D Layered</td>
 * <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0,
 * (1,TEXTURE1D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE1D_LAYERED_WIDTH), 0,
 * (1,SURFACE1D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>2D Layered</td>
 * <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT),
 * (1,TEXTURE2D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT),
 * (1,SURFACE2D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>Cubemap</td>
 * <td><small>{ (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }</small></td>
 * <td><small>{ (1,SURFACECUBEMAP_WIDTH),
 * (1,SURFACECUBEMAP_WIDTH), 6 }</small></td></tr>
 * <tr><td>Cubemap Layered</td>
 * <td><small>{ (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH),
 * (1,TEXTURECUBEMAP_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH),
 * (1,SURFACECUBEMAP_LAYERED_LAYERS) }</small></td></tr>
 * </table>
 *
 * Here are examples of CUDA array descriptions:
 *
 * Description for a CUDA array of 2048 floats:
 * \code
    CUDA_ARRAY3D_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 2048;
    desc.Height = 0;
    desc.Depth = 0;
 * \endcode
 *
 * Description for a 64 x 64 CUDA array of floats:
 * \code
    CUDA_ARRAY3D_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 64;
    desc.Height = 64;
    desc.Depth = 0;
 * \endcode
 *
 * Description for a \p width x \p height x \p depth CUDA array of 64-bit,
 * 4x16-bit float16's:
 * \code
    CUDA_ARRAY3D_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_HALF;
    desc.NumChannels = 4;
    desc.Width = width;
    desc.Height = height;
    desc.Depth = depth;
 * \endcode
 *
 * \param pHandle        - Returned array
 * \param pAllocateArray - 3D array descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuArray3DGetDescriptor, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaMalloc3DArray
 *)
function cuArray3DCreate(pHandle : PCUarray; const pAllocateArray : PCUDA_ARRAY3D_DESCRIPTOR):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get a 3D CUDA array descriptor
 *
 * Returns in \p *pArrayDescriptor a descriptor containing information on the
 * format and dimensions of the CUDA array \p hArray. It is useful for
 * subroutines that have been passed a CUDA array, but need to know the CUDA
 * array parameters for validation or other purposes.
 *
 * This function may be called on 1D and 2D arrays, in which case the \p Height
 * and/or \p Depth members of the descriptor struct will be set to 0.
 *
 * \param pArrayDescriptor - Returned 3D array descriptor
 * \param hArray           - 3D array to get descriptor of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 * \notefnerr
 *
 * \sa ::cuArray3DCreate, ::cuArrayCreate,
 * ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
 * ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
 * ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
 * ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
 * ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
 * ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
 * ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
 * ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
 * ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
 * ::cudaArrayGetInfo
 *)
function cuArray3DGetDescriptor(pArrayDescriptor : PCUDA_ARRAY3D_DESCRIPTOR; hArray : CUarray):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a CUDA mipmapped array
 *
 * Creates a CUDA mipmapped array according to the ::CUDA_ARRAY3D_DESCRIPTOR structure
 * \p pMipmappedArrayDesc and returns a handle to the new CUDA mipmapped array in \p *pHandle.
 * \p numMipmapLevels specifies the number of mipmap levels to be allocated. This value is
 * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 *
 * The ::CUDA_ARRAY3D_DESCRIPTOR is defined as:
 *
 * \code
    typedef struct {
        longint Width;
        longint Height;
        longint Depth;
        CUarray_format Format;
        longint NumChannels;
        longint Flags;
    } CUDA_ARRAY3D_DESCRIPTOR;
 * \endcode
 * where:
 *
 * - \p Width, \p Height, and \p Depth are the width, height, and depth of the
 * CUDA array (in elements); the following types of CUDA arrays can be allocated:
 *     - A 1D mipmapped array is allocated if \p Height and \p Depth extents are both zero.
 *     - A 2D mipmapped array is allocated if only \p Depth extent is zero.
 *     - A 3D mipmapped array is allocated if all three extents are non-zero.
 *     - A 1D layered CUDA mipmapped array is allocated if only \p Height is zero and the
 *       ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number
 *       of layers is determined by the depth extent.
 *     - A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and
 *       the ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number
 *       of layers is determined by the depth extent.
 *     - A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the
 *       ::CUDA_ARRAY3D_CUBEMAP flag is set. \p Width must be equal to \p Height, and
 *       \p Depth must be six. A cubemap is a special type of 2D layered CUDA array,
 *       where the six layers represent the six faces of a cube. The order of the six
 *       layers in memory is the same as that listed in ::CUarray_cubemap_face.
 *     - A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero,
 *       and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set.
 *       \p Width must be equal to \p Height, and \p Depth must be a multiple of six.
 *       A cubemap layered CUDA array is a special type of 2D layered CUDA array that
 *       consists of a collection of cubemaps. The first six layers represent the first
 *       cubemap, the next six layers form the second cubemap, and so on.
 *
 * - ::Format specifies the format of the elements; ::CUarray_format is
 * defined as:
 * \code
      CUarray_format_enum = (
        CU_AD_FORMAT_UNSIGNED_INT8 = $01,
        CU_AD_FORMAT_UNSIGNED_INT16 = $02,
        CU_AD_FORMAT_UNSIGNED_INT32 = $03,
        CU_AD_FORMAT_SIGNED_INT8 = $08,
        CU_AD_FORMAT_SIGNED_INT16 = $09,
        CU_AD_FORMAT_SIGNED_INT32 = $0a,
        CU_AD_FORMAT_HALF = $10,
        CU_AD_FORMAT_FLOAT = $20
      );
  CUarray_format = CUarray_format_enum;
 *  \endcode
 *
 * - \p NumChannels specifies the number of packed components per CUDA array
 * element; it may be 1, 2, or 4;
 *
 * - ::Flags may be set to
 *   - ::CUDA_ARRAY3D_LAYERED to enable creation of layered CUDA mipmapped arrays. If this flag is set,
 *     \p Depth specifies the number of layers, not the depth of a 3D array.
 *   - ::CUDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to individual mipmap levels of
 *     the CUDA mipmapped array. If this flag is not set, ::cuSurfRefSetArray will fail when attempting to
 *     bind a mipmap level of the CUDA mipmapped array to a surface reference.
  *   - ::CUDA_ARRAY3D_CUBEMAP to enable creation of mipmapped cubemaps. If this flag is set, \p Width must be
 *     equal to \p Height, and \p Depth must be six. If the ::CUDA_ARRAY3D_LAYERED flag is also set,
 *     then \p Depth must be a multiple of six.
 *   - ::CUDA_ARRAY3D_TEXTURE_GATHER to indicate that the CUDA mipmapped array will be used for texture gather.
 *     Texture gather can only be performed on 2D CUDA mipmapped arrays.
 *
 * \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table.
 * All values are specified in elements. Note that for brevity's sake, the full name of the device attribute
 * is not specified. For ex., TEXTURE1D_MIPMAPPED_WIDTH refers to the device attribute
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH.
 *
 * <table>
 * <tr><td><b>CUDA array type</b></td>
 * <td><b>Valid extents that must always be met<br>{(width range in elements), (height range),
 * (depth range)}</b></td>
 * <td><b>Valid extents with CUDA_ARRAY3D_SURFACE_LDST set<br>
 * {(width range in elements), (height range), (depth range)}</b></td></tr>
 * <tr><td>1D</td>
 * <td><small>{ (1,TEXTURE1D_MIPMAPPED_WIDTH), 0, 0 }</small></td>
 * <td><small>{ (1,SURFACE1D_WIDTH), 0, 0 }</small></td></tr>
 * <tr><td>2D</td>
 * <td><small>{ (1,TEXTURE2D_MIPMAPPED_WIDTH), (1,TEXTURE2D_MIPMAPPED_HEIGHT), 0 }</small></td>
 * <td><small>{ (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }</small></td></tr>
 * <tr><td>3D</td>
 * <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
 * <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE),
 * (1,TEXTURE3D_DEPTH_ALTERNATE) }</small></td>
 * <td><small>{ (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT),
 * (1,SURFACE3D_DEPTH) }</small></td></tr>
 * <tr><td>1D Layered</td>
 * <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0,
 * (1,TEXTURE1D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE1D_LAYERED_WIDTH), 0,
 * (1,SURFACE1D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>2D Layered</td>
 * <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT),
 * (1,TEXTURE2D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT),
 * (1,SURFACE2D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>Cubemap</td>
 * <td><small>{ (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }</small></td>
 * <td><small>{ (1,SURFACECUBEMAP_WIDTH),
 * (1,SURFACECUBEMAP_WIDTH), 6 }</small></td></tr>
 * <tr><td>Cubemap Layered</td>
 * <td><small>{ (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH),
 * (1,TEXTURECUBEMAP_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH),
 * (1,SURFACECUBEMAP_LAYERED_LAYERS) }</small></td></tr>
 * </table>
 *
 *
 * \param pHandle             - Returned mipmapped array
 * \param pMipmappedArrayDesc - mipmapped array descriptor
 * \param numMipmapLevels     - Number of mipmap levels
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuMipmappedArrayDestroy,
 * ::cuMipmappedArrayGetLevel,
 * ::cuArrayCreate,
 * ::cudaMallocMipmappedArray
 *)
function cuMipmappedArrayCreate(pHandle : PCUmipmappedArray; const pMipmappedArrayDesc : PCUDA_ARRAY3D_DESCRIPTOR; numMipmapLevels : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets a mipmap level of a CUDA mipmapped array
 *
 * Returns in \p *pLevelArray a CUDA array that represents a single mipmap level
 * of the CUDA mipmapped array \p hMipmappedArray.
 *
 * If \p level is greater than the maximum number of levels in this mipmapped array,
 * ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * \param pLevelArray     - Returned mipmap level CUDA array
 * \param hMipmappedArray - CUDA mipmapped array
 * \param level           - Mipmap level
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa
 * ::cuMipmappedArrayCreate,
 * ::cuMipmappedArrayDestroy,
 * ::cuArrayCreate,
 * ::cudaGetMipmappedArrayLevel
 *)
function cuMipmappedArrayGetLevel(pLevelArray : PCUarray; hMipmappedArray : CUmipmappedArray; level : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys a CUDA mipmapped array
 *
 * Destroys the CUDA mipmapped array \p hMipmappedArray.
 *
 * \param hMipmappedArray - Mipmapped array to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ARRAY_IS_MAPPED,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 * \notefnerr
 *
 * \sa
 * ::cuMipmappedArrayCreate,
 * ::cuMipmappedArrayGetLevel,
 * ::cuArrayCreate,
 * ::cudaFreeMipmappedArray
 *)
function cuMipmappedArrayDestroy(hMipmappedArray : CUmipmappedArray):CUresult;  WINAPI; external libcuda;

(** 
* \brief Retrieve handle for an address range 
* 
* Get a handle of the specified type to an address range. The address range
* must have been obtained by a prior call to either ::cuMemAlloc or ::cuMemAddressReserve.
* If the address range was obtained via ::cuMemAddressReserve, it must also be fully mapped via ::cuMemMap.
* 
* Users must ensure the \p dptr and \p size are aligned to the host page size.
* 
* When requesting CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
* users are expected to query for dma_buf support for the platform
* by using ::CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED device attribute before calling
* this API. The \p handle will be interpreted as a pointer to an integer to store the dma_buf file descriptor.
* Users must ensure the entire address range is backed and mapped when
* the address range is allocated by ::cuMemAddressReserve. All the physical
* allocations backing the address range must be resident on the same device and
* have identical allocation properties. Users are also expected to retrieve a
* new handle every time the underlying physical allocation(s) corresponding
* to a previously queried VA range are changed.
* 
* \param[out] handle     - Pointer to the location where the returned handle will be stored. 
* \param[in] dptr        - Pointer to a valid CUDA device allocation. Must be aligned to host page size.
* \param[in] size        - Length of the address range. Must be aligned to host page size.
* \param[in] handleType  - Type of handle requested (defines type and size of the \p handle output parameter)
* \param[in] flags       - Reserved, must be zero 
* 
* \return
* CUDA_SUCCESS 
* CUDA_ERROR_INVALID_VALUE 
* CUDA_ERROR_NOT_SUPPORTED 
*)
function cuMemGetHandleForAddressRange(handle : Pointer; dptr : CUdeviceptr; size : size_t; handleType : CUmemRangeHandleType; flags : uint64):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_MEM *)

(**
 * \defgroup CUDA_VA Virtual Memory Management
 *
 * ___MANBRIEF___ virtual memory management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the virtual memory management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
* \brief Allocate an address range reservation. 
* 
* Reserves a virtual address range based on the given parameters, giving
* the starting address of the range in \p ptr.  This API requires a system that
* supports UVA.  The size and address parameters must be a multiple of the
* host page size and the alignment must be a power of two or zero for default
* alignment.
*
* \param[out] ptr       - Resulting pointer to start of virtual address range allocated
* \param[in]  size      - Size of the reserved virtual address range requested
* \param[in]  alignment - Alignment of the reserved virtual address range requested
* \param[in]  addr      - Fixed starting address range requested
* \param[in]  flags     - Currently unused, must be zero
* \return
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_OUT_OF_MEMORY,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemAddressFree
*)
function cuMemAddressReserve(ptr : PCUdeviceptr; size : size_t; alignment : size_t; addr : CUdeviceptr; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
* \brief Free an address range reservation.
* 
* Frees a virtual address range reserved by cuMemAddressReserve.  The size
* must match what was given to memAddressReserve and the ptr given must
* match what was returned from memAddressReserve.
*
* \param[in] ptr  - Starting address of the virtual address range to free
* \param[in] size - Size of the virtual address region to free
* \return
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemAddressReserve
*)
function cuMemAddressFree(ptr : CUdeviceptr; size : size_t):CUresult;  WINAPI; external libcuda;

(**
* \brief Create a CUDA memory handle representing a memory allocation of a given size described by the given properties
*
* This creates a memory allocation on the target device specified through the
* \p prop structure. The created allocation will not have any device or host
* mappings. The generic memory \p handle for the allocation can be
* mapped to the address space of calling process via ::cuMemMap. This handle
* cannot be transmitted directly to other processes (see
* ::cuMemExportToShareableHandle).  On Windows, the caller must also pass
* an LPSECURITYATTRIBUTE in \p prop to be associated with this handle which
* limits or allows access to this handle for a recipient process (see
* ::CUmemAllocationProp::win32HandleMetaData for more).  The \p size of this
* allocation must be a multiple of the the value given via
* ::cuMemGetAllocationGranularity with the ::CU_MEM_ALLOC_GRANULARITY_MINIMUM
* flag.
* If ::CUmemAllocationProp::allocFlags::usage contains ::CU_MEM_CREATE_USAGE_TILE_POOL flag then
* the memory allocation is intended only to be used as backing tile pool for sparse CUDA arrays
* and sparse CUDA mipmapped arrays.
* (see ::cuMemMapArrayAsync).
*
* \param[out] handle - Value of handle returned. All operations on this allocation are to be performed using this handle.
* \param[in]  size   - Size of the allocation requested
* \param[in]  prop   - Properties of the allocation to create.
* \param[in]  flags  - flags for future use, must be zero now.
* \return
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_OUT_OF_MEMORY,
* ::CUDA_ERROR_INVALID_DEVICE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
* \notefnerr
*
* \sa ::cuMemRelease, ::cuMemExportToShareableHandle, ::cuMemImportFromShareableHandle
*)
function cuMemCreate(handle : CUmemGenericAllocationHandle; size : size_t; const prop : PCUmemAllocationProp; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
* \brief Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.
* 
* Frees the memory that was allocated on a device through cuMemCreate.
*
* The memory allocation will be freed when all outstanding mappings to the memory
* are unmapped and when all outstanding references to the handle (including it's
* shareable counterparts) are also released. The generic memory handle can be
* freed when there are still outstanding mappings made with this handle. Each
* time a recipient process imports a shareable handle, it needs to pair it with
* ::cuMemRelease for the handle to be freed.  If \p handle is not a valid handle
* the behavior is undefined. 
*
* \param[in] handle Value of handle which was returned previously by cuMemCreate.
* \return
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
* \notefnerr
*
* \sa ::cuMemCreate
*)
function cuMemRelease(handle : CUmemGenericAllocationHandle):CUresult;  WINAPI; external libcuda;

(**
* \brief Maps an allocation handle to a reserved virtual address range.
*
* Maps bytes of memory represented by \p handle starting from byte \p offset to
* \p size to address range [\p addr, \p addr + \p size]. This range must be an
* address reservation previously reserved with ::cuMemAddressReserve, and
* \p offset + \p size must be less than the size of the memory allocation.
* Both \p ptr, \p size, and \p offset must be a multiple of the value given via
* ::cuMemGetAllocationGranularity with the ::CU_MEM_ALLOC_GRANULARITY_MINIMUM flag.
* If \p handle represents a multicast object, \p ptr, \p size and \p offset must
* be aligned to the value returned by ::cuMulticastGetGranularity with the flag
* ::CU_MULTICAST_MINIMUM_GRANULARITY. For best performance however, it is
* recommended that \p ptr, \p size and \p offset be aligned to the value
* returned by ::cuMulticastGetGranularity with the flag
* ::CU_MULTICAST_RECOMMENDED_GRANULARITY.
* 
* Please note calling ::cuMemMap does not make the address accessible,
* the caller needs to update accessibility of a contiguous mapped VA
* range by calling ::cuMemSetAccess.
* 
* Once a recipient process obtains a shareable memory handle
* from ::cuMemImportFromShareableHandle, the process must
* use ::cuMemMap to map the memory into its address ranges before
* setting accessibility with ::cuMemSetAccess.
*  
* ::cuMemMap can only create mappings on VA range reservations 
* that are not currently mapped.
* 
* \param[in] ptr    - Address where memory will be mapped. 
* \param[in] size   - Size of the memory mapping. 
* \param[in] offset - Offset into the memory represented by 
*                   - \p handle from which to start mapping
*                   - Note: currently must be zero.
* \param[in] handle - Handle to a shareable memory 
* \param[in] flags  - flags for future use, must be zero now. 
* \return
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_INVALID_DEVICE,
* ::CUDA_ERROR_OUT_OF_MEMORY,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
* \notefnerr
*
* \sa ::cuMemUnmap, ::cuMemSetAccess, ::cuMemCreate, ::cuMemAddressReserve, ::cuMemImportFromShareableHandle
*)
function cuMemMap(ptr : CUdeviceptr; size : size_t; offset : size_t; handle : CUmemGenericAllocationHandle; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays
 *
 * Performs map or unmap operations on subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
 * Each operation is specified by a ::CUarrayMapInfo entry in the \p mapInfoList array of size \p count.
 * The structure ::CUarrayMapInfo is defined as follow:
 \code
     typedef struct CUarrayMapInfo_st {
        CUresourcetype resourceType;                   
        union {
            CUmipmappedArray mipmap;
            CUarray array;
        } resource;

        CUarraySparseSubresourceType subresourceType;   
        union {
            struct {
                longint level;                     
                longint layer;                     
                longint offsetX;                   
                longint offsetY;                   
                longint offsetZ;                   
                longint extentWidth;               
                longint extentHeight;              
                longint extentDepth;               
            } sparseLevel;
            struct {
                longint layer;
                uint64 offset;              
                uint64 size;                
            } miptail;
        } subresource;

        CUmemOperationType memOperationType;
        
        CUmemHandleType memHandleType;                  
        union {
            CUmemGenericAllocationHandle memHandle;
        } memHandle;

        uint64 offset;                      
        longint deviceBitMask;                     
        longint flags;                             
        longint reserved[2];                       
    } CUarrayMapInfo;
 \endcode
 *
 * where ::CUarrayMapInfo::resourceType specifies the type of resource to be operated on.
 * If ::CUarrayMapInfo::resourceType is set to ::CUresourcetype::CU_RESOURCE_TYPE_ARRAY then 
 * ::CUarrayMapInfo::resource::array must be set to a valid sparse CUDA array handle.
 * The CUDA array must be either a 2D, 2D layered or 3D CUDA array and must have been allocated using
 * ::cuArrayCreate or ::cuArray3DCreate with the flag ::CUDA_ARRAY3D_SPARSE
 * or ::CUDA_ARRAY3D_DEFERRED_MAPPING.
 * For CUDA arrays obtained using ::cuMipmappedArrayGetLevel, ::CUDA_ERROR_INVALID_VALUE will be returned.
 * If ::CUarrayMapInfo::resourceType is set to ::CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY 
 * then ::CUarrayMapInfo::resource::mipmap must be set to a valid sparse CUDA mipmapped array handle.
 * The CUDA mipmapped array must be either a 2D, 2D layered or 3D CUDA mipmapped array and must have been
 * allocated using ::cuMipmappedArrayCreate with the flag ::CUDA_ARRAY3D_SPARSE
 * or ::CUDA_ARRAY3D_DEFERRED_MAPPING.
 *
 * ::CUarrayMapInfo::subresourceType specifies the type of subresource within the resource. 
 * ::CUarraySparseSubresourceType_enum is defined as:
 \code
      CUarraySparseSubresourceType_enum = (
        CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
        CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
      );
  CUarraySparseSubresourceType = CUarraySparseSubresourceType_enum;
 \endcode
 *
 * where ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL indicates a
 * sparse-miplevel which spans at least one tile in every dimension. The remaining miplevels which
 * are too small to span at least one tile in any dimension constitute the mip tail region as indicated by 
 * ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL subresource type.
 *
 * If ::CUarrayMapInfo::subresourceType is set to ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
 * then ::CUarrayMapInfo::subresource::sparseLevel struct must contain valid array subregion offsets and extents.
 * The ::CUarrayMapInfo::subresource::sparseLevel::offsetX, ::CUarrayMapInfo::subresource::sparseLevel::offsetY
 * and ::CUarrayMapInfo::subresource::sparseLevel::offsetZ must specify valid X, Y and Z offsets respectively.
 * The ::CUarrayMapInfo::subresource::sparseLevel::extentWidth, ::CUarrayMapInfo::subresource::sparseLevel::extentHeight
 * and ::CUarrayMapInfo::subresource::sparseLevel::extentDepth must specify valid width, height and depth extents respectively.
 * These offsets and extents must be aligned to the corresponding tile dimension.
 * For CUDA mipmapped arrays ::CUarrayMapInfo::subresource::sparseLevel::level must specify a valid mip level index. Otherwise,
 * must be zero.
 * For layered CUDA arrays and layered CUDA mipmapped arrays ::CUarrayMapInfo::subresource::sparseLevel::layer must specify a valid layer index. Otherwise,
 * must be zero.
 * ::CUarrayMapInfo::subresource::sparseLevel::offsetZ must be zero and ::CUarrayMapInfo::subresource::sparseLevel::extentDepth
 * must be set to 1 for 2D and 2D layered CUDA arrays and CUDA mipmapped arrays.
 * Tile extents can be obtained by calling ::cuArrayGetSparseProperties and ::cuMipmappedArrayGetSparseProperties
 *
 * If ::CUarrayMapInfo::subresourceType is set to ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
 * then ::CUarrayMapInfo::subresource::miptail struct must contain valid mip tail offset in 
 * ::CUarrayMapInfo::subresource::miptail::offset and size in ::CUarrayMapInfo::subresource::miptail::size.
 * Both, mip tail offset and mip tail size must be aligned to the tile size. 
 * For layered CUDA mipmapped arrays which don't have the flag ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL set in ::CUDA_ARRAY_SPARSE_PROPERTIES::flags
 * as returned by ::cuMipmappedArrayGetSparseProperties, ::CUarrayMapInfo::subresource::miptail::layer must specify a valid layer index.
 * Otherwise, must be zero.
 *
 * If ::CUarrayMapInfo::resource::array or ::CUarrayMapInfo::resource::mipmap was created with ::CUDA_ARRAY3D_DEFERRED_MAPPING
 * flag set the ::CUarrayMapInfo::subresourceType and the contents of ::CUarrayMapInfo::subresource will be ignored.
 *
 * ::CUarrayMapInfo::memOperationType specifies the type of operation. ::CUmemOperationType is defined as:
 \code
      CUmemOperationType_enum = (
        CU_MEM_OPERATION_TYPE_MAP = 1,
        CU_MEM_OPERATION_TYPE_UNMAP = 2
      );
  CUmemOperationType = CUmemOperationType_enum;
 \endcode
 * If ::CUarrayMapInfo::memOperationType is set to ::CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP then the subresource 
 * will be mapped onto the tile pool memory specified by ::CUarrayMapInfo::memHandle at offset ::CUarrayMapInfo::offset. 
 * The tile pool allocation has to be created by specifying the ::CU_MEM_CREATE_USAGE_TILE_POOL flag when calling ::cuMemCreate. Also, 
 * ::CUarrayMapInfo::memHandleType must be set to ::CUmemHandleType::CU_MEM_HANDLE_TYPE_GENERIC.
 * 
 * If ::CUarrayMapInfo::memOperationType is set to ::CUmemOperationType::CU_MEM_OPERATION_TYPE_UNMAP then an unmapping operation
 * is performed. ::CUarrayMapInfo::memHandle must be NULL.
 *
 * ::CUarrayMapInfo::deviceBitMask specifies the list of devices that must map or unmap physical memory. 
 * Currently, this mask must have exactly one bit set, and the corresponding device must match the device associated with the stream. 
 * If ::CUarrayMapInfo::memOperationType is set to ::CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP, the device must also match 
 * the device associated with the tile pool memory allocation as specified by ::CUarrayMapInfo::memHandle.
 *
 * ::CUarrayMapInfo::flags and ::CUarrayMapInfo::reserved[] are unused and must be set to zero.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 *
 * \param[in] mapInfoList - List of ::CUarrayMapInfo
 * \param[in] count       - Count of ::CUarrayMapInfo  in \p mapInfoList
 * \param[in] hStream     - Stream identifier for the stream to use for map or unmap operations
 *
 * \sa ::cuMipmappedArrayCreate, ::cuArrayCreate, ::cuArray3DCreate, ::cuMemCreate, ::cuArrayGetSparseProperties, ::cuMipmappedArrayGetSparseProperties
 *)
function cuMemMapArrayAsync(mapInfoList : PCUarrayMapInfo; count : longword; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
* \brief Unmap the backing memory of a given address range.
*
* The range must be the entire contiguous address range that was mapped to.  In
* other words, ::cuMemUnmap cannot unmap a sub-range of an address range mapped
* by ::cuMemCreate / ::cuMemMap.  Any backing memory allocations will be freed
* if there are no existing mappings and there are no unreleased memory handles.
*
* When ::cuMemUnmap returns successfully the address range is converted to an
* address reservation and can be used for a future calls to ::cuMemMap.  Any new
* mapping to this virtual address will need to have access granted through
* ::cuMemSetAccess, as all mappings start with no accessibility setup.
*
* \param[in] ptr  - Starting address for the virtual address range to unmap
* \param[in] size - Size of the virtual address range to unmap
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
* \notefnerr
* \note_sync
*
* \sa ::cuMemCreate, ::cuMemAddressReserve
*)
function cuMemUnmap(ptr : CUdeviceptr; size : size_t):CUresult;  WINAPI; external libcuda;

(**
* \brief Set the access flags for each location specified in \p desc for the given virtual address range
* 
* Given the virtual address range via \p ptr and \p size, and the locations
* in the array given by \p desc and \p count, set the access flags for the
* target locations.  The range must be a fully mapped address range
* containing all allocations created by ::cuMemMap / ::cuMemCreate.
* When setting the access flags for a virtual address range mapping a multicast
* object, \p ptr and \p size must be aligned to the value returned by
* ::cuMulticastGetGranularity with the flag ::CU_MULTICAST_MINIMUM_GRANULARITY.
* For best performance however, it is recommended that \p ptr and \p size be
* aligned to the value returned by ::cuMulticastGetGranularity with the flag
* ::CU_MULTICAST_RECOMMENDED_GRANULARITY.
*
* \param[in] ptr   - Starting address for the virtual address range
* \param[in] size  - Length of the virtual address range
* \param[in] desc  - Array of ::CUmemAccessDesc that describe how to change the
*                  - mapping for each location specified
* \param[in] count - Number of ::CUmemAccessDesc in \p desc
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_INVALID_DEVICE,
* ::CUDA_ERROR_NOT_SUPPORTED
* \notefnerr
* \note_sync
*
* \sa ::cuMemSetAccess, ::cuMemCreate, :cuMemMap
*)
function cuMemSetAccess(ptr : CUdeviceptr; size : size_t; const desc : PCUmemAccessDesc; count : size_t):CUresult;  WINAPI; external libcuda;

(**
* \brief Get the access \p flags set for the given \p location and \p ptr
*
* \param[out] flags   - Flags set for this location
* \param[in] location - Location in which to check the flags for
* \param[in] ptr      - Address in which to check the access flags for
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_INVALID_DEVICE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemSetAccess
*)
function cuMemGetAccess(flags : Puint64; const location : PCUmemLocation; ptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
* \brief Exports an allocation to a requested shareable handle type
*
* Given a CUDA memory handle, create a shareable memory
* allocation handle that can be used to share the memory with other
* processes. The recipient process can convert the shareable handle back into a
* CUDA memory handle using ::cuMemImportFromShareableHandle and map
* it with ::cuMemMap. The implementation of what this handle is and how it
* can be transferred is defined by the requested handle type in \p handleType
*
* Once all shareable handles are closed and the allocation is released, the allocated
* memory referenced will be released back to the OS and uses of the CUDA handle afterward
* will lead to undefined behavior.
*
* This API can also be used in conjunction with other APIs (e.g. Vulkan, OpenGL)
* that support importing memory from the shareable type
*
* \param[out] shareableHandle - Pointer to the location in which to store the requested handle type
* \param[in] handle           - CUDA handle for the memory allocation
* \param[in] handleType       - Type of shareable handle requested (defines type and size of the \p shareableHandle output parameter)
* \param[in] flags            - Reserved, must be zero
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemImportFromShareableHandle
*)
function cuMemExportToShareableHandle(shareableHandle : Pointer; handle : CUmemGenericAllocationHandle; handleType : CUmemAllocationHandleType; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
* \brief Imports an allocation from a requested shareable handle type.
*
* If the current process cannot support the memory described by this shareable
* handle, this API will error as CUDA_ERROR_NOT_SUPPORTED.
*
* \note Importing shareable handles exported from some graphics APIs(VUlkan, OpenGL, etc)
* created on devices under an SLI group may not be supported, and thus this API will
* return CUDA_ERROR_NOT_SUPPORTED.
* There is no guarantee that the contents of \p handle will be the same CUDA memory handle
* for the same given OS shareable handle, or the same underlying allocation.
*
* \param[out] handle       - CUDA Memory handle for the memory allocation.
* \param[in]  osHandle     - Shareable Handle representing the memory allocation that is to be imported. 
* \param[in]  shHandleType - handle type of the exported handle ::CUmemAllocationHandleType.
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemExportToShareableHandle, ::cuMemMap, ::cuMemRelease
*)
function cuMemImportFromShareableHandle(handle : PCUmemGenericAllocationHandle; osHandle : Pointer; shHandleType : CUmemAllocationHandleType):CUresult;  WINAPI; external libcuda;

(**
* \brief Calculates either the minimal or recommended granularity 
*
* Calculates either the minimal or recommended granularity
* for a given allocation specification and returns it in granularity.  This
* granularity can be used as a multiple for alignment, size, or address mapping.
*
* \param[out] granularity Returned granularity.
* \param[in]  prop Property for which to determine the granularity for
* \param[in]  option Determines which granularity to return
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemCreate, ::cuMemMap
*)
function cuMemGetAllocationGranularity(granularity : Psize_t; const prop : PCUmemAllocationProp; option : CUmemAllocationGranularity_flags):CUresult;  WINAPI; external libcuda;

(**
* \brief Retrieve the contents of the property structure defining properties for this handle
*
* \param[out] prop  - Pointer to a properties structure which will hold the information about this handle
* \param[in] handle - Handle which to perform the query on
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemCreate, ::cuMemImportFromShareableHandle
*)
function cuMemGetAllocationPropertiesFromHandle(prop : PCUmemAllocationProp; handle : CUmemGenericAllocationHandle):CUresult;  WINAPI; external libcuda;

(**
* \brief Given an address \p addr, returns the allocation handle of the backing memory allocation.
*
* The handle is guaranteed to be the same handle value used to map the memory. If the address
* requested is not mapped, the function will fail. The returned handle must be released with
* corresponding number of calls to ::cuMemRelease.
*
* \note The address \p addr, can be any address in a range previously mapped
* by ::cuMemMap, and not necessarily the start address.
*
* \param[out] handle CUDA Memory handle for the backing memory allocation.
* \param[in] addr Memory address to query, that has been mapped previously.
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMemCreate, ::cuMemRelease, ::cuMemMap
*)
function cuMemRetainAllocationHandle(handle : PCUmemGenericAllocationHandle; addr : Pointer):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_VA *)

(**
 * \defgroup CUDA_MALLOC_ASYNC Stream Ordered Memory Allocator
 *
 * ___MANBRIEF___ Functions for performing allocation and free operations in stream order.
 *                Functions for controlling the behavior of the underlying allocator.
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream ordered memory allocator exposed by the
 * low-level CUDA driver application programming interface.
 *
 * @{
 *
 * \section CUDA_MALLOC_ASYNC_overview overview
 *
 * The asynchronous allocator allows the user to allocate and free in stream order.
 * All asynchronous accesses of the allocation must happen between
 * the stream executions of the allocation and the free. If the memory is accessed
 * outside of the promised stream order, a use before allocation / use after free error
 * will cause undefined behavior.
 *
 * The allocator is free to reallocate the memory as long as it can guarantee
 * that compliant memory accesses will not overlap temporally.
 * The allocator may refer to internal stream ordering as well as inter-stream dependencies
 * (such as CUDA events and null stream dependencies) when establishing the temporal guarantee.
 * The allocator may also insert inter-stream dependencies to establish the temporal guarantee. 
 *
 * \section CUDA_MALLOC_ASYNC_support Supported Platforms
 *
 * Whether or not a device supports the integrated stream ordered memory allocator
 * may be queried by calling ::cuDeviceGetAttribute() with the device attribute
 * ::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
 *)

(**
 * \brief Frees memory with stream ordered semantics
 *
 * Inserts a free operation into \p hStream.
 * The allocation must not be accessed after stream execution reaches the free.
 * After this API returns, accessing the memory from any subsequent work launched on the GPU
 * or querying its pointer attributes results in undefined behavior.
 *
 * \note During stream capture, this function results in the creation of a free node and
 *       must therefore be passed the address of a graph allocation.
 * 
 * \param dptr - memory to free
 * \param hStream - The stream establishing the stream ordering contract. 
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT (default stream specified with no current context),
 * ::CUDA_ERROR_NOT_SUPPORTED
 *)
function cuMemFreeAsync(dptr : CUdeviceptr; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates memory with stream ordered semantics
 *
 * Inserts an allocation operation into \p hStream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the memory pool current to the stream's device.
 *
 * \note The default memory pool of a device contains device memory from that device.
 * \note Basic stream ordering allows future work submitted into the same stream to use the allocation.
 *       Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
 *       operation completes before work submitted in a separate stream runs. 
 * \note During stream capture, this function results in the creation of an allocation node.  In this case,
 *       the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 *       are used to set the node's creation parameters.
 *
 * \param[out] dptr    - Returned device pointer
 * \param[in] bytesize - Number of bytes to allocate
 * \param[in] hStream  - The stream establishing the stream ordering contract and the memory pool to allocate from
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT (default stream specified with no current context),
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemAllocFromPoolAsync, ::cuMemFreeAsync, ::cuDeviceSetMemPool,
 *     ::cuDeviceGetDefaultMemPool, ::cuDeviceGetMemPool, ::cuMemPoolCreate,
 *     ::cuMemPoolSetAccess, ::cuMemPoolSetAttribute
 *)
function cuMemAllocAsync(dptr : PCUdeviceptr; bytesize : size_t; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Tries to release memory back to the OS
 *
 * Releases memory back to the OS until the pool contains fewer than minBytesToKeep
 * reserved bytes, or there is no more memory that the allocator can safely release.
 * The allocator cannot release OS allocations that back outstanding asynchronous allocations.
 * The OS allocations may happen at different granularity from the user allocations.
 *
 * \note: Allocations that have not been freed count as outstanding. 
 * \note: Allocations that have been asynchronously freed but whose completion has
 *        not been observed on the host (eg. by a synchronize) can count as outstanding.
 *
 * \param[in] pool           - The memory pool to trim
 * \param[in] minBytesToKeep - If the pool has less than minBytesToKeep reserved,
 * the TrimTo operation is a no-op.  Otherwise the pool will be guaranteed to have
 * at least minBytesToKeep bytes reserved after the operation.
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuMemAllocAsync, ::cuMemFreeAsync, ::cuDeviceGetDefaultMemPool,
 *     ::cuDeviceGetMemPool, ::cuMemPoolCreate
 *)
function cuMemPoolTrimTo(pool : CUmemoryPool; minBytesToKeep : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets attributes of a memory pool
 *
 * Supported attributes are:
 * - ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: (value type = cuuint64_t)
 *                    Amount of reserved memory in bytes to hold onto before trying
 *                    to release memory back to the OS. When more than the release
 *                    threshold bytes of memory are held by the memory pool, the
 *                    allocator will try to release memory back to the OS on the
 *                    next call to stream, event or context synchronize. (default 0)
 * - ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: (value type = longint)
 *                    Allow ::cuMemAllocAsync to use memory asynchronously freed
 *                    in another stream as long as a stream ordering dependency
 *                    of the allocating stream on the free action exists.
 *                    Cuda events and null stream interactions can create the required
 *                    stream ordered dependencies. (default enabled)
 * - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: (value type = longint)
 *                    Allow reuse of already completed frees when there is no dependency
 *                    between the free and allocation. (default enabled)
 * - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: (value type = longint)
 *                    Allow ::cuMemAllocAsync to insert new stream dependencies
 *                    in order to establish the stream ordering required to reuse
 *                    a piece of memory released by ::cuMemFreeAsync (default enabled).
 * - ::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: (value type = cuuint64_t)
 *                    Reset the high watermark that tracks the amount of backing memory that was
 *                    allocated for the memory pool. It is illegal to set this attribute to a non-zero value.
 * - ::CU_MEMPOOL_ATTR_USED_MEM_HIGH: (value type = cuuint64_t)
 *                    Reset the high watermark that tracks the amount of used memory that was
 *                    allocated for the memory pool.
 *
 * \param[in] pool  - The memory pool to modify
 * \param[in] attr  - The attribute to modify
 * \param[in] value - Pointer to the value to assign
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuMemAllocAsync, ::cuMemFreeAsync, ::cuDeviceGetDefaultMemPool,
 *     ::cuDeviceGetMemPool, ::cuMemPoolCreate
 *)
function cuMemPoolSetAttribute(pool : CUmemoryPool; attr : CUmemPool_attribute; value : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets attributes of a memory pool
 *
 * Supported attributes are:
 * - ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: (value type = cuuint64_t)
 *                    Amount of reserved memory in bytes to hold onto before trying
 *                    to release memory back to the OS. When more than the release
 *                    threshold bytes of memory are held by the memory pool, the
 *                    allocator will try to release memory back to the OS on the
 *                    next call to stream, event or context synchronize. (default 0)
 * - ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: (value type = longint)
 *                    Allow ::cuMemAllocAsync to use memory asynchronously freed
 *                    in another stream as long as a stream ordering dependency
 *                    of the allocating stream on the free action exists.
 *                    Cuda events and null stream interactions can create the required
 *                    stream ordered dependencies. (default enabled)
 * - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: (value type = longint)
 *                    Allow reuse of already completed frees when there is no dependency
 *                    between the free and allocation. (default enabled)
 * - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: (value type = longint)
 *                    Allow ::cuMemAllocAsync to insert new stream dependencies
 *                    in order to establish the stream ordering required to reuse
 *                    a piece of memory released by ::cuMemFreeAsync (default enabled).
 * - ::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: (value type = cuuint64_t)
 *                    Amount of backing memory currently allocated for the mempool
 * - ::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: (value type = cuuint64_t)
 *                    High watermark of backing memory allocated for the mempool since the
 *                    last time it was reset.
 * - ::CU_MEMPOOL_ATTR_USED_MEM_CURRENT: (value type = cuuint64_t)
 *                    Amount of memory from the pool that is currently in use by the application.
 * - ::CU_MEMPOOL_ATTR_USED_MEM_HIGH: (value type = cuuint64_t)
 *                    High watermark of the amount of memory from the pool that was in use by the application.
 *
 * \param[in] pool   - The memory pool to get attributes of
 * \param[in] attr   - The attribute to get 
 * \param[out] value - Retrieved value
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuMemAllocAsync, ::cuMemFreeAsync, ::cuDeviceGetDefaultMemPool,
 *     ::cuDeviceGetMemPool, ::cuMemPoolCreate
 *)
function cuMemPoolGetAttribute(pool : CUmemoryPool; attr : CUmemPool_attribute; value : Pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Controls visibility of pools between devices
 *
 * \param[in] pool  - The pool being modified
 * \param[in] map   - Array of access descriptors. Each descriptor instructs the access to enable for a single gpu.
 * \param[in] count - Number of descriptors in the map array.
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuMemAllocAsync, ::cuMemFreeAsync, ::cuDeviceGetDefaultMemPool,
 *     ::cuDeviceGetMemPool, ::cuMemPoolCreate
 *)
function cuMemPoolSetAccess(pool : CUmemoryPool; const map : PCUmemAccessDesc; count : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the accessibility of a pool from a device
 *
 * Returns the accessibility of the pool's memory from the specified location. 
 *
 * \param[out] flags   - the accessibility of the pool from the specified location
 * \param[in] memPool  - the pool being queried
 * \param[in] location - the location accessing the pool
 *
 * \sa ::cuMemAllocAsync, ::cuMemFreeAsync, ::cuDeviceGetDefaultMemPool,
 *     ::cuDeviceGetMemPool, ::cuMemPoolCreate
 *)
function cuMemPoolGetAccess(flags : PCUmemAccess_flags; memPool : CUmemoryPool; location : PCUmemLocation):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a memory pool
 *
 * Creates a CUDA memory pool and returns the handle in \p pool.  The \p poolProps determines
 * the properties of the pool such as the backing device and IPC capabilities. 
 *
 * By default, the pool's memory will be accessible from the device it is allocated on.
 *
 * \note Specifying CU_MEM_HANDLE_TYPE_NONE creates a memory pool that will not support IPC.
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_NOT_SUPPORTED
 *
 * \sa ::cuDeviceSetMemPool, ::cuDeviceGetMemPool, ::cuDeviceGetDefaultMemPool,
 *     ::cuMemAllocFromPoolAsync, ::cuMemPoolExportToShareableHandle
 *)
function cuMemPoolCreate(pool : PCUmemoryPool; const poolProps : PCUmemPoolProps):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys the specified memory pool
 *
 * If any pointers obtained from this pool haven't been freed or
 * the pool has free operations that haven't completed
 * when ::cuMemPoolDestroy is invoked, the function will return immediately and the
 * resources associated with the pool will be released automatically
 * once there are no more outstanding allocations. 
 *
 * Destroying the current mempool of a device sets the default mempool of
 * that device as the current mempool for that device.
 *
 * \note A device's default memory pool cannot be destroyed.
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuMemFreeAsync, ::cuDeviceSetMemPool, ::cuDeviceGetMemPool,
 *     ::cuDeviceGetDefaultMemPool, ::cuMemPoolCreate
 *)
function cuMemPoolDestroy(pool : CUmemoryPool):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allocates memory from a specified pool with stream ordered semantics.
 *
 * Inserts an allocation operation into \p hStream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the specified memory pool.
 *
 * \note
 *    -  The specified memory pool may be from a device different than that of the specified \p hStream. 
 * 
 *    -  Basic stream ordering allows future work submitted into the same stream to use the allocation.
 *       Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
 *       operation completes before work submitted in a separate stream runs. 
 *
 * \note During stream capture, this function results in the creation of an allocation node.  In this case,
 *       the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 *       are used to set the node's creation parameters.
 *
 * \param[out] dptr    - Returned device pointer
 * \param[in] bytesize - Number of bytes to allocate
 * \param[in] pool     - The pool to allocate from 
 * \param[in] hStream  - The stream establishing the stream ordering semantic
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT (default stream specified with no current context),
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemAllocAsync, ::cuMemFreeAsync, ::cuDeviceGetDefaultMemPool,
 *     ::cuDeviceGetMemPool, ::cuMemPoolCreate, ::cuMemPoolSetAccess,
 *     ::cuMemPoolSetAttribute
 *)
function cuMemAllocFromPoolAsync(dptr : PCUdeviceptr; bytesize : size_t; pool : CUmemoryPool; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Exports a memory pool to the requested handle type.
 *
 * Given an IPC capable mempool, create an OS handle to share the pool with another process.
 * A recipient process can convert the shareable handle into a mempool with ::cuMemPoolImportFromShareableHandle.
 * Individual pointers can then be shared with the ::cuMemPoolExportPointer and ::cuMemPoolImportPointer APIs.
 * The implementation of what the shareable handle is and how it can be transferred is defined by the requested
 * handle type.
 *
 * \note: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than CU_MEM_HANDLE_TYPE_NONE.
 *
 * \param[out] handle_out  - Returned OS handle 
 * \param[in] pool         - pool to export 
 * \param[in] handleType   - the type of handle to create 
 * \param[in] flags        - must be 0 
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemPoolImportFromShareableHandle, ::cuMemPoolExportPointer,
 *     ::cuMemPoolImportPointer, ::cuMemAllocAsync, ::cuMemFreeAsync,
 *     ::cuDeviceGetDefaultMemPool, ::cuDeviceGetMemPool, ::cuMemPoolCreate,
 *     ::cuMemPoolSetAccess, ::cuMemPoolSetAttribute
 *)
function cuMemPoolExportToShareableHandle(handle_out : Pointer; pool : CUmemoryPool; handleType : CUmemAllocationHandleType; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief imports a memory pool from a shared handle.
 *
 * Specific allocations can be imported from the imported pool with cuMemPoolImportPointer.
 *
 * \note Imported memory pools do not support creating new allocations.
 *       As such imported memory pools may not be used in cuDeviceSetMemPool
 *       or ::cuMemAllocFromPoolAsync calls.
 *
 * \param[out] pool_out    - Returned memory pool
 * \param[in] handle       - OS handle of the pool to open 
 * \param[in] handleType   - The type of handle being imported 
 * \param[in] flags        - must be 0 
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemPoolExportToShareableHandle, ::cuMemPoolExportPointer, ::cuMemPoolImportPointer
 *)
function cuMemPoolImportFromShareableHandle(
        pool_out : PCUmemoryPool;
        handle : Pointer;
        handleType : CUmemAllocationHandleType;
        flags : uint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Export data to share a memory pool allocation between processes.
 *
 * Constructs \p shareData_out for sharing a specific allocation from an already shared memory pool.
 * The recipient process can import the allocation with the ::cuMemPoolImportPointer api.
 * The data is not a handle and may be shared through any IPC mechanism.
 *
 * \param[out] shareData_out - Returned export data  
 * \param[in] ptr            - pointer to memory being exported
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemPoolExportToShareableHandle, ::cuMemPoolImportFromShareableHandle, ::cuMemPoolImportPointer
 *)
function cuMemPoolExportPointer(shareData_out : PCUmemPoolPtrExportData; ptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Import a memory pool allocation from another process.
 *
 * Returns in \p ptr_out a pointer to the imported memory.
 * The imported memory must not be accessed before the allocation operation completes
 * in the exporting process. The imported memory must be freed from all importing processes before
 * being freed in the exporting process. The pointer may be freed with cuMemFree
 * or cuMemFreeAsync.  If cuMemFreeAsync is used, the free must be completed
 * on the importing process before the free operation on the exporting process.
 *
 * \note The cuMemFreeAsync api may be used in the exporting process before
 *       the cuMemFreeAsync operation completes in its stream as long as the
 *       cuMemFreeAsync in the exporting process specifies a stream with
 *       a stream dependency on the importing process's cuMemFreeAsync.
 *
 * \param[out] ptr_out  - pointer to imported memory
 * \param[in] pool      - pool from which to import
 * \param[in] shareData - data specifying the memory to import
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemPoolExportToShareableHandle, ::cuMemPoolImportFromShareableHandle, ::cuMemPoolExportPointer
 *)
function cuMemPoolImportPointer(ptr_out : PCUdeviceptr; pool : CUmemoryPool; shareData : PCUmemPoolPtrExportData):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_MALLOC_ASYNC *)

(**
 * \defgroup CUDA_MULTICAST Multicast Object Management
 *
 * ___MANBRIEF___ Functions for creating multicast objects, adding devices to them and binding/unbinding memory
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the CUDA multicast object operations exposed by the
 * low-level CUDA driver application programming interface.
 *
 * @{
 *
 * \section CUDA_MULTICAST_overview overview
 *
 * A multicast object created via ::cuMulticastCreate enables certain memory
 * operations to be broadcasted to a team of devices. Devices can be added to a
 * multicast object via ::cuMulticastAddDevice. Memory can be bound on each
 * participating device via either ::cuMulticastBindMem or ::cuMulticastBindAddr.
 * Multicast objects can be mapped into a device's virtual address space using
 * the virtual memmory management APIs (see ::cuMemMap and ::cuMemSetAccess).
 *
 * \section CUDA_MULTICAST_support Supported Platforms
 *
 * Support for multicast on a specific device can be queried using the device
 * attribute ::CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
 *)

(**
 * \brief Create a generic allocation handle representing a multicast object described by the given properties.
 *
 * This creates a multicast object as described by \p prop. The number of
 * participating devices is specified by ::CUmulticastObjectProp::numDevices.
 * Devices can be added to the multicast object via ::cuMulticastAddDevice.
 * All participating devices must be added to the multicast object before memory
 * can be bound to it. Memory is bound to the multicast object via either
 * ::cuMulticastBindMem or ::cuMulticastBindAddr, and can be unbound via
 * ::cuMulticastUnbind. The total amount of memory that can be bound per device
 * is specified by :CUmulticastObjectProp::size. This size must be a multiple of
 * the value returned by ::cuMulticastGetGranularity with the flag
 * ::CU_MULTICAST_GRANULARITY_MINIMUM. For best performance however, the size
 * should be aligned to the value returned by ::cuMulticastGetGranularity with
 * the flag ::CU_MULTICAST_GRANULARITY_RECOMMENDED.
 *
 * After all participating devices have been added, multicast objects can also
 * be mapped to a device's virtual address space using the virtual memory
 * management APIs (see ::cuMemMap and ::cuMemSetAccess). Multicast objects can
 * also be shared with other processes by requesting a shareable handle via
 * ::cuMemExportToShareableHandle. Note that the desired types of shareable
 * handles must be specified in the bitmask ::CUmulticastObjectProp::handleTypes.
 * Multicast objects can be released using the virtual memory management API
 * ::cuMemRelease.
 *
 * \param[out] mcHandle     Value of handle returned.
 * \param[in]  prop         Properties of the multicast object to create.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_NOT_SUPPORTED
 *
 * \sa ::cuMulticastAddDevice, ::cuMulticastBindMem, ::cuMulticastBindAddr, ::cuMulticastUnbind
 * \sa ::cuMemCreate, ::cuMemRelease, ::cuMemExportToShareableHandle, ::cuMemImportFromShareableHandle
 *)
function cuMulticastCreate(mcHandle : PCUmemGenericAllocationHandle; const prop : PCUmulticastObjectProp):CUresult;  WINAPI; external libcuda;

(**
 * \brief Associate a device to a multicast object.
 *
 * Associates a device to a multicast object. The added device will be a part of
 * the multicast team of size specified by CUmulticastObjectProp::numDevices
 * during ::cuMulticastCreate.
 * The association of the device to the multicast object is permanent during
 * the life time of the multicast object.
 * All devices must be added to the multicast team before any memory can be
 * bound to any device in the team. Any calls to ::cuMulticastBindMem or
 * ::cuMulticastBindAddr will block until all devices have been added.
 * Similarly all devices must be added to the multicast team before a virtual
 * address range can be mapped to the multicast object. A call to ::cuMemMap
 * will block until all devices have been added.
 *
 * \param[in] mcHandle     Handle representing a multicast object.
 * \param[in] dev          Device that will be associated to the multicast
 *                         object.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_NOT_SUPPORTED
 *
 * \sa ::cuMulticastCreate, ::cuMulticastBindMem, ::cuMulticastBindAddr
 *)
function cuMulticastAddDevice(mcHandle : CUmemGenericAllocationHandle; dev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Bind a memory allocation represented by a handle to a multicast object.
 *
 * Binds a memory allocation specified by \p memHandle and created via
 * ::cuMemCreate to a multicast object represented by \p mcHandle and created
 * via ::cuMulticastCreate. The intended \p size of the bind, the offset in the
 * multicast range \p mcOffset as well as the offset in the memory \p memOffset
 * must be a multiple of the value returned by ::cuMulticastGetGranularity with
 * the flag ::CU_MULTICAST_GRANULARITY_MINIMUM. For best performance however,
 * \p size, \p mcOffset and \p memOffset should be aligned to the granularity of
 * the memory allocation(see ::cuMemGetAllocationGranularity) or to the value
 * returned by ::cuMulticastGetGranularity with the flag
 * ::CU_MULTICAST_GRANULARITY_RECOMMENDED.
 *
 * The \p size + \p memOffset must be smaller than the size of the allocated
 * memory. Similarly the \p size + \p mcOffset must be smaller than the size
 * of the multicast object.
 * The memory allocation must have beeen created on one of the devices
 * that was added to the multicast team via ::cuMulticastAddDevice.
 * Externally shareable as well as imported multicast objects can be bound only
 * to externally shareable memory.
 * Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are
 * insufficient resources required to perform the bind. This call may also
 * return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not
 * initialized or running.
 *
 * \param[in]  mcHandle     Handle representing a multicast object.
 * \param[in]  mcOffset     Offset into the multicast object for attachment.
 * \param[in]  memHandle    Handle representing a memory allocation.
 * \param[in]  memOffset    Offset into the memory for attachment.
 * \param[in]  size         Size of the memory that will be bound to the
 *                          multicast object.
 * \param[in]  flags        Flags for future use, must be zero for now.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_SYSTEM_NOT_READY
 *
 * \sa ::cuMulticastCreate, ::cuMulticastAddDevice, ::cuMemCreate
 *)
function cuMulticastBindMem(mcHandle : CUmemGenericAllocationHandle; mcOffset : size_t; memHandle : CUmemGenericAllocationHandle; memOffset : size_t; size : size_t; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Bind a memory allocation represented by a virtual address to a multicast object.
 *
 * Binds a memory allocation specified by its mapped address \p memptr to a
 * multicast object represented by \p mcHandle.
 * The memory must have been allocated via ::cuMemCreate or ::cudaMallocAsync.
 * The intended \p size of the bind, the offset in the multicast range
 * \p mcOffset and \p memptr must be a multiple of the value returned by
 * ::cuMulticastGetGranularity with the flag ::CU_MULTICAST_GRANULARITY_MINIMUM.
 * For best performance however, \p size, \p mcOffset and \p memptr should be
 * aligned to the value returned by ::cuMulticastGetGranularity with the flag
 * ::CU_MULTICAST_GRANULARITY_RECOMMENDED.
 *
 * The \p size must be smaller than the size of the allocated memory.
 * Similarly the \p size + \p mcOffset must be smaller than the total size
 * of the multicast object.
 * The memory allocation must have beeen created on one of the devices
 * that was added to the multicast team via ::cuMulticastAddDevice.
 * Externally shareable as well as imported multicast objects can be bound only
 * to externally shareable memory.
 * Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are
 * insufficient resources required to perform the bind. This call may also
 * return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not
 * initialized or running.
 *
 * \param[in]  mcHandle     Handle representing a multicast object.
 * \param[in]  mcOffset     Offset into multicast va range for attachment.
 * \param[in]  memptr       Virtual address of the memory allocation.
 * \param[in]  size         Size of memory that will be bound to the
 *                          multicast object.
 * \param[in]  flags        Flags for future use, must be zero now.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_SYSTEM_NOT_READY
 *
 * \sa ::cuMulticastCreate, ::cuMulticastAddDevice, ::cuMemCreate
 *)
function cuMulticastBindAddr(mcHandle : CUmemGenericAllocationHandle; mcOffset : size_t; memptr : CUdeviceptr; size : size_t; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.
 *
 * Unbinds any memory allocations hosted on \p dev and bound to a multicast
 * object at \p mcOffset and upto a given \p size.
 * The intended \p size of the unbind and the offset in the multicast range
 * ( \p mcOffset ) must be a multiple of the value returned by
 * ::cuMulticastGetGranularity flag ::CU_MULTICAST_GRANULARITY_MINIMUM.
 * The \p size + \p mcOffset must be smaller than the total size of the
 * multicast object.
 *
 * \note 
 * Warning:
 * The \p mcOffset and the \p size must match the corresponding values specified
 * during the bind call. Any other values may result in undefined behavior.
 *
 * \param[in]  mcHandle     Handle representing a multicast object.
 * \param[in]  dev          Device that hosts the memory allocation.
 * \param[in]  mcOffset     Offset into the multicast object.
 * \param[in]  size         Desired size to unbind.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_NOT_SUPPORTED
 *
 * \sa ::cuMulticastBindMem, ::cuMulticastBindAddr
 *)
function cuMulticastUnbind(mcHandle : CUmemGenericAllocationHandle; dev : CUdevice; mcOffset : size_t; size : size_t):CUresult;  WINAPI; external libcuda;

(**
* \brief Calculates either the minimal or recommended granularity for multicast object
*
* Calculates either the minimal or recommended granularity for a given set of
* multicast object properties and returns it in granularity.  This granularity
* can be used as a multiple for size, bind offsets and address mappings of the
* multicast object.
*
* \param[out] granularity Returned granularity.
* \param[in]  prop        Properties of the multicast object.
* \param[in]  option      Determines which granularity to return.
*
* \returns
* ::CUDA_SUCCESS,
* ::CUDA_ERROR_INVALID_VALUE,
* ::CUDA_ERROR_NOT_INITIALIZED,
* ::CUDA_ERROR_DEINITIALIZED,
* ::CUDA_ERROR_NOT_PERMITTED,
* ::CUDA_ERROR_NOT_SUPPORTED
*
* \sa ::cuMulticastCreate, ::cuMulticastBindMem, ::cuMulticastBindAddr, ::cuMulticastUnbind
*)
function cuMulticastGetGranularity(granularity : Psize_t; const prop : PCUmulticastObjectProp; option : CUmulticastGranularity_flags):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_MULTICAST *)

(**
 * \defgroup CUDA_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 *
 * \section CUDA_UNIFIED_overview Overview
 *
 * CUDA devices can share a unified address space with the host.
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be
 * used to access memory from the host program and from a kernel
 * running on the device (with exceptions enumerated below).
 *
 * \section CUDA_UNIFIED_support Supported Platforms
 *
 * Whether or not a device supports unified addressing may be
 * queried by calling ::cuDeviceGetAttribute() with the device
 * attribute ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.
 *
 * Unified addressing is automatically enabled in 64-bit processes
 *
 * \section CUDA_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device
 * memory, one may want to know on which CUDA device the memory
 * resides.  These properties may be queried using the function
 * ::cuPointerGetAttribute()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to the various copy functions in the
 * CUDA API.  The function ::cuMemcpy() may be used to perform a copy
 * between two pointers, ignoring whether they point to host or device
 * memory (making ::cuMemcpyHtoD(), ::cuMemcpyDtoD(), and ::cuMemcpyDtoH()
 * unnecessary for devices supporting unified addressing).  For
 * multidimensional copies, the memory type ::CU_MEMORYTYPE_UNIFIED may be
 * used to specify that the CUDA driver should infer the location of the
 * pointer from its value.
 *
 * \section CUDA_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated in all contexts using ::cuMemAllocHost() and
 * ::cuMemHostAlloc() is always directly accessible from all contexts on
 * all devices that support unified addressing.  This is the case regardless
 * of whether or not the flags ::CU_MEMHOSTALLOC_PORTABLE and
 * ::CU_MEMHOSTALLOC_DEVICEMAP are specified.
 *
 * The pointer value through which allocated host memory may be accessed
 * in kernels on all devices that support unified addressing is the same
 * as the pointer value through which that memory is accessed on the host,
 * so it is not necessary to call ::cuMemHostGetDevicePointer() to get the device
 * pointer for these allocations.
 *
 * Note that this is not the case for memory allocated using the flag
 * ::CU_MEMHOSTALLOC_WRITECOMBINED, as discussed below.
 *
 * \section CUDA_UNIFIED_autopeerregister Automatic Registration of Peer Memory
 *
 * Upon enabling direct access from a context that supports unified addressing
 * to another peer context that supports unified addressing using
 * ::cuCtxEnablePeerAccess() all memory allocated in the peer context using
 * ::cuMemAlloc() and ::cuMemAllocPitch() will immediately be accessible
 * by the current context.  The device pointer value through
 * which any peer memory may be accessed in the current context
 * is the same pointer value through which that memory may be
 * accessed in the peer context.
 *
 * \section CUDA_UNIFIED_exceptions Exceptions, Disjoint Addressing
 *
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::cuMemHostRegister() and host memory
 * allocated using the flag ::CU_MEMHOSTALLOC_WRITECOMBINED.  For these
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all
 * contexts that support unified addressing.
 *
 * This device address may be queried using ::cuMemHostGetDevicePointer()
 * when a context using unified addressing is current.  Either the host
 * or the unified device pointer value may be used to refer to this memory
 * through ::cuMemcpy() and similar functions using the
 * ::CU_MEMORYTYPE_UNIFIED memory type.
 *
 *)

(**
 * \brief Returns information about a pointer
 *
 * The supported attributes are:
 *
 * - ::CU_POINTER_ATTRIBUTE_CONTEXT:
 *
 *      Returns in \p *data the ::CUcontext in which \p ptr was allocated or
 *      registered.
 *      The type of \p data must be ::CUcontext *.
 *
 *      If \p ptr was not allocated by, mapped by, or registered with
 *      a ::CUcontext which uses unified virtual addressing then
 *      ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
 *
 *      Returns in \p *data the physical memory type of the memory that
 *      \p ptr addresses as a ::CUmemorytype enumerated value.
 *      The type of \p data must be longint.
 *
 *      If \p ptr addresses device memory then \p *data is set to
 *      ::CU_MEMORYTYPE_DEVICE.  The particular ::CUdevice on which the
 *      memory resides is the ::CUdevice of the ::CUcontext returned by the
 *      ::CU_POINTER_ATTRIBUTE_CONTEXT attribute of \p ptr.
 *
 *      If \p ptr addresses host memory then \p *data is set to
 *      ::CU_MEMORYTYPE_HOST.
 *
 *      If \p ptr was not allocated by, mapped by, or registered with
 *      a ::CUcontext which uses unified virtual addressing then
 *      ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 *      If the current ::CUcontext does not support unified virtual
 *      addressing then ::CUDA_ERROR_INVALID_CONTEXT is returned.
 *
 * - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
 *
 *      Returns in \p *data the device pointer value through which
 *      \p ptr may be accessed by kernels running in the current
 *      ::CUcontext.
 *      The type of \p data must be CUdeviceptr *.
 *
 *      If there exists no device pointer value through which
 *      kernels running in the current ::CUcontext may access
 *      \p ptr then ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 *      If there is no current ::CUcontext then
 *      ::CUDA_ERROR_INVALID_CONTEXT is returned.
 *
 *      Except in the exceptional disjoint addressing cases discussed
 *      below, the value returned in \p *data will equal the input
 *      value \p ptr.
 *
 * - ::CU_POINTER_ATTRIBUTE_HOST_POINTER:
 *
 *      Returns in \p *data the host pointer value through which
 *      \p ptr may be accessed by by the host program.
 *      The type of \p data must be pointer *.
 *      If there exists no host pointer value through which
 *      the host program may directly access \p ptr then
 *      ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 *      Except in the exceptional disjoint addressing cases discussed
 *      below, the value returned in \p *data will equal the input
 *      value \p ptr.
 *
 * - ::CU_POINTER_ATTRIBUTE_P2P_TOKENS:
 *
 *      Returns in \p *data two tokens for use with the nv-p2p.h Linux
 *      kernel interface. \p data must be a struct of type
 *      CUDA_POINTER_ATTRIBUTE_P2P_TOKENS.
 *
 *      \p ptr must be a pointer to memory obtained from :cuMemAlloc().
 *      Note that p2pToken and vaSpaceToken are only valid for the
 *      lifetime of the source allocation. A subsequent allocation at
 *      the same address may return completely different tokens.
 *      Querying this attribute has a side effect of setting the attribute
 *      ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS for the region of memory that
 *      \p ptr points to.
 *
 * - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
 *
 *      A boolean attribute which when set, ensures that synchronous memory operations
 *      initiated on the region of memory that \p ptr points to will always synchronize.
 *      See further documentation in the section titled "API synchronization behavior"
 *      to learn more about cases when synchronous memory operations can
 *      exhibit asynchronous behavior.
 *
 * - ::CU_POINTER_ATTRIBUTE_BUFFER_ID:
 *
 *      Returns in \p *data a buffer ID which is guaranteed to be unique within the process.
 *      \p data must point to an uint64.
 *
 *      \p ptr must be a pointer to memory obtained from a CUDA memory allocation API.
 *      Every memory allocation from any of the CUDA memory allocation APIs will
 *      have a unique ID over a process lifetime. Subsequent allocations do not reuse IDs
 *      from previous freed allocations. IDs are only unique within a single process.
 *
 *
 * - ::CU_POINTER_ATTRIBUTE_IS_MANAGED:
 *
 *      Returns in \p *data a boolean that indicates whether the pointer points to
 *      managed memory or not.
 *
 *      If \p ptr is not a valid CUDA pointer then ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * - ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
 *
 *      Returns in \p *data an integer representing a device ordinal of a device against
 *      which the memory was allocated or registered.
 *
 * - ::CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
 *
 *      Returns in \p *data a boolean that indicates if this pointer maps to
 *      an allocation that is suitable for ::cudaIpcGetMemHandle.
 *
 * - ::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
 *
 *      Returns in \p *data the starting address for the allocation referenced
 *      by the device pointer \p ptr.  Note that this is not necessarily the
 *      address of the mapped region, but the address of the mappable address
 *      range \p ptr references (e.g. from ::cuMemAddressReserve).
 *
 * - ::CU_POINTER_ATTRIBUTE_RANGE_SIZE:
 *
 *      Returns in \p *data the size for the allocation referenced by the device
 *      pointer \p ptr.  Note that this is not necessarily the size of the mapped
 *      region, but the size of the mappable address range \p ptr references
 *      (e.g. from ::cuMemAddressReserve).  To retrieve the size of the mapped
 *      region, see ::cuMemGetAddressRange
 *
 * - ::CU_POINTER_ATTRIBUTE_MAPPED:
 *
 *      Returns in \p *data a boolean that indicates if this pointer is in a
 *      valid address range that is mapped to a backing allocation.
 *
 * - ::CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
 *
 *      Returns a bitmask of the allowed handle types for an allocation that may
 *      be passed to ::cuMemExportToShareableHandle.
 * 
 * - ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
 * 
 *      Returns in \p *data the handle to the mempool that the allocation was obtained from.
 *
 * \par
 *
 * Note that for most allocations in the unified virtual address space
 * the host and device pointer for accessing the allocation will be the
 * same.  The exceptions to this are
 *  - user memory registered using ::cuMemHostRegister
 *  - host memory allocated using ::cuMemHostAlloc with the
 *    ::CU_MEMHOSTALLOC_WRITECOMBINED flag
 * For these types of allocation there will exist separate, disjoint host
 * and device addresses for accessing the allocation.  In particular
 *  - The host address will correspond to an invalid unmapped device address
 *    (which will result in an exception if accessed from the device)
 *  - The device address will correspond to an invalid unmapped host address
 *    (which will result in an exception if accessed from the host).
 * For these types of allocations, querying ::CU_POINTER_ATTRIBUTE_HOST_POINTER
 * and ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER may be used to retrieve the host
 * and device addresses from either address.
 *
 * \param data      - Returned pointer attribute value
 * \param attribute - Pointer attribute to query
 * \param ptr       - Pointer
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuPointerSetAttribute,
 * ::cuMemAlloc,
 * ::cuMemFree,
 * ::cuMemAllocHost,
 * ::cuMemFreeHost,
 * ::cuMemHostAlloc,
 * ::cuMemHostRegister,
 * ::cuMemHostUnregister,
 * ::cudaPointerGetAttributes
 *)
function cuPointerGetAttribute(data : Pointer; attribute : CUpointer_attribute; ptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Prefetches memory to the specified destination device
 *
 * Prefetches memory to the specified destination device.  \p devPtr is the
 * base device pointer of the memory to be prefetched and \p dstDevice is the
 * destination device. \p count specifies the number of bytes to copy. \p hStream
 * is the stream in which the operation is enqueued. The memory range must refer
 * to managed memory allocated via ::cuMemAllocManaged or declared via __managed__ variables.
 *
 * Passing in CU_DEVICE_CPU for \p dstDevice will prefetch the data to host memory. If
 * \p dstDevice is a GPU, then the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
 * must be non-zero. Additionally, \p hStream must be associated with a device that has a
 * non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
 *
 * The start address and end address of the memory range will be rounded down and rounded up
 * respectively to be aligned to CPU page size before the prefetch operation is enqueued
 * in the stream.
 *
 * If no physical memory has been allocated for this region, then this memory region
 * will be populated and mapped on the destination device. If there's insufficient
 * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
 * ::cuMemAllocManaged allocations to host memory in order to make room. Device memory
 * allocated using ::cuMemAlloc or ::cuArrayCreate will not be evicted.
 *
 * By default, any mappings to the previous location of the migrated pages are removed and
 * mappings for the new location are only setup on \p dstDevice. The exact behavior however
 * also depends on the settings applied to this memory range via ::cuMemAdvise as described
 * below:
 *
 * If ::CU_MEM_ADVISE_SET_READ_MOSTLY was set on any subset of this memory range,
 * then that subset will create a read-only copy of the pages on \p dstDevice.
 *
 * If ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION was called on any subset of this memory
 * range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
 * preferred location of any pages in the memory range.
 *
 * If ::CU_MEM_ADVISE_SET_ACCESSED_BY was called on any subset of this memory range,
 * then mappings to those pages from all the appropriate processors are updated to
 * refer to the new location if establishing such a mapping is possible. Otherwise,
 * those mappings are cleared.
 *
 * Note that this API is not required for functionality and only serves to improve performance
 * by allowing the application to migrate data to a suitable location before it is accessed.
 * Memory accesses to this range are always coherent and are allowed even when the data is
 * actively being migrated.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param devPtr    - Pointer to be prefetched
 * \param count     - Size in bytes
 * \param dstDevice - Destination device to prefetch to
 * \param hStream    - Stream to enqueue prefetch operation
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuMemcpy, ::cuMemcpyPeer, ::cuMemcpyAsync,
 * ::cuMemcpy3DPeerAsync, ::cuMemAdvise,
 * ::cudaMemPrefetchAsync
 *)
function cuMemPrefetchAsync(devPtr : CUdeviceptr; count : size_t; dstDevice : CUdevice; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Advise about the usage of a given memory range
 *
 * Advise the Unified Memory subsystem about the usage pattern for the memory range
 * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
 * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
 * advice is applied. The memory range must refer to managed memory allocated via ::cuMemAllocManaged
 * or declared via __managed__ variables. The memory range could also refer to system-allocated pageable
 * memory provided it represents a valid, host-accessible region of memory and all additional constraints
 * imposed by \p advice as outlined below are also satisfied. Specifying an invalid system-allocated pageable
 * memory range results in an error being returned.
 *
 * The \p advice parameter can take the following values:
 * - ::CU_MEM_ADVISE_SET_READ_MOSTLY: This implies that the data is mostly going to be read
 * from and only occasionally written to. Any read accesses from any processor to this region will create a
 * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::cuMemPrefetchAsync
 * is called on this region, it will create a read-only copy of the data on the destination processor.
 * If any processor writes to this region, all copies of the corresponding page will be invalidated
 * except for the one where the write occurred. The \p device argument is ignored for this advice.
 * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
 * that has a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
 * Also, if a context is created on a device that does not have the device attribute
 * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS set, then read-duplication will not occur until
 * all such contexts are destroyed.
 * If the memory region refers to valid system-allocated pageable memory, then the accessing device must
 * have a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS for a read-only
 * copy to be created on that device. Note however that if the accessing device also has a non-zero value for the
 * device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, then setting this advice
 * will not create a read-only copy when that device accesses this memory region.
 *
 * - ::CU_MEM_ADVISE_UNSET_READ_MOSTLY:  Undoes the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY and also prevents the
 * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
 * copies of the data will be collapsed into a single copy. The location for the collapsed
 * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
 * copies was resident at that location. Otherwise, the location chosen is arbitrary.
 *
 * - ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION: This advice sets the preferred location for the
 * data to be the memory belonging to \p device. Passing in CU_DEVICE_CPU for \p device sets the
 * preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
 * device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Setting the preferred location
 * does not cause data to migrate to that location immediately. Instead, it guides the migration policy
 * when a fault occurs on that memory region. If the data is already in its preferred location and the
 * faulting processor can establish a mapping without requiring the data to be migrated, then
 * data migration will be avoided. On the other hand, if the data is not in its preferred location
 * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
 * it. It is important to note that setting the preferred location does not prevent data prefetching
 * done using ::cuMemPrefetchAsync.
 * Having a preferred location can override the page thrash detection and resolution logic in the Unified
 * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
 * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
 * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
 * If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice, unless read accesses from
 * \p device will not result in a read-only copy being created on that device as outlined in description for
 * the advice ::CU_MEM_ADVISE_SET_READ_MOSTLY.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. Additionally, if \p device has
 * a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
 * then this call has no effect. Note however that this behavior may change in the future.
 *
 * - ::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: Undoes the effect of ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION
 * and changes the preferred location to none.
 *
 * - ::CU_MEM_ADVISE_SET_ACCESSED_BY: This advice implies that the data will be accessed by \p device.
 * Passing in ::CU_DEVICE_CPU for \p device will set the advice for the CPU. If \p device is a GPU, then
 * the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS must be non-zero.
 * This advice does not cause data migration and has no impact on the location of the data per se. Instead,
 * it causes the data to always be mapped in the specified processor's page tables, as long as the
 * location of the data permits a mapping to be established. If the data gets migrated for any reason,
 * the mappings are updated accordingly.
 * This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
 * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
 * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
 * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
 * migration may be too high. But preventing faults can still help improve performance, and so having
 * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
 * to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
 * ::CU_MEM_ADVISE_SET_ACCESSED_BY flag set for this data will now have its mapping updated to point to the
 * page in host memory.
 * If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice. Additionally, if the
 * preferred location of this memory region or any subset of it is also \p device, then the policies
 * associated with ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of this advice.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. Additionally, if \p device has
 * a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
 * then this call has no effect.
 *
 * - ::CU_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of ::CU_MEM_ADVISE_SET_ACCESSED_BY. Any mappings to
 * the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. Additionally, if \p device has
 * a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
 * then this call has no effect.
 *
 * \param devPtr - Pointer to memory to set the advice for
 * \param count  - Size in bytes of the memory range
 * \param advice - Advice to be applied for the specified memory range
 * \param device - Device to apply the advice for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuMemcpy, ::cuMemcpyPeer, ::cuMemcpyAsync,
 * ::cuMemcpy3DPeerAsync, ::cuMemPrefetchAsync,
 * ::cudaMemAdvise
 *)
function cuMemAdvise(devPtr : CUdeviceptr; count : size_t; advice : CUmem_advise; device : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query an attribute of a given memory range
 *
 * Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::cuMemAllocManaged or declared via
 * __managed__ variables.
 *
 * The \p attribute parameter can take the following values:
 * - ::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: If this attribute is specified, \p data will be interpreted
 * as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
 * memory range have read-duplication enabled, or 0 otherwise.
 * - ::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: If this attribute is specified, \p data will be
 * interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
 * id if all pages in the memory range have that GPU as their preferred location, or it will be CU_DEVICE_CPU
 * if all pages in the memory range have the CPU as their preferred location, or it will be CU_DEVICE_INVALID
 * if either all the pages don't have the same preferred location or some of the pages don't have a
 * preferred location at all. Note that the actual location of the pages in the memory range at the time of
 * the query may be different from the preferred location.
 * - ::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: If this attribute is specified, \p data will be interpreted
 * as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
 * will be a list of device ids that had ::CU_MEM_ADVISE_SET_ACCESSED_BY set for that entire memory range.
 * If any device does not have that advice set for the entire memory range, that device will not be included.
 * If \p data is larger than the number of devices that have that advice set for that memory range,
 * CU_DEVICE_INVALID will be returned in all the extra space provided. For ex., if \p dataSize is 12
 * (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
 * { 0, CU_DEVICE_INVALID, CU_DEVICE_INVALID }. If \p data is smaller than the number of devices that have
 * that advice set, then only as many devices will be returned as can fit in the array. There is no
 * guarantee on which specific devices will be returned, however.
 * - ::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: If this attribute is specified, \p data will be
 * interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
 * to which all pages in the memory range were prefetched explicitly via ::cuMemPrefetchAsync. This will either be
 * a GPU id or CU_DEVICE_CPU depending on whether the last location for prefetch was a GPU or the CPU
 * respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
 * prefetched to the same location, CU_DEVICE_INVALID will be returned. Note that this simply returns the
 * last location that the application requested to prefetch the memory range to. It gives no indication as to
 * whether the prefetch operation to that location has completed or even begun.
 *
 * \param data      - A pointers to a memory location where the result
 *                    of each attribute query will be written to.
 * \param dataSize  - Array containing the size of data
 * \param attribute - The attribute to query
 * \param devPtr    - Start of the range to query
 * \param count     - Size of the range to query
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::cuMemRangeGetAttributes, ::cuMemPrefetchAsync,
 * ::cuMemAdvise,
 * ::cudaMemRangeGetAttribute
 *)
function cuMemRangeGetAttribute(data : Pointer; dataSize : size_t; attribute : CUmem_range_attribute; devPtr : CUdeviceptr; count : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query attributes of a given memory range.
 *
 * Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::cuMemAllocManaged or declared via
 * __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
 * entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
 * The results of the query will be stored in \p data.
 *
 * The list of supported attributes are given below. Please refer to ::cuMemRangeGetAttribute for
 * attribute descriptions and restrictions.
 *
 * - ::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
 * - ::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
 * - ::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
 * - ::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
 *
 * \param data          - A two-dimensional array containing pointers to memory
 *                        locations where the result of each attribute query will be written to.
 * \param dataSizes     - Array containing the sizes of each result
 * \param attributes    - An array of attributes to query
 *                        (numAttributes and the number of attributes in this array should match)
 * \param numAttributes - Number of attributes to query
 * \param devPtr        - Start of the range to query
 * \param count         - Size of the range to query
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa ::cuMemRangeGetAttribute, ::cuMemAdvise,
 * ::cuMemPrefetchAsync,
 * ::cudaMemRangeGetAttributes
 *)
function cuMemRangeGetAttributes(data : Ppointer; dataSizes : Psize_t; attributes : PCUmem_range_attribute; numAttributes : size_t; devPtr : CUdeviceptr; count : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Set attributes on a previously allocated memory region
 *
 * The supported attributes are:
 *
 * - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
 *
 *      A boolean attribute that can either be set (1) or unset (0). When set,
 *      the region of memory that \p ptr points to is guaranteed to always synchronize
 *      memory operations that are synchronous. If there are some previously initiated
 *      synchronous memory operations that are pending when this attribute is set, the
 *      function does not return until those memory operations are complete.
 *      See further documentation in the section titled "API synchronization behavior"
 *      to learn more about cases when synchronous memory operations can
 *      exhibit asynchronous behavior.
 *      \p value will be considered as a pointer to an unsigned integer to which this attribute is to be set.
 *
 * \param value     - Pointer to memory containing the value to be set
 * \param attribute - Pointer attribute to set
 * \param ptr       - Pointer to a memory region allocated using CUDA memory allocation APIs
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa ::cuPointerGetAttribute,
 * ::cuPointerGetAttributes,
 * ::cuMemAlloc,
 * ::cuMemFree,
 * ::cuMemAllocHost,
 * ::cuMemFreeHost,
 * ::cuMemHostAlloc,
 * ::cuMemHostRegister,
 * ::cuMemHostUnregister
 *)
function cuPointerSetAttribute(const value : Pointer; attribute : CUpointer_attribute; ptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns information about a pointer.
 *
 * The supported attributes are (refer to ::cuPointerGetAttribute for attribute descriptions and restrictions):
 *
 * - ::CU_POINTER_ATTRIBUTE_CONTEXT
 * - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE
 * - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER
 * - ::CU_POINTER_ATTRIBUTE_HOST_POINTER
 * - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
 * - ::CU_POINTER_ATTRIBUTE_BUFFER_ID
 * - ::CU_POINTER_ATTRIBUTE_IS_MANAGED
 * - ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
 * - ::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
 * - ::CU_POINTER_ATTRIBUTE_RANGE_SIZE
 * - ::CU_POINTER_ATTRIBUTE_MAPPED
 * - ::CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE
 * - ::CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
 * - ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
 *
 * \param numAttributes - Number of attributes to query
 * \param attributes    - An array of attributes to query
 *                      (numAttributes and the number of attributes in this array should match)
 * \param data          - A two-dimensional array containing pointers to memory
 *                      locations where the result of each attribute query will be written to.
 * \param ptr           - Pointer to query
 *
 * Unlike ::cuPointerGetAttribute, this function will not return an error when the \p ptr
 * encountered is not a valid CUDA pointer. Instead, the attributes are assigned default NULL values
 * and CUDA_SUCCESS is returned.
 *
 * If \p ptr was not allocated by, mapped by, or registered with a ::CUcontext which uses UVA
 * (Unified Virtual Addressing), ::CUDA_ERROR_INVALID_CONTEXT is returned.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuPointerGetAttribute,
 * ::cuPointerSetAttribute,
 * ::cudaPointerGetAttributes
 *)
function cuPointerGetAttributes(numAttributes : longword; attributes : PCUpointer_attribute; data : Ppointer; ptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_UNIFIED *)

(**
 * \defgroup CUDA_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Create a stream
 *
 * Creates a stream and returns a handle in \p phStream.  The \p Flags argument
 * determines behaviors of the stream.
 *
 * Valid values for \p Flags are:
 * - ::CU_STREAM_DEFAULT: Default stream creation flag.
 * - ::CU_STREAM_NON_BLOCKING: Specifies that work running in the created
 *   stream may run concurrently with work in stream 0 (the NULL stream), and that
 *   the created stream should perform no implicit synchronization with stream 0.
 *
 * \param phStream - Returned newly created stream
 * \param Flags    - Parameters for stream creation
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuStreamDestroy,
 * ::cuStreamCreateWithPriority,
 * ::cuStreamGetPriority,
 * ::cuStreamGetFlags,
 * ::cuStreamWaitEvent,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamAddCallback,
 * ::cudaStreamCreate,
 * ::cudaStreamCreateWithFlags
 *)
function cuStreamCreate(phStream : PCUstream; Flags : CUstream_flags):CUresult;  WINAPI; external libcuda;

(**
 * \brief Create a stream with the given priority
 *
 * Creates a stream with the specified priority and returns a handle in \p phStream.
 * This API alters the scheduler priority of work in the stream. Work in a higher
 * priority stream may preempt work already executing in a low priority stream.
 *
 * \p priority follows a convention where lower numbers represent higher priorities.
 * '0' represents default priority. The range of meaningful numerical priorities can
 * be queried using ::cuCtxGetStreamPriorityRange. If the specified priority is
 * outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
 * it will automatically be clamped to the lowest or the highest number in the range.
 *
 * \param phStream    - Returned newly created stream
 * \param flags       - Flags for stream creation. See ::cuStreamCreate for a list of
 *                      valid flags
 * \param priority    - Stream priority. Lower numbers represent higher priorities.
 *                      See ::cuCtxGetStreamPriorityRange for more information about
 *                      meaningful stream priorities that can be passed.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \note Stream priorities are supported only on GPUs
 * with compute capability 3.5 or higher.
 *
 * \note In the current implementation, only compute kernels launched in
 * priority streams are affected by the stream's priority. Stream priorities have
 * no effect on host-to-device and device-to-host memory operations.
 *
 * \sa ::cuStreamDestroy,
 * ::cuStreamCreate,
 * ::cuStreamGetPriority,
 * ::cuCtxGetStreamPriorityRange,
 * ::cuStreamGetFlags,
 * ::cuStreamWaitEvent,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamAddCallback,
 * ::cudaStreamCreateWithPriority
 *)
function cuStreamCreateWithPriority(phStream : PCUstream; flags : longword; priority : longint):CUresult;  WINAPI; external libcuda;


(**
 * \brief Query the priority of a given stream
 *
 * Query the priority of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
 * and return the priority in \p priority. Note that if the stream was created with a
 * priority outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
 * this function returns the clamped priority.
 * See ::cuStreamCreateWithPriority for details about priority clamping.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param priority   - Pointer to a signed integer in which the stream's priority is returned
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuStreamDestroy,
 * ::cuStreamCreate,
 * ::cuStreamCreateWithPriority,
 * ::cuCtxGetStreamPriorityRange,
 * ::cuStreamGetFlags,
 * ::cudaStreamGetPriority
 *)
function cuStreamGetPriority(hStream : CUstream; priority : Plongint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query the flags of a given stream
 *
 * Query the flags of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
 * and return the flags in \p flags.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param flags      - Pointer to an unsigned integer in which the stream's flags are returned
 *                     The value returned in \p flags is a logical 'OR' of all flags that
 *                     were used while creating this stream. See ::cuStreamCreate for the list
 *                     of valid flags
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuStreamDestroy,
 * ::cuStreamCreate,
 * ::cuStreamGetPriority,
 * ::cudaStreamGetFlags
 *)
function cuStreamGetFlags(hStream : CUstream; flags : Plongword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the unique Id associated with the stream handle supplied
 *
 * Returns in \p streamId the unique Id which is associated with the given stream handle.
 * The Id is unique for the life of the program.
 * 
 * The stream handle \p hStream can refer to any of the following:
 * <ul>
 *   <li>a stream created via any of the CUDA driver APIs such as ::cuStreamCreate
 *   and ::cuStreamCreateWithPriority, or their runtime API equivalents such as
 *   ::cudaStreamCreate, ::cudaStreamCreateWithFlags and ::cudaStreamCreateWithPriority.
 *   Passing an invalid handle will result in undefined behavior.</li>
 *   <li>any of the special streams such as the NULL stream, ::CU_STREAM_LEGACY and
 *   ::CU_STREAM_PER_THREAD. The runtime API equivalents of these are also accepted,
 *   which are NULL, ::cudaStreamLegacy and ::cudaStreamPerThread respectively.</li>
 * </ul>
 *
 * \param hStream    - Handle to the stream to be queried
 * \param streamId   - Pointer to store the Id of the stream
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuStreamDestroy,
 * ::cuStreamCreate,
 * ::cuStreamGetPriority,
 * ::cudaStreamGetId
 *)
function cuStreamGetId(hStream : CUstream; streamId : Puint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query the context associated with a stream
 *
 * Returns the CUDA context that the stream is associated with.
 *
 * The stream handle \p hStream can refer to any of the following:
 * <ul>
 *   <li>a stream created via any of the CUDA driver APIs such as ::cuStreamCreate
 *   and ::cuStreamCreateWithPriority, or their runtime API equivalents such as
 *   ::cudaStreamCreate, ::cudaStreamCreateWithFlags and ::cudaStreamCreateWithPriority.
 *   The returned context is the context that was active in the calling thread when the
 *   stream was created. Passing an invalid handle will result in undefined behavior.</li>
 *   <li>any of the special streams such as the NULL stream, ::CU_STREAM_LEGACY and
 *   ::CU_STREAM_PER_THREAD. The runtime API equivalents of these are also accepted,
 *   which are NULL, ::cudaStreamLegacy and ::cudaStreamPerThread respectively.
 *   Specifying any of the special handles will return the context current to the
 *   calling thread. If no context is current to the calling thread,
 *   ::CUDA_ERROR_INVALID_CONTEXT is returned.</li>
 * </ul>
 *
 * \param hStream - Handle to the stream to be queried
 * \param pctx    - Returned context associated with the stream
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * \notefnerr
 *
 * \sa ::cuStreamDestroy,
 * ::cuStreamCreateWithPriority,
 * ::cuStreamGetPriority,
 * ::cuStreamGetFlags,
 * ::cuStreamWaitEvent,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamAddCallback,
 * ::cudaStreamCreate,
 * ::cudaStreamCreateWithFlags
 *)
function cuStreamGetCtx(hStream : CUstream; pctx : PCUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Make a compute stream wait on an event
 *
 * Makes all future work submitted to \p hStream wait for all work captured in
 * \p hEvent.  See ::cuEventRecord() for details on what is captured by an event.
 * The synchronization will be performed efficiently on the device when applicable.
 * \p hEvent may be from a different context or device than \p hStream.
 *
 * flags include:
 * - ::CU_EVENT_WAIT_DEFAULT: Default event creation flag.
 * - ::CU_EVENT_WAIT_EXTERNAL: Event is captured in the graph as an external
 *   event node when performing stream capture. This flag is invalid outside
 *   of stream capture.
 *
 * \param hStream - Stream to wait
 * \param hEvent  - Event to wait on (may not be NULL)
 * \param Flags   - See ::CUevent_capture_flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuEventRecord,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamAddCallback,
 * ::cuStreamDestroy,
 * ::cudaStreamWaitEvent
 *)
function cuStreamWaitEvent(hStream : CUstream; hEvent : CUevent; Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Add a callback to a compute stream
 *
 * \note This function is slated for eventual deprecation and removal. If
 * you do not require the callback to execute in case of a device error,
 * consider using ::cuLaunchHostFunc. Additionally, this function is not
 * supported with ::cuStreamBeginCapture and ::cuStreamEndCapture, unlike
 * ::cuLaunchHostFunc.
 *
 * Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each
 * cuStreamAddCallback call, the callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 *
 * The callback may be passed ::CUDA_SUCCESS or an error code.  In the event
 * of a device error, all subsequently executed callbacks will receive an
 * appropriate ::CUresult.
 *
 * Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
 * will result in ::CUDA_ERROR_NOT_PERMITTED.  Callbacks must not perform any
 * synchronization that may depend on outstanding device work or other callbacks
 * that are not mandated to run earlier.  Callbacks without a mandated order
 * (in independent streams) execute in undefined order and may be serialized.
 *
 * For the purposes of Unified Memory, callback execution makes a number of
 * guarantees:
 * <ul>
 *   <li>The callback stream is considered idle for the duration of the
 *   callback.  Thus, for example, a callback may always use memory attached
 *   to the callback stream.</li>
 *   <li>The start of execution of a callback has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the callback.  It thus synchronizes streams which have been "joined"
 *   prior to the callback.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding host functions and stream callbacks
 *   have executed.  Thus, for
 *   example, a callback might use global attached memory even if work has
 *   been added to another stream, if the work has been ordered behind the
 *   callback with an event.</li>
 *   <li>Completion of a callback does not cause a stream to become
 *   active except as described above.  The callback stream will remain idle
 *   if no device work follows the callback, and will remain idle across
 *   consecutive callbacks without device work in between.  Thus, for example,
 *   stream synchronization can be done by signaling from a callback at the
 *   end of the stream.</li>
 * </ul>
 *
 * \param hStream  - Stream to add callback to
 * \param callback - The function to call once preceding stream operations are complete
 * \param userData - User specified data to be passed to the callback function
 * \param flags    - Reserved for future use, must be 0
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamWaitEvent,
 * ::cuStreamDestroy,
 * ::cuMemAllocManaged,
 * ::cuStreamAttachMemAsync,
 * ::cuLaunchHostFunc,
 * ::cudaStreamAddCallback
 *)
function cuStreamAddCallback(hStream : CUstream; callback : CUstreamCallback; userData : Pointer; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Begins graph capture on a stream
 *
 * Begin graph capture on \p hStream. When a stream is in capture mode, all operations
 * pushed into the stream will not be executed, but will instead be captured into
 * a graph, which will be returned via ::cuStreamEndCapture. Capture may not be initiated
 * if \p stream is CU_STREAM_LEGACY. Capture must be ended on the same stream in which
 * it was initiated, and it may only be initiated if the stream is not already in capture
 * mode. The capture mode may be queried via ::cuStreamIsCapturing. A unique id
 * representing the capture sequence may be queried via ::cuStreamGetCaptureInfo.
 *
 * If \p mode is not ::CU_STREAM_CAPTURE_MODE_RELAXED, ::cuStreamEndCapture must be
 * called on this stream from the same thread.
 *
 * \param hStream - Stream in which to initiate capture
 * \param mode    - Controls the interaction of this capture sequence with other API
 *                  calls that are potentially unsafe. For more details see
 *                  ::cuThreadExchangeStreamCaptureMode.
 *
 * \note Kernels captured using this API must not use texture and surface references.
 *       Reading or writing through any texture or surface reference is undefined
 *       behavior. This restriction does not apply to texture and surface objects.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cuStreamCreate,
 * ::cuStreamIsCapturing,
 * ::cuStreamEndCapture,
 * ::cuThreadExchangeStreamCaptureMode
 *)
function cuStreamBeginCapture(hStream : CUstream; mode : CUstreamCaptureMode):CUresult;  WINAPI; external libcuda;

(**
 * \brief Swaps the stream capture interaction mode for a thread
 *
 * Sets the calling thread's stream capture interaction mode to the value contained
 * in \p *mode, and overwrites \p *mode with the previous mode for the thread. To
 * facilitate deterministic behavior across function or module boundaries, callers
 * are encouraged to use this API in a push-pop fashion: \code
     CUstreamCaptureMode mode = desiredMode;
     cuThreadExchangeStreamCaptureMode(&mode);
     ...
     cuThreadExchangeStreamCaptureMode(&mode); // restore previous mode
 * \endcode
 *
 * During stream capture (see ::cuStreamBeginCapture), some actions, such as a call
 * to ::cudaMalloc, may be unsafe. In the case of ::cudaMalloc, the operation is
 * not enqueued asynchronously to a stream, and is not observed by stream capture.
 * Therefore, if the sequence of operations captured via ::cuStreamBeginCapture
 * depended on the allocation being replayed whenever the graph is launched, the
 * captured graph would be invalid.
 *
 * Therefore, stream capture places restrictions on API calls that can be made within
 * or concurrently to a ::cuStreamBeginCapture-::cuStreamEndCapture sequence. This
 * behavior can be controlled via this API and flags to ::cuStreamBeginCapture.
 *
 * A thread's mode is one of the following:
 * - \p CU_STREAM_CAPTURE_MODE_GLOBAL: This is the default mode. If the local thread has
 *   an ongoing capture sequence that was not initiated with
 *   \p CU_STREAM_CAPTURE_MODE_RELAXED at \p cuStreamBeginCapture, or if any other thread
 *   has a concurrent capture sequence initiated with \p CU_STREAM_CAPTURE_MODE_GLOBAL,
 *   this thread is prohibited from potentially unsafe API calls.
 * - \p CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: If the local thread has an ongoing capture
 *   sequence not initiated with \p CU_STREAM_CAPTURE_MODE_RELAXED, it is prohibited
 *   from potentially unsafe API calls. Concurrent capture sequences in other threads
 *   are ignored.
 * - \p CU_STREAM_CAPTURE_MODE_RELAXED: The local thread is not prohibited from potentially
 *   unsafe API calls. Note that the thread is still prohibited from API calls which
 *   necessarily conflict with stream capture, for example, attempting ::cuEventQuery
 *   on an event that was last recorded inside a capture sequence.
 *
 * \param mode - Pointer to mode value to swap with the current mode
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cuStreamBeginCapture
 *)
function cuThreadExchangeStreamCaptureMode(mode : PCUstreamCaptureMode):CUresult;  WINAPI; external libcuda;

(**
 * \brief Ends capture on a stream, returning the captured graph
 *
 * End capture on \p hStream, returning the captured graph via \p phGraph.
 * Capture must have been initiated on \p hStream via a call to ::cuStreamBeginCapture.
 * If capture was invalidated, due to a violation of the rules of stream capture, then
 * a NULL graph will be returned.
 *
 * If the \p mode argument to ::cuStreamBeginCapture was not
 * ::CU_STREAM_CAPTURE_MODE_RELAXED, this call must be from the same thread as
 * ::cuStreamBeginCapture.
 *
 * \param hStream - Stream to query
 * \param phGraph - The captured graph
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD
 * \notefnerr
 *
 * \sa
 * ::cuStreamCreate,
 * ::cuStreamBeginCapture,
 * ::cuStreamIsCapturing
 *)
function cuStreamEndCapture(hStream : CUstream; phGraph : PCUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a stream's capture status
 *
 * Return the capture status of \p hStream via \p captureStatus. After a successful
 * call, \p *captureStatus will contain one of the following:
 * - ::CU_STREAM_CAPTURE_STATUS_NONE: The stream is not capturing.
 * - ::CU_STREAM_CAPTURE_STATUS_ACTIVE: The stream is capturing.
 * - ::CU_STREAM_CAPTURE_STATUS_INVALIDATED: The stream was capturing but an error
 *   has invalidated the capture sequence. The capture sequence must be terminated
 *   with ::cuStreamEndCapture on the stream where it was initiated in order to
 *   continue using \p hStream.
 *
 * Note that, if this is called on ::CU_STREAM_LEGACY (the "null stream") while
 * a blocking stream in the same context is capturing, it will return
 * ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT and \p *captureStatus is unspecified
 * after the call. The blocking stream capture is not invalidated.
 *
 * When a blocking stream is capturing, the legacy stream is in an
 * unusable state until the blocking stream capture is terminated. The legacy
 * stream is not supported for stream capture, but attempted use would have an
 * implicit dependency on the capturing stream(s).
 *
 * \param hStream       - Stream to query
 * \param captureStatus - Returns the stream's capture status
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
 * \notefnerr
 *
 * \sa
 * ::cuStreamCreate,
 * ::cuStreamBeginCapture,
 * ::cuStreamEndCapture
 *)
function cuStreamIsCapturing(hStream : CUstream; captureStatus : PCUstreamCaptureStatus):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query a stream's capture state
 *
 * Query stream state related to stream capture.
 *
 * If called on ::CU_STREAM_LEGACY (the "null stream") while a stream not created 
 * with ::CU_STREAM_NON_BLOCKING is capturing, returns ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT.
 *
 * Valid data (other than capture status) is returned only if both of the following are true:
 * - the call returns CUDA_SUCCESS
 * - the returned capture status is ::CU_STREAM_CAPTURE_STATUS_ACTIVE
 *
 * \param hStream - The stream to query
 * \param captureStatus_out - Location to return the capture status of the stream; required
 * \param id_out - Optional location to return an id for the capture sequence, which is
 *           unique over the lifetime of the process
 * \param graph_out - Optional location to return the graph being captured into. All
 *           operations other than destroy and node removal are permitted on the graph
 *           while the capture sequence is in progress. This API does not transfer
 *           ownership of the graph, which is transferred or destroyed at
 *           ::cuStreamEndCapture. Note that the graph handle may be invalidated before
 *           end of capture for certain errors. Nodes that are or become
 *           unreachable from the original stream at ::cuStreamEndCapture due to direct
 *           actions on the graph do not trigger ::CUDA_ERROR_STREAM_CAPTURE_UNJOINED.
 * \param dependencies_out - Optional location to store a pointer to an array of nodes.
 *           The next node to be captured in the stream will depend on this set of nodes,
 *           absent operations such as event wait which modify this set. The array pointer
 *           is valid until the next API call which operates on the stream or until end of
 *           capture. The node handles may be copied out and are valid until they or the
 *           graph is destroyed. The driver-owned array may also be passed directly to
 *           APIs that operate on the graph (not the stream) without copying.
 * \param numDependencies_out - Optional location to store the size of the array
 *           returned in dependencies_out.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuStreamBeginCapture,
 * ::cuStreamIsCapturing,
 * ::cuStreamUpdateCaptureDependencies
 *)
function cuStreamGetCaptureInfo(hStream : CUstream; captureStatus_out : PCUstreamCaptureStatus;
        id_out : Pcuuint64_t; graph_out : PCUgraph; const dependencies_out : PPCUgraphNode; numDependencies_out : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Update the set of dependencies in a capturing stream (11.3+)
 *
 * Modifies the dependency set of a capturing stream. The dependency set is the set
 * of nodes that the next captured node in the stream will depend on.
 *
 * Valid flags are ::CU_STREAM_ADD_CAPTURE_DEPENDENCIES and
 * ::CU_STREAM_SET_CAPTURE_DEPENDENCIES. These control whether the set passed to
 * the API is added to the existing set or replaces it. A flags value of 0 defaults
 * to ::CU_STREAM_ADD_CAPTURE_DEPENDENCIES.
 *
 * Nodes that are removed from the dependency set via this API do not result in
 * ::CUDA_ERROR_STREAM_CAPTURE_UNJOINED if they are unreachable from the stream at
 * ::cuStreamEndCapture.
 *
 * Returns ::CUDA_ERROR_ILLEGAL_STATE if the stream is not capturing.
 *
 * This API is new in CUDA 11.3. Developers requiring compatibility across minor
 * versions to CUDA 11.0 should not use this API or provide a fallback.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_ILLEGAL_STATE
 *
 * \sa
 * ::cuStreamBeginCapture,
 * ::cuStreamGetCaptureInfo,
 *)
function cuStreamUpdateCaptureDependencies(hStream : CUstream; dependencies : PCUgraphNode; numDependencies : size_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p hStream to specify stream association of
 * \p length bytes of memory starting from \p dptr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p dptr must point to one of the following types of memories:
 * - managed memory declared using the __managed__ keyword or allocated with
 *   ::cuMemAllocManaged.
 * - a valid host-accessible region of system-allocated pageable memory. This
 *   type of memory may only be specified if the device associated with the
 *   stream reports a non-zero value for the device attribute
 *   ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS.
 *
 * For managed allocations, \p length must be either zero or the entire
 * allocation's size. Both indicate that the entire allocation's stream
 * association is being changed. Currently, it is not possible to change stream
 * association for a portion of a managed allocation.
 *
 * For pageable host allocations, \p length must be non-zero.
 *
 * The stream association is specified using \p flags which must be
 * one of ::CUmemAttach_flags.
 * If the ::CU_MEM_ATTACH_GLOBAL flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream on a device that
 * has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
 * If the ::CU_MEM_ATTACH_SINGLE flag is specified and \p hStream is associated with
 * a device that has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
 * the program makes a guarantee that it will only access the memory on the device
 * from \p hStream. It is illegal to attach singly to the NULL stream, because the
 * NULL stream is a virtual global stream and not a specific stream. An error will
 * be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p hStream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownership of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region.
 *
 * It is a program's responsibility to order calls to ::cuStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p hStream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::cuMemAllocManaged. For __managed__ variables, the default
 * association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param hStream - Stream in which to enqueue the attach operation
 * \param dptr    - Pointer to memory (must be a pointer to managed memory or
 *                  to a valid host-accessible region of system-allocated
 *                  pageable memory)
 * \param length  - Length of memory
 * \param flags   - Must be one of ::CUmemAttach_flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamWaitEvent,
 * ::cuStreamDestroy,
 * ::cuMemAllocManaged,
 * ::cudaStreamAttachMemAsync
 *)
function cuStreamAttachMemAsync(hStream : CUstream; dptr : CUdeviceptr; length : size_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Determine status of a compute stream
 *
 * Returns ::CUDA_SUCCESS if all operations in the stream specified by
 * \p hStream have completed, or ::CUDA_ERROR_NOT_READY if not.
 *
 * For the purposes of Unified Memory, a return value of ::CUDA_SUCCESS
 * is equivalent to having called ::cuStreamSynchronize().
 *
 * \param hStream - Stream to query status of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_READY
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuStreamWaitEvent,
 * ::cuStreamDestroy,
 * ::cuStreamSynchronize,
 * ::cuStreamAddCallback,
 * ::cudaStreamQuery
 *)
function cuStreamQuery(hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Wait until a stream's tasks are completed
 *
 * Waits until the device has completed all operations in the stream specified
 * by \p hStream. If the context was created with the
 * ::CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will block until the
 * stream is finished with all of its tasks.
 *
 * \param hStream - Stream to wait for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE

 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuStreamDestroy,
 * ::cuStreamWaitEvent,
 * ::cuStreamQuery,
 * ::cuStreamAddCallback,
 * ::cudaStreamSynchronize
 *)
function cuStreamSynchronize(hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys a stream
 *
 * Destroys the stream specified by \p hStream.
 *
 * In case the device is still doing work in the stream \p hStream
 * when ::cuStreamDestroy() is called, the function will return immediately
 * and the resources associated with \p hStream will be released automatically
 * once the device has completed all work in \p hStream.
 *
 * \param hStream - Stream to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuStreamWaitEvent,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamAddCallback,
 * ::cudaStreamDestroy
 *)
function cuStreamDestroy(hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies attributes from source stream to destination stream.
 *
 * Copies attributes from source stream \p src to destination stream \p dst.
 * Both streams must have the same context.
 *
 * \param[out] dst Destination stream
 * \param[in] src Source stream
 * For list of attributes see ::CUstreamAttrID
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuStreamCopyAttributes(dst : CUstream; src : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Queries stream attribute.
 *
 * Queries attribute \p attr from \p hStream and stores it in corresponding
 * member of \p value_out.
 *
 * \param[in] hStream
 * \param[in] attr
 * \param[out] value_out
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuStreamGetAttribute(hStream : CUstream; attr : CUstreamAttrID;
                                      value_out : PCUstreamAttrValue):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets stream attribute.
 *
 * Sets attribute \p attr on \p hStream from corresponding attribute of
 * \p value. The updated attribute will be applied to subsequent work
 * submitted to the stream. It will not affect previously submitted work.
 *
 * \param[out] hStream
 * \param[in] attr
 * \param[in] value
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuStreamSetAttribute(hStream : CUstream; attr : CUstreamAttrID;
                                      const value : PCUstreamAttrValue):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_STREAM *)


(**
 * \defgroup CUDA_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Creates an event
 *
 * Creates an event *phEvent for the current context with the flags specified via
 * \p Flags. Valid flags include:
 * - ::CU_EVENT_DEFAULT: Default event creation flag.
 * - ::CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking
 *   synchronization.  A CPU thread that uses ::cuEventSynchronize() to wait on
 *   an event created with this flag will block until the event has actually
 *   been recorded.
 * - ::CU_EVENT_DISABLE_TIMING: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::CU_EVENT_BLOCKING_SYNC flag not specified will provide the best
 *   performance when used with ::cuStreamWaitEvent() and ::cuEventQuery().
 * - ::CU_EVENT_INTERPROCESS: Specifies that the created event may be used as an
 *   interprocess event by ::cuIpcGetEventHandle(). ::CU_EVENT_INTERPROCESS must
 *   be specified along with ::CU_EVENT_DISABLE_TIMING.
 *
 * \param phEvent - Returns newly created event
 * \param Flags   - Event creation flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa
 * ::cuEventRecord,
 * ::cuEventQuery,
 * ::cuEventSynchronize,
 * ::cuEventDestroy,
 * ::cuEventElapsedTime,
 * ::cudaEventCreate,
 * ::cudaEventCreateWithFlags
 *)
function cuEventCreate(phEvent : PCUevent; Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Records an event
 *
 * Captures in \p hEvent the contents of \p hStream at the time of this call.
 * \p hEvent and \p hStream must be from the same context.
 * Calls such as ::cuEventQuery() or ::cuStreamWaitEvent() will then
 * examine or wait for completion of the work that was captured. Uses of
 * \p hStream after this call do not modify \p hEvent. See note on default
 * stream behavior for what is captured in the default case.
 *
 * ::cuEventRecord() can be called multiple times on the same event and
 * will overwrite the previously captured state. Other APIs such as
 * ::cuStreamWaitEvent() use the most recently captured state at the time
 * of the API call, and are not affected by later calls to
 * ::cuEventRecord(). Before the first call to ::cuEventRecord(), an
 * event represents an empty set of work, so for example ::cuEventQuery()
 * would return ::CUDA_SUCCESS.
 *
 * \param hEvent  - Event to record
 * \param hStream - Stream to record event for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuEventCreate,
 * ::cuEventQuery,
 * ::cuEventSynchronize,
 * ::cuStreamWaitEvent,
 * ::cuEventDestroy,
 * ::cuEventElapsedTime,
 * ::cudaEventRecord,
 * ::cuEventRecordWithFlags
 *)
function cuEventRecord(hEvent : CUevent; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Records an event
 *
 * Captures in \p hEvent the contents of \p hStream at the time of this call.
 * \p hEvent and \p hStream must be from the same context.
 * Calls such as ::cuEventQuery() or ::cuStreamWaitEvent() will then
 * examine or wait for completion of the work that was captured. Uses of
 * \p hStream after this call do not modify \p hEvent. See note on default
 * stream behavior for what is captured in the default case.
 *
 * ::cuEventRecordWithFlags() can be called multiple times on the same event and
 * will overwrite the previously captured state. Other APIs such as
 * ::cuStreamWaitEvent() use the most recently captured state at the time
 * of the API call, and are not affected by later calls to
 * ::cuEventRecordWithFlags(). Before the first call to ::cuEventRecordWithFlags(), an
 * event represents an empty set of work, so for example ::cuEventQuery()
 * would return ::CUDA_SUCCESS.
 *
 * flags include:
 * - ::CU_EVENT_RECORD_DEFAULT: Default event creation flag.
 * - ::CU_EVENT_RECORD_EXTERNAL: Event is captured in the graph as an external
 *   event node when performing stream capture. This flag is invalid outside
 *   of stream capture.
 *
 * \param hEvent  - Event to record
 * \param hStream - Stream to record event for
 * \param flags   - See ::CUevent_capture_flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuEventCreate,
 * ::cuEventQuery,
 * ::cuEventSynchronize,
 * ::cuStreamWaitEvent,
 * ::cuEventDestroy,
 * ::cuEventElapsedTime,
 * ::cuEventRecord,
 * ::cudaEventRecord
 *)
function cuEventRecordWithFlags(hEvent : CUevent; hStream : CUstream; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Queries an event's status
 *
 * Queries the status of all work currently captured by \p hEvent. See
 * ::cuEventRecord() for details on what is captured by an event.
 *
 * Returns ::CUDA_SUCCESS if all captured work has been completed, or
 * ::CUDA_ERROR_NOT_READY if any captured work is incomplete.
 *
 * For the purposes of Unified Memory, a return value of ::CUDA_SUCCESS
 * is equivalent to having called ::cuEventSynchronize().
 *
 * \param hEvent - Event to query
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_READY
 * \notefnerr
 *
 * \sa ::cuEventCreate,
 * ::cuEventRecord,
 * ::cuEventSynchronize,
 * ::cuEventDestroy,
 * ::cuEventElapsedTime,
 * ::cudaEventQuery
 *)
function cuEventQuery(hEvent : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Waits for an event to complete
 *
 * Waits until the completion of all work currently captured in \p hEvent.
 * See ::cuEventRecord() for details on what is captured by an event.
 *
 * Waiting for an event that was created with the ::CU_EVENT_BLOCKING_SYNC
 * flag will cause the calling CPU thread to block until the event has
 * been completed by the device.  If the ::CU_EVENT_BLOCKING_SYNC flag has
 * not been set, then the CPU thread will busy-wait until the event has
 * been completed by the device.
 *
 * \param hEvent - Event to wait for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuEventCreate,
 * ::cuEventRecord,
 * ::cuEventQuery,
 * ::cuEventDestroy,
 * ::cuEventElapsedTime,
 * ::cudaEventSynchronize
 *)
function cuEventSynchronize(hEvent : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys an event
 *
 * Destroys the event specified by \p hEvent.
 *
 * An event may be destroyed before it is complete (i.e., while
 * ::cuEventQuery() would return ::CUDA_ERROR_NOT_READY). In this case, the
 * call does not block on completion of the event, and any associated
 * resources will automatically be released asynchronously at completion.
 *
 * \param hEvent - Event to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuEventCreate,
 * ::cuEventRecord,
 * ::cuEventQuery,
 * ::cuEventSynchronize,
 * ::cuEventElapsedTime,
 * ::cudaEventDestroy
 *)
function cuEventDestroy(hEvent : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Computes the elapsed time between two events
 *
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle). This
 * happens because the ::cuEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could execute
 * in between the two measured events, thus altering the timing in a significant
 * way.
 *
 * If ::cuEventRecord() has not been called on either event then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If ::cuEventRecord() has been called
 * on both events but one or both of them has not yet been completed (that is,
 * ::cuEventQuery() would return ::CUDA_ERROR_NOT_READY on at least one of the
 * events), ::CUDA_ERROR_NOT_READY is returned. If either event was created with
 * the ::CU_EVENT_DISABLE_TIMING flag, then this function will return
 * ::CUDA_ERROR_INVALID_HANDLE.
 *
 * \param pMilliseconds - Time between \p hStart and \p hEnd in ms
 * \param hStart        - Starting event
 * \param hEnd          - Ending event
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_READY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuEventCreate,
 * ::cuEventRecord,
 * ::cuEventQuery,
 * ::cuEventSynchronize,
 * ::cuEventDestroy,
 * ::cudaEventElapsedTime
 *)
function cuEventElapsedTime(pMilliseconds : PSingle; hStart : CUevent; hEnd : CUevent):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_EVENT *)

(**
 * \defgroup CUDA_EXTRES_INTEROP External Resource Interoperability
 *
 * ___MANBRIEF___ External resource interoperability functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the external resource interoperability functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

 (**
 * \brief Imports an external memory object
 *
 * Imports an externally allocated memory object and returns
 * a handle to that in \p extMem_out.
 *
 * The properties of the handle being imported must be described in
 * \p memHandleDesc. The ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC structure
 * is defined as follows:
 *
 * \code
        typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
            CUexternalMemoryHandleType type;
            union {
                longint fd;
                struct {
                    Pointer handle;
                    const Pointer name;
                } win32;
                const Pointer nvSciBufObject;
            } handle;
            uint64 size;
            longint flags;
        } CUDA_EXTERNAL_MEMORY_HANDLE_DESC;
 * \endcode
 *
 * where ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type specifies the type
 * of handle being imported. ::CUexternalMemoryHandleType is
 * defined as:
 *
 * \code
          CUexternalMemoryHandleType_enum = (
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD          = 1,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32       = 2,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT   = 3,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP         = 4,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE     = 5,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE     = 6,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF           = 8
          );
  CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum;
 * \endcode
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD, then
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::fd must be a valid
 * file descriptor referencing a memory object. Ownership of
 * the file descriptor is transferred to the CUDA driver when the
 * handle is imported successfully. Performing any operations on the
 * file descriptor after it is imported results in undefined behavior.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32, then exactly one
 * of ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be
 * NULL. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a memory object. Ownership of this handle is
 * not transferred to CUDA after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a memory object.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT, then
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must
 * be non-NULL and
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
 * must be NULL. The handle specified must be a globally shared KMT
 * handle. This handle does not hold a reference to the underlying
 * object, and thus will be invalid when all references to the
 * memory object are destroyed.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, then exactly one
 * of ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be
 * NULL. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Heap object. This handle holds a reference to the underlying
 * object. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D12Heap object.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, then exactly one
 * of ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be
 * NULL. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Resource object. This handle holds a reference to the
 * underlying object. If
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D12Resource object.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE, then
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must
 * represent a valid shared NT handle that is returned by
 * IDXGIResource1::CreateSharedHandle when referring to a
 * ID3D11Resource object. If
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D11Resource object.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT, then
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must
 * represent a valid shared KMT handle that is returned by
 * IDXGIResource::GetSharedHandle when referring to a
 * ID3D11Resource object and
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
 * must be NULL.
 *
 * If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF, then
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::nvSciBufObject must be non-NULL
 * and reference a valid NvSciBuf object.
 * If the NvSciBuf object imported into CUDA is also mapped by other drivers, then the
 * application must use ::cuWaitExternalSemaphoresAsync or ::cuSignalExternalSemaphoresAsync
 * as appropriate barriers to maintain coherence between CUDA and the other drivers.
 * See ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC and ::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC
 * for memory synchronization.
 *
 *
 * The size of the memory object must be specified in
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::size.
 *
 * Specifying the flag ::CUDA_EXTERNAL_MEMORY_DEDICATED in
 * ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::flags indicates that the
 * resource is a dedicated resource. The definition of what a
 * dedicated resource is outside the scope of this extension.
 * This flag must be set if ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type
 * is one of the following:
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
 *
 * \param extMem_out    - Returned handle to an external memory object
 * \param memHandleDesc - Memory import handle descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OPERATING_SYSTEM
 * \notefnerr
 *
 * \note If the Vulkan memory imported into CUDA is mapped on the CPU then the
 * application must use vkInvalidateMappedMemoryRanges/vkFlushMappedMemoryRanges
 * as well as appropriate Vulkan pipeline barriers to maintain coherence between
 * CPU and GPU. For more information on these APIs, please refer to "Synchronization
 * and Cache Control" chapter from Vulkan specification.
 *
 * \sa ::cuDestroyExternalMemory,
 * ::cuExternalMemoryGetMappedBuffer,
 * ::cuExternalMemoryGetMappedMipmappedArray
 *)
function cuImportExternalMemory(extMem_out : PCUexternalMemory; const memHandleDesc : PCUDA_EXTERNAL_MEMORY_HANDLE_DESC):CUresult;  WINAPI; external libcuda;

(**
 * \brief Maps a buffer onto an imported memory object
 *
 * Maps a buffer onto an imported memory object and returns a device
 * pointer in \p devPtr.
 *
 * The properties of the buffer being mapped must be described in
 * \p bufferDesc. The ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC structure is
 * defined as follows:
 *
 * \code
          CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st=packed record
            uint64 offset;
            uint64 size;
            longint flags;
          end;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
 * \endcode
 *
 * where ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC::offset is the offset in
 * the memory object where the buffer's base address is.
 * ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC::size is the size of the buffer.
 * ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC::flags must be zero.
 *
 * The offset and size have to be suitably aligned to match the
 * requirements of the external API. Mapping two buffers whose ranges
 * overlap may or may not result in the same virtual address being
 * returned for the overlapped portion. In such cases, the application
 * must ensure that all accesses to that region from the GPU are
 * volatile. Otherwise writes made via one address are not guaranteed
 * to be visible via the other address, even if they're issued by the
 * same thread. It is recommended that applications map the combined
 * range instead of mapping separate buffers and then apply the
 * appropriate offsets to the returned pointer to derive the
 * individual buffers.
 *
 * The returned pointer \p devPtr must be freed using ::cuMemFree.
 *
 * \param devPtr     - Returned device pointer to buffer
 * \param extMem     - Handle to external memory object
 * \param bufferDesc - Buffer descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuImportExternalMemory,
 * ::cuDestroyExternalMemory,
 * ::cuExternalMemoryGetMappedMipmappedArray
 *)
function cuExternalMemoryGetMappedBuffer(devPtr : PCUdeviceptr; extMem : CUexternalMemory; const bufferDesc : PCUDA_EXTERNAL_MEMORY_BUFFER_DESC):CUresult;  WINAPI; external libcuda;

(**
 * \brief Maps a CUDA mipmapped array onto an external memory object
 *
 * Maps a CUDA mipmapped array onto an external object and returns a
 * handle to it in \p mipmap.
 *
 * The properties of the CUDA mipmapped array being mapped must be
 * described in \p mipmapDesc. The structure
 * ::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC is defined as follows:
 *
 * \code
          CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st=packed record
            uint64 offset;
            CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
            longint numLevels;
          end;
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
 * \endcode
 *
 * where ::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::offset is the
 * offset in the memory object where the base level of the mipmap
 * chain is.
 * ::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::arrayDesc describes
 * the format, dimensions and type of the base level of the mipmap
 * chain. For further details on these parameters, please refer to the
 * documentation for ::cuMipmappedArrayCreate. Note that if the mipmapped
 * array is bound as a color target in the graphics API, then the flag
 * ::CUDA_ARRAY3D_COLOR_ATTACHMENT must be specified in
 * ::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::arrayDesc::Flags.
 * ::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::numLevels specifies
 * the total number of levels in the mipmap chain.
 *
 * If \p extMem was imported from a handle of type ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF, then
 * ::CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::numLevels must be equal to 1.
 *
 * The returned CUDA mipmapped array must be freed using ::cuMipmappedArrayDestroy.
 *
 * \param mipmap     - Returned CUDA mipmapped array
 * \param extMem     - Handle to external memory object
 * \param mipmapDesc - CUDA array descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuImportExternalMemory,
 * ::cuDestroyExternalMemory,
 * ::cuExternalMemoryGetMappedBuffer
 *)
function cuExternalMemoryGetMappedMipmappedArray(mipmap : PCUmipmappedArray; extMem : CUexternalMemory; const mipmapDesc : PCUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys an external memory object.
 *
 * Destroys the specified external memory object. Any existing buffers
 * and CUDA mipmapped arrays mapped onto this object must no longer be
 * used and must be explicitly freed using ::cuMemFree and
 * ::cuMipmappedArrayDestroy respectively.
 *
 * \param extMem - External memory object to be destroyed
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuImportExternalMemory,
 * ::cuExternalMemoryGetMappedBuffer,
 * ::cuExternalMemoryGetMappedMipmappedArray
 *)
function cuDestroyExternalMemory(extMem : CUexternalMemory):CUresult;  WINAPI; external libcuda;

(**
 * \brief Imports an external semaphore
 *
 * Imports an externally allocated synchronization object and returns
 * a handle to that in \p extSem_out.
 *
 * The properties of the handle being imported must be described in
 * \p semHandleDesc. The ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC is
 * defined as follows:
 *
 * \code
        typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
            CUexternalSemaphoreHandleType type;
            union {
                longint fd;
                struct {
                    Pointer handle;
                    const Pointer name;
                } win32;
                const pointer  NvSciSyncObj;
            } handle;
            longint flags;
        } CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;
 * \endcode
 *
 * where ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type specifies the type of
 * handle being imported. ::CUexternalSemaphoreHandleType is defined
 * as:
 *
 * \code
          CUexternalSemaphoreHandleType_enum = (
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD                = 1,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32             = 2,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT         = 3,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE              = 4,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE              = 5,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC                = 6,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX        = 7,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT    = 8,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD    = 9,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10
          );
  CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum;
 * \endcode
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid
 * file descriptor referencing a synchronization object. Ownership of
 * the file descriptor is transferred to the CUDA driver when the
 * handle is imported successfully. Performing any operations on the
 * file descriptor after it is imported results in undefined behavior.
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32, then exactly one
 * of ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be
 * NULL. If
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a synchronization object. Ownership of this handle is
 * not transferred to CUDA after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must name a valid synchronization object.
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle must
 * be non-NULL and
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
 * must be NULL. The handle specified must be a globally shared KMT
 * handle. This handle does not hold a reference to the underlying
 * object, and thus will be invalid when all references to the
 * synchronization object are destroyed.
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE, then exactly one
 * of ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be
 * NULL. If
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Fence object. This handle holds a reference to the underlying
 * object. If
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid ID3D12Fence object.
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
 * represents a valid shared NT handle that is returned by
 * ID3D11Fence::CreateSharedHandle. If
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid ID3D11Fence object.
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::nvSciSyncObj
 * represents a valid NvSciSyncObj.
 *
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
 * represents a valid shared NT handle that
 * is returned by IDXGIResource1::CreateSharedHandle when referring to
 * a IDXGIKeyedMutex object. If
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid IDXGIKeyedMutex object.
 *
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
 * represents a valid shared KMT handle that
 * is returned by IDXGIResource::GetSharedHandle when referring to
 * a IDXGIKeyedMutex object and
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must be NULL.
 * 
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD, then
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid
 * file descriptor referencing a synchronization object. Ownership of
 * the file descriptor is transferred to the CUDA driver when the
 * handle is imported successfully. Performing any operations on the
 * file descriptor after it is imported results in undefined behavior.
 * 
 * If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32, then exactly one
 * of ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be
 * NULL. If
 * ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a synchronization object. Ownership of this handle is
 * not transferred to CUDA after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
 * is not NULL, then it must name a valid synchronization object.
 *
 * \param extSem_out    - Returned handle to an external semaphore
 * \param semHandleDesc - Semaphore import handle descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OPERATING_SYSTEM
 * \notefnerr
 *
 * \sa ::cuDestroyExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuImportExternalSemaphore(extSem_out : PCUexternalSemaphore; const semHandleDesc : PCUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC):CUresult;  WINAPI; external libcuda;

(**
 * \brief Signals a set of external semaphore objects
 *
 * Enqueues a signal operation on a set of externally allocated
 * semaphore object in the specified stream. The operations will be
 * executed when all prior operations in the stream complete.
 *
 * The exact semantics of signaling a semaphore depends on the type of
 * the object.
 *
 * If the semaphore object is any one of the following types:
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
 * then signaling the semaphore will set it to the signaled state.
 *
 * If the semaphore object is any one of the following types:
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32
 * then the semaphore will be set to the value specified in
 * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::fence::value.
 *
 * If the semaphore object is of the type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC
 * this API sets ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence
 * to a value that can be used by subsequent waiters of the same NvSciSync object
 * to order operations with those currently submitted in \p stream. Such an update
 * will overwrite previous contents of
 * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence. By default,
 * signaling such an external semaphore object causes appropriate memory synchronization
 * operations to be performed over all external memory objects that are imported as
 * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. This ensures that any subsequent accesses
 * made by other importers of the same set of NvSciBuf memory object(s) are coherent.
 * These operations can be skipped by specifying the flag
 * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC, which can be used as a
 * performance optimization when data coherency is not required. But specifying this
 * flag in scenarios where data coherency is required results in undefined behavior.
 * Also, for semaphore object of the type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC,
 * if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
 * ::cuDeviceGetNvSciSyncAttributes to CUDA_NVSCISYNC_ATTR_SIGNAL, this API will return
 * CUDA_ERROR_NOT_SUPPORTED.
 * NvSciSyncFence associated with semaphore object of the type 
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC can be deterministic. For this the 
 * NvSciSyncAttrList used to create the semaphore object must have value of 
 * NvSciSyncAttrKey_RequireDeterministicFences key set to true. Deterministic fences 
 * allow users to enqueue a wait over the semaphore object even before corresponding
 * signal is enqueued. For such a semaphore object, CUDA guarantees that each signal 
 * operation will increment the fence value by '1'. Users are expected to track count 
 * of signals enqueued on the semaphore object and insert waits accordingly. When such 
 * a semaphore object is signaled from multiple streams, due to concurrent stream 
 * execution, it is possible that the order in which the semaphore gets signaled is 
 * indeterministic. This could lead to waiters of the semaphore getting unblocked 
 * incorrectly. Users are expected to handle such situations, either by not using the 
 * same semaphore object with deterministic fence support enabled in different streams 
 * or by adding explicit dependency amongst such streams so that the semaphore is 
 * signaled in order.
 *
 * If the semaphore object is any one of the following types:
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
 * then the keyed mutex will be released with the key specified in
 * ::CUDA_EXTERNAL_SEMAPHORE_PARAMS::params::keyedmutex::key.
 *
 * \param extSemArray - Set of external semaphores to be signaled
 * \param paramsArray - Array of semaphore parameters
 * \param numExtSems  - Number of semaphores to signal
 * \param stream      - Stream to enqueue the signal operations in
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuImportExternalSemaphore,
 * ::cuDestroyExternalSemaphore,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuSignalExternalSemaphoresAsync(const extSemArray : PCUexternalSemaphore; const paramsArray : PCUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS; numExtSems : longword; stream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Waits on a set of external semaphore objects
 *
 * Enqueues a wait operation on a set of externally allocated
 * semaphore object in the specified stream. The operations will be
 * executed when all prior operations in the stream complete.
 *
 * The exact semantics of waiting on a semaphore depends on the type
 * of the object.
 *
 * If the semaphore object is any one of the following types:
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
 * then waiting on the semaphore will wait until the semaphore reaches
 * the signaled state. The semaphore will then be reset to the
 * unsignaled state. Therefore for every signal operation, there can
 * only be one wait operation.
 *
 * If the semaphore object is any one of the following types:
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32
 * then waiting on the semaphore will wait until the value of the
 * semaphore is greater than or equal to
 * ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::fence::value.
 *
 * If the semaphore object is of the type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC
 * then, waiting on the semaphore will wait until the
 * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence is signaled by the
 * signaler of the NvSciSyncObj that was associated with this semaphore object.
 * By default, waiting on such an external semaphore object causes appropriate
 * memory synchronization operations to be performed over all external memory objects
 * that are imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. This ensures that
 * any subsequent accesses made by other importers of the same set of NvSciBuf memory
 * object(s) are coherent. These operations can be skipped by specifying the flag
 * ::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC, which can be used as a
 * performance optimization when data coherency is not required. But specifying this
 * flag in scenarios where data coherency is required results in undefined behavior.
 * Also, for semaphore object of the type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC,
 * if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
 * ::cuDeviceGetNvSciSyncAttributes to CUDA_NVSCISYNC_ATTR_WAIT, this API will return
 * CUDA_ERROR_NOT_SUPPORTED.
 *
 * If the semaphore object is any one of the following types:
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX,
 * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
 * then the keyed mutex will be acquired when it is released with the key 
 * specified in ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::keyedmutex::key 
 * or until the timeout specified by
 * ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::keyedmutex::timeoutMs
 * has lapsed. The timeout interval can either be a finite value
 * specified in milliseconds or an infinite value. In case an infinite
 * value is specified the timeout never elapses. The windows INFINITE
 * macro must be used to specify infinite timeout.
 *
 * \param extSemArray - External semaphores to be waited on
 * \param paramsArray - Array of semaphore parameters
 * \param numExtSems  - Number of semaphores to wait on
 * \param stream      - Stream to enqueue the wait operations in
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_TIMEOUT
 * \notefnerr
 *
 * \sa ::cuImportExternalSemaphore,
 * ::cuDestroyExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync
 *)
function cuWaitExternalSemaphoresAsync(const extSemArray : PCUexternalSemaphore; const paramsArray : PCUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS; numExtSems : longword; stream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys an external semaphore
 *
 * Destroys an external semaphore object and releases any references
 * to the underlying resource. Any outstanding signals or waits must
 * have completed before the semaphore is destroyed.
 *
 * \param extSem - External semaphore to be destroyed
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuImportExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuDestroyExternalSemaphore(extSem : CUexternalSemaphore):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_EXTRES_INTEROP *)

(**
 * \defgroup CUDA_MEMOP Stream Memory Operations
 *
 * ___MANBRIEF___ Stream memory operations of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream memory operations of the low-level CUDA
 * driver application programming interface.
 *
 * Support for the ::CU_STREAM_WAIT_VALUE_NOR flag can be queried with
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.
 *
 * Support for the ::cuStreamWriteValue64() and ::cuStreamWaitValue64()
 * functions, as well as for the ::CU_STREAM_MEM_OP_WAIT_VALUE_64 and
 * ::CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.
 *
 * Support for both ::CU_STREAM_WAIT_VALUE_FLUSH and
 * ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform
 * hardware features and can be queried with ::cuDeviceGetAttribute() and
 * ::CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.
 *
 * Note that all memory pointers passed as parameters to these operations
 * are device pointers. Where necessary a device pointer should be
 * obtained, for example with ::cuMemHostGetDevicePointer().
 *
 * None of the operations accepts pointers to managed memory buffers
 * (::cuMemAllocManaged).
 *
 * \note
 * Warning:
 * Improper use of these APIs may deadlock the application. Synchronization 
 * ordering established through these APIs is not visible to CUDA. CUDA tasks 
 * that are (even indirectly) ordered by these APIs should also have that order
 * expressed with CUDA-visible dependencies such as events. This ensures that 
 * the scheduler does not serialize them in an improper order.
 *
 * @{
 *)

(**
 * \brief Wait on a memory location
 *
 * Enqueues a synchronization of the stream on the given memory location. Work
 * ordered after the operation will block until the given condition on the
 * memory is satisfied. By default, the condition is to wait for
 * (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal.
 * Other condition types can be specified via \p flags.
 *
 * If the memory was registered via ::cuMemHostRegister(), the device pointer
 * should be obtained with ::cuMemHostGetDevicePointer(). This function cannot
 * be used with managed memory (::cuMemAllocManaged).
 *
 * Support for CU_STREAM_WAIT_VALUE_NOR can be queried with ::cuDeviceGetAttribute() and
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.
 *
 * \note
 * Warning:
 * Improper use of this API may deadlock the application. Synchronization 
 * ordering established through this API is not visible to CUDA. CUDA tasks 
 * that are (even indirectly) ordered by this API should also have that order
 * expressed with CUDA-visible dependencies such as events. This ensures that 
 * the scheduler does not serialize them in an improper order.
 *
 * \param stream The stream to synchronize on the memory location.
 * \param addr The memory location to wait on.
 * \param value The value to compare with the memory location.
 * \param flags See ::CUstreamWaitValue_flags.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuStreamWaitValue64,
 * ::cuStreamWriteValue32,
 * ::cuStreamWriteValue64,
 * ::cuStreamBatchMemOp,
 * ::cuMemHostRegister,
 * ::cuStreamWaitEvent
 *)
function cuStreamWaitValue32(stream : CUstream; addr : CUdeviceptr; value : cuuint32_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Wait on a memory location
 *
 * Enqueues a synchronization of the stream on the given memory location. Work
 * ordered after the operation will block until the given condition on the
 * memory is satisfied. By default, the condition is to wait for
 * (int64_t)(*addr - value) >= 0, a cyclic greater-or-equal.
 * Other condition types can be specified via \p flags.
 *
 * If the memory was registered via ::cuMemHostRegister(), the device pointer
 * should be obtained with ::cuMemHostGetDevicePointer().
 *
 * Support for this can be queried with ::cuDeviceGetAttribute() and
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.
 *
 * \note
 * Warning:
 * Improper use of this API may deadlock the application. Synchronization 
 * ordering established through this API is not visible to CUDA. CUDA tasks 
 * that are (even indirectly) ordered by this API should also have that order
 * expressed with CUDA-visible dependencies such as events. This ensures that 
 * the scheduler does not serialize them in an improper order.
 *
 * \param stream The stream to synchronize on the memory location.
 * \param addr The memory location to wait on.
 * \param value The value to compare with the memory location.
 * \param flags See ::CUstreamWaitValue_flags.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuStreamWaitValue32,
 * ::cuStreamWriteValue32,
 * ::cuStreamWriteValue64,
 * ::cuStreamBatchMemOp,
 * ::cuMemHostRegister,
 * ::cuStreamWaitEvent
 *)
function cuStreamWaitValue64(stream : CUstream; addr : CUdeviceptr; value : cuuint64_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Write a value to memory
 *
 * Write a value to memory.
 *
 * If the memory was registered via ::cuMemHostRegister(), the device pointer
 * should be obtained with ::cuMemHostGetDevicePointer(). This function cannot
 * be used with managed memory (::cuMemAllocManaged).
 *
 * \param stream The stream to do the write in.
 * \param addr The device address to write to.
 * \param value The value to write.
 * \param flags See ::CUstreamWriteValue_flags.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuStreamWriteValue64,
 * ::cuStreamWaitValue32,
 * ::cuStreamWaitValue64,
 * ::cuStreamBatchMemOp,
 * ::cuMemHostRegister,
 * ::cuEventRecord
 *)
function cuStreamWriteValue32(stream : CUstream; addr : CUdeviceptr; value : cuuint32_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Write a value to memory
 *
 * Write a value to memory.
 *
 * If the memory was registered via ::cuMemHostRegister(), the device pointer
 * should be obtained with ::cuMemHostGetDevicePointer().
 *
 * Support for this can be queried with ::cuDeviceGetAttribute() and
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.
 *
 * \param stream The stream to do the write in.
 * \param addr The device address to write to.
 * \param value The value to write.
 * \param flags See ::CUstreamWriteValue_flags.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuStreamWriteValue32,
 * ::cuStreamWaitValue32,
 * ::cuStreamWaitValue64,
 * ::cuStreamBatchMemOp,
 * ::cuMemHostRegister,
 * ::cuEventRecord
 *)
function cuStreamWriteValue64(stream : CUstream; addr : CUdeviceptr; value : cuuint64_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Batch operations to synchronize the stream via memory operations
 *
 * This is a batch version of ::cuStreamWaitValue32() and ::cuStreamWriteValue32().
 * Batching operations may avoid some performance overhead in both the API call
 * and the device execution versus adding them to the stream in separate API
 * calls. The operations are enqueued in the order they appear in the array.
 *
 * See ::CUstreamBatchMemOpType for the full set of supported operations, and
 * ::cuStreamWaitValue32(), ::cuStreamWaitValue64(), ::cuStreamWriteValue32(),
 * and ::cuStreamWriteValue64() for details of specific operations.
 *
 * See related APIs for details on querying support for specific operations.
 *
 * \note
 * Warning:
 * Improper use of this API may deadlock the application. Synchronization 
 * ordering established through this API is not visible to CUDA. CUDA tasks 
 * that are (even indirectly) ordered by this API should also have that order
 * expressed with CUDA-visible dependencies such as events. This ensures that 
 * the scheduler does not serialize them in an improper order. For more 
 * information, see the Stream Memory Operations section in the programming 
 * guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
 *
 * \param stream The stream to enqueue the operations in.
 * \param count The number of operations in the array. Must be less than 256.
 * \param paramArray The types and parameters of the individual operations.
 * \param flags Reserved for future expansion; must be 0.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::cuStreamWaitValue32,
 * ::cuStreamWaitValue64,
 * ::cuStreamWriteValue32,
 * ::cuStreamWriteValue64,
 * ::cuMemHostRegister
 *)
function cuStreamBatchMemOp(stream : CUstream; count : longword; paramArray : PCUstreamBatchMemOpParams; flags : longword):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_MEMOP *)

(**
 * \defgroup CUDA_EXEC Execution Control
 *
 * ___MANBRIEF___ execution control functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the execution control functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns information about a function
 *
 * Returns in \p *pi the integer value of the attribute \p attrib on the kernel
 * given by \p hfunc. The supported attributes are:
 * - ::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: The maximum number of threads
 *   per block, beyond which a launch of the function would fail. This number
 *   depends on both the function and the device on which the function is
 *   currently loaded.
 * - ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: The size in bytes of
 *   statically-allocated shared memory per block required by this function.
 *   This does not include dynamically-allocated shared memory requested by
 *   the user at runtime.
 * - ::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: The size in bytes of user-allocated
 *   constant memory required by this function.
 * - ::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: The size in bytes of local memory
 *   used by each thread of this function.
 * - ::CU_FUNC_ATTRIBUTE_NUM_REGS: The number of registers used by each thread
 *   of this function.
 * - ::CU_FUNC_ATTRIBUTE_PTX_VERSION: The PTX virtual architecture version for
 *   which the function was compiled. This value is the major PTX version * 10
 *   + the minor PTX version, so a PTX version 1.3 function would return the
 *   value 13. Note that this may return the undefined value of 0 for cubins
 *   compiled prior to CUDA 3.0.
 * - ::CU_FUNC_ATTRIBUTE_BINARY_VERSION: The binary architecture version for
 *   which the function was compiled. This value is the major binary
 *   version * 10 + the minor binary version, so a binary version 1.3 function
 *   would return the value 13. Note that this will return a value of 10 for
 *   legacy cubins that do not have a properly-encoded binary architecture
 *   version.
 * - ::CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the function has
 *   been compiled with user specified option "-Xptxas --dlcm=ca" set .
 * - ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: The maximum size in bytes of
 *   dynamically-allocated shared memory.
 * - ::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: Preferred shared memory-L1
 *   cache split ratio in percent of total shared memory.
 * - ::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: If this attribute is set, the
 *   kernel must launch with a valid cluster size specified.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: The required cluster width in
 *   blocks.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: The required cluster height in
 *   blocks.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: The required cluster depth in
 *   blocks.
 * - ::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: Indicates whether
 *   the function can be launched with non-portable cluster size. 1 is allowed,
 *   0 is disallowed. A non-portable cluster size may only function on the
 *   specific SKUs the program is tested on. The launch might fail if the
 *   program is run on a different hardware platform. CUDA API provides
 *   cudaOccupancyMaxActiveClusters to assist with checking whether the desired
 *   size can be launched on the current device. A portable cluster size is
 *   guaranteed to be functional on all compute capabilities higher than the
 *   target compute capability. The portable cluster size for sm_90 is 8 blocks
 *   per cluster. This value may increase for future compute capabilities. The
 *   specific hardware unit may support higher cluster sizes that’s not
 *   guaranteed to be portable.
 * - ::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block
 *   scheduling policy of a function. The value type is CUclusterSchedulingPolicy.
 *
 * \param pi     - Returned attribute value
 * \param attrib - Attribute requested
 * \param hfunc  - Function to query attribute of
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncSetCacheConfig,
 * ::cuLaunchKernel,
 * ::cudaFuncGetAttributes,
 * ::cudaFuncSetAttribute,
 * ::cuKernelGetAttribute
 *)
function cuFuncGetAttribute(pi : Plongint; attrib : CUfunction_attribute; hfunc : CUfunction):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets information about a function
 *
 * This call sets the value of a specified attribute \p attrib on the kernel given
 * by \p hfunc to an integer value specified by \p val
 * This function returns CUDA_SUCCESS if the new value of the attribute could be
 * successfully set. If the set fails, this call will return an error.
 * Not all attributes can have values set. Attempting to set a value on a read-only
 * attribute will result in an error (CUDA_ERROR_INVALID_VALUE)
 *
 * Supported attributes for the cuFuncSetAttribute call are:
 * - ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This maximum size in bytes of
 *   dynamically-allocated shared memory. The value should contain the requested
 *   maximum size of dynamically-allocated shared memory. The sum of this value and
 *   the function attribute ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the
 *   device attribute ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
 *   The maximal size of requestable dynamic shared memory may differ by GPU
 *   architecture.
 * - ::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On devices where the L1
 *   cache and shared memory use the same hardware resources, this sets the shared memory
 *   carveout preference, in percent of the total shared memory. 
 *   See ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
 *   This is only a hint, and the driver can choose a different ratio if required to execute the function.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: The required cluster width in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: The required cluster height in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
 * - ::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: The required cluster depth in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
 * - ::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: The block
 *   scheduling policy of a function. The value type is CUclusterSchedulingPolicy.
 *
 * \param hfunc  - Function to query attribute of
 * \param attrib - Attribute requested
 * \param value   - The value to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncSetCacheConfig,
 * ::cuLaunchKernel,
 * ::cudaFuncGetAttributes,
 * ::cudaFuncSetAttribute,
 * ::cuKernelSetAttribute
 *)
function cuFuncSetAttribute(hfunc : CUfunction; attrib : CUfunction_attribute; value : longint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p config the preferred cache configuration for
 * the device function \p hfunc. This is only a preference. The driver will use
 * the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute \p hfunc.  Any context-wide preference
 * set via ::cuCtxSetCacheConfig() will be overridden by this per-function
 * setting unless the per-function setting is ::CU_FUNC_CACHE_PREFER_NONE. In
 * that case, the current context-wide setting will be used.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 *
 * The supported cache configurations are:
 * - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \param hfunc  - Kernel to configure cache for
 * \param config - Requested cache configuration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cuLaunchKernel,
 * ::cudaFuncSetCacheConfig,
 * ::cuKernelSetCacheConfig
 *)
function cuFuncSetCacheConfig(hfunc : CUfunction; config : CUfunc_cache):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the shared memory configuration for a device function.
 *
 * On devices with configurable shared memory banks, this function will
 * force all subsequent launches of the specified device function to have
 * the given shared memory bank size configuration. On any given launch of the
 * function, the shared memory configuration of the device will be temporarily
 * changed if needed to suit the function's preferred configuration. Changes in
 * shared memory configuration between subsequent launches of functions,
 * may introduce a device side synchronization point.
 *
 * Any per-function setting of shared memory bank size set via
 * ::cuFuncSetSharedMemConfig will override the context wide setting set with
 * ::cuCtxSetSharedMemConfig.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance.
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: use the context's shared memory
 *   configuration when launching this function.
 * - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively four bytes when launching this function.
 * - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively eight bytes when launching this function.
 *
 * \param hfunc  - kernel to be given a shared memory config
 * \param config - requested shared memory configuration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuCtxGetSharedMemConfig,
 * ::cuCtxSetSharedMemConfig,
 * ::cuFuncGetAttribute,
 * ::cuLaunchKernel,
 * ::cudaFuncSetSharedMemConfig
 *)
function cuFuncSetSharedMemConfig(hfunc : CUfunction; config : CUsharedconfig):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a module handle
 *
 * Returns in \p *hmod the handle of the module that function \p hfunc
 * is located in. The lifetime of the module corresponds to the lifetime of
 * the context it was loaded in or until the module is explicitly unloaded.
 *
 * The CUDA runtime manages its own modules loaded into the primary context.
 * If the handle returned by this API refers to a module loaded by the CUDA runtime,
 * calling ::cuModuleUnload() on that module will result in undefined behavior.
 *
 * \param hmod - Returned module handle
 * \param hfunc   - Function to retrieve module for
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 *)
function cuFuncGetModule(hmod : PCUmodule; hfunc : CUfunction):CUresult;  WINAPI; external libcuda;

(**
 * \brief Launches a CUDA function ::CUfunction or a CUDA kernel ::CUkernel
 *
 * Invokes the function ::CUfunction or the kernel ::CUkernel \p f
 * on a \p gridDimX x \p gridDimY x \p gridDimZ grid of blocks.
 * Each block contains \p blockDimX x \p blockDimY x
 * \p blockDimZ threads.
 *
 * \p sharedMemBytes sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * Kernel parameters to \p f can be specified in one of two ways:
 *
 * 1) Kernel parameters can be specified via \p kernelParams.  If \p f
 * has N parameters, then \p kernelParams needs to be an array of N
 * pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
 * must point to a region of memory from which the actual kernel
 * parameter will be copied.  The number of kernel parameters and their
 * offsets and sizes do not need to be specified as that information is
 * retrieved directly from the kernel's image.
 *
 * 2) Kernel parameters can also be packaged by the application into
 * a single buffer that is passed in via the \p extra parameter.
 * This places the burden on the application of knowing each kernel
 * parameter's size and alignment/padding within the buffer.  Here is
 * an example of using the \p extra parameter in this manner:
 * \code
    size_t argBufferSize;
    char argBuffer[256];

    // populate argBuffer and argBufferSize

    Pointer config[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
        CU_LAUNCH_PARAM_END
    };
    status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);
 * \endcode
 *
 * The \p extra parameter exists to allow ::cuLaunchKernel to take
 * additional less commonly used arguments.  \p extra specifies a list of
 * names of extra settings and their corresponding values.  Each extra
 * setting name is immediately followed by the corresponding value.  The
 * list must be terminated with either NULL or ::CU_LAUNCH_PARAM_END.
 *
 * - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
 *   array;
 * - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
 *   value in \p extra will be a pointer to a buffer containing all
 *   the kernel parameters for launching kernel \p f;
 * - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
 *   value in \p extra will be a pointer to a size_t containing the
 *   size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER;
 *
 * The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel
 * parameters are specified with both \p kernelParams and \p extra
 * (i.e. both \p kernelParams and \p extra are non-NULL).
 *
 * Calling ::cuLaunchKernel() invalidates the persistent function state
 * set through the following deprecated APIs:
 *  ::cuFuncSetBlockShape(),
 *  ::cuFuncSetSharedSize(),
 *  ::cuParamSetSize(),
 *  ::cuParamSeti(),
 *  ::cuParamSetf(),
 *  ::cuParamSetv().
 *
 * Note that to use ::cuLaunchKernel(), the kernel \p f must either have
 * been compiled with toolchain version 3.2 or later so that it will
 * contain kernel parameter information, or have no kernel parameters.
 * If either of these conditions is not met, then ::cuLaunchKernel() will
 * return ::CUDA_ERROR_INVALID_IMAGE.
 *
 * Note that the API can also be used to launch context-less kernel ::CUkernel
 * by querying the handle using ::cuLibraryGetKernel() and then passing it
 * to the API by casting to ::CUfunction. Here, the context to launch
 * the kernel on will either be taken from the specified stream \p hStream
 * or the current context in case of NULL stream.
 *
 * \param f              - Function ::CUfunction or Kernel ::CUkernel to launch
 * \param gridDimX       - Width of grid in blocks
 * \param gridDimY       - Height of grid in blocks
 * \param gridDimZ       - Depth of grid in blocks
 * \param blockDimX      - X dimension of each thread block
 * \param blockDimY      - Y dimension of each thread block
 * \param blockDimZ      - Z dimension of each thread block
 * \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
 * \param hStream        - Stream identifier
 * \param kernelParams   - Array of pointers to kernel parameters
 * \param extra          - Extra options
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_IMAGE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_NOT_FOUND
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cudaLaunchKernel,
 * ::cuLibraryGetKernel,
 * ::cuKernelSetCacheConfig,
 * ::cuKernelGetAttribute,
 * ::cuKernelSetAttribute
 *)
function cuLaunchKernel(f : CUfunction;
                                gridDimX : longword;
                                gridDimY : longword;
                                gridDimZ : longword;
                                blockDimX : longword;
                                blockDimY : longword;
                                blockDimZ : longword;
                                sharedMemBytes : longword;
                                hStream : CUstream;
                                kernelParams : Ppointer;
                                extra : Ppointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Launches a CUDA function ::CUfunction or a CUDA kernel ::CUkernel with launch-time configuration
 *
 * Invokes the function ::CUfunction or the kernel ::CUkernel \p f with the specified launch-time configuration
 * \p config.
 *
 * The ::CUlaunchConfig structure is defined as:
 * \code
          CUlaunchConfig_st=packed record
            longint gridDimX;
            longint gridDimY;
            longint gridDimZ;
            longint blockDimX;
            longint blockDimY;
            longint blockDimZ;
            longint sharedMemBytes;
            CUstream hStream;
            CUlaunchAttribute *attrs;
            longint numAttrs;
          end;
  CUlaunchConfig = CUlaunchConfig_st;
 * \endcode
 * where:
 * - ::CUlaunchConfig::gridDimX is the width of the grid in blocks.
 * - ::CUlaunchConfig::gridDimY is the height of the grid in blocks.
 * - ::CUlaunchConfig::gridDimZ is the depth of the grid in blocks.
 * - ::CUlaunchConfig::blockDimX is the X dimension of each thread block.
 * - ::CUlaunchConfig::blockDimX is the Y dimension of each thread block.
 * - ::CUlaunchConfig::blockDimZ is the Z dimension of each thread block.
 * - ::CUlaunchConfig::sharedMemBytes is the dynamic shared-memory size per
 *   thread block in bytes.
 * - ::CUlaunchConfig::hStream is the handle to the stream to perform the launch
 *   in. The CUDA context associated with this stream must match that associated
 *   with function f.
 * - ::CUlaunchConfig::attrs is an array of ::CUlaunchConfig::numAttrs
 *   continguous ::CUlaunchAttribute elements. The value of this pointer is not
 *   considered if ::CUlaunchConfig::numAttrs is zero. However, in that case, it
 *   is recommended to set the pointer to NULL.
 * - ::CUlaunchConfig::numAttrs is the numbers of attributes populating the
 *   first ::CUlaunchConfig::numAttrs positions of the ::CUlaunchConfig::attrs
 *   array.
 *
 * Launch-time configuration is specified by adding entries to
 * ::CUlaunchConfig::attrs. Each entry is an attribute ID and a corresponding
 * attribute value.
 *
 * The ::CUlaunchAttribute structure is defined as:
 * \code
          CUlaunchAttribute_st=packed record
            CUlaunchAttributeID id;
            CUlaunchAttributeValue value;
          end;
  CUlaunchAttribute = CUlaunchAttribute_st;
 * \endcode
 * where:
 * - ::CUlaunchAttribute::id is a unique enum identifying the attribute.
 * - ::CUlaunchAttribute::value is a union that hold the attribute value.
 *
 * An example of using the \p config parameter:
 * \code
        CUlaunchAttribute coopAttr = {.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                                      .value = 1};
        CUlaunchConfig config = {... // set block and grid dimensions
                                 .attrs = &coopAttr,
                                 .numAttrs = 1};

        cuLaunchKernelEx(&config, kernel, NULL, NULL);
 * \endcode
 *
 * The ::CUlaunchAttributeID enum is defined as:
 * \code
          CUlaunchAttributeID_enum = (
            CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
            CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW   = 1,
            CU_LAUNCH_ATTRIBUTE_COOPERATIVE            = 2,
            CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
            CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION                    = 4,
            CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
            CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION    = 6,
            CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT                   = 7,
          );
  CUlaunchAttributeID = CUlaunchAttributeID_enum;
 * \endcode
 *
 * and the corresponding ::CUlaunchAttributeValue union as :
 * \code
        typedef union CUlaunchAttributeValue_union {
            cuuint64_t pad[8];
            CUaccessPolicyWindow accessPolicyWindow;
            longint cooperative;
            CUsynchronizationPolicy syncPolicy;
            struct {
                longint x;
                longint y;
                longint z;
            } clusterDim;
            CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
            longint programmaticStreamSerializationAllowed;
            struct {
                CUevent event;
                longint flags;
                longint triggerAtBlockStart;
            } programmaticEvent;
        } CUlaunchAttributeValue;
 * \endcode
 *
 * Setting ::CU_LAUNCH_ATTRIBUTE_COOPERATIVE to a non-zero value causes the
 * kernel launch to be a cooperative launch, with exactly the same usage and
 * semantics of ::cuLaunchCooperativeKernel.
 *
 * Setting ::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION to a non-zero
 * values causes the kernel to use programmatic means to resolve its stream
 * dependency -- enabling the CUDA runtime to opportunistically allow the grid's
 * execution to overlap with the previous kernel in the stream, if that kernel
 * requests the overlap.
 *
 * ::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT records an event along with the
 * kernel launch. Event recorded through this launch attribute is guaranteed to
 * only trigger after all block in the associated kernel trigger the event. A
 * block can trigger the event through PTX launchdep.release or CUDA builtin
 * function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also be
 * inserted at the beginning of each block's execution if triggerAtBlockStart is
 * set to non-0. Note that dependents (including the CPU thread calling
 * cuEventSynchronize()) are not guaranteed to observe the release precisely
 * when it is released. For example, cuEventSynchronize() may only observe the
 * event trigger long after the associated kernel has completed. This recording
 * type is primarily meant for establishing programmatic dependency between
 * device tasks. The event supplied must not be an interprocess or interop
 * event. The event must disable timing (i.e. created with
 * ::CU_EVENT_DISABLE_TIMING flag set).
 *
 * The effect of other attributes is consistent with their effect when set via
 * persistent APIs.
 *
 * See ::cuStreamSetAttribute for
 * - ::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
 * - ::CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY
 *
 * See ::cuFunctionSetAttribute for
 * - ::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
 * - ::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
 *
 * Kernel parameters to \p f can be specified in the same ways that they can be
 * using ::cuLaunchKernel.
 *
 * Note that the API can also be used to launch context-less kernel ::CUkernel
 * by querying the handle using ::cuLibraryGetKernel() and then passing it
 * to the API by casting to ::CUfunction. Here, the context to launch
 * the kernel on will either be taken from the specified stream ::CUlaunchConfig::hStream
 * or the current context in case of NULL stream.
 *
 * \param config         - Config to launch
 * \param f              - Function ::CUfunction or Kernel ::CUkernel to launch
 * \param kernelParams   - Array of pointers to kernel parameters
 * \param extra          - Extra options
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_IMAGE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_NOT_FOUND
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cudaLaunchKernel,
 * ::cudaLaunchKernelEx,
 * ::cuLibraryGetKernel,
 * ::cuKernelSetCacheConfig,
 * ::cuKernelGetAttribute,
 * ::cuKernelSetAttribute
 *)
function cuLaunchKernelEx(const config : PCUlaunchConfig;
                                  f : CUfunction;
                                  kernelParams : Ppointer;
                                  extra : Ppointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Launches a CUDA function ::CUfunction or a CUDA kernel ::CUkernel where thread blocks
 * can cooperate and synchronize as they execute
 *
 * Invokes the function ::CUfunction or the kernel ::CUkernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
 * grid of blocks. Each block contains \p blockDimX x \p blockDimY x
 * \p blockDimZ threads.
 *
 * Note that the API can also be used to launch context-less kernel ::CUkernel
 * by querying the handle using ::cuLibraryGetKernel() and then passing it
 * to the API by casting to ::CUfunction. Here, the context to launch
 * the kernel on will either be taken from the specified stream \p hStream
 * or the current context in case of NULL stream.
 *
 * \p sharedMemBytes sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * The device on which this kernel is invoked must have a non-zero value for
 * the device attribute ::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::cuOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
 *
 * The kernel cannot make use of CUDA dynamic parallelism.
 *
 * Kernel parameters must be specified via \p kernelParams.  If \p f
 * has N parameters, then \p kernelParams needs to be an array of N
 * pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
 * must point to a region of memory from which the actual kernel
 * parameter will be copied.  The number of kernel parameters and their
 * offsets and sizes do not need to be specified as that information is
 * retrieved directly from the kernel's image.
 *
 * Calling ::cuLaunchCooperativeKernel() sets persistent function state that is
 * the same as function state set through ::cuLaunchKernel API
 *
 * When the kernel \p f is launched via ::cuLaunchCooperativeKernel(), the previous
 * block shape, shared size and parameter info associated with \p f
 * is overwritten.
 *
 * Note that to use ::cuLaunchCooperativeKernel(), the kernel \p f must either have
 * been compiled with toolchain version 3.2 or later so that it will
 * contain kernel parameter information, or have no kernel parameters.
 * If either of these conditions is not met, then ::cuLaunchCooperativeKernel() will
 * return ::CUDA_ERROR_INVALID_IMAGE.
 *
 * Note that the API can also be used to launch context-less kernel ::CUkernel
 * by querying the handle using ::cuLibraryGetKernel() and then passing it
 * to the API by casting to ::CUfunction. Here, the context to launch
 * the kernel on will either be taken from the specified stream \p hStream
 * or the current context in case of NULL stream.
 *
 * \param f              - Function ::CUfunction or Kernel ::CUkernel to launch
 * \param gridDimX       - Width of grid in blocks
 * \param gridDimY       - Height of grid in blocks
 * \param gridDimZ       - Depth of grid in blocks
 * \param blockDimX      - X dimension of each thread block
 * \param blockDimY      - Y dimension of each thread block
 * \param blockDimZ      - Z dimension of each thread block
 * \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
 * \param hStream        - Stream identifier
 * \param kernelParams   - Array of pointers to kernel parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_IMAGE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::CUDA_ERROR_NOT_FOUND
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cuLaunchCooperativeKernelMultiDevice,
 * ::cudaLaunchCooperativeKernel,
 * ::cuLibraryGetKernel,
 * ::cuKernelSetCacheConfig,
 * ::cuKernelGetAttribute,
 * ::cuKernelSetAttribute
 *)
function cuLaunchCooperativeKernel(f : CUfunction;
                                gridDimX : longword;
                                gridDimY : longword;
                                gridDimZ : longword;
                                blockDimX : longword;
                                blockDimY : longword;
                                blockDimZ : longword;
                                sharedMemBytes : longword;
                                hStream : CUstream;
                                kernelParams : Ppointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute
 *
 * \deprecated This function is deprecated as of CUDA 11.3.
 *
 * Invokes kernels as specified in the \p launchParamsList array where each element
 * of the array specifies all the parameters required to perform a single kernel launch.
 * These kernels can cooperate and synchronize as they execute. The size of the array is
 * specified by \p numDevices.
 *
 * No two kernels can be launched on the same device. All the devices targeted by this
 * multi-device launch must be identical. All devices must have a non-zero value for the
 * device attribute ::CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH.
 *
 * All kernels launched must be identical with respect to the compiled code. Note that
 * any __device__, __constant__ or __managed__ variables present in the module that owns
 * the kernel launched on each device, are independently instantiated on every device.
 * It is the application's responsibility to ensure these variables are initialized and
 * used appropriately.
 *
 * The size of the grids as specified in blocks, the size of the blocks themselves
 * and the amount of shared memory used by each thread block must also match across
 * all launched kernels.
 *
 * The streams used to launch these kernels must have been created via either ::cuStreamCreate
 * or ::cuStreamCreateWithPriority. The NULL stream or ::CU_STREAM_LEGACY or ::CU_STREAM_PER_THREAD
 * cannot be used.
 *
 * The total number of blocks launched per kernel cannot exceed the maximum number of blocks
 * per multiprocessor as returned by ::cuOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT. Since the
 * total number of blocks launched per device has to match across all devices, the maximum
 * number of blocks that can be launched per device will be limited by the device with the
 * least number of multiprocessors.
 *
 * The kernels cannot make use of CUDA dynamic parallelism.
 *
 * The ::CUDA_LAUNCH_PARAMS structure is defined as:
 * \code
          CUDA_LAUNCH_PARAMS_st=packed record
            CUfunction function;
            longint gridDimX;
            longint gridDimY;
            longint gridDimZ;
            longint blockDimX;
            longint blockDimY;
            longint blockDimZ;
            longint sharedMemBytes;
            CUstream hStream;
            pointer *kernelParams;
          end;
  CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_st;
 * \endcode
 * where:
 * - ::CUDA_LAUNCH_PARAMS::function specifies the kernel to be launched. All functions must
 *   be identical with respect to the compiled code.
 *   Note that you can also specify context-less kernel ::CUkernel by querying the handle
 *   using ::cuLibraryGetKernel() and then casting to ::CUfunction. In this case, the context to
 *   launch the kernel on be taken from the specified stream ::CUDA_LAUNCH_PARAMS::hStream.
 * - ::CUDA_LAUNCH_PARAMS::gridDimX is the width of the grid in blocks. This must match across
 *   all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::gridDimY is the height of the grid in blocks. This must match across
 *   all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::gridDimZ is the depth of the grid in blocks. This must match across
 *   all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::blockDimX is the X dimension of each thread block. This must match across
 *   all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::blockDimX is the Y dimension of each thread block. This must match across
 *   all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::blockDimZ is the Z dimension of each thread block. This must match across
 *   all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::sharedMemBytes is the dynamic shared-memory size per thread block in bytes.
 *   This must match across all kernels launched.
 * - ::CUDA_LAUNCH_PARAMS::hStream is the handle to the stream to perform the launch in. This cannot
 *   be the NULL stream or ::CU_STREAM_LEGACY or ::CU_STREAM_PER_THREAD. The CUDA context associated
 *   with this stream must match that associated with ::CUDA_LAUNCH_PARAMS::function.
 * - ::CUDA_LAUNCH_PARAMS::kernelParams is an array of pointers to kernel parameters. If
 *   ::CUDA_LAUNCH_PARAMS::function has N parameters, then ::CUDA_LAUNCH_PARAMS::kernelParams
 *   needs to be an array of N pointers. Each of ::CUDA_LAUNCH_PARAMS::kernelParams[0] through
 *   ::CUDA_LAUNCH_PARAMS::kernelParams[N-1] must point to a region of memory from which the actual
 *   kernel parameter will be copied. The number of kernel parameters and their offsets and sizes
 *   do not need to be specified as that information is retrieved directly from the kernel's image.
 *
 * By default, the kernel won't begin execution on any GPU until all prior work in all the specified
 * streams has completed. This behavior can be overridden by specifying the flag
 * ::CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC. When this flag is specified, each kernel
 * will only wait for prior work in the stream corresponding to that GPU to complete before it begins
 * execution.
 *
 * Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
 * execution until the kernels on all GPUs have completed. This behavior can be overridden by specifying
 * the flag ::CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC. When this flag is specified,
 * any subsequent work pushed in any of the specified streams will only wait for the kernel launched
 * on the GPU corresponding to that stream to complete before it begins execution.
 *
 * Calling ::cuLaunchCooperativeKernelMultiDevice() sets persistent function state that is
 * the same as function state set through ::cuLaunchKernel API when called individually for each
 * element in \p launchParamsList.
 *
 * When kernels are launched via ::cuLaunchCooperativeKernelMultiDevice(), the previous
 * block shape, shared size and parameter info associated with each ::CUDA_LAUNCH_PARAMS::function
 * in \p launchParamsList is overwritten.
 *
 * Note that to use ::cuLaunchCooperativeKernelMultiDevice(), the kernels must either have
 * been compiled with toolchain version 3.2 or later so that it will
 * contain kernel parameter information, or have no kernel parameters.
 * If either of these conditions is not met, then ::cuLaunchCooperativeKernelMultiDevice() will
 * return ::CUDA_ERROR_INVALID_IMAGE.
 *
 * \param launchParamsList - List of launch parameters, one per device
 * \param numDevices       - Size of the \p launchParamsList array
 * \param flags            - Flags to control launch behavior
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_IMAGE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuCtxGetCacheConfig,
 * ::cuCtxSetCacheConfig,
 * ::cuFuncSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cuLaunchCooperativeKernel,
 * ::cudaLaunchCooperativeKernelMultiDevice
 *)
function cuLaunchCooperativeKernelMultiDevice(launchParamsList : PCUDA_LAUNCH_PARAMS; numDevices : longword; flags : longword):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Enqueues a host function call in a stream
 *
 * Enqueues a host function to run in a stream.  The function will be called
 * after currently enqueued work and will block work added after it.
 *
 * The host function must not make any CUDA API calls.  Attempting to use a
 * CUDA API may result in ::CUDA_ERROR_NOT_PERMITTED, but this is not required.
 * The host function must not perform any synchronization that may depend on
 * outstanding CUDA work not mandated to run earlier.  Host functions without a
 * mandated order (such as in independent streams) execute in undefined order
 * and may be serialized.
 *
 * For the purposes of Unified Memory, execution makes a number of guarantees:
 * <ul>
 *   <li>The stream is considered idle for the duration of the function's
 *   execution.  Thus, for example, the function may always use memory attached
 *   to the stream it was enqueued in.</li>
 *   <li>The start of execution of the function has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the function.  It thus synchronizes streams which have been "joined"
 *   prior to the function.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding host functions and stream callbacks
 *   have executed.  Thus, for
 *   example, a function might use global attached memory even if work has
 *   been added to another stream, if the work has been ordered behind the
 *   function call with an event.</li>
 *   <li>Completion of the function does not cause a stream to become
 *   active except as described above.  The stream will remain idle
 *   if no device work follows the function, and will remain idle across
 *   consecutive host functions or stream callbacks without device work in
 *   between.  Thus, for example,
 *   stream synchronization can be done by signaling from a host function at the
 *   end of the stream.</li>
 * </ul>
 *
 * Note that, in contrast to ::cuStreamAddCallback, the function will not be
 * called in the event of an error in the CUDA context.
 *
 * \param hStream  - Stream to enqueue function call in
 * \param fn       - The function to call once preceding stream operations are complete
 * \param userData - User-specified data to be passed to the function
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuStreamCreate,
 * ::cuStreamQuery,
 * ::cuStreamSynchronize,
 * ::cuStreamWaitEvent,
 * ::cuStreamDestroy,
 * ::cuMemAllocManaged,
 * ::cuStreamAttachMemAsync,
 * ::cuStreamAddCallback
 *)
function cuLaunchHostFunc(hStream : CUstream; fn : CUhostFn; userData : Pointer):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_EXEC *)

(**
 * \defgroup CUDA_EXEC_DEPRECATED Execution Control [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated execution control functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated execution control functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Sets the block-dimensions for the function
 *
 * \deprecated
 *
 * Specifies the \p x, \p y, and \p z dimensions of the thread blocks that are
 * created when the kernel given by \p hfunc is launched.
 *
 * \param hfunc - Kernel to specify dimensions of
 * \param x     - X dimension
 * \param y     - Y dimension
 * \param z     - Z dimension
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuFuncSetSharedSize,
 * ::cuFuncSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSeti,
 * ::cuParamSetf,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuFuncSetBlockShape(hfunc : CUfunction; x : longint; y : longint; z : longint):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the dynamic shared-memory size for the function
 *
 * \deprecated
 *
 * Sets through \p bytes the amount of dynamic shared memory that will be
 * available to each thread block when the kernel given by \p hfunc is launched.
 *
 * \param hfunc - Kernel to specify dynamic shared-memory size for
 * \param bytes - Dynamic shared-memory size per thread in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetCacheConfig,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSeti,
 * ::cuParamSetf,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuFuncSetSharedSize(hfunc : CUfunction; bytes : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameter size for the function
 *
 * \deprecated
 *
 * Sets through \p numbytes the total size in bytes needed by the function
 * parameters of the kernel corresponding to \p hfunc.
 *
 * \param hfunc    - Kernel to set parameter size for
 * \param numbytes - Size of parameter list in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetf,
 * ::cuParamSeti,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuParamSetSize(hfunc : CUfunction; numbytes : longword):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Adds an integer parameter to the function's argument list
 *
 * \deprecated
 *
 * Sets an integer parameter that will be specified the next time the
 * kernel corresponding to \p hfunc will be invoked. \p offset is a byte offset.
 *
 * \param hfunc  - Kernel to add parameter to
 * \param offset - Offset to add parameter to argument list
 * \param value  - Value of parameter
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSetf,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuParamSeti(hfunc : CUfunction; offset : longint; value : longword):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Adds a floating-point parameter to the function's argument list
 *
 * \deprecated
 *
 * Sets a floating-point parameter that will be specified the next time the
 * kernel corresponding to \p hfunc will be invoked. \p offset is a byte offset.
 *
 * \param hfunc  - Kernel to add parameter to
 * \param offset - Offset to add parameter to argument list
 * \param value  - Value of parameter
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSeti,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuParamSetf(hfunc : CUfunction; offset : longint; value : single):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Adds arbitrary data to the function's argument list
 *
 * \deprecated
 *
 * Copies an arbitrary amount of data (specified in \p numbytes) from \p ptr
 * into the parameter space of the kernel corresponding to \p hfunc. \p offset
 * is a byte offset.
 *
 * \param hfunc    - Kernel to add data to
 * \param offset   - Offset to add data to argument list
 * \param ptr      - Pointer to arbitrary data
 * \param numbytes - Size of data to copy in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSetf,
 * ::cuParamSeti,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuParamSetv(hfunc : CUfunction; offset : longint; ptr : Pointer; numbytes : longword):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Launches a CUDA function
 *
 * \deprecated
 *
 * Invokes the kernel \p f on a 1 x 1 x 1 grid of blocks. The block
 * contains the number of threads specified by a previous call to
 * ::cuFuncSetBlockShape().
 *
 * The block shape, dynamic shared memory size, and parameter information
 * must be set using
 *  ::cuFuncSetBlockShape(),
 *  ::cuFuncSetSharedSize(),
 *  ::cuParamSetSize(),
 *  ::cuParamSeti(),
 *  ::cuParamSetf(), and
 *  ::cuParamSetv()
 * prior to calling this function.
 *
 * Launching a function via ::cuLaunchKernel() invalidates the function's
 * block shape, dynamic shared memory size, and parameter information. After
 * launching via cuLaunchKernel, this state must be re-initialized prior to
 * calling this function. Failure to do so results in undefined behavior.
 *
 * \param f - Kernel to launch
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSetf,
 * ::cuParamSeti,
 * ::cuParamSetv,
 * ::cuLaunchGrid,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuLaunch(f : CUfunction):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Launches a CUDA function
 *
 * \deprecated
 *
 * Invokes the kernel \p f on a \p grid_width x \p grid_height grid of
 * blocks. Each block contains the number of threads specified by a previous
 * call to ::cuFuncSetBlockShape().
 *
 * The block shape, dynamic shared memory size, and parameter information
 * must be set using
 *  ::cuFuncSetBlockShape(),
 *  ::cuFuncSetSharedSize(),
 *  ::cuParamSetSize(),
 *  ::cuParamSeti(),
 *  ::cuParamSetf(), and
 *  ::cuParamSetv()
 * prior to calling this function.
 *
 * Launching a function via ::cuLaunchKernel() invalidates the function's
 * block shape, dynamic shared memory size, and parameter information. After
 * launching via cuLaunchKernel, this state must be re-initialized prior to
 * calling this function. Failure to do so results in undefined behavior.
 *
 * \param f           - Kernel to launch
 * \param grid_width  - Width of grid in blocks
 * \param grid_height - Height of grid in blocks
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSetf,
 * ::cuParamSeti,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGridAsync,
 * ::cuLaunchKernel
 *)
function cuLaunchGrid(f : CUfunction; grid_width : longint; grid_height : longint):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Launches a CUDA function
 *
 * \deprecated
 *
 * Invokes the kernel \p f on a \p grid_width x \p grid_height grid of
 * blocks. Each block contains the number of threads specified by a previous
 * call to ::cuFuncSetBlockShape().
 *
 * The block shape, dynamic shared memory size, and parameter information
 * must be set using
 *  ::cuFuncSetBlockShape(),
 *  ::cuFuncSetSharedSize(),
 *  ::cuParamSetSize(),
 *  ::cuParamSeti(),
 *  ::cuParamSetf(), and
 *  ::cuParamSetv()
 * prior to calling this function.
 *
 * Launching a function via ::cuLaunchKernel() invalidates the function's
 * block shape, dynamic shared memory size, and parameter information. After
 * launching via cuLaunchKernel, this state must be re-initialized prior to
 * calling this function. Failure to do so results in undefined behavior.
 *
 * \param f           - Kernel to launch
 * \param grid_width  - Width of grid in blocks
 * \param grid_height - Height of grid in blocks
 * \param hStream     - Stream identifier
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_LAUNCH_FAILED,
 * ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::CUDA_ERROR_LAUNCH_TIMEOUT,
 * ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
 *
 * \note In certain cases where cubins are created with no ABI (i.e., using \p ptxas \p --abi-compile \p no),
 *       this function may serialize kernel launches. The CUDA driver retains asynchronous behavior by
 *       growing the per-thread stack as needed per launch and not shrinking it afterwards.
 *
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::cuFuncSetBlockShape,
 * ::cuFuncSetSharedSize,
 * ::cuFuncGetAttribute,
 * ::cuParamSetSize,
 * ::cuParamSetf,
 * ::cuParamSeti,
 * ::cuParamSetv,
 * ::cuLaunch,
 * ::cuLaunchGrid,
 * ::cuLaunchKernel
 *)
function cuLaunchGridAsync(f : CUfunction; grid_width : longint; grid_height : longint; hStream : CUstream):CUresult;  WINAPI; external libcuda;deprecated;


(**
 * \brief Adds a texture-reference to the function's argument list
 *
 * \deprecated
 *
 * Makes the CUDA array or linear memory bound to the texture reference
 * \p hTexRef available to a device program as a texture. In this version of
 * CUDA, the texture-reference must be obtained via ::cuModuleGetTexRef() and
 * the \p texunit parameter must be set to ::CU_PARAM_TR_DEFAULT.
 *
 * \param hfunc   - Kernel to add texture-reference to
 * \param texunit - Texture unit (must be ::CU_PARAM_TR_DEFAULT)
 * \param hTexRef - Texture-reference to add to argument list
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *)
function cuParamSetTexRef(hfunc : CUfunction; texunit : longint; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;deprecated;
(** @} *) (* END CUDA_EXEC_DEPRECATED *)

(**
 * \defgroup CUDA_GRAPH Graph Management
 *
 * ___MANBRIEF___ graph management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graph management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Creates a graph
 *
 * Creates an empty graph, which is returned via \p phGraph.
 *
 * \param phGraph - Returns newly created graph
 * \param flags   - Graph creation flags, must be 0
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode,
 * ::cuGraphInstantiate,
 * ::cuGraphDestroy,
 * ::cuGraphGetNodes,
 * ::cuGraphGetRootNodes,
 * ::cuGraphGetEdges,
 * ::cuGraphClone
 *)
function cuGraphCreate(phGraph : PCUgraph; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a kernel execution node and adds it to a graph
 *
 * Creates a new kernel execution node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * The CUDA_KERNEL_NODE_PARAMS structure is defined as:
 *
 * \code
 *    CUDA_KERNEL_NODE_PARAMS_st=packed record
 *      CUfunction func;
 *      longword gridDimX;
 *      longword gridDimY;
 *      longword gridDimZ;
 *      longword blockDimX;
 *      longword blockDimY;
 *      longword blockDimZ;
 *      longword sharedMemBytes;
 *      pointer *kernelParams;
 *      pointer *extra;
 *    end;
  CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_st;
 * \endcode
 *
 * When the graph is launched, the node will invoke kernel \p func on a (\p gridDimX x
 * \p gridDimY x \p gridDimZ) grid of blocks. Each block contains
 * (\p blockDimX x \p blockDimY x \p blockDimZ) threads.
 *
 * \p sharedMemBytes sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * Kernel parameters to \p func can be specified in one of two ways:
 *
 * 1) Kernel parameters can be specified via \p kernelParams. If the kernel has N
 * parameters, then \p kernelParams needs to be an array of N pointers. Each pointer,
 * from \p kernelParams[0] to \p kernelParams[N-1], points to the region of memory from which the actual
 * parameter will be copied. The number of kernel parameters and their offsets and sizes do not need
 * to be specified as that information is retrieved directly from the kernel's image.
 *
 * 2) Kernel parameters for non-cooperative kernels can also be packaged by the application into a single
 * buffer that is passed in via \p extra. This places the burden on the application of knowing each
 * kernel parameter's size and alignment/padding within the buffer. The \p extra parameter exists
 * to allow this function to take additional less commonly used arguments. \p extra specifies
 * a list of names of extra settings and their corresponding values. Each extra setting name is
 * immediately followed by the corresponding value. The list must be terminated with either NULL or
 * CU_LAUNCH_PARAM_END.
 *
 * - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
 *   array;
 * - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
 *   value in \p extra will be a pointer to a buffer
 *   containing all the kernel parameters for launching kernel
 *   \p func;
 * - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
 *   value in \p extra will be a pointer to a size_t
 *   containing the size of the buffer specified with
 *   ::CU_LAUNCH_PARAM_BUFFER_POINTER;
 *
 * The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters are specified with both
 * \p kernelParams and \p extra (i.e. both \p kernelParams and \p extra are non-NULL).
 * ::CUDA_ERROR_INVALID_VALUE will be returned if \p extra is used for a cooperative kernel.
 *
 * The \p kernelParams or \p extra array, as well as the argument values it points to,
 * are copied during this call.
 *
 * \note Kernels launched using graphs must not use texture and surface references. Reading or
 *       writing through any texture or surface reference is undefined behavior.
 *       This restriction does not apply to texture and surface objects.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the GPU execution node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchKernel,
 * ::cuLaunchCooperativeKernel,
 * ::cuGraphKernelNodeGetParams,
 * ::cuGraphKernelNodeSetParams,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddKernelNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const nodeParams : PCUDA_KERNEL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a kernel node's parameters
 *
 * Returns the parameters of kernel node \p hNode in \p nodeParams.
 * The \p kernelParams or \p extra array returned in \p nodeParams,
 * as well as the argument values it points to, are owned by the node.
 * This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cuGraphKernelNodeSetParams to update the
 * parameters of this node.
 *
 * The params will contain either \p kernelParams or \p extra,
 * according to which of these was most recently set on the node.
 *
 * \param hNode      - Node to get the parameters for
 * \param nodeParams - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchKernel,
 * ::cuGraphAddKernelNode,
 * ::cuGraphKernelNodeSetParams
 *)
function cuGraphKernelNodeGetParams(hNode : CUgraphNode; nodeParams : PCUDA_KERNEL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets a kernel node's parameters
 *
 * Sets the parameters of kernel node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchKernel,
 * ::cuGraphAddKernelNode,
 * ::cuGraphKernelNodeGetParams
 *)
function cuGraphKernelNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_KERNEL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a memcpy node and adds it to a graph
 *
 * Creates a new memcpy node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the graph is launched, the node will perform the memcpy described by \p copyParams.
 * See ::cuMemcpy3D() for a description of the structure and its restrictions.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. If one or more of the operands refer
 * to managed memory, then using the memory type ::CU_MEMORYTYPE_UNIFIED is disallowed
 * for those operand(s). The managed memory will be treated as residing on either the
 * host or the device, depending on which memory type is specified.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param copyParams      - Parameters for the memory copy
 * \param ctx             - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuMemcpy3D,
 * ::cuGraphMemcpyNodeGetParams,
 * ::cuGraphMemcpyNodeSetParams,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddMemcpyNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const copyParams : PCUDA_MEMCPY3D; ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a memcpy node's parameters
 *
 * Returns the parameters of memcpy node \p hNode in \p nodeParams.
 *
 * \param hNode      - Node to get the parameters for
 * \param nodeParams - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuMemcpy3D,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphMemcpyNodeSetParams
 *)
function cuGraphMemcpyNodeGetParams(hNode : CUgraphNode; nodeParams : PCUDA_MEMCPY3D):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets a memcpy node's parameters
 *
 * Sets the parameters of memcpy node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuMemcpy3D,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphMemcpyNodeGetParams
 *)
function cuGraphMemcpyNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_MEMCPY3D):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a memset node and adds it to a graph
 *
 * Creates a new memset node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * The element size must be 1, 2, or 4 bytes.
 * When the graph is launched, the node will perform the memset described by \p memsetParams.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param memsetParams    - Parameters for the memory set
 * \param ctx             - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuMemsetD2D32,
 * ::cuGraphMemsetNodeGetParams,
 * ::cuGraphMemsetNodeSetParams,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemcpyNode
 *)
function cuGraphAddMemsetNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const memsetParams : PCUDA_MEMSET_NODE_PARAMS; ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a memset node's parameters
 *
 * Returns the parameters of memset node \p hNode in \p nodeParams.
 *
 * \param hNode      - Node to get the parameters for
 * \param nodeParams - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuMemsetD2D32,
 * ::cuGraphAddMemsetNode,
 * ::cuGraphMemsetNodeSetParams
 *)
function cuGraphMemsetNodeGetParams(hNode : CUgraphNode; nodeParams : PCUDA_MEMSET_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets a memset node's parameters
 *
 * Sets the parameters of memset node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuMemsetD2D32,
 * ::cuGraphAddMemsetNode,
 * ::cuGraphMemsetNodeGetParams
 *)
function cuGraphMemsetNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_MEMSET_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a host execution node and adds it to a graph
 *
 * Creates a new CPU execution node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the graph is launched, the node will invoke the specified CPU function.
 * Host nodes are not supported under MPS with pre-Volta GPUs.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the host node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchHostFunc,
 * ::cuGraphHostNodeGetParams,
 * ::cuGraphHostNodeSetParams,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddHostNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const nodeParams : PCUDA_HOST_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a host node's parameters
 *
 * Returns the parameters of host node \p hNode in \p nodeParams.
 *
 * \param hNode      - Node to get the parameters for
 * \param nodeParams - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchHostFunc,
 * ::cuGraphAddHostNode,
 * ::cuGraphHostNodeSetParams
 *)
function cuGraphHostNodeGetParams(hNode : CUgraphNode; nodeParams : PCUDA_HOST_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets a host node's parameters
 *
 * Sets the parameters of host node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchHostFunc,
 * ::cuGraphAddHostNode,
 * ::cuGraphHostNodeGetParams
 *)
function cuGraphHostNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_HOST_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a child graph node and adds it to a graph
 *
 * Creates a new node which executes an embedded graph, and adds it to \p hGraph with
 * \p numDependencies dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * If \p hGraph contains allocation or free nodes, this call will return an error.
 *
 * The node executes an embedded child graph. The child graph is cloned in this call.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param childGraph      - The graph to clone into this node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphChildGraphNodeGetGraph,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode,
 * ::cuGraphClone
 *)
function cuGraphAddChildGraphNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; childGraph : CUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Gets a handle to the embedded graph of a child graph node
 *
 * Gets a handle to the embedded graph in a child graph node. This call
 * does not clone the graph. Changes to the graph will be reflected in
 * the node, and the node retains ownership of the graph.
 *
 * Allocation and free nodes cannot be added to the returned graph.
 * Attempting to do so will return an error.
 *
 * \param hNode   - Node to get the embedded graph for
 * \param phGraph - Location to store a handle to the graph
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphNodeFindInClone
 *)
function cuGraphChildGraphNodeGetGraph(hNode : CUgraphNode; phGraph : PCUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an empty node and adds it to a graph
 *
 * Creates a new node which performs no operation, and adds it to \p hGraph with
 * \p numDependencies dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * An empty node performs no operation during execution, but can be used for
 * transitive ordering. For example, a phased execution graph with 2 groups of n
 * nodes with a barrier between them can be represented using an empty node and
 * 2*n dependency edges, rather than no empty node and n^2 dependency edges.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddEmptyNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an event record node and adds it to a graph
 *
 * Creates a new event record node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and event specified in \p event.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * Each launch of the graph will record \p event to capture execution of the
 * node's dependencies.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param event           - Event for the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventWaitNode,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddEventRecordNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; event : CUevent):CUresult;  WINAPI; external libcuda;
 
(**
 * \brief Returns the event associated with an event record node
 *
 * Returns the event of event record node \p hNode in \p event_out.
 *
 * \param hNode     - Node to get the event for
 * \param event_out - Pointer to return the event
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphEventRecordNodeSetEvent,
 * ::cuGraphEventWaitNodeGetEvent,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent
 *)
function cuGraphEventRecordNodeGetEvent(hNode : CUgraphNode; event_out : PCUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets an event record node's event
 *
 * Sets the event of event record node \p hNode to \p event.
 *
 * \param hNode - Node to set the event for
 * \param event - Event to use
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphEventRecordNodeGetEvent,
 * ::cuGraphEventWaitNodeSetEvent,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent
 *)
function cuGraphEventRecordNodeSetEvent(hNode : CUgraphNode; event : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an event wait node and adds it to a graph
 *
 * Creates a new event wait node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and event specified in \p event.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * The graph node will wait for all work captured in \p event.  See ::cuEventRecord()
 * for details on what is captured by an event. \p event may be from a different context
 * or device than the launch stream.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param event           - Event for the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventRecordNode,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddEventWaitNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; event : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns the event associated with an event wait node
 *
 * Returns the event of event wait node \p hNode in \p event_out.
 *
 * \param hNode     - Node to get the event for
 * \param event_out - Pointer to return the event
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphEventWaitNodeSetEvent,
 * ::cuGraphEventRecordNodeGetEvent,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent
 *)
function cuGraphEventWaitNodeGetEvent(hNode : CUgraphNode; event_out : PCUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets an event wait node's event
 *
 * Sets the event of event wait node \p hNode to \p event.
 *
 * \param hNode - Node to set the event for
 * \param event - Event to use
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphEventWaitNodeGetEvent,
 * ::cuGraphEventRecordNodeSetEvent,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent
 *)
function cuGraphEventWaitNodeSetEvent(hNode : CUgraphNode; event : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an external semaphore signal node and adds it to a graph
 *
 * Creates a new external semaphore signal node and adds it to \p hGraph with \p
 * numDependencies dependencies specified via \p dependencies and arguments specified
 * in \p nodeParams. It is possible for \p numDependencies to be 0, in which case the
 * node will be placed at the root of the graph. \p dependencies may not have any
 * duplicate entries. A handle to the new node will be returned in \p phGraphNode.
 *
 * Performs a signal operation on a set of externally allocated semaphore objects
 * when the node is launched.  The operation(s) will occur after all of the node's
 * dependencies have completed.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphExternalSemaphoresSignalNodeGetParams,
 * ::cuGraphExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuImportExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddExternalSemaphoresSignalNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const nodeParams : PCUDA_EXT_SEM_SIGNAL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns an external semaphore signal node's parameters
 *
 * Returns the parameters of an external semaphore signal node \p hNode in \p params_out.
 * The \p extSemArray and \p paramsArray returned in \p params_out,
 * are owned by the node.  This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cuGraphExternalSemaphoresSignalNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchKernel,
 * ::cuGraphAddExternalSemaphoresSignalNode,
 * ::cuGraphExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuGraphExternalSemaphoresSignalNodeGetParams(hNode : CUgraphNode; params_out : PCUDA_EXT_SEM_SIGNAL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets an external semaphore signal node's parameters
 *
 * Sets the parameters of an external semaphore signal node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddExternalSemaphoresSignalNode,
 * ::cuGraphExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuGraphExternalSemaphoresSignalNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_EXT_SEM_SIGNAL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an external semaphore wait node and adds it to a graph
 *
 * Creates a new external semaphore wait node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p phGraphNode.
 *
 * Performs a wait operation on a set of externally allocated semaphore objects
 * when the node is launched.  The node's dependencies will not be launched until
 * the wait operation has completed.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphExternalSemaphoresWaitNodeGetParams,
 * ::cuGraphExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphAddExternalSemaphoresSignalNode,
 * ::cuImportExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddExternalSemaphoresWaitNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const nodeParams : PCUDA_EXT_SEM_WAIT_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns an external semaphore wait node's parameters
 *
 * Returns the parameters of an external semaphore wait node \p hNode in \p params_out.
 * The \p extSemArray and \p paramsArray returned in \p params_out,
 * are owned by the node.  This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cuGraphExternalSemaphoresSignalNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuLaunchKernel,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuGraphExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuGraphExternalSemaphoresWaitNodeGetParams(hNode : CUgraphNode; params_out : PCUDA_EXT_SEM_WAIT_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets an external semaphore wait node's parameters
 *
 * Sets the parameters of an external semaphore wait node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuGraphExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync
 *)
function cuGraphExternalSemaphoresWaitNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_EXT_SEM_WAIT_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a batch memory operation node and adds it to a graph
 *
 * Creates a new batch memory operation node and adds it to \p hGraph with \p
 * numDependencies dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the node is added, the paramArray inside \p nodeParams is copied and therefore it can be
 * freed after the call returns.
 *
 * \note
 * Warning:
 * Improper use of this API may deadlock the application. Synchronization 
 * ordering established through this API is not visible to CUDA. CUDA tasks 
 * that are (even indirectly) ordered by this API should also have that order
 * expressed with CUDA-visible dependencies such as events. This ensures that 
 * the scheduler does not serialize them in an improper order. For more 
 * information, see the Stream Memory Operations section in the programming 
 * guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuStreamBatchMemOp,
 * ::cuStreamWaitValue32,
 * ::cuStreamWriteValue32,
 * ::cuStreamWaitValue64,
 * ::cuStreamWriteValue64,
 * ::cuGraphBatchMemOpNodeGetParams,
 * ::cuGraphBatchMemOpNodeSetParams,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddBatchMemOpNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; const nodeParams : PCUDA_BATCH_MEM_OP_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a batch mem op node's parameters
 *
 * Returns the parameters of batch mem op node \p hNode in \p nodeParams_out.
 * The \p paramArray returned in \p nodeParams_out is owned by the node.
 * This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cuGraphBatchMemOpNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode          - Node to get the parameters for
 * \param nodeParams_out - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuStreamBatchMemOp,
 * ::cuGraphAddBatchMemOpNode,
 * ::cuGraphBatchMemOpNodeSetParams
 *)
function cuGraphBatchMemOpNodeGetParams(hNode : CUgraphNode; nodeParams_out : PCUDA_BATCH_MEM_OP_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets a batch mem op node's parameters
 *
 * Sets the parameters of batch mem op node \p hNode to \p nodeParams.
 *
 * The paramArray inside \p nodeParams is copied and therefore it can be
 * freed after the call returns.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuStreamBatchMemOp,
 * ::cuGraphAddBatchMemOpNode,
 * ::cuGraphBatchMemOpNodeGetParams
 *)
function cuGraphBatchMemOpNodeSetParams(hNode : CUgraphNode; const nodeParams : PCUDA_BATCH_MEM_OP_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for a batch mem op node in the given graphExec
 *
 * Sets the parameters of a batch mem op node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The following fields on operations may be modified on an executable graph:
 *
 *  op.waitValue.address
 *  op.waitValue.value[64]
 *  op.waitValue.flags bits corresponding to wait type (i.e. CU_STREAM_WAIT_VALUE_FLUSH bit cannot be modified)
 *  op.writeValue.address
 *  op.writeValue.value[64]
 *
 * Other fields, such as the context, count or type of operations, and other types of operations such as membars, 
 * may not be modified.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * The paramArray inside \p nodeParams is copied and therefore it can be
 * freed after the call returns.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Batch mem op node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuStreamBatchMemOp,
 * ::cuGraphAddBatchMemOpNode,
 * ::cuGraphBatchMemOpNodeGetParams,
 * ::cuGraphBatchMemOpNodeSetParams,
 * ::cuGraphInstantiate
 *)
function cuGraphExecBatchMemOpNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const nodeParams : PCUDA_BATCH_MEM_OP_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an allocation node and adds it to a graph
 *
 * Creates a new allocation node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p phGraphNode.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * When ::cuGraphAddMemAllocNode creates an allocation node, it returns the address of the allocation in
 * \p nodeParams.dptr.  The allocation's address remains fixed across instantiations and launches.
 *
 * If the allocation is freed in the same graph, by creating a free node using ::cuGraphAddMemFreeNode,
 * the allocation can be accessed by nodes ordered after the allocation node but before the free node.
 * These allocations cannot be freed outside the owning graph, and they can only be freed once in the
 * owning graph.
 *
 * If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the
 * graph which are ordered after the allocation node, but also by stream operations ordered after the
 * graph's execution but before the allocation is freed.
 *
 * Allocations which are not freed in the same graph can be freed by:
 * - passing the allocation to ::cuMemFreeAsync or ::cuMemFree;
 * - launching a graph with a free node for that allocation; or
 * - specifying ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH during instantiation, which makes
 * each launch behave as though it called ::cuMemFreeAsync for every unfreed allocation.
 * 
 * It is not possible to free an allocation in both the owning graph and another graph.  If the allocation
 * is freed in the same graph, a free node cannot be added to another graph.  If the allocation is freed
 * in another graph, a free node can no longer be added to the owning graph.
 *
 * The following restrictions apply to graphs which contain allocation and/or memory free nodes:
 * - Nodes and edges of the graph cannot be deleted.
 * - The graph cannot be used in a child node.
 * - Only one instantiation of the graph may exist at any point in time.
 * - The graph cannot be cloned.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddMemFreeNode,
 * ::cuGraphMemAllocNodeGetParams,
 * ::cuDeviceGraphMemTrim,
 * ::cuDeviceGetGraphMemAttribute,
 * ::cuDeviceSetGraphMemAttribute,
 * ::cuMemAllocAsync,
 * ::cuMemFreeAsync,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphAddExternalSemaphoresSignalNode,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddMemAllocNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; nodeParams : PCUDA_MEM_ALLOC_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a memory alloc node's parameters
 *
 * Returns the parameters of a memory alloc node \p hNode in \p params_out.
 * The \p poolProps and \p accessDescs returned in \p params_out, are owned by the
 * node.  This memory remains valid until the node is destroyed.  The returned
 * parameters must not be modified.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddMemAllocNode,
 * ::cuGraphMemFreeNodeGetParams
 *)
function cuGraphMemAllocNodeGetParams(hNode : CUgraphNode; params_out : PCUDA_MEM_ALLOC_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates a memory free node and adds it to a graph
 *
 * Creates a new memory free node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p phGraphNode.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dptr            - Address of memory to free
 *
 * ::cuGraphAddMemFreeNode will return ::CUDA_ERROR_INVALID_VALUE if the user attempts to free:
 * - an allocation twice in the same graph.
 * - an address that was not returned by an allocation node.
 * - an invalid address.
 *
 * The following restrictions apply to graphs which contain allocation and/or memory free nodes:
 * - Nodes and edges of the graph cannot be deleted.
 * - The graph cannot be used in a child node.
 * - Only one instantiation of the graph may exist at any point in time.
 * - The graph cannot be cloned.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NOT_SUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddMemAllocNode,
 * ::cuGraphMemFreeNodeGetParams,
 * ::cuDeviceGraphMemTrim,
 * ::cuDeviceGetGraphMemAttribute,
 * ::cuDeviceSetGraphMemAttribute,
 * ::cuMemAllocAsync,
 * ::cuMemFreeAsync,
 * ::cuGraphCreate,
 * ::cuGraphDestroyNode,
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphAddExternalSemaphoresSignalNode,
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphAddMemFreeNode(phGraphNode : PCUgraphNode; hGraph : CUgraph; const dependencies : PCUgraphNode; numDependencies : size_t; dptr : CUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a memory free node's parameters
 *
 * Returns the address of a memory free node \p hNode in \p dptr_out.
 *
 * \param hNode    - Node to get the parameters for
 * \param dptr_out - Pointer to return the device address
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddMemFreeNode,
 * ::cuGraphMemAllocNodeGetParams
 *)
function cuGraphMemFreeNodeGetParams(hNode : CUgraphNode; dptr_out : PCUdeviceptr):CUresult;  WINAPI; external libcuda;

(**
 * \brief Free unused memory that was cached on the specified device for use with graphs back to the OS.
 *
 * Blocks which are not in use by a graph that is either currently executing or scheduled to execute are
 * freed back to the operating system.
 *
 * \param device - The device for which cached memory should be freed.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_DEVICE
 *
 * \sa
 * ::cuGraphAddMemAllocNode,
 * ::cuGraphAddMemFreeNode,
 * ::cuDeviceSetGraphMemAttribute,
 * ::cuDeviceGetGraphMemAttribute
 *)
function cuDeviceGraphMemTrim(device : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query asynchronous allocation attributes related to graphs
 *
 * Valid attributes are:
 *
 * - ::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT: Amount of memory, in bytes, currently associated with graphs
 * - ::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: High watermark of memory, in bytes, associated with graphs since the
 *   last time it was reset.  High watermark can only be reset to zero.
 * - ::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT: Amount of memory, in bytes, currently allocated for use by
 *   the CUDA graphs asynchronous allocator.
 * - ::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: High watermark of memory, in bytes, currently allocated for use by
 *   the CUDA graphs asynchronous allocator.
 *
 * \param device - Specifies the scope of the query
 * \param attr - attribute to get
 * \param value - retrieved value
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_DEVICE
 *
 * \sa
 * ::cuDeviceSetGraphMemAttribute,
 * ::cuGraphAddMemAllocNode,
 * ::cuGraphAddMemFreeNode
 *)
function cuDeviceGetGraphMemAttribute(device : CUdevice; attr : CUgraphMem_attribute; value : pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Set asynchronous allocation attributes related to graphs
 *
 * Valid attributes are:
 *
 * - ::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: High watermark of memory, in bytes, associated with graphs since the
 *   last time it was reset.  High watermark can only be reset to zero.
 * - ::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: High watermark of memory, in bytes, currently allocated for use by
 *   the CUDA graphs asynchronous allocator.
 *
 * \param device - Specifies the scope of the query
 * \param attr - attribute to get
 * \param value - pointer to value to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_DEVICE
 *
 * \sa
 * ::cuDeviceGetGraphMemAttribute,
 * ::cuGraphAddMemAllocNode,
 * ::cuGraphAddMemFreeNode
 *)
function cuDeviceSetGraphMemAttribute(device : CUdevice; attr : CUgraphMem_attribute; value : pointer):CUresult;  WINAPI; external libcuda;

(**
 * \brief Clones a graph
 *
 * This function creates a copy of \p originalGraph and returns it in \p phGraphClone.
 * All parameters are copied into the cloned graph. The original graph may be modified
 * after this call without affecting the clone.
 *
 * Child graph nodes in the original graph are recursively copied into the clone.
 *
 * \param phGraphClone  - Returns newly created cloned graph
 * \param originalGraph - Graph to clone
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphCreate,
 * ::cuGraphNodeFindInClone
 *)
function cuGraphClone(phGraphClone : PCUgraph; originalGraph : CUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Finds a cloned version of a node
 *
 * This function returns the node in \p hClonedGraph corresponding to \p hOriginalNode
 * in the original graph.
 *
 * \p hClonedGraph must have been cloned from \p hOriginalGraph via ::cuGraphClone.
 * \p hOriginalNode must have been in \p hOriginalGraph at the time of the call to
 * ::cuGraphClone, and the corresponding cloned node in \p hClonedGraph must not have
 * been removed. The cloned node is then returned via \p phClonedNode.
 *
 * \param phNode  - Returns handle to the cloned node
 * \param hOriginalNode - Handle to the original node
 * \param hClonedGraph - Cloned graph to query
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphClone
 *)
function cuGraphNodeFindInClone(phNode : PCUgraphNode; hOriginalNode : CUgraphNode; hClonedGraph : CUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a node's type
 *
 * Returns the node type of \p hNode in \p type.
 *
 * \param hNode - Node to query
 * \param type  - Pointer to return the node type
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphGetNodes,
 * ::cuGraphGetRootNodes,
 * ::cuGraphChildGraphNodeGetGraph,
 * ::cuGraphKernelNodeGetParams,
 * ::cuGraphKernelNodeSetParams,
 * ::cuGraphHostNodeGetParams,
 * ::cuGraphHostNodeSetParams,
 * ::cuGraphMemcpyNodeGetParams,
 * ::cuGraphMemcpyNodeSetParams,
 * ::cuGraphMemsetNodeGetParams,
 * ::cuGraphMemsetNodeSetParams
 *)
function cuGraphNodeGetType(hNode : CUgraphNode; &type : PCUgraphNodeType):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a graph's nodes
 *
 * Returns a list of \p hGraph's nodes. \p nodes may be NULL, in which case this
 * function will return the number of nodes in \p numNodes. Otherwise,
 * \p numNodes entries will be filled in. If \p numNodes is higher than the actual
 * number of nodes, the remaining entries in \p nodes will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p numNodes.
 *
 * \param hGraph   - Graph to query
 * \param nodes    - Pointer to return the nodes
 * \param numNodes - See description
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphCreate,
 * ::cuGraphGetRootNodes,
 * ::cuGraphGetEdges,
 * ::cuGraphNodeGetType,
 * ::cuGraphNodeGetDependencies,
 * ::cuGraphNodeGetDependentNodes
 *)
function cuGraphGetNodes(hGraph : CUgraph; nodes : PCUgraphNode; numNodes : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a graph's root nodes
 *
 * Returns a list of \p hGraph's root nodes. \p rootNodes may be NULL, in which case this
 * function will return the number of root nodes in \p numRootNodes. Otherwise,
 * \p numRootNodes entries will be filled in. If \p numRootNodes is higher than the actual
 * number of root nodes, the remaining entries in \p rootNodes will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p numRootNodes.
 *
 * \param hGraph       - Graph to query
 * \param rootNodes    - Pointer to return the root nodes
 * \param numRootNodes - See description
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphCreate,
 * ::cuGraphGetNodes,
 * ::cuGraphGetEdges,
 * ::cuGraphNodeGetType,
 * ::cuGraphNodeGetDependencies,
 * ::cuGraphNodeGetDependentNodes
 *)
function cuGraphGetRootNodes(hGraph : CUgraph; rootNodes : PCUgraphNode; numRootNodes : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a graph's dependency edges
 *
 * Returns a list of \p hGraph's dependency edges. Edges are returned via corresponding
 * indices in \p from and \p to; that is, the node in \p to[i] has a dependency on the
 * node in \p from[i]. \p from and \p to may both be NULL, in which
 * case this function only returns the number of edges in \p numEdges. Otherwise,
 * \p numEdges entries will be filled in. If \p numEdges is higher than the actual
 * number of edges, the remaining entries in \p from and \p to will be set to NULL, and
 * the number of edges actually returned will be written to \p numEdges.
 *
 * \param hGraph   - Graph to get the edges from
 * \param from     - Location to return edge endpoints
 * \param to       - Location to return edge endpoints
 * \param numEdges - See description
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphGetNodes,
 * ::cuGraphGetRootNodes,
 * ::cuGraphAddDependencies,
 * ::cuGraphRemoveDependencies,
 * ::cuGraphNodeGetDependencies,
 * ::cuGraphNodeGetDependentNodes
 *)
function cuGraphGetEdges(hGraph : CUgraph; from : PCUgraphNode; &to : PCUgraphNode; numEdges : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a node's dependencies
 *
 * Returns a list of \p node's dependencies. \p dependencies may be NULL, in which case this
 * function will return the number of dependencies in \p numDependencies. Otherwise,
 * \p numDependencies entries will be filled in. If \p numDependencies is higher than the actual
 * number of dependencies, the remaining entries in \p dependencies will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p numDependencies.
 *
 * \param hNode           - Node to query
 * \param dependencies    - Pointer to return the dependencies
 * \param numDependencies - See description
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphNodeGetDependentNodes,
 * ::cuGraphGetNodes,
 * ::cuGraphGetRootNodes,
 * ::cuGraphGetEdges,
 * ::cuGraphAddDependencies,
 * ::cuGraphRemoveDependencies
 *)
function cuGraphNodeGetDependencies(hNode : CUgraphNode; dependencies : PCUgraphNode; numDependencies : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a node's dependent nodes
 *
 * Returns a list of \p node's dependent nodes. \p dependentNodes may be NULL, in which
 * case this function will return the number of dependent nodes in \p numDependentNodes.
 * Otherwise, \p numDependentNodes entries will be filled in. If \p numDependentNodes is
 * higher than the actual number of dependent nodes, the remaining entries in
 * \p dependentNodes will be set to NULL, and the number of nodes actually obtained will
 * be returned in \p numDependentNodes.
 *
 * \param hNode             - Node to query
 * \param dependentNodes    - Pointer to return the dependent nodes
 * \param numDependentNodes - See description
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphNodeGetDependencies,
 * ::cuGraphGetNodes,
 * ::cuGraphGetRootNodes,
 * ::cuGraphGetEdges,
 * ::cuGraphAddDependencies,
 * ::cuGraphRemoveDependencies
 *)
function cuGraphNodeGetDependentNodes(hNode : CUgraphNode; dependentNodes : PCUgraphNode; numDependentNodes : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Adds dependency edges to a graph
 *
 * The number of dependencies to be added is defined by \p numDependencies
 * Elements in \p from and \p to at corresponding indices define a dependency.
 * Each node in \p from and \p to must belong to \p hGraph.
 *
 * If \p numDependencies is 0, elements in \p from and \p to will be ignored.
 * Specifying an existing dependency will return an error.
 *
 * \param hGraph - Graph to which dependencies are added
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param numDependencies - Number of dependencies to be added
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphRemoveDependencies,
 * ::cuGraphGetEdges,
 * ::cuGraphNodeGetDependencies,
 * ::cuGraphNodeGetDependentNodes
 *)
function cuGraphAddDependencies(hGraph : CUgraph; const from : PCUgraphNode; const &to : PCUgraphNode; numDependencies : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Removes dependency edges from a graph
 *
 * The number of \p dependencies to be removed is defined by \p numDependencies.
 * Elements in \p from and \p to at corresponding indices define a dependency.
 * Each node in \p from and \p to must belong to \p hGraph.
 *
 * If \p numDependencies is 0, elements in \p from and \p to will be ignored.
 * Specifying a non-existing dependency will return an error.
 *
 * Dependencies cannot be removed from graphs which contain allocation or free nodes.
 * Any attempt to do so will return an error.
 *
 * \param hGraph - Graph from which to remove dependencies
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param numDependencies - Number of dependencies to be removed
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddDependencies,
 * ::cuGraphGetEdges,
 * ::cuGraphNodeGetDependencies,
 * ::cuGraphNodeGetDependentNodes
 *)
function cuGraphRemoveDependencies(hGraph : CUgraph; const from : PCUgraphNode; const &to : PCUgraphNode; numDependencies : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Remove a node from the graph
 *
 * Removes \p hNode from its graph. This operation also severs any dependencies of other nodes
 * on \p hNode and vice versa.
 *
 * Nodes which belong to a graph which contains allocation or free nodes cannot be destroyed.
 * Any attempt to do so will return an error.
 *
 * \param hNode  - Node to remove
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphAddEmptyNode,
 * ::cuGraphAddKernelNode,
 * ::cuGraphAddHostNode,
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphAddMemsetNode
 *)
function cuGraphDestroyNode(hNode : CUgraphNode):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p hGraph as an executable graph. The graph is validated for any
 * structural constraints or intra-node constraints which were not previously
 * validated. If instantiation is successful, a handle to the instantiated graph
 * is returned in \p phGraphExec.
 *
 * The \p flags parameter controls the behavior of instantiation and subsequent
 * graph launches.  Valid flags are:
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH, which configures a
 * graph containing memory allocation nodes to automatically free any
 * unfreed memory allocations before the graph is relaunched.
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH, which configures the graph for launch
 * from the device. If this flag is passed, the executable graph handle returned can be
 * used to launch the graph from both the host and device. This flag can only be used
 * on platforms which support unified addressing. This flag cannot be used in
 * conjunction with ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, which causes the graph
 * to use the priorities from the per-node attributes rather than the priority
 * of the launch stream during execution. Note that priorities are only available
 * on kernel nodes, and are copied from stream priority during stream capture.
 *
 * If \p hGraph contains any allocation or free nodes, there can be at most one
 * executable graph in existence for that graph at a time. An attempt to instantiate
 * a second executable graph before destroying the first with ::cuGraphExecDestroy
 * will result in an error.
 *
 * If \p hGraph contains kernels which call device-side cudaGraphLaunch() from multiple
 * contexts, this will result in an error.
 *
 * Graphs instantiated for launch on the device have additional restrictions which do not
 * apply to host graphs:
 *
 * - The graph's nodes must reside on a single context.
 * - The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
 *   Operation-specific restrictions are outlined below.
 * - Kernel nodes:
 *   - Use of CUDA Dynamic Parallelism is not permitted.
 *   - Cooperative launches are permitted as long as MPS is not in use.
 * - Memcpy nodes:
 *   - Only copies involving device memory and/or pinned device-mapped host memory are permitted.
 *   - Copies involving CUDA arrays are not permitted.
 *   - Both operands must be accessible from the current context, and the current context must
 *     match the context of other nodes in the graph.
 *
 * \param phGraphExec - Returns instantiated graph
 * \param hGraph      - Graph to instantiate
 * \param flags       - Flags to control instantiation.  See ::CUgraphInstantiate_flags.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphInstantiate,
 * ::cuGraphCreate,
 * ::cuGraphUpload,
 * ::cuGraphLaunch,
 * ::cuGraphExecDestroy
 *)
function cuGraphInstantiate(phGraphExec : PCUgraphExec; hGraph : CUgraph; flags : uint64):CUresult;  WINAPI; external libcuda;

(**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p hGraph as an executable graph according to the \p instantiateParams structure.
 * The graph is validated for any structural constraints or intra-node constraints
 * which were not previously validated. If instantiation is successful, a handle to
 * the instantiated graph is returned in \p phGraphExec.
 *
 * \p instantiateParams controls the behavior of instantiation and subsequent
 * graph launches, as well as returning more detailed information in the event of an error.
 * ::CUDA_GRAPH_INSTANTIATE_PARAMS is defined as:
 *
 * \code
    typedef struct {
        cuuint64_t flags;
        CUstream hUploadStream;
        CUgraphNode hErrNode_out;
        CUgraphInstantiateResult result_out;
    } CUDA_GRAPH_INSTANTIATE_PARAMS;
 * \endcode
 *
 * The \p flags field controls the behavior of instantiation and subsequent
 * graph launches. Valid flags are:
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH, which configures a
 * graph containing memory allocation nodes to automatically free any
 * unfreed memory allocations before the graph is relaunched.
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD, which will perform an upload of the graph
 * into \p hUploadStream once the graph has been instantiated.
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH, which configures the graph for launch
 * from the device. If this flag is passed, the executable graph handle returned can be
 * used to launch the graph from both the host and device. This flag can only be used
 * on platforms which support unified addressing. This flag cannot be used in
 * conjunction with ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.
 *
 * - ::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, which causes the graph
 * to use the priorities from the per-node attributes rather than the priority
 * of the launch stream during execution. Note that priorities are only available
 * on kernel nodes, and are copied from stream priority during stream capture.
 *
 * If \p hGraph contains any allocation or free nodes, there can be at most one
 * executable graph in existence for that graph at a time. An attempt to instantiate a
 * second executable graph before destroying the first with ::cuGraphExecDestroy will
 * result in an error.
 *
 * If \p hGraph contains kernels which call device-side cudaGraphLaunch() from multiple
 * contexts, this will result in an error.
 *
 * Graphs instantiated for launch on the device have additional restrictions which do not
 * apply to host graphs:
 *
 * - The graph's nodes must reside on a single context.
 * - The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
 *   Operation-specific restrictions are outlined below.
 * - Kernel nodes:
 *   - Use of CUDA Dynamic Parallelism is not permitted.
 *   - Cooperative launches are permitted as long as MPS is not in use.
 * - Memcpy nodes:
 *   - Only copies involving device memory and/or pinned device-mapped host memory are permitted.
 *   - Copies involving CUDA arrays are not permitted.
 *   - Both operands must be accessible from the current context, and the current context must
 *     match the context of other nodes in the graph.
 *
 * In the event of an error, the \p result_out and \p hErrNode_out fields will contain more
 * information about the nature of the error. Possible error reporting includes:
 *
 * - ::CUDA_GRAPH_INSTANTIATE_ERROR, if passed an invalid value or if an unexpected error occurred
 *   which is described by the return value of the function. \p hErrNode_out will be set to NULL.
 * - ::CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE, if the graph structure is invalid. \p hErrNode_out
 *   will be set to one of the offending nodes.
 * - ::CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED, if the graph is instantiated for device
 *   launch but contains a node of an unsupported node type, or a node which performs unsupported
 *   operations, such as use of CUDA dynamic parallelism within a kernel node. \p hErrNode_out will
 *   be set to this node.
 * - ::CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED, if the graph is instantiated for device
 *   launch but a node’s context differs from that of another node. This error can also be returned
 *   if a graph is not instantiated for device launch and it contains kernels which call device-side
 *   cudaGraphLaunch() from multiple contexts. \p hErrNode_out will be set to this node.
 *
 * If instantiation is successful, \p result_out will be set to ::CUDA_GRAPH_INSTANTIATE_SUCCESS,
 * and \p hErrNode_out will be set to NULL.
 *
 * \param phGraphExec       - Returns instantiated graph
 * \param hGraph            - Graph to instantiate
 * \param instantiateParams - Instantiation parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphCreate,
 * ::cuGraphInstantiate,
 * ::cuGraphExecDestroy
 *)
function cuGraphInstantiateWithParams(phGraphExec : PCUgraphExec; hGraph : CUgraph; instantiateParams : PCUDA_GRAPH_INSTANTIATE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query the instantiation flags of an executable graph
 *
 * Returns the flags that were passed to instantiation for the given executable graph.
 * ::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD will not be returned by this API as it does
 * not affect the resulting executable graph.
 *
 * \param hGraphExec - The executable graph to query
 * \param flags      - Returns the instantiation flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphInstantiate,
 * ::cuGraphInstantiateWithParams
 *)
function cuGraphExecGetFlags(hGraphExec : CUgraphExec; flags : Pcuuint64_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for a kernel node in the given graphExec
 *
 * Sets the parameters of a kernel node in an executable graph \p hGraphExec. 
 * The node is identified by the corresponding node \p hNode in the 
 * non-executable graph, from which the executable graph was instantiated. 
 *
 * \p hNode must not have been removed from the original graph. All \p nodeParams 
 * fields may change, but the following restrictions apply to \p func updates: 
 *
 *   - The owning context of the function cannot change.
 *   - A node whose function originally did not use CUDA dynamic parallelism cannot be updated
 *     to a function which uses CDP
 *   - If \p hGraphExec was not instantiated for device launch, a node whose function originally
 *     did not use device-side cudaGraphLaunch() cannot be updated to a function which uses
 *     device-side cudaGraphLaunch() unless the node resides on the same context as nodes which
 *     contained such calls at instantiate-time. If no such calls were present at instantiation,
 *     these updates cannot be performed at all.
 *
 * The modifications only affect future launches of \p hGraphExec. Already 
 * enqueued or running launches of \p hGraphExec are not affected by this call. 
 * \p hNode is also not modified by this call.
 * 
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param hNode       - kernel node from the graph from which graphExec was instantiated
 * \param nodeParams  - Updated Parameters to set
 * 
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddKernelNode,
 * ::cuGraphKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecKernelNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const nodeParams : PCUDA_KERNEL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for a memcpy node in the given graphExec.
 *
 * Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
 * contained \p copyParams at instantiation.  hNode must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.
 *
 * The source and destination memory in \p copyParams must be allocated from the same 
 * contexts as the original source and destination memory.  Both the instantiation-time 
 * memory operands and the memory operands in \p copyParams must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  hNode is also 
 * not modified by this call.
 *
 * Returns CUDA_ERROR_INVALID_VALUE if the memory operands' mappings changed or
 * either the original or new memory operands are multidimensional.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Memcpy node from the graph which was used to instantiate graphExec
 * \param copyParams - The updated parameters to set
 * \param ctx        - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddMemcpyNode,
 * ::cuGraphMemcpyNodeSetParams,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecMemcpyNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const copyParams : PCUDA_MEMCPY3D; ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for a memset node in the given graphExec.
 *
 * Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
 * contained \p memsetParams at instantiation.  hNode must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.
 *
 * The destination memory in \p memsetParams must be allocated from the same 
 * contexts as the original destination memory.  Both the instantiation-time 
 * memory operand and the memory operand in \p memsetParams must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  hNode is also 
 * not modified by this call.
 *
 * Returns CUDA_ERROR_INVALID_VALUE if the memory operand's mappings changed or
 * either the original or new memory operand are multidimensional.
 *
 * \param hGraphExec   - The executable graph in which to set the specified node
 * \param hNode        - Memset node from the graph which was used to instantiate graphExec
 * \param memsetParams - The updated parameters to set
 * \param ctx          - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddMemsetNode,
 * ::cuGraphMemsetNodeSetParams,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecMemsetNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const memsetParams : PCUDA_MEMSET_NODE_PARAMS; ctx : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for a host node in the given graphExec.
 *
 * Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
 * contained \p nodeParams at instantiation.  hNode must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  hNode is also 
 * not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Host node from the graph which was used to instantiate graphExec
 * \param nodeParams - The updated parameters to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddHostNode,
 * ::cuGraphHostNodeSetParams,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecHostNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const nodeParams : PCUDA_HOST_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Updates node parameters in the child graph node in the given graphExec.
 *
 * Updates the work represented by \p hNode in \p hGraphExec as though the nodes contained
 * in \p hNode's graph had the parameters contained in \p childGraph's nodes at instantiation.
 * \p hNode must remain in the graph which was used to instantiate \p hGraphExec.
 * Changed edges to and from \p hNode are ignored.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p hNode is also 
 * not modified by this call.
 *
 * The topology of \p childGraph, as well as the node insertion order,  must match that
 * of the graph contained in \p hNode.  See ::cuGraphExecUpdate() for a list of restrictions
 * on what can be updated in an instantiated graph.  The update is recursive, so child graph
 * nodes contained within the top level child graph will also be updated.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Host node from the graph which was used to instantiate graphExec
 * \param childGraph - The graph supplying the updated parameters
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddChildGraphNode,
 * ::cuGraphChildGraphNodeGetGraph,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecChildGraphNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; childGraph : CUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the event for an event record node in the given graphExec
 *
 * Sets the event of an event record node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - event record node from the graph from which graphExec was instantiated
 * \param event      - Updated event to use
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventRecordNode,
 * ::cuGraphEventRecordNodeGetEvent,
 * ::cuGraphEventWaitNodeSetEvent,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecEventRecordNodeSetEvent(hGraphExec : CUgraphExec; hNode : CUgraphNode; event : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the event for an event wait node in the given graphExec
 *
 * Sets the event of an event wait node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - event wait node from the graph from which graphExec was instantiated
 * \param event      - Updated event to use
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddEventWaitNode,
 * ::cuGraphEventWaitNodeGetEvent,
 * ::cuGraphEventRecordNodeSetEvent,
 * ::cuEventRecordWithFlags,
 * ::cuStreamWaitEvent,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecEventWaitNodeSetEvent(hGraphExec : CUgraphExec; hNode : CUgraphNode; event : CUevent):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for an external semaphore signal node in the given graphExec
 *
 * Sets the parameters of an external semaphore signal node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * Changing \p nodeParams->numExtSems is not supported.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - semaphore signal node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddExternalSemaphoresSignalNode,
 * ::cuImportExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const nodeParams : PCUDA_EXT_SEM_SIGNAL_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Sets the parameters for an external semaphore wait node in the given graphExec
 *
 * Sets the parameters of an external semaphore wait node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * Changing \p nodeParams->numExtSems is not supported.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - semaphore wait node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddExternalSemaphoresWaitNode,
 * ::cuImportExternalSemaphore,
 * ::cuSignalExternalSemaphoresAsync,
 * ::cuWaitExternalSemaphoresAsync,
 * ::cuGraphExecKernelNodeSetParams,
 * ::cuGraphExecMemcpyNodeSetParams,
 * ::cuGraphExecMemsetNodeSetParams,
 * ::cuGraphExecHostNodeSetParams,
 * ::cuGraphExecChildGraphNodeSetParams,
 * ::cuGraphExecEventRecordNodeSetEvent,
 * ::cuGraphExecEventWaitNodeSetEvent,
 * ::cuGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 *)
function cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec : CUgraphExec; hNode : CUgraphNode; const nodeParams : PCUDA_EXT_SEM_WAIT_NODE_PARAMS):CUresult;  WINAPI; external libcuda;

(**
 * \brief Enables or disables the specified node in the given graphExec
 *
 * Sets \p hNode to be either enabled or disabled. Disabled nodes are functionally equivalent 
 * to empty nodes until they are reenabled. Existing node parameters are not affected by 
 * disabling/enabling the node.
 *  
 * The node is identified by the corresponding node \p hNode in the non-executable 
 * graph, from which the executable graph was instantiated.   
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \note Currently only kernel, memset and memcpy nodes are supported. 
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Node from the graph from which graphExec was instantiated
 * \param isEnabled  - Node is enabled if != 0, otherwise the node is disabled
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphNodeGetEnabled,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 * ::cuGraphLaunch
 *)
function cuGraphNodeSetEnabled(hGraphExec : CUgraphExec; hNode : CUgraphNode; isEnabled : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Query whether a node in the given graphExec is enabled
 *
 * Sets isEnabled to 1 if \p hNode is enabled, or 0 if \p hNode is disabled.
 *
 * The node is identified by the corresponding node \p hNode in the non-executable 
 * graph, from which the executable graph was instantiated.   
 *
 * \p hNode must not have been removed from the original graph.
 *
 * \note Currently only kernel, memset and memcpy nodes are supported. 
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Node from the graph from which graphExec was instantiated
 * \param isEnabled  - Location to return the enabled status of the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphNodeSetEnabled,
 * ::cuGraphExecUpdate,
 * ::cuGraphInstantiate
 * ::cuGraphLaunch
 *)
function cuGraphNodeGetEnabled(hGraphExec : CUgraphExec; hNode : CUgraphNode; isEnabled : Plongword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Uploads an executable graph in a stream
 *
 * Uploads \p hGraphExec to the device in \p hStream without executing it. Uploads of
 * the same \p hGraphExec will be serialized. Each upload is ordered behind both any
 * previous work in \p hStream and any previous launches of \p hGraphExec.
 * Uses memory cached by \p stream to back the allocations owned by \p hGraphExec.
 *
 * \param hGraphExec - Executable graph to upload
 * \param hStream    - Stream in which to upload the graph
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphInstantiate,
 * ::cuGraphLaunch,
 * ::cuGraphExecDestroy
 *)
function cuGraphUpload(hGraphExec : CUgraphExec; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Launches an executable graph in a stream
 *
 * Executes \p hGraphExec in \p hStream. Only one instance of \p hGraphExec may be executing
 * at a time. Each launch is ordered behind both any previous work in \p hStream
 * and any previous launches of \p hGraphExec. To execute a graph concurrently, it must be
 * instantiated multiple times into multiple executable graphs.
 *
 * If any allocations created by \p hGraphExec remain unfreed (from a previous launch) and
 * \p hGraphExec was not instantiated with ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
 * the launch will fail with ::CUDA_ERROR_INVALID_VALUE.
 *
 * \param hGraphExec - Executable graph to launch
 * \param hStream    - Stream in which to launch the graph
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphInstantiate,
 * ::cuGraphUpload,
 * ::cuGraphExecDestroy
 *)
function cuGraphLaunch(hGraphExec : CUgraphExec; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys an executable graph
 *
 * Destroys the executable graph specified by \p hGraphExec, as well
 * as all of its executable nodes. If the executable graph is
 * in-flight, it will not be terminated, but rather freed
 * asynchronously on completion.
 *
 * \param hGraphExec - Executable graph to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphInstantiate,
 * ::cuGraphUpload,
 * ::cuGraphLaunch
 *)
function cuGraphExecDestroy(hGraphExec : CUgraphExec):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys a graph
 *
 * Destroys the graph specified by \p hGraph, as well as all of its nodes.
 *
 * \param hGraph - Graph to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphCreate
 *)
function cuGraphDestroy(hGraph : CUgraph):CUresult;  WINAPI; external libcuda;

(**
 * \brief Check whether an executable graph can be updated with a graph and perform the update if possible
 *
 * Updates the node parameters in the instantiated graph specified by \p hGraphExec with the
 * node parameters in a topologically identical graph specified by \p hGraph.
 *
 * Limitations:
 *
 * - Kernel nodes:
 *   - The owning context of the function cannot change.
 *   - A node whose function originally did not use CUDA dynamic parallelism cannot be updated
 *     to a function which uses CDP.
 *   - A cooperative node cannot be updated to a non-cooperative node, and vice-versa.
 *   - If the graph was instantiated with CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, the
 *     priority attribute cannot change. Equality is checked on the originally requested
 *     priority values, before they are clamped to the device's supported range.
 *   - If \p hGraphExec was not instantiated for device launch, a node whose function originally
 *     did not use device-side cudaGraphLaunch() cannot be updated to a function which uses
 *     device-side cudaGraphLaunch() unless the node resides on the same context as nodes which
 *     contained such calls at instantiate-time. If no such calls were present at instantiation,
 *     these updates cannot be performed at all.
 * - Memset and memcpy nodes:
 *   - The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.
 *   - The source/destination memory must be allocated from the same contexts as the original
 *     source/destination memory.
 *   - Only 1D memsets can be changed.
 * - Additional memcpy node restrictions:
 *   - Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE,
 *     CU_MEMORYTYPE_ARRAY, etc.) is not supported.
 * - External semaphore wait nodes and record nodes:
 *   - Changing the number of semaphores is not supported.
 *
 * Note:  The API may add further restrictions in future releases.  The return code should always be checked.
 *
 * cuGraphExecUpdate sets the result member of \p resultInfo to CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED
 * under the following conditions:
 * - The count of nodes directly in \p hGraphExec and \p hGraph differ, in which case resultInfo->errorNode
 *   is set to NULL.
 * - \p hGraph has more exit nodes than \p hGraph, in which case resultInfo->errorNode is set to one of
 *   the exit nodes in hGraph. 
 * - A node in \p hGraph has a different number of dependencies than the node from \p hGraphExec it is paired with,
 *   in which case resultInfo->errorNode is set to the node from \p hGraph.
 * - A node in \p hGraph has a dependency that does not match with the corresponding dependency of the paired node
 *   from \p hGraphExec. resultInfo->errorNode will be set to the node from \p hGraph. resultInfo->errorFromNode
 *   will be set to the mismatched dependency. The dependencies are paired based on edge order and a dependency
 *   does not match when the nodes are already paired based on other edges examined in the graph.
 *
 * cuGraphExecUpdate sets the result member of \p resultInfo to: 
 * - CU_GRAPH_EXEC_UPDATE_ERROR if passed an invalid value.
 * - CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED if the graph topology changed
 * - CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED if the type of a node changed, in which case
 *   \p hErrorNode_out is set to the node from \p hGraph.
 * - CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE if the function changed in an unsupported
 *   way(see note above), in which case \p hErrorNode_out is set to the node from \p hGraph
 * - CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED if any parameters to a node changed in a way 
 *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph.
 * - CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED if any attributes of a node changed in a way
 *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph.
 * - CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED if something about a node is unsupported, like 
 *   the node's type or configuration, in which case \p hErrorNode_out is set to the node from \p hGraph
 *
 * If the update fails for a reason not listed above, the result member of \p resultInfo will be set
 * to CU_GRAPH_EXEC_UPDATE_ERROR. If the update succeeds, the result member will be set to CU_GRAPH_EXEC_UPDATE_SUCCESS.
 *
 * cuGraphExecUpdate returns CUDA_SUCCESS when the updated was performed successfully.  It returns
 * CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE if the graph update was not performed because it included 
 * changes which violated constraints specific to instantiated graph update.
 *
 * \param hGraphExec The instantiated graph to be updated
 * \param hGraph The graph containing the updated parameters
 * \param resultInfo the error info structure 
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphInstantiate
 *)
function cuGraphExecUpdate(hGraphExec : CUgraphExec; hGraph : CUgraph; resultInfo : PCUgraphExecUpdateResultInfo):CUresult;  WINAPI; external libcuda;

(**
 * \brief Copies attributes from source node to destination node.
 *
 * Copies attributes from source node \p src to destination node \p dst.
 * Both node must have the same context.
 *
 * \param[out] dst Destination node
 * \param[in] src Source node
 * For list of attributes see ::CUkernelNodeAttrID
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuGraphKernelNodeCopyAttributes(dst : CUgraphNode; src : CUgraphNode):CUresult;  WINAPI; external libcuda;

(**
 * \brief Queries node attribute.
 * 
 * Queries attribute \p attr from node \p hNode and stores it in corresponding
 * member of \p value_out.
 *
 * \param[in] hNode
 * \param[in] attr
 * \param[out] value_out 
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *  
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuGraphKernelNodeGetAttribute(hNode : CUgraphNode; attr : CUkernelNodeAttrID;
                                      value_out : PCUkernelNodeAttrValue):CUresult;  WINAPI; external libcuda;
 
(**
 * \brief Sets node attribute.
 * 
 * Sets attribute \p attr on node \p hNode from corresponding attribute of
 * \p value.
 *
 * \param[out] hNode
 * \param[in] attr
 * \param[out] value
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa
 * ::CUaccessPolicyWindow
 *)
function cuGraphKernelNodeSetAttribute(hNode : CUgraphNode; attr : CUkernelNodeAttrID;
                                      const value : PCUkernelNodeAttrValue):CUresult;  WINAPI; external libcuda;

(**
 * \brief Write a DOT file describing graph structure
 *
 * Using the provided \p hGraph, write to \p path a DOT formatted description of the graph.
 * By default this includes the graph topology, node types, node id, kernel names and memcpy direction.
 * \p flags can be specified to write more detailed information about each node type such as
 * parameter values, kernel attributes, node and function handles.
 *
 * \param hGraph - The graph to create a DOT file from
 * \param path   - The path to write the DOT file to
 * \param flags  - Flags from CUgraphDebugDot_flags for specifying which additional node information to write
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OPERATING_SYSTEM
 *)
function cuGraphDebugDotPrint(hGraph : CUgraph; const path : PAnsiChar; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Create a user object
 *
 * Create a user object with the specified destructor callback and initial reference count. The
 * initial references are owned by the caller.
 *
 * Destructor callbacks cannot make CUDA API calls and should avoid blocking behavior, as they
 * are executed by a shared internal thread. Another thread may be signaled to perform such
 * actions, if it does not block forward progress of tasks scheduled through CUDA.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param object_out      - Location to return the user object handle
 * \param ptr             - The pointer to pass to the destroy function
 * \param destroy         - Callback to free the user object when it is no longer in use
 * \param initialRefcount - The initial refcount to create the object with, typically 1. The
 *                          initial references are owned by the calling thread.
 * \param flags           - Currently it is required to pass ::CU_USER_OBJECT_NO_DESTRUCTOR_SYNC,
 *                          which is the only defined flag. This indicates that the destroy
 *                          callback cannot be waited on by any CUDA API. Users requiring
 *                          synchronization of the callback should signal its completion
 *                          manually.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuUserObjectRetain,
 * ::cuUserObjectRelease,
 * ::cuGraphRetainUserObject,
 * ::cuGraphReleaseUserObject,
 * ::cuGraphCreate
 *)
function cuUserObjectCreate(object_out : PCUuserObject; ptr : Pointer; destroy : CUhostFn;
                                    initialRefcount : longword; flags: longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Retain a reference to a user object
 *
 * Retains new references to a user object. The new references are owned by the caller.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param object - The object to retain
 * \param count  - The number of references to retain, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuUserObjectCreate,
 * ::cuUserObjectRelease,
 * ::cuGraphRetainUserObject,
 * ::cuGraphReleaseUserObject,
 * ::cuGraphCreate
 *)
function cuUserObjectRetain(&object : CUuserObject; count : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Release a reference to a user object
 *
 * Releases user object references owned by the caller. The object's destructor is invoked if
 * the reference count reaches zero.
 *
 * It is undefined behavior to release references not owned by the caller, or to use a user
 * object handle after all references are released.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param object - The object to release
 * \param count  - The number of references to release, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuUserObjectCreate,
 * ::cuUserObjectRetain,
 * ::cuGraphRetainUserObject,
 * ::cuGraphReleaseUserObject,
 * ::cuGraphCreate
 *)
function cuUserObjectRelease(&object : CUuserObject; count : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Retain a reference to a user object from a graph
 *
 * Creates or moves user object references that will be owned by a CUDA graph.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param graph  - The graph to associate the reference with
 * \param object - The user object to retain a reference for
 * \param count  - The number of references to add to the graph, typically 1. Must be
 *                 nonzero and not larger than INT_MAX.
 * \param flags  - The optional flag ::CU_GRAPH_USER_OBJECT_MOVE transfers references
 *                 from the calling thread, rather than create new references. Pass 0
 *                 to create new references.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuUserObjectCreate,
 * ::cuUserObjectRetain,
 * ::cuUserObjectRelease,
 * ::cuGraphReleaseUserObject,
 * ::cuGraphCreate
 *)
function cuGraphRetainUserObject(graph : CUgraph; &object : CUuserObject; count : longword; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Release a user object reference from a graph
 *
 * Releases user object references owned by a graph.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param graph  - The graph that will release the reference
 * \param object - The user object to release a reference for
 * \param count  - The number of references to release, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuUserObjectCreate,
 * ::cuUserObjectRetain,
 * ::cuUserObjectRelease,
 * ::cuGraphRetainUserObject,
 * ::cuGraphCreate
 *)
function cuGraphReleaseUserObject(graph : CUgraph; &object : CUuserObject; count : longword):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_GRAPH *)

(**
 * \defgroup CUDA_OCCUPANCY Occupancy
 *
 * ___MANBRIEF___ occupancy calculation functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the occupancy calculation functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns occupancy of a function
 *
 * Returns in \p *numBlocks the number of the maximum active blocks per
 * streaming multiprocessor.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel for which occupancy is calculated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *)
function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks : Plongint; func : CUfunction; blockSize : longint; dynamicSMemSize : size_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns occupancy of a function
 *
 * Returns in \p *numBlocks the number of the maximum active blocks per
 * streaming multiprocessor.
 *
 * The \p Flags parameter controls how special cases are handled. The
 * valid flags are:
 *
 * - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
 *   ::cuOccupancyMaxActiveBlocksPerMultiprocessor;
 *
 * - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
 *   default behavior on platform where global caching affects
 *   occupancy. On such platforms, if caching is enabled, but
 *   per-block SM resource usage would result in zero occupancy, the
 *   occupancy calculator will calculate the occupancy as if caching
 *   is disabled. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE makes
 *   the occupancy calculator to return 0 in such cases. More information
 *   can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel for which occupancy is calculated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param flags           - Requested behavior for the occupancy calculator
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 *)
function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks : Plongint; func : CUfunction; blockSize : longint; dynamicSMemSize : size_t; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Suggest a launch configuration with reasonable occupancy
 *
 * Returns in \p *blockSize a reasonable block size that can achieve
 * the maximum occupancy (or, the maximum number of active warps with
 * the fewest blocks per multiprocessor), and in \p *minGridSize the
 * minimum grid size to achieve the maximum occupancy.
 *
 * If \p blockSizeLimit is 0, the configurator will use the maximum
 * block size permitted by the device / function instead.
 *
 * If per-block dynamic shared memory allocation is not needed, the
 * user should leave both \p blockSizeToDynamicSMemSize and \p
 * dynamicSMemSize as 0.
 *
 * If per-block dynamic shared memory allocation is needed, then if
 * the dynamic shared memory size is constant regardless of block
 * size, the size should be passed through \p dynamicSMemSize, and \p
 * blockSizeToDynamicSMemSize should be NULL.
 *
 * Otherwise, if the per-block dynamic shared memory size varies with
 * different block sizes, the user needs to provide a unary function
 * through \p blockSizeToDynamicSMemSize that computes the dynamic
 * shared memory needed by \p func for any given block size. \p
 * dynamicSMemSize is ignored. An example signature is:
 *
 * \code
 *    // Take block size, returns dynamic shared memory needed
 *    size_t blockToSmem(longint blockSize);
 * \endcode
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
 * \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
 * \param func        - Kernel for which launch configuration is calculated
 * \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
 * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
 * \param blockSizeLimit  - The maximum block size \p func is designed to handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cudaOccupancyMaxPotentialBlockSize
 *)
function cuOccupancyMaxPotentialBlockSize(minGridSize : Plongint; blockSize : Plongint; func : CUfunction; blockSizeToDynamicSMemSize : CUoccupancyB2DSize; dynamicSMemSize : size_t; blockSizeLimit : longint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Suggest a launch configuration with reasonable occupancy
 *
 * An extended version of ::cuOccupancyMaxPotentialBlockSize. In
 * addition to arguments passed to ::cuOccupancyMaxPotentialBlockSize,
 * ::cuOccupancyMaxPotentialBlockSizeWithFlags also takes a \p Flags
 * parameter.
 *
 * The \p Flags parameter controls how special cases are handled. The
 * valid flags are:
 *
 * - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
 *   ::cuOccupancyMaxPotentialBlockSize;
 *
 * - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
 *   default behavior on platform where global caching affects
 *   occupancy. On such platforms, the launch configurations that
 *   produces maximal occupancy might not support global
 *   caching. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
 *   guarantees that the the produced launch configuration is global
 *   caching compatible at a potential cost of occupancy. More information
 *   can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
 * \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
 * \param func        - Kernel for which launch configuration is calculated
 * \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
 * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
 * \param blockSizeLimit  - The maximum block size \p func is designed to handle
 * \param flags       - Options
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 *)
function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize : Plongint; blockSize : Plongint; func : CUfunction; blockSizeToDynamicSMemSize : CUoccupancyB2DSize; dynamicSMemSize : size_t; blockSizeLimit : longint; flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns dynamic shared memory available per block when launching \p numBlocks blocks on SM 
 *
 * Returns in \p *dynamicSmemSize the maximum size of dynamic shared memory to allow \p numBlocks blocks per SM. 
 *
 * \param dynamicSmemSize - Returned maximum dynamic shared memory 
 * \param func            - Kernel function for which occupancy is calculated
 * \param numBlocks       - Number of blocks to fit on SM 
 * \param blockSize       - Size of the blocks
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *)
function cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize : Psize_t; func : CUfunction; numBlocks : longint; blockSize : longint):CUresult;  WINAPI; external libcuda;

(**
 * \brief Given the kernel function (\p func) and launch configuration
 * (\p config), return the maximum cluster size in \p *clusterSize.
 *
 * The cluster dimensions in \p config are ignored. If func has a required
 * cluster size set (see ::cudaFuncGetAttributes / ::cuFuncGetAttribute),\p
 * *clusterSize will reflect the required cluster size.
 *
 * By default this function will always return a value that's portable on
 * future hardware. A higher value may be returned if the kernel function
 * allows non-portable cluster sizes.
 *
 * This function will respect the compile time launch bounds.
 *
 * \param clusterSize - Returned maximum cluster size that can be launched
 *                      for the given kernel function and launch configuration
 * \param func        - Kernel function for which maximum cluster
 *                      size is calculated
 * \param config      - Launch configuration for the given kernel function
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cudaFuncGetAttributes,
 * ::cuFuncGetAttribute
 *)
function cuOccupancyMaxPotentialClusterSize(clusterSize : Plongint; func : CUfunction; const config : PCUlaunchConfig):CUresult;  WINAPI; external libcuda;

(**
 * \brief Given the kernel function (\p func) and launch configuration
 * (\p config), return the maximum number of clusters that could co-exist
 * on the target device in \p *numClusters.
 *
 * If the function has required cluster size already set (see
 * ::cudaFuncGetAttributes / ::cuFuncGetAttribute), the cluster size
 * from config must either be unspecified or match the required size.
 * Without required sizes, the cluster size must be specified in config,
 * else the function will return an error.
 *
 * Note that various attributes of the kernel function may affect occupancy
 * calculation. Runtime environment may affect how the hardware schedules
 * the clusters, so the calculated occupancy is not guaranteed to be achievable.
 *
 * \param numClusters - Returned maximum number of clusters that
 *                      could co-exist on the target device
 * \param func        - Kernel function for which maximum number
 *                      of clusters are calculated
 * \param config      - Launch configuration for the given kernel function
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_CLUSTER_SIZE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cudaFuncGetAttributes,
 * ::cuFuncGetAttribute
 *)
function cuOccupancyMaxActiveClusters(numClusters : Plongint; func : CUfunction; const config : PCUlaunchConfig):CUresult;  WINAPI; external libcuda;
(** @} *) (* END CUDA_OCCUPANCY *)

(**
 * \defgroup CUDA_TEXREF_DEPRECATED Texture Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated texture reference management functions of the
 * low-level CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated texture reference management
 * functions of the low-level CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Binds an array as a texture reference
 *
 * \deprecated
 *
 * Binds the CUDA array \p hArray to the texture reference \p hTexRef. Any
 * previous address or CUDA array state associated with the texture reference
 * is superseded by this function. \p Flags must be set to
 * ::CU_TRSA_OVERRIDE_FORMAT. Any CUDA array previously bound to \p hTexRef is
 * unbound.
 *
 * \param hTexRef - Texture reference to bind
 * \param hArray  - Array to bind
 * \param Flags   - Options (must be ::CU_TRSA_OVERRIDE_FORMAT)
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetArray(hTexRef : CUtexref; hArray : CUarray; Flags : longword):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Binds a mipmapped array to a texture reference
 *
 * \deprecated
 *
 * Binds the CUDA mipmapped array \p hMipmappedArray to the texture reference \p hTexRef.
 * Any previous address or CUDA array state associated with the texture reference
 * is superseded by this function. \p Flags must be set to ::CU_TRSA_OVERRIDE_FORMAT.
 * Any CUDA array previously bound to \p hTexRef is unbound.
 *
 * \param hTexRef         - Texture reference to bind
 * \param hMipmappedArray - Mipmapped array to bind
 * \param Flags           - Options (must be ::CU_TRSA_OVERRIDE_FORMAT)
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetMipmappedArray(hTexRef : CUtexref; hMipmappedArray : CUmipmappedArray; Flags : longword):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Binds an address as a texture reference
 *
 * \deprecated
 *
 * Binds a linear address range to the texture reference \p hTexRef. Any
 * previous address or CUDA array state associated with the texture reference
 * is superseded by this function. Any memory previously bound to \p hTexRef
 * is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses, ::cuTexRefSetAddress() passes back a byte offset in
 * \p *ByteOffset that must be applied to texture fetches in order to read from
 * the desired memory. This offset must be divided by the texel size and
 * passed to kernels that read from the texture so they can be applied to the
 * ::tex1Dfetch() function.
 *
 * If the device memory pointer was returned from ::cuMemAlloc(), the offset
 * is guaranteed to be 0 and NULL may be passed as the \p ByteOffset parameter.
 *
 * The total number of elements (or texels) in the linear address range
 * cannot exceed ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH.
 * The number of elements is computed as (\p bytes / bytesPerElement),
 * where bytesPerElement is determined from the data format and number of
 * components set using ::cuTexRefSetFormat().
 *
 * \param ByteOffset - Returned byte offset
 * \param hTexRef    - Texture reference to bind
 * \param dptr       - Device pointer to bind
 * \param bytes      - Size of memory to bind in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetAddress(ByteOffset : Psize_t; hTexRef : CUtexref; dptr : CUdeviceptr; bytes : size_t):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Binds an address as a 2D texture reference
 *
 * \deprecated
 *
 * Binds a linear address range to the texture reference \p hTexRef. Any
 * previous address or CUDA array state associated with the texture reference
 * is superseded by this function. Any memory previously bound to \p hTexRef
 * is unbound.
 *
 * Using a ::tex2D() function inside a kernel requires a call to either
 * ::cuTexRefSetArray() to bind the corresponding texture reference to an
 * array, or ::cuTexRefSetAddress2D() to bind the texture reference to linear
 * memory.
 *
 * Function calls to ::cuTexRefSetFormat() cannot follow calls to
 * ::cuTexRefSetAddress2D() for the same texture reference.
 *
 * It is required that \p dptr be aligned to the appropriate hardware-specific
 * texture alignment. You can query this value using the device attribute
 * ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. If an unaligned \p dptr is
 * supplied, ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * \p Pitch has to be aligned to the hardware-specific texture pitch alignment.
 * This value can be queried using the device attribute
 * ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. If an unaligned \p Pitch is
 * supplied, ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * Width and Height, which are specified in elements (or texels), cannot exceed
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
 * \p Pitch, which is specified in bytes, cannot exceed
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
 *
 * \param hTexRef - Texture reference to bind
 * \param desc    - Descriptor of CUDA array
 * \param dptr    - Device pointer to bind
 * \param Pitch   - Line pitch in bytes
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetAddress2D(hTexRef : CUtexref; const desc : PCUDA_ARRAY_DESCRIPTOR; dptr : CUdeviceptr; Pitch : size_t):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the format for a texture reference
 *
 * \deprecated
 *
 * Specifies the format of the data to be read by the texture reference
 * \p hTexRef. \p fmt and \p NumPackedComponents are exactly analogous to the
 * ::Format and ::NumChannels members of the ::CUDA_ARRAY_DESCRIPTOR structure:
 * They specify the format of each component and the number of components per
 * array element.
 *
 * \param hTexRef             - Texture reference
 * \param fmt                 - Format to set
 * \param NumPackedComponents - Number of components per array element
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
 * ::cudaCreateChannelDesc
 *)
function cuTexRefSetFormat(hTexRef : CUtexref; fmt : CUarray_format; NumPackedComponents : longint):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the addressing mode for a texture reference
 *
 * \deprecated
 *
 * Specifies the addressing mode \p am for the given dimension \p dim of the
 * texture reference \p hTexRef. If \p dim is zero, the addressing mode is
 * applied to the first parameter of the functions used to fetch from the
 * texture; if \p dim is 1, the second, and so on. ::CUaddress_mode is defined
 * as:
 * \code
     CUaddress_mode_enum = (
      CU_TR_ADDRESS_MODE_WRAP = 0,
      CU_TR_ADDRESS_MODE_CLAMP = 1,
      CU_TR_ADDRESS_MODE_MIRROR = 2,
      CU_TR_ADDRESS_MODE_BORDER = 3
     );
  CUaddress_mode = CUaddress_mode_enum;
 * \endcode
 *
 * Note that this call has no effect if \p hTexRef is bound to linear memory.
 * Also, if the flag, ::CU_TRSF_NORMALIZED_COORDINATES, is not set, the only
 * supported address mode is ::CU_TR_ADDRESS_MODE_CLAMP.
 *
 * \param hTexRef - Texture reference
 * \param dim     - Dimension
 * \param am      - Addressing mode to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetAddressMode(hTexRef : CUtexref; dim : longint; am : CUaddress_mode):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the filtering mode for a texture reference
 *
 * \deprecated
 *
 * Specifies the filtering mode \p fm to be used when reading memory through
 * the texture reference \p hTexRef. ::CUfilter_mode_enum is defined as:
 *
 * \code
     CUfilter_mode_enum = (
      CU_TR_FILTER_MODE_POINT = 0,
      CU_TR_FILTER_MODE_LINEAR = 1
     );
  CUfilter_mode = CUfilter_mode_enum;
 * \endcode
 *
 * Note that this call has no effect if \p hTexRef is bound to linear memory.
 *
 * \param hTexRef - Texture reference
 * \param fm      - Filtering mode to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetFilterMode(hTexRef : CUtexref; fm : CUfilter_mode):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the mipmap filtering mode for a texture reference
 *
 * \deprecated
 *
 * Specifies the mipmap filtering mode \p fm to be used when reading memory through
 * the texture reference \p hTexRef. ::CUfilter_mode_enum is defined as:
 *
 * \code
     CUfilter_mode_enum = (
      CU_TR_FILTER_MODE_POINT = 0,
      CU_TR_FILTER_MODE_LINEAR = 1
     );
  CUfilter_mode = CUfilter_mode_enum;
 * \endcode
 *
 * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 *
 * \param hTexRef - Texture reference
 * \param fm      - Filtering mode to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetMipmapFilterMode(hTexRef : CUtexref; fm : CUfilter_mode):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the mipmap level bias for a texture reference
 *
 * \deprecated
 *
 * Specifies the mipmap level bias \p bias to be added to the specified mipmap level when
 * reading memory through the texture reference \p hTexRef.
 *
 * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 *
 * \param hTexRef - Texture reference
 * \param bias    - Mipmap level bias
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetMipmapLevelBias(hTexRef : CUtexref; bias : single):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the mipmap min/max mipmap level clamps for a texture reference
 *
 * \deprecated
 *
 * Specifies the min/max mipmap level clamps, \p minMipmapLevelClamp and \p maxMipmapLevelClamp
 * respectively, to be used when reading memory through the texture reference
 * \p hTexRef.
 *
 * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 *
 * \param hTexRef        - Texture reference
 * \param minMipmapLevelClamp - Mipmap min level clamp
 * \param maxMipmapLevelClamp - Mipmap max level clamp
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetMipmapLevelClamp(hTexRef : CUtexref; minMipmapLevelClamp : single; maxMipmapLevelClamp : single):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the maximum anisotropy for a texture reference
 *
 * \deprecated
 *
 * Specifies the maximum anisotropy \p maxAniso to be used when reading memory through
 * the texture reference \p hTexRef.
 *
 * Note that this call has no effect if \p hTexRef is bound to linear memory.
 *
 * \param hTexRef  - Texture reference
 * \param maxAniso - Maximum anisotropy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetMaxAnisotropy(hTexRef : CUtexref; maxAniso : longword):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the border color for a texture reference
 *
 * \deprecated
 *
 * Specifies the value of the RGBA color via the \p pBorderColor to the texture reference
 * \p hTexRef. The color value supports only float type and holds color components in
 * the following sequence:
 * pBorderColor[0] holds 'R' component
 * pBorderColor[1] holds 'G' component
 * pBorderColor[2] holds 'B' component
 * pBorderColor[3] holds 'A' component
 *
 * Note that the color values can be set only when the Address mode is set to
 * CU_TR_ADDRESS_MODE_BORDER using ::cuTexRefSetAddressMode.
 * Applications using integer border color values have to "reinterpret_cast" their values to float.
 *
 * \param hTexRef       - Texture reference
 * \param pBorderColor  - RGBA color
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddressMode,
 * ::cuTexRefGetAddressMode, ::cuTexRefGetBorderColor
 *)
function cuTexRefSetBorderColor(hTexRef : CUtexref; pBorderColor : Psingle):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Sets the flags for a texture reference
 *
 * \deprecated
 *
 * Specifies optional flags via \p Flags to specify the behavior of data
 * returned through the texture reference \p hTexRef. The valid flags are:
 *
 * - ::CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of
 *   having the texture promote integer data to floating point data in the
 *   range [0, 1]. Note that texture with 32-bit integer format
 *   would not be promoted, regardless of whether or not this
 *   flag is specified;
 * - ::CU_TRSF_NORMALIZED_COORDINATES, which suppresses the
 *   default behavior of having the texture coordinates range
 *   from [0, Dim) where Dim is the width or height of the CUDA
 *   array. Instead, the texture coordinates [0, 1.0) reference
 *   the entire breadth of the array dimension;
 * - ::CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION, which disables any trilinear
 *   filtering optimizations. Trilinear optimizations improve texture filtering
 *   performance by allowing bilinear filtering on textures in scenarios where
 *   it can closely approximate the expected results.
 *
 * \param hTexRef - Texture reference
 * \param Flags   - Optional flags to set
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefSetFlags(hTexRef : CUtexref; Flags : longword):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the address associated with a texture reference
 *
 * \deprecated
 *
 * Returns in \p *pdptr the base address bound to the texture reference
 * \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
 * is not bound to any device memory range.
 *
 * \param pdptr   - Returned device address
 * \param hTexRef - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetAddress(pdptr : PCUdeviceptr; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the array bound to a texture reference
 *
 * \deprecated
 *
 * Returns in \p *phArray the CUDA array bound to the texture reference
 * \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
 * is not bound to any CUDA array.
 *
 * \param phArray - Returned array
 * \param hTexRef - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetArray(phArray : PCUarray; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;  deprecated;

(**
 * \brief Gets the mipmapped array bound to a texture reference
 *
 * \deprecated
 *
 * Returns in \p *phMipmappedArray the CUDA mipmapped array bound to the texture
 * reference \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
 * is not bound to any CUDA mipmapped array.
 *
 * \param phMipmappedArray - Returned mipmapped array
 * \param hTexRef          - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetMipmappedArray(phMipmappedArray : PCUmipmappedArray; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Gets the addressing mode used by a texture reference
 *
 * \deprecated
 *
 * Returns in \p *pam the addressing mode corresponding to the
 * dimension \p dim of the texture reference \p hTexRef. Currently, the only
 * valid value for \p dim are 0 and 1.
 *
 * \param pam     - Returned addressing mode
 * \param hTexRef - Texture reference
 * \param dim     - Dimension
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetAddressMode(pam : PCUaddress_mode; hTexRef : CUtexref; dim : longint):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the filter-mode used by a texture reference
 *
 * \deprecated
 *
 * Returns in \p *pfm the filtering mode of the texture reference
 * \p hTexRef.
 *
 * \param pfm     - Returned filtering mode
 * \param hTexRef - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetFilterMode(pfm : PCUfilter_mode; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the format used by a texture reference
 *
 * \deprecated
 *
 * Returns in \p *pFormat and \p *pNumChannels the format and number
 * of components of the CUDA array bound to the texture reference \p hTexRef.
 * If \p pFormat or \p pNumChannels is NULL, it will be ignored.
 *
 * \param pFormat      - Returned format
 * \param pNumChannels - Returned number of components
 * \param hTexRef      - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags
 *)
function cuTexRefGetFormat(pFormat : PCUarray_format; pNumChannels : Plongint; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the mipmap filtering mode for a texture reference
 *
 * \deprecated
 *
 * Returns the mipmap filtering mode in \p pfm that's used when reading memory through
 * the texture reference \p hTexRef.
 *
 * \param pfm     - Returned mipmap filtering mode
 * \param hTexRef - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetMipmapFilterMode(pfm : PCUfilter_mode; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the mipmap level bias for a texture reference
 *
 * \deprecated
 *
 * Returns the mipmap level bias in \p pBias that's added to the specified mipmap
 * level when reading memory through the texture reference \p hTexRef.
 *
 * \param pbias   - Returned mipmap level bias
 * \param hTexRef - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetMipmapLevelBias(pbias : Psingle; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda;deprecated;

(**
 * \brief Gets the min/max mipmap level clamps for a texture reference
 *
 * \deprecated
 *
 * Returns the min/max mipmap level clamps in \p pminMipmapLevelClamp and \p pmaxMipmapLevelClamp
 * that's used when reading memory through the texture reference \p hTexRef.
 *
 * \param pminMipmapLevelClamp - Returned mipmap min level clamp
 * \param pmaxMipmapLevelClamp - Returned mipmap max level clamp
 * \param hTexRef              - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp : PSingle; pmaxMipmapLevelClamp : PSingle; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Gets the maximum anisotropy for a texture reference
 *
 * \deprecated
 *
 * Returns the maximum anisotropy in \p pmaxAniso that's used when reading memory through
 * the texture reference \p hTexRef.
 *
 * \param pmaxAniso - Returned maximum anisotropy
 * \param hTexRef   - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat
 *)
function cuTexRefGetMaxAnisotropy(pmaxAniso : Plongint; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Gets the border color used by a texture reference
 *
 * \deprecated
 *
 * Returns in \p pBorderColor, values of the RGBA color used by
 * the texture reference \p hTexRef.
 * The color value is of type float and holds color components in
 * the following sequence:
 * pBorderColor[0] holds 'R' component
 * pBorderColor[1] holds 'G' component
 * pBorderColor[2] holds 'B' component
 * pBorderColor[3] holds 'A' component
 *
 * \param hTexRef  - Texture reference
 * \param pBorderColor   - Returned Type and Value of RGBA color
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddressMode,
 * ::cuTexRefSetAddressMode, ::cuTexRefSetBorderColor
 *)
function cuTexRefGetBorderColor(pBorderColor : PSingle; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Gets the flags used by a texture reference
 *
 * \deprecated
 *
 * Returns in \p *pFlags the flags of the texture reference \p hTexRef.
 *
 * \param pFlags  - Returned flags
 * \param hTexRef - Texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefSetAddress,
 * ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
 * ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
 * ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
 * ::cuTexRefGetFilterMode, ::cuTexRefGetFormat
 *)
function cuTexRefGetFlags(pFlags : Plongword; hTexRef : CUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Creates a texture reference
 *
 * \deprecated
 *
 * Creates a texture reference and returns its handle in \p *pTexRef. Once
 * created, the application must call ::cuTexRefSetArray() or
 * ::cuTexRefSetAddress() to associate the reference with allocated memory.
 * Other texture reference functions are used to specify the format and
 * interpretation (addressing, filtering, etc.) to be used when the memory is
 * read through this texture reference.
 *
 * \param pTexRef - Returned texture reference
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefDestroy
 *)
function cuTexRefCreate(pTexRef : PCUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Destroys a texture reference
 *
 * \deprecated
 *
 * Destroys the texture reference specified by \p hTexRef.
 *
 * \param hTexRef - Texture reference to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuTexRefCreate
 *)
function cuTexRefDestroy(hTexRef : CUtexref):CUresult;  WINAPI; external libcuda; deprecated;

(** @} *) (* END CUDA_TEXREF_DEPRECATED *)


(**
 * \defgroup CUDA_SURFREF_DEPRECATED Surface Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ surface reference management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the surface reference management functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Sets the CUDA array for a surface reference.
 *
 * \deprecated
 *
 * Sets the CUDA array \p hArray to be read and written by the surface reference
 * \p hSurfRef.  Any previous CUDA array state associated with the surface
 * reference is superseded by this function.  \p Flags must be set to 0.
 * The ::CUDA_ARRAY3D_SURFACE_LDST flag must have been set for the CUDA array.
 * Any CUDA array previously bound to \p hSurfRef is unbound.

 * \param hSurfRef - Surface reference handle
 * \param hArray - CUDA array handle
 * \param Flags - set to 0
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuModuleGetSurfRef,
 * ::cuSurfRefGetArray
 *)
function cuSurfRefSetArray(hSurfRef:CUsurfref;  hArray:CUarray;  Flags : longword):CUresult;  WINAPI; external libcuda; deprecated;

(**
 * \brief Passes back the CUDA array bound to a surface reference.
 *
 * \deprecated
 *
 * Returns in \p *phArray the CUDA array bound to the surface reference
 * \p hSurfRef, or returns ::CUDA_ERROR_INVALID_VALUE if the surface reference
 * is not bound to any CUDA array.

 * \param phArray - Surface reference handle
 * \param hSurfRef - Surface reference handle
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa ::cuModuleGetSurfRef, ::cuSurfRefSetArray
 *)
function cuSurfRefGetArray(phArray : PCUarray; hSurfRef : CUsurfref):CUresult;  WINAPI; external libcuda; deprecated;

(** @} *) (* END CUDA_SURFREF_DEPRECATED *)

(**
 * \defgroup CUDA_TEXOBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the texture object management functions of the
 * low-level CUDA driver application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 *)

(**
 * \brief Creates a texture object
 *
 * Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
 * the data to texture from. \p pTexDesc describes how the data should be sampled.
 * \p pResViewDesc is an optional argument that specifies an alternate format for
 * the data described by \p pResDesc, and also describes the subresource region
 * to restrict access to when texturing. \p pResViewDesc can only be specified if
 * the type of resource is a CUDA array or a CUDA mipmapped array.
 *
 * Texture objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a texture object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * The ::CUDA_RESOURCE_DESC structure is defined as:
 * \code
        typedef struct CUDA_RESOURCE_DESC_st
        {
            CUresourcetype resType;

            union {
                struct {
                    CUarray hArray;
                } array;
                struct {
                    CUmipmappedArray hMipmappedArray;
                } mipmap;
                struct {
                    CUdeviceptr devPtr;
                    CUarray_format format;
                    longint numChannels;
                    size_t sizeInBytes;
                } linear;
                struct {
                    CUdeviceptr devPtr;
                    CUarray_format format;
                    longint numChannels;
                    size_t width;
                    size_t height;
                    size_t pitchInBytes;
                } pitch2D;
            } res;

            longint flags;
        } CUDA_RESOURCE_DESC;

 * \endcode
 * where:
 * - ::CUDA_RESOURCE_DESC::resType specifies the type of resource to texture from.
 * CUresourceType is defined as:
 * \code
          CUresourcetype_enum = (
            CU_RESOURCE_TYPE_ARRAY           = $00,
            CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = $01,
            CU_RESOURCE_TYPE_LINEAR          = $02,
            CU_RESOURCE_TYPE_PITCH2D         = $03
          );
  CUresourcetype = CUresourcetype_enum;
 * \endcode
 *
 * \par
 * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_ARRAY, ::CUDA_RESOURCE_DESC::res::array::hArray
 * must be set to a valid CUDA array handle.
 *
 * \par
 * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY, ::CUDA_RESOURCE_DESC::res::mipmap::hMipmappedArray
 * must be set to a valid CUDA mipmapped array handle.
 *
 * \par
 * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_LINEAR, ::CUDA_RESOURCE_DESC::res::linear::devPtr
 * must be set to a valid device pointer, that is aligned to ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
 * ::CUDA_RESOURCE_DESC::res::linear::format and ::CUDA_RESOURCE_DESC::res::linear::numChannels
 * describe the format of each component and the number of components per array element. ::CUDA_RESOURCE_DESC::res::linear::sizeInBytes
 * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of elements is computed as (sizeInBytes / (sizeof(format) * numChannels)).
 *
 * \par
 * If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_PITCH2D, ::CUDA_RESOURCE_DESC::res::pitch2D::devPtr
 * must be set to a valid device pointer, that is aligned to ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
 * ::CUDA_RESOURCE_DESC::res::pitch2D::format and ::CUDA_RESOURCE_DESC::res::pitch2D::numChannels
 * describe the format of each component and the number of components per array element. ::CUDA_RESOURCE_DESC::res::pitch2D::width
 * and ::CUDA_RESOURCE_DESC::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
 * ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
 * ::CUDA_RESOURCE_DESC::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to
 * ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. Pitch cannot exceed ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
 *
 * - ::flags must be set to zero.
 *
 *
 * The ::CUDA_TEXTURE_DESC struct is defined as
 * \code
          CUDA_TEXTURE_DESC_st=packed record
            CUaddress_mode addressMode[3];
            CUfilter_mode filterMode;
            longint flags;
            longint maxAnisotropy;
            CUfilter_mode mipmapFilterMode;
            float mipmapLevelBias;
            float minMipmapLevelClamp;
            float maxMipmapLevelClamp;
          end;
  CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_st;
 * \endcode
 * where
 * - ::CUDA_TEXTURE_DESC::addressMode specifies the addressing mode for each dimension of the texture data. ::CUaddress_mode is defined as:
 *   \code
          CUaddress_mode_enum = (
            CU_TR_ADDRESS_MODE_WRAP = 0,
            CU_TR_ADDRESS_MODE_CLAMP = 1,
            CU_TR_ADDRESS_MODE_MIRROR = 2,
            CU_TR_ADDRESS_MODE_BORDER = 3
          );
  CUaddress_mode = CUaddress_mode_enum;
 *   \endcode
 *   This is ignored if ::CUDA_RESOURCE_DESC::resType is ::CU_RESOURCE_TYPE_LINEAR. Also, if the flag, ::CU_TRSF_NORMALIZED_COORDINATES
 *   is not set, the only supported address mode is ::CU_TR_ADDRESS_MODE_CLAMP.
 *
 * - ::CUDA_TEXTURE_DESC::filterMode specifies the filtering mode to be used when fetching from the texture. CUfilter_mode is defined as:
 *   \code
          CUfilter_mode_enum = (
            CU_TR_FILTER_MODE_POINT = 0,
            CU_TR_FILTER_MODE_LINEAR = 1
          );
  CUfilter_mode = CUfilter_mode_enum;
 *   \endcode
 *   This is ignored if ::CUDA_RESOURCE_DESC::resType is ::CU_RESOURCE_TYPE_LINEAR.
 *
 * - ::CUDA_TEXTURE_DESC::flags can be any combination of the following:
 *   - ::CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of
 *   having the texture promote integer data to floating point data in the
 *   range [0, 1]. Note that texture with 32-bit integer format would not be 
 *   promoted, regardless of whether or not this flag is specified.
 *   - ::CU_TRSF_NORMALIZED_COORDINATES, which suppresses the default behavior
 *   of having the texture coordinates range from [0, Dim) where Dim is the 
 *   width or height of the CUDA array. Instead, the texture coordinates 
 *   [0, 1.0) reference the entire breadth of the array dimension; Note that
 *   for CUDA mipmapped arrays, this flag has to be set.
 *   - ::CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION, which disables any trilinear
 *   filtering optimizations. Trilinear optimizations improve texture filtering
 *   performance by allowing bilinear filtering on textures in scenarios where
 *   it can closely approximate the expected results.
 *   - ::CU_TRSF_SEAMLESS_CUBEMAP, which enables seamless cube map filtering. 
 *   This flag can only be specified if the underlying resource is a CUDA array 
 *   or a CUDA mipmapped array that was created with the flag ::CUDA_ARRAY3D_CUBEMAP.
 *   When seamless cube map filtering is enabled, texture address modes specified 
 *   by ::CUDA_TEXTURE_DESC::addressMode are ignored. Instead, if the ::CUDA_TEXTURE_DESC::filterMode 
 *   is set to ::CU_TR_FILTER_MODE_POINT the address mode ::CU_TR_ADDRESS_MODE_CLAMP 
 *   will be applied for all dimensions. If the ::CUDA_TEXTURE_DESC::filterMode is 
 *   set to ::CU_TR_FILTER_MODE_LINEAR seamless cube map filtering will be performed
 *   when sampling along the cube face borders.
 *
 * - ::CUDA_TEXTURE_DESC::maxAnisotropy specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be
 *   clamped to the range [1,16].
 *
 * - ::CUDA_TEXTURE_DESC::mipmapFilterMode specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
 *
 * - ::CUDA_TEXTURE_DESC::mipmapLevelBias specifies the offset to be applied to the calculated mipmap level.
 *
 * - ::CUDA_TEXTURE_DESC::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 *
 * - ::CUDA_TEXTURE_DESC::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 *
 *
 * The ::CUDA_RESOURCE_VIEW_DESC struct is defined as
 * \code
          CUDA_RESOURCE_VIEW_DESC_st=packed record
            CUresourceViewFormat format;
            size_t width;
            size_t height;
            size_t depth;
            longint firstMipmapLevel;
            longint lastMipmapLevel;
            longint firstLayer;
            longint lastLayer;
          end;
  CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_st;
 * \endcode
 * where:
 * - ::CUDA_RESOURCE_VIEW_DESC::format specifies how the data contained in the CUDA array or CUDA mipmapped array should
 *   be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block
 *   compressed format, then the underlying CUDA array or CUDA mipmapped array has to have a base of format ::CU_AD_FORMAT_UNSIGNED_INT32.
 *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying CUDA array to have
 *   a format of ::CU_AD_FORMAT_UNSIGNED_INT32 with 2 channels. The other BC formats require the underlying resource to have the same base
 *   format but with 4 channels.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::width specifies the new width of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::height specifies the new height of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::depth specifies the new depth of the texture data. This value has to be equal to that of the
 *   original resource.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
 *   For non-mipmapped resources, this value has to be zero.::CUDA_TEXTURE_DESC::minMipmapLevelClamp and ::CUDA_TEXTURE_DESC::maxMipmapLevelClamp
 *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
 *   then the actual minimum mipmap level clamp will be 3.2.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
 *   has to be zero.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
 *   For non-layered resources, this value has to be zero.
 *
 * - ::CUDA_RESOURCE_VIEW_DESC::lastLayer specifies the last layer index for layered textures. For non-layered resources,
 *   this value has to be zero.
 *
 *
 * \param pTexObject   - Texture object to create
 * \param pResDesc     - Resource descriptor
 * \param pTexDesc     - Texture descriptor
 * \param pResViewDesc - Resource view descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexObjectDestroy,
 * ::cudaCreateTextureObject
 *)
function cuTexObjectCreate(pTexObject : PCUtexObject; const pResDesc : PCUDA_RESOURCE_DESC; const pTexDesc : PCUDA_TEXTURE_DESC; const pResViewDesc : PCUDA_RESOURCE_VIEW_DESC):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys a texture object
 *
 * Destroys the texture object specified by \p texObject.
 *
 * \param texObject - Texture object to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexObjectCreate,
 * ::cudaDestroyTextureObject
 *)
function cuTexObjectDestroy(texObject : CUtexObject):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a texture object's resource descriptor
 *
 * Returns the resource descriptor for the texture object specified by \p texObject.
 *
 * \param pResDesc  - Resource descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexObjectCreate,
 * ::cudaGetTextureObjectResourceDesc,
 *)
function cuTexObjectGetResourceDesc(pResDesc : PCUDA_RESOURCE_DESC; texObject : CUtexObject):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a texture object's texture descriptor
 *
 * Returns the texture descriptor for the texture object specified by \p texObject.
 *
 * \param pTexDesc  - Texture descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexObjectCreate,
 * ::cudaGetTextureObjectTextureDesc
 *)
function cuTexObjectGetTextureDesc(pTexDesc : PCUDA_TEXTURE_DESC; texObject : CUtexObject):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a texture object's resource view descriptor
 *
 * Returns the resource view descriptor for the texture object specified by \p texObject.
 * If no resource view was set for \p texObject, the ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * \param pResViewDesc - Resource view descriptor
 * \param texObject    - Texture object
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTexObjectCreate,
 * ::cudaGetTextureObjectResourceViewDesc
 *)
function cuTexObjectGetResourceViewDesc(pResViewDesc : PCUDA_RESOURCE_VIEW_DESC; texObject : CUtexObject):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_TEXOBJECT *)

(**
 * \defgroup CUDA_SURFOBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the surface object management functions of the
 * low-level CUDA driver application programming interface. The surface
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 *)

(**
 * \brief Creates a surface object
 *
 * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
 * the data to perform surface load/stores on. ::CUDA_RESOURCE_DESC::resType must be
 * ::CU_RESOURCE_TYPE_ARRAY and  ::CUDA_RESOURCE_DESC::res::array::hArray
 * must be set to a valid CUDA array handle. ::CUDA_RESOURCE_DESC::flags must be set to zero.
 *
 * Surface objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a surface object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * \param pSurfObject - Surface object to create
 * \param pResDesc    - Resource descriptor
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuSurfObjectDestroy,
 * ::cudaCreateSurfaceObject
 *)
function cuSurfObjectCreate(pSurfObject : PCUsurfObject; const pResDesc : PCUDA_RESOURCE_DESC):CUresult;  WINAPI; external libcuda;

(**
 * \brief Destroys a surface object
 *
 * Destroys the surface object specified by \p surfObject.
 *
 * \param surfObject - Surface object to destroy
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuSurfObjectCreate,
 * ::cudaDestroySurfaceObject
 *)
function cuSurfObjectDestroy(surfObject : CUsurfObject):CUresult;  WINAPI; external libcuda;

(**
 * \brief Returns a surface object's resource descriptor
 *
 * Returns the resource descriptor for the surface object specified by \p surfObject.
 *
 * \param pResDesc   - Resource descriptor
 * \param surfObject - Surface object
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuSurfObjectCreate,
 * ::cudaGetSurfaceObjectResourceDesc
 *)
function cuSurfObjectGetResourceDesc(pResDesc : PCUDA_RESOURCE_DESC; surfObject : CUsurfObject):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_SURFOBJECT *)

(**
 * \defgroup CUDA_TENSOR_MEMORY Tensor Map Object Managment
 *
 * ___MANBRIEF___ tensor map object management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the tensor map object management functions of the
 * low-level CUDA driver application programming interface. The tensor
 * core API is only supported on devices of compute capability 9.0 or higher.
 *
 * @{
 *)

(**
 * \brief Create a tensor map descriptor object representing tiled memory region
 *
 * Creates a descriptor for Tensor Memory Access (TMA) object specified
 * by the parameters describing a tiled region and returns it in \p tensorMap.
 *
 * Tensor map objects are only supported on devices of compute capability 9.0 or higher.
 * Additionally, a tensor map object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * The parameters passed are bound to the following requirements:
 *
 * - \p tensorMap address must be aligned to 64 bytes.
 *
 * - \p tensorDataType has to be an enum from ::CUtensorMapDataType which is defined as:
 * \code
      CUtensorMapDataType_enum = (
        CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,       // 1 byte
        CU_TENSOR_MAP_DATA_TYPE_UINT16,          // 2 bytes
        CU_TENSOR_MAP_DATA_TYPE_UINT32,          // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_INT32,           // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_UINT64,          // 8 bytes
        CU_TENSOR_MAP_DATA_TYPE_INT64,           // 8 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,         // 2 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,         // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT64,         // 8 bytes
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,        // 2 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,     // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,        // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ     // 4 bytes
      );
  CUtensorMapDataType = CUtensorMapDataType_enum;
 * \endcode
 *
 * - \p tensorRank must be non-zero and less than or equal to the maximum supported dimensionality of 5. If \p interleave is not
 * ::CU_TENSOR_MAP_INTERLEAVE_NONE, then \p tensorRank must additionally be greater than or equal to 3.
 *
 * - \p globalAddress, which specifies the starting address of the memory region described, must be 32 byte aligned when \p interleave is
 * ::CU_TENSOR_MAP_INTERLEAVE_32B and 16 byte aligned otherwise.
 *
 * - \p globalDim array, which specifies tensor size of each of the \p tensorRank dimensions, must be non-zero and less than or
 * equal to 2^32.
 *
 * - \p globalStrides array, which specifies tensor stride of each of the lower \p tensorRank - 1 dimensions in bytes, must be a
 * multiple of 16 and less than 2^40. Additionally, the stride must be a multiple of 32 when \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_32B.
 * Each following dimension specified includes previous dimension stride:
 * \code
    globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0];
    for (i = 1; i < tensorRank - 1; i++)
        globalStrides[i] = globalStrides[i – 1] * (globalDim[i] + padding[i]);
        assert(globalStrides[i] >= globalDim[i]);
 * \endcode
 *
 * - \p boxDim array, which specifies number of elements to be traversed along each of the \p tensorRank dimensions, must be non-zero
 * and less than or equal to 256.
 * When \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_NONE, { \p boxDim[0] * elementSizeInBytes( \p tensorDataType ) } must be a multiple
 * of 16 bytes.
 *
 * - \p elementStrides array, which specifies the iteration step along each of the \p tensorRank dimensions, must be non-zero and less
 * than or equal to 8. Note that when \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_NONE, the first element of this array is ignored since
 * TMA doesn’t support the stride for dimension zero.
 * When all elemets of \p elementStrides array is one, \p boxDim specifies the number of elements to load. However, if the \p elementStrides[i]
 * is not equal to one, then TMA loads ceil( \p boxDim[i] / \p elementStrides[i]) number of elements along i-th dimension. To load N elements along
 * i-th dimension, \p boxDim[i] must be set to N * \p elementStrides[i].
 *
 * - \p interleave specifies the interleaved layout of type ::CUtensorMapInterleave, which is defined as:
 * \code
      CUtensorMapInterleave_enum = (
        CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
        CU_TENSOR_MAP_INTERLEAVE_16B,
        CU_TENSOR_MAP_INTERLEAVE_32B
      );
  CUtensorMapInterleave = CUtensorMapInterleave_enum;
 * \endcode
 * TMA supports interleaved layouts like NC/8HWC8 where C8 utilizes 16 bytes in memory assuming 2 byte per channel or NC/16HWC16 where C16
 * uses 32 bytes.
 * When \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_NONE and \p swizzle is not ::CU_TENSOR_MAP_SWIZZLE_NONE, the bounding box inner dimension
 * (computed as \p boxDim[0] multiplied by element size derived from \p tensorDataType) must be less than or equal to the swizzle size.
 *    - CU_TENSOR_MAP_SWIZZLE_32B implies the bounding box inner dimension will be <= 32.
 *    - CU_TENSOR_MAP_SWIZZLE_64B implies the bounding box inner dimension will be <= 64.
 *    - CU_TENSOR_MAP_SWIZZLE_128B implies the bounding box inner dimension will be <= 128.
 *
 * - \p swizzle, which specifies the shared memory bank swizzling pattern, has to be of type ::CUtensorMapSwizzle which is defined as:
 * \code
      CUtensorMapSwizzle_enum = (
        CU_TENSOR_MAP_SWIZZLE_NONE = 0,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_SWIZZLE_128B
      );
  CUtensorMapSwizzle = CUtensorMapSwizzle_enum;
 * \endcode
 * Data is organized in specific order in global memory; however, it may not match the order in which data are accessed by application in
 * the shared memory. This difference in data organization may cause bank conflicts when shared memory is accessed. In order to avoid this
 * problem, data can be loaded to shard memory with shuffling across shared memory banks.
 * Note that it’s expected that when \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_32B, \p swizzle should be ::CU_TENSOR_MAP_SWIZZLE_32B mode.
 * Other interleave modes can have any swizzling patterns.
 *
 * - \p l2Promotion specifies L2 fetch size which indicates the byte granurality at which L2 requests is filled from DRAM. It must be of
 * type ::CUtensorMapL2promotion, which is defined as:
 * \code
      CUtensorMapL2promotion_enum = (
        CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B
      );
  CUtensorMapL2promotion = CUtensorMapL2promotion_enum;
 * \endcode
 *
 * - \p oobFill, which indicates whether zero or a special NaN constant should be used to fill out-of-bound elements, must be of type
 * ::CUtensorMapFloatOOBfill which is defined as:
 * \code
      CUtensorMapFloatOOBfill_enum = (
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
      );
  CUtensorMapFloatOOBfill = CUtensorMapFloatOOBfill_enum;
 * \endcode
 * Note that ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can only be used when \p tensorDataType represents a floating data type.
 *
 * \param tensorMap         - Tensor map object to create
 * \param tensorDataType    - Tensor data type
 * \param tensorRank        - Dimensionality of tensor
 * \param globalAddress     - Starting address of memory region described by tensor
 * \param globalDim         - Array containing tensor size (number of elements) along each of the \p tensorRank dimensions
 * \param globalStrides     - Array containing stride size (in bytes) along each of the \p tensorRank - 1 dimensions
 * \param boxDim            - Array containing traversal box size (number of elments) along each of the \p tensorRank dimensions. Specifies how many elements to be traversed along each tensor dimension.
 * \param elementStrides    - Array containing traversal stride in each of the \p tensorRank dimensions
 * \param interleave        - Type of interleaved layout the tensor addresses
 * \param swizzle           - Bank swizzling pattern inside shared memory
 * \param l2Promotion       - L2 promotion size
 * \param oobFill           - Indicate whether zero or special NaN constant must be used to fill out-of-bound elements
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTensorMapEncodeIm2col,
 * ::cuTensorMapReplaceAddress
 *)
function cuTensorMapEncodeTiled(tensorMap : PCUtensorMap; tensorDataType:CUtensorMapDataType;  tensorRank:cuuint32_t;  globalAddress:Pointer;  const globalDim : Pcuuint64_t; const globalStrides : Pcuuint64_t; const boxDim : Pcuuint32_t; const elementStrides : Pcuuint32_t; interleave:CUtensorMapInterleave;  swizzle:CUtensorMapSwizzle;  l2Promotion:CUtensorMapL2promotion;  oobFill : CUtensorMapFloatOOBfill):CUresult;  WINAPI; external libcuda;


(**
 * \brief Create a tensor map descriptor object representing im2col memory region
 *
 * Creates a descriptor for Tensor Memory Access (TMA) object specified
 * by the parameters describing a im2col memory layout and returns it in \p tensorMap.
 *
 * Tensor map objects are only supported on devices of compute capability 9.0 or higher.
 * Additionally, a tensor map object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * The parameters passed are bound to the following requirements:
 *
 * - \p tensorMap address must be aligned to 64 bytes.
 *
 * - \p tensorDataType has to be an enum from ::CUtensorMapDataType which is defined as:
 * \code
      CUtensorMapDataType_enum = (
        CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,       // 1 byte
        CU_TENSOR_MAP_DATA_TYPE_UINT16,          // 2 bytes
        CU_TENSOR_MAP_DATA_TYPE_UINT32,          // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_INT32,           // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_UINT64,          // 8 bytes
        CU_TENSOR_MAP_DATA_TYPE_INT64,           // 8 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,         // 2 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,         // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT64,         // 8 bytes
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,        // 2 bytes
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,     // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,        // 4 bytes
        CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ     // 4 bytes
      );
  CUtensorMapDataType = CUtensorMapDataType_enum;
 * \endcode
 *
 * - \p tensorRank must be one of dimensions 3, 4, or 5.
 *
 * - \p globalAddress, which specifies the starting address of the memory region described, must be 32 byte aligned when \p interleave is
 * ::CU_TENSOR_MAP_INTERLEAVE_32B and 16 byte aligned otherwise.
 *
 * - \p globalDim array, which specifies tensor size of each of the \p tensorRank dimensions, must be non-zero and less than or
 * equal to 2^32.
 *
 * - \p globalStrides array, which specifies tensor stride of each of the lower \p tensorRank - 1 dimensions in bytes, must be a
 * multiple of 16 and less than 2^40. Additionally, the stride must be a multiple of 32 when \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_32B.
 * Each following dimension specified includes previous dimension stride:
 * \code
    globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0];
    for (i = 1; i < tensorRank - 1; i++)
        globalStrides[i] = globalStrides[i – 1] * (globalDim[i] + padding[i]);
        assert(globalStrides[i] >= globalDim[i]);
 * \endcode
 *
 * - \p pixelBoxLowerCorner array specifies the coordinate offsets {D, H, W} of the bounding box from top/left/front corner. The number of
 * offsets and their precision depends on the tensor dimensionality:
 *    - When \p tensorRank is 3, one signed offset within range [-32768, 32767] is supported.
 *    - When \p tensorRank is 4, two signed offsets each within range [-128, 127] are supported.
 *    - When \p tensorRank is 5, three offsets each within range [-16, 15] are supported.
 *
 * - \p pixelBoxUpperCorner array specifies the coordinate offsets {D, H, W} of the bounding box from bottom/right/back corner. The number of
 * offsets and their precision depends on the tensor dimensionality:
 *    - When \p tensorRank is 3, one signed offset within range [-32768, 32767] is supported.
 *    - When \p tensorRank is 4, two signed offsets each within range [-128, 127] are supported.
 *    - When \p tensorRank is 5, three offsets each within range [-16, 15] are supported.
 * The bounding box specified by \p pixelBoxLowerCorner and \p pixelBoxUpperCorner must have non-zero area.
 *
 * - \p channelsPerPixel, which specifies the number of elements which must be accessed along C dimension, must be less than or equal to 256.
 *
 * - \p pixelsPerColumn, which specifies the number of elements that must be accessed along the {N, D, H, W} dimensions, must be less than or
 * equal to 1024.
 *
 * - \p elementStrides array, which specifies the iteration step along each of the \p tensorRank dimensions, must be non-zero and less
 * than or equal to 8. Note that when \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_NONE, the first element of this array is ignored since
 * TMA doesn’t support the stride for dimension zero.
 * When all elemets of \p elementStrides array is one, \p boxDim specifies the number of elements to load. However, if the \p elementStrides[i]
 * is not equal to one, then TMA loads ceil( \p boxDim[i] / \p elementStrides[i]) number of elements along i-th dimension. To load N elements along
 * i-th dimension, \p boxDim[i] must be set to N * \p elementStrides[i].
 *
 * - \p interleave specifies the interleaved layout of type ::CUtensorMapInterleave, which is defined as:
 * \code
      CUtensorMapInterleave_enum = (
        CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
        CU_TENSOR_MAP_INTERLEAVE_16B,
        CU_TENSOR_MAP_INTERLEAVE_32B
      );
  CUtensorMapInterleave = CUtensorMapInterleave_enum;
 * \endcode
 * TMA supports interleaved layouts like NC/8HWC8 where C8 utilizes 16 bytes in memory assuming 2 byte per channel or NC/16HWC16 where C16
 * uses 32 bytes.
 * When \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_NONE and \p swizzle is not ::CU_TENSOR_MAP_SWIZZLE_NONE, the bounding box inner dimension
 * (computed as \p boxDim[0] multiplied by element size derived from \p tensorDataType) must be less than or equal to the swizzle size.
 *    - CU_TENSOR_MAP_SWIZZLE_32B implies the bounding box inner dimension will be <= 32.
 *    - CU_TENSOR_MAP_SWIZZLE_64B implies the bounding box inner dimension will be <= 64.
 *    - CU_TENSOR_MAP_SWIZZLE_128B implies the bounding box inner dimension will be <= 128.
 *
 * - \p swizzle, which specifies the shared memory bank swizzling pattern, has to be of type ::CUtensorMapSwizzle which is defined as:
 * \code
      CUtensorMapSwizzle_enum = (
        CU_TENSOR_MAP_SWIZZLE_NONE = 0,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_SWIZZLE_128B
      );
  CUtensorMapSwizzle = CUtensorMapSwizzle_enum;
 * \endcode
 * Data is organized in specific order in global memory; however, it may not match the order in which data are accessed by application in
 * the shared memory. This difference in data organization may cause bank conflicts when shared memory is accessed. In order to avoid this
 * problem, data can be loaded to shard memory with shuffling across shared memory banks.
 * Note that it’s expected that when \p interleave is ::CU_TENSOR_MAP_INTERLEAVE_32B, \p swizzle should be ::CU_TENSOR_MAP_SWIZZLE_32B mode.
 * Other interleave modes can have any swizzling patterns.
 *
 * - \p l2Promotion specifies L2 fetch size which indicates the byte granurality at which L2 requests is filled from DRAM. It must be of
 * type ::CUtensorMapL2promotion, which is defined as:
 * \code
      CUtensorMapL2promotion_enum = (
        CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B
      );
  CUtensorMapL2promotion = CUtensorMapL2promotion_enum;
 * \endcode
 *
 * - \p oobFill, which indicates whether zero or a special NaN constant should be used to fill out-of-bound elements, must be of type
 * ::CUtensorMapFloatOOBfill which is defined as:
 * \code
      CUtensorMapFloatOOBfill_enum = (
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
      );
  CUtensorMapFloatOOBfill = CUtensorMapFloatOOBfill_enum;
 * \endcode
 * Note that ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can only be used when \p tensorDataType represents a floating data type.
 *
 * \param tensorMap             - Tensor map object to create
 * \param tensorDataType        - Tensor data type
 * \param tensorRank            - Dimensionality of tensor, needs to be at least of dimension 3
 * \param globalAddress         - Starting address of memory region described by tensor
 * \param globalDim             - Array containing tensor size (number of elements) along each of the \p tensorRank dimensions
 * \param globalStrides         - Array containing stride size (in bytes) along each of the \p tensorRank - 1 dimensions
 * \param pixelBoxLowerCorner   - Array containing DHW dimentions of lower box corner
 * \param pixelBoxUpperCorner   - Array containing DHW dimentions of upper box corner
 * \param channelsPerPixel      - Number of channels per pixel
 * \param pixelsPerColumn       - Number of pixels per column
 * \param elementStrides        - Array containing traversal stride in each of the \p tensorRank dimensions
 * \param interleave            - Type of interleaved layout the tensor addresses
 * \param swizzle               - Bank swizzling pattern inside shared memory
 * \param l2Promotion           - L2 promotion size
 * \param oobFill               - Indicate whether zero or special NaN constant must be used to fill out-of-bound elements
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTensorMapEncodeTiled,
 * ::cuTensorMapReplaceAddress
 *)
function cuTensorMapEncodeIm2col(tensorMap : PCUtensorMap; tensorDataType:CUtensorMapDataType;  tensorRank:cuuint32_t;  globalAddress:Pointer;  const globalDim : Pcuuint64_t; const globalStrides : Pcuuint64_t; const pixelBoxLowerCorner : Plongint; const pixelBoxUpperCorner : Plongint; channelsPerPixel:cuuint32_t;  pixelsPerColumn:cuuint32_t;  const elementStrides : Pcuuint32_t; interleave:CUtensorMapInterleave;  swizzle:CUtensorMapSwizzle;  l2Promotion:CUtensorMapL2promotion;  oobFill : CUtensorMapFloatOOBfill):CUresult;  WINAPI; external libcuda;

(**
 * \brief Modify an existing tensor map descriptor with an updated global address
 *
 * Modifies the descriptor for Tensor Memory Access (TMA) object passed in \p tensorMap with
 * an updated \p globalAddress.
 *
 * Tensor map objects are only supported on devices of compute capability 9.0 or higher.
 * Additionally, a tensor map object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * \param tensorMap             - Tensor map object to modify
 * \param globalAddress         - Starting address of memory region described by tensor, must follow previous alignment requirements
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuTensorMapEncodeTiled,
 * ::cuTensorMapEncodeIm2col
 *)
function cuTensorMapReplaceAddress(tensorMap : PCUtensorMap; globalAddress : Pointer):CUresult;  WINAPI; external libcuda;

(** @} *)
(* END CUDA_TENSOR_MEMORY *)

(**
 * \defgroup CUDA_PEER_ACCESS Peer Context Memory Access
 *
 * ___MANBRIEF___ direct peer context memory access functions of the low-level
 * CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the direct peer context memory access functions
 * of the low-level CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Queries if a device may directly access a peer device's memory.
 *
 * Returns in \p *canAccessPeer a value of 1 if contexts on \p dev are capable of
 * directly accessing memory from contexts on \p peerDev and 0 otherwise.
 * If direct access of \p peerDev from \p dev is possible, then access may be
 * enabled on two specific contexts by calling ::cuCtxEnablePeerAccess().
 *
 * \param canAccessPeer - Returned access capability
 * \param dev           - Device from which allocations on \p peerDev are to
 *                        be directly accessed.
 * \param peerDev       - Device on which the allocations to be directly accessed
 *                        by \p dev reside.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::cuCtxEnablePeerAccess,
 * ::cuCtxDisablePeerAccess,
 * ::cudaDeviceCanAccessPeer
 *)
function cuDeviceCanAccessPeer(canAccessPeer : Plongint; dev:CUdevice;  peerDev : CUdevice):CUresult;  WINAPI; external libcuda;

(**
 * \brief Enables direct access to memory allocations in a peer context.
 *
 * If both the current context and \p peerContext are on devices which support unified
 * addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) and same
 * major compute capability, then on success all allocations from \p peerContext will
 * immediately be accessible by the current context.  See \ref CUDA_UNIFIED for additional
 * details.
 *
 * Note that access granted by this call is unidirectional and that in order to access
 * memory from the current context in \p peerContext, a separate symmetric call
 * to ::cuCtxEnablePeerAccess() is required.
 *
 * Note that there are both device-wide and system-wide limitations per system
 * configuration, as noted in the CUDA Programming Guide under the section
 * "Peer-to-Peer Memory Access".
 *
 * Returns ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED if ::cuDeviceCanAccessPeer() indicates
 * that the ::CUdevice of the current context cannot directly access memory
 * from the ::CUdevice of \p peerContext.
 *
 * Returns ::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED if direct access of
 * \p peerContext from the current context has already been enabled.
 *
 * Returns ::CUDA_ERROR_TOO_MANY_PEERS if direct peer access is not possible
 * because hardware resources required for peer access have been exhausted.
 *
 * Returns ::CUDA_ERROR_INVALID_CONTEXT if there is no current context, \p peerContext
 * is not a valid context, or if the current context is \p peerContext.
 *
 * Returns ::CUDA_ERROR_INVALID_VALUE if \p Flags is not 0.
 *
 * \param peerContext - Peer context to enable direct access to from the current context
 * \param Flags       - Reserved for future use and must be set to 0
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
 * ::CUDA_ERROR_TOO_MANY_PEERS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cuDeviceCanAccessPeer,
 * ::cuCtxDisablePeerAccess,
 * ::cudaDeviceEnablePeerAccess
 *)
function cuCtxEnablePeerAccess(peerContext:CUcontext;  Flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Disables direct access to memory allocations in a peer context and
 * unregisters any registered allocations.
 *
  Returns ::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has
 * not yet been enabled from \p peerContext to the current context.
 *
 * Returns ::CUDA_ERROR_INVALID_CONTEXT if there is no current context, or if
 * \p peerContext is not a valid context.
 *
 * \param peerContext - Peer context to disable direct access to
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * \notefnerr
 *
 * \sa
 * ::cuDeviceCanAccessPeer,
 * ::cuCtxEnablePeerAccess,
 * ::cudaDeviceDisablePeerAccess
 *)
function cuCtxDisablePeerAccess(peerContext : CUcontext):CUresult;  WINAPI; external libcuda;

(**
 * \brief Queries attributes of the link between two devices.
 *
 * Returns in \p *value the value of the requested attribute \p attrib of the
 * link between \p srcDevice and \p dstDevice. The supported attributes are:
 * - ::CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: A relative value indicating the
 *   performance of the link between two devices.
 * - ::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED P2P: 1 if P2P Access is enable.
 * - ::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: 1 if Atomic operations over
 *   the link are supported.
 * - ::CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED: 1 if cudaArray can
 *   be accessed over the link.
 *
 * Returns ::CUDA_ERROR_INVALID_DEVICE if \p srcDevice or \p dstDevice are not valid
 * or if they represent the same device.
 *
 * Returns ::CUDA_ERROR_INVALID_VALUE if \p attrib is not valid or if \p value is
 * a null pointer.
 *
 * \param value         - Returned value of the requested attribute
 * \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
 * \param srcDevice     - The source device of the target link.
 * \param dstDevice     - The destination device of the target link.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::cuCtxEnablePeerAccess,
 * ::cuCtxDisablePeerAccess,
 * ::cuDeviceCanAccessPeer,
 * ::cudaDeviceGetP2PAttribute
 *)
function cuDeviceGetP2PAttribute(value :PLongint; attrib:CUdevice_P2PAttribute;  srcDevice:CUdevice;  dstDevice : CUdevice):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_PEER_ACCESS *)

(**
 * \defgroup CUDA_GRAPHICS Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Unregisters a graphics resource for access by CUDA
 *
 * Unregisters the graphics resource \p resource so it is not accessible by
 * CUDA unless registered again.
 *
 * If \p resource is invalid then ::CUDA_ERROR_INVALID_HANDLE is
 * returned.
 *
 * \param resource - Resource to unregister
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsD3D9RegisterResource,
 * ::cuGraphicsD3D10RegisterResource,
 * ::cuGraphicsD3D11RegisterResource,
 * ::cuGraphicsGLRegisterBuffer,
 * ::cuGraphicsGLRegisterImage,
 * ::cudaGraphicsUnregisterResource
 *)
function cuGraphicsUnregisterResource(resource : CUgraphicsResource):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * Returns in \p *pArray an array through which the subresource of the mapped
 * graphics resource \p resource which corresponds to array index \p arrayIndex
 * and mipmap level \p mipLevel may be accessed.  The value set in \p *pArray may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
 * If \p arrayIndex is not a valid array index for \p resource then
 * ::CUDA_ERROR_INVALID_VALUE is returned.
 * If \p mipLevel is not a valid mipmap level for \p resource then
 * ::CUDA_ERROR_INVALID_VALUE is returned.
 * If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
 *
 * \param pArray      - Returned array through which a subresource of \p resource may be accessed
 * \param resource    - Mapped resource to access
 * \param arrayIndex  - Array index for array textures or cubemap face
 *                      index as defined by ::CUarray_cubemap_face for
 *                      cubemap textures for the subresource to access
 * \param mipLevel    - Mipmap level for the subresource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED,
 * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsSubResourceGetMappedArray
 *)
function cuGraphicsSubResourceGetMappedArray(pArray : PCUarray; resource:CUgraphicsResource;  arrayIndex:longword;  mipLevel : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get a mipmapped array through which to access a mapped graphics resource.
 *
 * Returns in \p *pMipmappedArray a mipmapped array through which the mapped graphics
 * resource \p resource. The value set in \p *pMipmappedArray may change every time
 * that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via a mipmapped array and
 * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
 * If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
 *
 * \param pMipmappedArray - Returned mipmapped array through which \p resource may be accessed
 * \param resource        - Mapped resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED,
 * ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsResourceGetMappedMipmappedArray
 *)
function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray : PCUmipmappedArray; resource : CUgraphicsResource):CUresult;  WINAPI; external libcuda;

(**
 * \brief Get a device pointer through which to access a mapped graphics resource.
 *
 * Returns in \p *pDevPtr a pointer through which the mapped graphics resource
 * \p resource may be accessed.
 * Returns in \p pSize the size of the memory in bytes which may be accessed from that pointer.
 * The value set in \p pPointer may change every time that \p resource is mapped.
 *
 * If \p resource is not a buffer then it cannot be accessed via a pointer and
 * ::CUDA_ERROR_NOT_MAPPED_AS_POINTER is returned.
 * If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
 * *
 * \param pDevPtr    - Returned pointer through which \p resource may be accessed
 * \param pSize      - Returned size of the buffer accessible starting at \p *pPointer
 * \param resource   - Mapped resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED,
 * ::CUDA_ERROR_NOT_MAPPED_AS_POINTER
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsMapResources,
 * ::cuGraphicsSubResourceGetMappedArray,
 * ::cudaGraphicsResourceGetMappedPointer
 *)
function cuGraphicsResourceGetMappedPointer(pDevPtr : PCUdeviceptr; pSize : Psize_t; resource : CUgraphicsResource):CUresult;  WINAPI; external libcuda;

(**
 * \brief Set usage flags for mapping a graphics resource
 *
 * Set \p flags for mapping the graphics resource \p resource.
 *
 * Changes to \p flags will take effect the next time \p resource is mapped.
 * The \p flags argument may be any of the following:

 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA kernels.  This is the default value.
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which
 *   access this resource will not write to this resource.
 * - ::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels
 *   which access this resource will not read from this resource and will
 *   write over the entire contents of the resource, so none of the data
 *   previously stored in the resource will be preserved.
 *
 * If \p resource is presently mapped for access by CUDA then
 * ::CUDA_ERROR_ALREADY_MAPPED is returned.
 * If \p flags is not one of the above values then ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * \param resource - Registered resource to set flags for
 * \param flags    - Parameters for resource mapping
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsMapResources,
 * ::cudaGraphicsResourceSetMapFlags
 *)
function cuGraphicsResourceSetMapFlags(resource:CUgraphicsResource;  flags : longword):CUresult;  WINAPI; external libcuda;

(**
 * \brief Map graphics resources for access by CUDA
 *
 * Maps the \p count graphics resources in \p resources for access by CUDA.
 *
 * The resources in \p resources may be accessed by CUDA until they
 * are unmapped. The graphics API from which \p resources were registered
 * should not access any resources while they are mapped by CUDA. If an
 * application does so, the results are undefined.
 *
 * This function provides the synchronization guarantee that any graphics calls
 * issued before ::cuGraphicsMapResources() will complete before any subsequent CUDA
 * work issued in \p stream begins.
 *
 * If \p resources includes any duplicate entries then ::CUDA_ERROR_INVALID_HANDLE is returned.
 * If any of \p resources are presently mapped for access by CUDA then ::CUDA_ERROR_ALREADY_MAPPED is returned.
 *
 * \param count      - Number of resources to map
 * \param resources  - Resources to map for CUDA usage
 * \param hStream    - Stream with which to synchronize
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_UNKNOWN
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsResourceGetMappedPointer,
 * ::cuGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsUnmapResources,
 * ::cudaGraphicsMapResources
 *)
function cuGraphicsMapResources(count:longword;  resources : PCUgraphicsResource; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(**
 * \brief Unmap graphics resources.
 *
 * Unmaps the \p count graphics resources in \p resources.
 *
 * Once unmapped, the resources in \p resources may not be accessed by CUDA
 * until they are mapped again.
 *
 * This function provides the synchronization guarantee that any CUDA work issued
 * in \p stream before ::cuGraphicsUnmapResources() will complete before any
 * subsequently issued graphics work begins.
 *
 *
 * If \p resources includes any duplicate entries then ::CUDA_ERROR_INVALID_HANDLE is returned.
 * If any of \p resources are not presently mapped for access by CUDA then ::CUDA_ERROR_NOT_MAPPED is returned.
 *
 * \param count      - Number of resources to unmap
 * \param resources  - Resources to unmap
 * \param hStream    - Stream with which to synchronize
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED,
 * ::CUDA_ERROR_UNKNOWN
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsMapResources,
 * ::cudaGraphicsUnmapResources
 *)
function cuGraphicsUnmapResources(count:longword;  resources : PCUgraphicsResource; hStream : CUstream):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_GRAPHICS *)

(**
 * \defgroup CUDA_DRIVER_ENTRY_POINT Driver Entry Point Access 
 *
 * ___MANBRIEF___ driver entry point access functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the driver entry point access functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * \brief Returns the requested driver API function pointer
 *
 * Returns in \p **pfn the address of the CUDA driver function for the requested
 * CUDA version and flags.
 *
 * The CUDA version is specified as (1000 * major + 10 * minor), so CUDA 11.2
 * should be specified as 11020. For a requested driver symbol, if the specified
 * CUDA version is greater than or equal to the CUDA version in which the driver symbol
 * was introduced, this API will return the function pointer to the corresponding
 * versioned function.
 *
 * The pointer returned by the API should be cast to a function pointer matching the
 * requested driver function's definition in the API header file. The function pointer
 * typedef can be picked up from the corresponding typedefs header file. For example,
 * cudaTypedefs.h consists of function pointer typedefs for driver APIs defined in cuda.h.
 *
 * The API will return ::CUDA_SUCCESS and set the returned \p pfn to NULL if the 
 * requested driver function is not supported on the platform, no ABI 
 * compatible driver function exists for the specified \p cudaVersion or if the 
 * driver symbol is invalid.
 *
 * It will also set the optional \p symbolStatus to one of the values in
 * ::CUdriverProcAddressQueryResult with the following meanings:
 * - ::CU_GET_PROC_ADDRESS_SUCCESS - The requested symbol was succesfully found based
 *   on input arguments and \p pfn is valid
 * - ::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND - The requested symbol was not found
 * - ::CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT - The requested symbol was found but is
 *   not supported by cudaVersion specified
 *
 * The requested flags can be:
 * - ::CU_GET_PROC_ADDRESS_DEFAULT: This is the default mode. This is equivalent to
 *   ::CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM if the code is compiled with
 *   --default-stream per-thread compilation flag or the macro CUDA_API_PER_THREAD_DEFAULT_STREAM
 *   is defined; ::CU_GET_PROC_ADDRESS_LEGACY_STREAM otherwise.
 * - ::CU_GET_PROC_ADDRESS_LEGACY_STREAM: This will enable the search for all driver symbols
 *   that match the requested driver symbol name except the corresponding per-thread versions.
 * - ::CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM: This will enable the search for all
 *   driver symbols that match the requested driver symbol name including the per-thread
 *   versions. If a per-thread version is not found, the API will return the legacy version
 *   of the driver function.
 *
 * \param symbol - The base name of the driver API function to look for. As an example,
 *                 for the driver API ::cuMemAlloc_v2, \p symbol would be cuMemAlloc and
 *                 \p cudaVersion would be the ABI compatible CUDA version for the _v2 variant. 
 * \param pfn - Location to return the function pointer to the requested driver function
 * \param cudaVersion - The CUDA version to look for the requested driver symbol 
 * \param flags -  Flags to specify search options.
 * \param symbolStatus - Optional location to store the status of the search for
 *                       \p symbol based on \p cudaVersion. See ::CUdriverProcAddressQueryResult
 *                       for possible values.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_SUPPORTED
 * \note_version_mixing
 *
 * \sa
 * ::cudaGetDriverEntryPoint
 *)
function cuGetProcAddress(symbol:PAnsiChar;  pfn : Ppointer; cudaVersion:longint;  flags : cuuint64_t; symbolStatus : PCUdriverProcAddressQueryResult):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_DRIVER_ENTRY_POINT *)


(**
 * \defgroup CUDA_COREDUMP Coredump Attributes Control API
 *
 * ___MANBRIEF___ coredump attribute control functions for the low-level CUDA API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the coredump attribute control functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 *)

(**
 * Flags for choosing a coredump attribute to get/set
 *)
type
  CUcoredumpSettings_enum = (
    CU_COREDUMP_ENABLE_ON_EXCEPTION = 1,
    CU_COREDUMP_TRIGGER_HOST,
    CU_COREDUMP_LIGHTWEIGHT,
    CU_COREDUMP_ENABLE_USER_TRIGGER,
    CU_COREDUMP_FILE,
    CU_COREDUMP_PIPE,
    CU_COREDUMP_MAX
  );
  CUcoredumpSettings = CUcoredumpSettings_enum;

(**
 * \brief Allows caller to fetch a coredump attribute value for the current context
 *
 * Returns in \p *value the requested value specified by \p attrib. It is up to the caller
 * to ensure that the data type and size of \p *value matches the request.
 *
 * If the caller calls this function with \p *value equal to NULL, the size of the memory
 * region (in bytes) expected for \p attrib will be placed in \p size.
 *
 * The supported attributes are:
 * - ::CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where ::true means that GPU exceptions from
 *      this context will create a coredump at the location specified by ::CU_COREDUMP_FILE.
 *      The default value is ::false unless set to ::true globally or locally, or the
 *      CU_CTX_USER_COREDUMP_ENABLE flag was set during context creation.
 * - ::CU_COREDUMP_TRIGGER_HOST: Bool where ::true means that the host CPU will
 *      also create a coredump. The default value is ::true unless set to ::false globally or
 *      or locally.
 * - ::CU_COREDUMP_LIGHTWEIGHT: Bool where ::true means that any resulting coredumps
 *      will not have a dump of GPU memory or non-reloc ELF images. The default value is
 *      ::false unless set to ::true globally or locally.
 * - ::CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where ::true means that a coredump can be
 *      created by writing to the system pipe specified by ::CU_COREDUMP_PIPE. The default
 *      value is ::false unless set to ::true globally or locally.
 * - ::CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where
 *      any coredumps generated by this context will be written. The default value is
 *      ::core.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine running
 *      the CUDA applications and ::PID is the process ID of the CUDA application.
 * - ::CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe
 *      that will be monitored if user-triggered coredumps are enabled. The default value is
 *      ::corepipe.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine running
 *      the CUDA application and ::PID is the process ID of the CUDA application.
 *
 * \param attrib - The enum defining which value to fetch.
 * \param value - pointer  containing the requested data.
 * \param size - The size of the memory region \p value points to.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED
 *
 * \sa
 * ::cuCoredumpGetAttributeGlobal,
 * ::cuCoredumpSetAttribute,
 * ::cuCoredumpSetAttributeGlobal
 *)
function cuCoredumpGetAttribute(attrib:CUcoredumpSettings;  value : pointer; size : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allows caller to fetch a coredump attribute value for the entire application
 *
 * Returns in \p *value the requested value specified by \p attrib. It is up to the caller
 * to ensure that the data type and size of \p *value matches the request.
 *
 * If the caller calls this function with \p *value equal to NULL, the size of the memory
 * region (in bytes) expected for \p attrib will be placed in \p size.
 *
 * The supported attributes are:
 * - ::CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where ::true means that GPU exceptions from
 *      this context will create a coredump at the location specified by ::CU_COREDUMP_FILE.
 *      The default value is ::false.
 * - ::CU_COREDUMP_TRIGGER_HOST: Bool where ::true means that the host CPU will
 *      also create a coredump. The default value is ::true.
 * - ::CU_COREDUMP_LIGHTWEIGHT: Bool where ::true means that any resulting coredumps
 *      will not have a dump of GPU memory or non-reloc ELF images. The default value is
 *      ::false.
 * - ::CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where ::true means that a coredump can be
 *      created by writing to the system pipe specified by ::CU_COREDUMP_PIPE. The default
 *      value is ::false.
 * - ::CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where
 *      any coredumps generated by this context will be written. The default value is
 *      ::core.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine running
 *      the CUDA applications and ::PID is the process ID of the CUDA application.
 * - ::CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe
 *      that will be monitored if user-triggered coredumps are enabled. The default value is
 *      ::corepipe.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine running
 *      the CUDA application and ::PID is the process ID of the CUDA application.
 *
 * \param attrib - The enum defining which value to fetch.
 * \param value - pointer  containing the requested data.
 * \param size - The size of the memory region \p value points to.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE
 *
 * \sa
 * ::cuCoredumpGetAttribute,
 * ::cuCoredumpSetAttribute,
 * ::cuCoredumpSetAttributeGlobal
 *)
function cuCoredumpGetAttributeGlobal(attrib:CUcoredumpSettings;  value : Pointer; size : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allows caller to set a coredump attribute value for the current context
 *
 * This function should be considered an alternate interface to the CUDA-GDB environment
 * variables defined in this document: https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump
 *
 * An important design decision to note is that any coredump environment variable values
 * set before CUDA initializes will take permanent precedence over any values set with this
 * this function. This decision was made to ensure no change in behavior for any users that
 * may be currently using these variables to get coredumps.
 *
 * \p *value shall contain the requested value specified by \p set. It is up to the caller
 * to ensure that the data type and size of \p *value matches the request.
 *
 * If the caller calls this function with \p *value equal to NULL, the size of the memory
 * region (in bytes) expected for \p set will be placed in \p size.
 *
 * /note This function will return ::CUDA_ERROR_NOT_SUPPORTED if the caller attempts to set
 * ::CU_COREDUMP_ENABLE_ON_EXCEPTION on a GPU of with Compute Capability < 6.0. ::cuCoredumpSetAttributeGlobal
 * works on those platforms as an alternative.
 *
 * /note ::CU_COREDUMP_ENABLE_USER_TRIGGER and ::CU_COREDUMP_PIPE cannot be set on a per-context basis.
 *
 * The supported attributes are:
 * - ::CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where ::true means that GPU exceptions from
 *      this context will create a coredump at the location specified by ::CU_COREDUMP_FILE.
 *      The default value is ::false.
 * - ::CU_COREDUMP_TRIGGER_HOST: Bool where ::true means that the host CPU will
 *      also create a coredump. The default value is ::true.
 * - ::CU_COREDUMP_LIGHTWEIGHT: Bool where ::true means that any resulting coredumps
 *      will not have a dump of GPU memory or non-reloc ELF images. The default value is
 *      ::false.
 * - ::CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where
 *      any coredumps generated by this context will be written. The default value is
 *      ::core.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine running
 *      the CUDA applications and ::PID is the process ID of the CUDA application.
 *
 * \param attrib - The enum defining which value to set.
 * \param value - pointer  containing the requested data.
 * \param size - The size of the memory region \p value points to.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_PERMITTED,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_CONTEXT_IS_DESTROYED,
 * ::CUDA_ERROR_NOT_SUPPORTED
 *
 * \sa
 * ::cuCoredumpGetAttributeGlobal,
 * ::cuCoredumpGetAttribute,
 * ::cuCoredumpSetAttributeGlobal
 *)
function cuCoredumpSetAttribute(attrib:CUcoredumpSettings;  value : pointer; size : Psize_t):CUresult;  WINAPI; external libcuda;

(**
 * \brief Allows caller to set a coredump attribute value globally
 *
 * This function should be considered an alternate interface to the CUDA-GDB environment
 * variables defined in this document: https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump
 *
 * An important design decision to note is that any coredump environment variable values
 * set before CUDA initializes will take permanent precedence over any values set with this
 * this function. This decision was made to ensure no change in behavior for any users that
 * may be currently using these variables to get coredumps.
 *
 * \p *value shall contain the requested value specified by \p set. It is up to the caller
 * to ensure that the data type and size of \p *value matches the request.
 *
 * If the caller calls this function with \p *value equal to NULL, the size of the memory
 * region (in bytes) expected for \p set will be placed in \p size.
 *
 * The supported attributes are:
 * - ::CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where ::true means that GPU exceptions from
 *      this context will create a coredump at the location specified by ::CU_COREDUMP_FILE.
 *      The default value is ::false.
 * - ::CU_COREDUMP_TRIGGER_HOST: Bool where ::true means that the host CPU will
 *      also create a coredump. The default value is ::true.
 * - ::CU_COREDUMP_LIGHTWEIGHT: Bool where ::true means that any resulting coredumps
 *      will not have a dump of GPU memory or non-reloc ELF images. The default value is
 *      ::false.
 * - ::CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where ::true means that a coredump can be
 *      created by writing to the system pipe specified by ::CU_COREDUMP_PIPE. The default
 *      value is ::false.
 * - ::CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where
 *      any coredumps generated by this context will be written. The default value is
 *      ::core.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine running
 *      the CUDA applications and ::PID is the process ID of the CUDA application.
 * - ::CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe
 *      that will be monitored if user-triggered coredumps are enabled. This value may not be
 *      changed after ::CU_COREDUMP_ENABLE_USER_TRIGGER is set to ::true. The default
 *      value is ::corepipe.cuda.HOSTNAME.PID where ::HOSTNAME is the host name of the machine
 *      running the CUDA application and ::PID is the process ID of the CUDA application.
 *
 * \param attrib - The enum defining which value to set.
 * \param value - pointer  containing the requested data.
 * \param size - The size of the memory region \p value points to.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_PERMITTED
 *
 * \sa
 * ::cuCoredumpGetAttribute,
 * ::cuCoredumpGetAttributeGlobal,
 * ::cuCoredumpSetAttribute
 *)
function cuCoredumpSetAttributeGlobal(attrib:CUcoredumpSettings;  value : Pointer; size : Psize_t):CUresult;  WINAPI; external libcuda;

(** @} *) (* END CUDA_COREDUMP *)

function cuGetExportTable(const ppExportTable : Ppointer; const pExportTableId : PCUuuid):CUresult;  WINAPI; external libcuda;

(**
 * CUDA API versioning support
 *)
 (*
 static inline CUresult cuGetProcAddress_v2_ptsz(const PAnsiChar symbol, funcPtr : Ppointer, longint driverVersion, cuuint64_t flags, symbolStatus : PCUdriverProcAddressQueryResult) {
    const longint procAddressMask = (CU_GET_PROC_ADDRESS_LEGACY_STREAM|
                                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
    if ((flags & procAddressMask) == 0) {
        flags |= CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
    }
    return cuGetProcAddress_v2(symbol, funcPtr, driverVersion, flags, symbolStatus); 
}
*)

implementation

end.
