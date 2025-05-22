unit cudarttypes;
interface
{
  Automatically converted by H2Pas 1.0.0 from driver_types.h
  The following command line parameters were used:
    driver_types.h
}

{$IFDEF FPC}
{$mode delphi}
{$PACKRECORDS C}
{$ModeSwitch advancedrecords}
{$ENDIF}
//{$PackEnum 4}

  {
   * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
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
    }
//{$include "vector_types.h"}
  {*
   * \defgroup CUDART_TYPES Data types used by CUDA Runtime
   * \ingroup CUDART
   *
   * @
    }
  {******************************************************************************
  *                                                                              *
  *  TYPE DEFINITIONS USED BY RUNTIME API                                        *
  *                                                                              *
  ****************************************************************************** }
  {*< Default page-locked allocation flag  }

  const
    cudaHostAllocDefault = $00;    
  {*< Pinned memory accessible by all CUDA contexts  }
    cudaHostAllocPortable = $01;    
  {*< Map allocation into device space  }
    cudaHostAllocMapped = $02;    
  {*< Write-combined memory  }
    cudaHostAllocWriteCombined = $04;    
  {*< Default host memory registration flag  }
    cudaHostRegisterDefault = $00;    
  {*< Pinned memory accessible by all CUDA contexts  }
    cudaHostRegisterPortable = $01;    
  {*< Map registered memory into device space  }
    cudaHostRegisterMapped = $02;    
  {*< Memory-mapped I/O space  }
    cudaHostRegisterIoMemory = $04;    
  {*< Memory-mapped read-only  }
    cudaHostRegisterReadOnly = $08;    
  {*< Default peer addressing enable flag  }
    cudaPeerAccessDefault = $00;    
  {*< Default stream flag  }
    cudaStreamDefault = $00;    
  {*< Stream does not synchronize with stream 0 (the NULL stream)  }
    cudaStreamNonBlocking = $01;    
  {*
   * Legacy stream handle
   *
   * Stream handle that can be passed as a cudaStream_t to use an implicit stream
   * with legacy synchronization behavior.
   *
   * See details of the \link_sync_behavior
    }
    cudaStreamLegacy = $1;    
  {*
   * Per-thread stream handle
   *
   * Stream handle that can be passed as a cudaStream_t to use an implicit stream
   * with per-thread synchronization behavior.
   *
   * See details of the \link_sync_behavior
    }
    cudaStreamPerThread = $2;    
  {*< Default event flag  }
    cudaEventDefault = $00;    
  {*< Event uses blocking synchronization  }
    cudaEventBlockingSync = $01;    
  {*< Event will not record timing data  }
    cudaEventDisableTiming = $02;    
  {*< Event is suitable for interprocess use. cudaEventDisableTiming must be set  }
    cudaEventInterprocess = $04;    
  {*< Default event record flag  }
    cudaEventRecordDefault = $00;    
  {*< Event is captured in the graph as an external event node when performing stream capture  }
    cudaEventRecordExternal = $01;    
  {*< Default event wait flag  }
    cudaEventWaitDefault = $00;    
  {*< Event is captured in the graph as an external event node when performing stream capture  }
    cudaEventWaitExternal = $01;    
  {*< Device flag - Automatic scheduling  }
    cudaDeviceScheduleAuto = $00;    
  {*< Device flag - Spin default scheduling  }
    cudaDeviceScheduleSpin = $01;    
  {*< Device flag - Yield default scheduling  }
    cudaDeviceScheduleYield = $02;    
  {*< Device flag - Use blocking synchronization  }
    cudaDeviceScheduleBlockingSync = $04;    
  {*< Device flag - Use blocking synchronization 
                                                      *  \deprecated This flag was deprecated as of CUDA 4.0 and
                                                      *  replaced with ::cudaDeviceScheduleBlockingSync.  }
    cudaDeviceBlockingSync = $04;    
  {*< Device schedule flags mask  }
    cudaDeviceScheduleMask = $07;    
  {*< Device flag - Support mapped pinned allocations  }
    cudaDeviceMapHost = $08;    
  {*< Device flag - Keep local memory allocation after launch  }
    cudaDeviceLmemResizeToMax = $10;    
  {*< Device flag - Use synchronous behavior for cudaMemcpy/cudaMemset  }
    cudaDeviceSyncMemops = $80;    
  {*< Device flags mask  }
    cudaDeviceMask = $ff;    
  {*< Default CUDA array allocation flag  }
    cudaArrayDefault = $00;    
  {*< Must be set in cudaMalloc3DArray to create a layered CUDA array  }
    cudaArrayLayered = $01;    
  {*< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array  }
    cudaArraySurfaceLoadStore = $02;    
  {*< Must be set in cudaMalloc3DArray to create a cubemap CUDA array  }
    cudaArrayCubemap = $04;    
  {*< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array  }
    cudaArrayTextureGather = $08;    
  {*< Must be set in cudaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API  }
    cudaArrayColorAttachment = $20;    
  {*< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a sparse CUDA array or CUDA mipmapped array  }
    cudaArraySparse = $40;    
  {*< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a deferred mapping CUDA array or CUDA mipmapped array  }
    cudaArrayDeferredMapping = $80;    
  {*< Automatically enable peer access between remote devices as needed  }
    cudaIpcMemLazyEnablePeerAccess = $01;    
  {*< Memory can be accessed by any stream on any device }
    cudaMemAttachGlobal = $01;    
  {*< Memory cannot be accessed by any stream on any device  }
    cudaMemAttachHost = $02;    
  {*< Memory can only be accessed by a single stream on the associated device  }
    cudaMemAttachSingle = $04;    
  {*< Default behavior  }
    cudaOccupancyDefault = $00;    
  {*< Assume global caching is enabled and cannot be automatically turned off  }
    cudaOccupancyDisableCachingOverride = $01;    
  {*< Device id that represents the CPU  }
    cudaCpuDeviceId = -(1);    
  {*< Device id that represents an invalid device  }
    cudaInvalidDeviceId = -(2);    
  {*< Tell the CUDA runtime that DeviceFlags is being set in cudaInitDevice call  }
    cudaInitDeviceFlagsAreValid = $01;    
  {*
   * If set, each kernel launched as part of ::cudaLaunchCooperativeKernelMultiDevice only
   * waits for prior work in the stream corresponding to that GPU to complete before the
   * kernel begins execution.
    }
    cudaCooperativeLaunchMultiDeviceNoPreSync = $01;    
  {*
   * If set, any subsequent work pushed in a stream that participated in a call to
   * ::cudaLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
   * the GPU corresponding to that stream to complete before it begins execution.
    }
    cudaCooperativeLaunchMultiDeviceNoPostSync = $02;    
  {******************************************************************************
  *                                                                              *
  *                                                                              *
  *                                                                              *
  ****************************************************************************** }
  {*
   * CUDA error types
    }
  {*
       * The API call returned with no errors. In the case of query calls, this
       * also means that the operation being queried is complete (see
       * ::cudaEventQuery() and ::cudaStreamQuery()).
        }
  {*
       * This indicates that one or more of the parameters passed to the API call
       * is not within an acceptable range of values.
        }
  {*
       * The API call failed because it was unable to allocate enough memory to
       * perform the requested operation.
        }
  {*
       * The API call failed because the CUDA driver and runtime could not be
       * initialized.
        }
  {*
       * This indicates that a CUDA Runtime API call cannot be executed because
       * it is being called during process shut down, at a point in time after
       * CUDA driver has been unloaded.
        }
  {*
       * This indicates profiler is not initialized for this run. This can
       * happen when the application is running with external profiling tools
       * like visual profiler.
        }
  {*
       * \deprecated
       * This error return is deprecated as of CUDA 5.0. It is no longer an error
       * to attempt to enable/disable the profiling via ::cudaProfilerStart or
       * ::cudaProfilerStop without initialization.
        }
  {*
       * \deprecated
       * This error return is deprecated as of CUDA 5.0. It is no longer an error
       * to call cudaProfilerStart() when profiling is already enabled.
        }
  {*
       * \deprecated
       * This error return is deprecated as of CUDA 5.0. It is no longer an error
       * to call cudaProfilerStop() when profiling is already disabled.
        }
  {*
       * This indicates that a kernel launch is requesting resources that can
       * never be satisfied by the current device. Requesting more shared memory
       * per block than the device supports will trigger this error, as will
       * requesting too many threads or blocks. See ::cudaDeviceProp for more
       * device limitations.
        }
  {*
       * This indicates that one or more of the pitch-related parameters passed
       * to the API call is not within the acceptable range for pitch.
        }
  {*
       * This indicates that the symbol name/identifier passed to the API call
       * is not a valid name or identifier.
        }
  {*
       * This indicates that at least one host pointer passed to the API call is
       * not a valid host pointer.
       * \deprecated
       * This error return is deprecated as of CUDA 10.1.
        }
  {*
       * This indicates that at least one device pointer passed to the API call is
       * not a valid device pointer.
       * \deprecated
       * This error return is deprecated as of CUDA 10.1.
        }
  {*
       * This indicates that the texture passed to the API call is not a valid
       * texture.
        }
  {*
       * This indicates that the texture binding is not valid. This occurs if you
       * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
        }
  {*
       * This indicates that the channel descriptor passed to the API call is not
       * valid. This occurs if the format is not one of the formats specified by
       * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
        }
  {*
       * This indicates that the direction of the memcpy passed to the API call is
       * not one of the types specified by ::cudaMemcpyKind.
        }
  {*
       * This indicated that the user has taken the address of a constant variable,
       * which was forbidden up until the CUDA 3.1 release.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Variables in constant
       * memory may now have their address taken by the runtime via
       * ::cudaGetSymbolAddress().
        }
  {*
       * This indicated that a texture fetch was not able to be performed.
       * This was previously used for device emulation of texture operations.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Device emulation mode was
       * removed with the CUDA 3.1 release.
        }
  {*
       * This indicated that a texture was not bound for access.
       * This was previously used for device emulation of texture operations.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Device emulation mode was
       * removed with the CUDA 3.1 release.
        }
  {*
       * This indicated that a synchronization operation had failed.
       * This was previously used for some device emulation functions.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Device emulation mode was
       * removed with the CUDA 3.1 release.
        }
  {*
       * This indicates that a non-float texture was being accessed with linear
       * filtering. This is not supported by CUDA.
        }
  {*
       * This indicates that an attempt was made to read a non-float texture as a
       * normalized float. This is not supported by CUDA.
        }
  {*
       * Mixing of device and device emulation code was not allowed.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Device emulation mode was
       * removed with the CUDA 3.1 release.
        }
  {*
       * This indicates that the API call is not yet implemented. Production
       * releases of CUDA will never return this error.
       * \deprecated
       * This error return is deprecated as of CUDA 4.1.
        }
  {*
       * This indicated that an emulated device pointer exceeded the 32-bit address
       * range.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Device emulation mode was
       * removed with the CUDA 3.1 release.
        }
  {*
       * This indicates that the CUDA driver that the application has loaded is a
       * stub library. Applications that run with the stub rather than a real
       * driver loaded will result in CUDA API returning this error.
        }
  {*
       * This indicates that the installed NVIDIA CUDA driver is older than the
       * CUDA runtime library. This is not a supported configuration. Users should
       * install an updated NVIDIA display driver to allow the application to run.
        }
  {*
       * This indicates that the API call requires a newer CUDA driver than the one
       * currently installed. Users should install an updated NVIDIA CUDA driver
       * to allow the API call to succeed.
        }
  {*
       * This indicates that the surface passed to the API call is not a valid
       * surface.
        }
  {*
       * This indicates that multiple global or constant variables (across separate
       * CUDA source files in the application) share the same string name.
        }
  {*
       * This indicates that multiple textures (across separate CUDA source
       * files in the application) share the same string name.
        }
  {*
       * This indicates that multiple surfaces (across separate CUDA source
       * files in the application) share the same string name.
        }
  {*
       * This indicates that all CUDA devices are busy or unavailable at the current
       * time. Devices are often busy/unavailable due to use of
       * ::cudaComputeModeProhibited, ::cudaComputeModeExclusiveProcess, or when long
       * running CUDA kernels have filled up the GPU and are blocking new work
       * from starting. They can also be unavailable due to memory constraints
       * on a device that already has active CUDA work being performed.
        }
  {*
       * This indicates that the current context is not compatible with this
       * the CUDA Runtime. This can only occur if you are using CUDA
       * Runtime/Driver interoperability and have created an existing Driver
       * context using the driver API. The Driver context may be incompatible
       * either because the Driver context was created using an older version 
       * of the API, because the Runtime API call expects a primary driver 
       * context and the Driver context is not primary, or because the Driver 
       * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
       * with the CUDA Driver API" for more information.
        }
  {*
       * The device function being invoked (usually via ::cudaLaunchKernel()) was not
       * previously configured via the ::cudaConfigureCall() function.
        }
  {*
       * This indicated that a previous kernel launch failed. This was previously
       * used for device emulation of kernel launches.
       * \deprecated
       * This error return is deprecated as of CUDA 3.1. Device emulation mode was
       * removed with the CUDA 3.1 release.
        }
  {*
       * This error indicates that a device runtime grid launch did not occur 
       * because the depth of the child grid would exceed the maximum supported
       * number of nested grid launches. 
        }
  {*
       * This error indicates that a grid launch did not occur because the kernel 
       * uses file-scoped textures which are unsupported by the device runtime. 
       * Kernels launched via the device runtime only support textures created with 
       * the Texture Object API's.
        }
  {*
       * This error indicates that a grid launch did not occur because the kernel 
       * uses file-scoped surfaces which are unsupported by the device runtime.
       * Kernels launched via the device runtime only support surfaces created with
       * the Surface Object API's.
        }
  {*
       * This error indicates that a call to ::cudaDeviceSynchronize made from
       * the device runtime failed because the call was made at grid depth greater
       * than than either the default (2 levels of grids) or user specified device
       * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
       * launched grids at a greater depth successfully, the maximum nested
       * depth at which ::cudaDeviceSynchronize will be called must be specified
       * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
       * api before the host-side launch of a kernel using the device runtime.
       * Keep in mind that additional levels of sync depth require the runtime
       * to reserve large amounts of device memory that cannot be used for
       * user allocations. Note that ::cudaDeviceSynchronize made from device
       * runtime is only supported on devices of compute capability < 9.0.
        }
  {*
       * This error indicates that a device runtime grid launch failed because
       * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
       * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
       * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
       * than the upper bound of outstanding launches that can be issued to the
       * device runtime. Keep in mind that raising the limit of pending device
       * runtime launches will require the runtime to reserve device memory that
       * cannot be used for user allocations.
        }
  {*
       * The requested device function does not exist or is not compiled for the
       * proper device architecture.
        }
  {*
       * This indicates that no CUDA-capable devices were detected by the installed
       * CUDA driver.
        }
  {*
       * This indicates that the device ordinal supplied by the user does not
       * correspond to a valid CUDA device or that the action requested is
       * invalid for the specified device.
        }
  {*
       * This indicates that the device doesn't have a valid Grid License.
        }
  {*
      * By default, the CUDA runtime may perform a minimal set of self-tests,
      * as well as CUDA driver tests, to establish the validity of both.
      * Introduced in CUDA 11.2, this error return indicates that at least one
      * of these tests has failed and the validity of either the runtime
      * or the driver could not be established.
       }
  {*
       * This indicates an internal startup failure in the CUDA runtime.
        }
  {*
       * This indicates that the device kernel image is invalid.
        }
  {*
       * This most frequently indicates that there is no context bound to the
       * current thread. This can also be returned if the context passed to an
       * API call is not a valid handle (such as a context that has had
       * ::cuCtxDestroy() invoked on it). This can also be returned if a user
       * mixes different API versions (i.e. 3010 context with 3020 API calls).
       * See ::cuCtxGetApiVersion() for more details.
        }
  {*
       * This indicates that the buffer object could not be mapped.
        }
  {*
       * This indicates that the buffer object could not be unmapped.
        }
  {*
       * This indicates that the specified array is currently mapped and thus
       * cannot be destroyed.
        }
  {*
       * This indicates that the resource is already mapped.
        }
  {*
       * This indicates that there is no kernel image available that is suitable
       * for the device. This can occur when a user specifies code generation
       * options for a particular CUDA source file that do not include the
       * corresponding device configuration.
        }
  {*
       * This indicates that a resource has already been acquired.
        }
  {*
       * This indicates that a resource is not mapped.
        }
  {*
       * This indicates that a mapped resource is not available for access as an
       * array.
        }
  {*
       * This indicates that a mapped resource is not available for access as a
       * pointer.
        }
  {*
       * This indicates that an uncorrectable ECC error was detected during
       * execution.
        }
  {*
       * This indicates that the ::cudaLimit passed to the API call is not
       * supported by the active device.
        }
  {*
       * This indicates that a call tried to access an exclusive-thread device that 
       * is already in use by a different thread.
        }
  {*
       * This error indicates that P2P access is not supported across the given
       * devices.
        }
  {*
       * A PTX compilation failed. The runtime may fall back to compiling PTX if
       * an application does not contain a suitable binary for the current device.
        }
  {*
       * This indicates an error with the OpenGL or DirectX context.
        }
  {*
       * This indicates that an uncorrectable NVLink error was detected during the
       * execution.
        }
  {*
       * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
       * library is used for PTX compilation. The runtime may fall back to compiling PTX
       * if an application does not contain a suitable binary for the current device.
        }
  {*
       * This indicates that the provided PTX was compiled with an unsupported toolchain.
       * The most common reason for this, is the PTX was generated by a compiler newer
       * than what is supported by the CUDA driver and PTX JIT compiler.
        }
  {*
       * This indicates that the JIT compilation was disabled. The JIT compilation compiles
       * PTX. The runtime may fall back to compiling PTX if an application does not contain
       * a suitable binary for the current device.
        }
  {*
       * This indicates that the provided execution affinity is not supported by the device.
        }
  {*
       * This indicates that the code to be compiled by the PTX JIT contains
       * unsupported call to cudaDeviceSynchronize.
        }
  {*
       * This indicates that the device kernel source is invalid.
        }
  {*
       * This indicates that the file specified was not found.
        }
  {*
       * This indicates that a link to a shared object failed to resolve.
        }
  {*
       * This indicates that initialization of a shared object failed.
        }
  {*
       * This error indicates that an OS call failed.
        }
  {*
       * This indicates that a resource handle passed to the API call was not
       * valid. Resource handles are opaque types like ::cudaStream_t and
       * ::cudaEvent_t.
        }
  {*
       * This indicates that a resource required by the API call is not in a
       * valid state to perform the requested operation.
        }
  {*
       * This indicates that a named symbol was not found. Examples of symbols
       * are global/constant variable names, driver function names, texture names,
       * and surface names.
        }
  {*
       * This indicates that asynchronous operations issued previously have not
       * completed yet. This result is not actually an error, but must be indicated
       * differently than ::cudaSuccess (which indicates completion). Calls that
       * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
        }
  {*
       * The device encountered a load or store instruction on an invalid memory address.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * This indicates that a launch did not occur because it did not have
       * appropriate resources. Although this error is similar to
       * ::cudaErrorInvalidConfiguration, this error usually indicates that the
       * user has attempted to pass too many arguments to the device kernel, or the
       * kernel launch specifies too many threads for the kernel's register count.
        }
  {*
       * This indicates that the device kernel took too long to execute. This can
       * only occur if timeouts are enabled - see the device property
       * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
       * for more information.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * This error indicates a kernel launch that uses an incompatible texturing
       * mode.
        }
  {*
       * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
       * trying to re-enable peer addressing on from a context which has already
       * had peer addressing enabled.
        }
  {*
       * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
       * disable peer addressing which has not been enabled yet via 
       * ::cudaDeviceEnablePeerAccess().
        }
  {*
       * This indicates that the user has called ::cudaSetValidDevices(),
       * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
       * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
       * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
       * calling non-device management operations (allocating memory and
       * launching kernels are examples of non-device management operations).
       * This error can also be returned if using runtime/driver
       * interoperability and there is an existing ::CUcontext active on the
       * host thread.
        }
  {*
       * This error indicates that the context current to the calling thread
       * has been destroyed using ::cuCtxDestroy, or is a primary context which
       * has not yet been initialized.
        }
  {*
       * An assert triggered in device code during kernel execution. The device
       * cannot be used again. All existing allocations are invalid. To continue
       * using CUDA, the process must be terminated and relaunched.
        }
  {*
       * This error indicates that the hardware resources required to enable
       * peer access have been exhausted for one or more of the devices 
       * passed to ::cudaEnablePeerAccess().
        }
  {*
       * This error indicates that the memory range passed to ::cudaHostRegister()
       * has already been registered.
        }
  {*
       * This error indicates that the pointer passed to ::cudaHostUnregister()
       * does not correspond to any currently registered memory region.
        }
  {*
       * Device encountered an error in the call stack during kernel execution,
       * possibly due to stack corruption or exceeding the stack size limit.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * The device encountered an illegal instruction during kernel execution
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * The device encountered a load or store instruction
       * on a memory address which is not aligned.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * While executing a kernel, the device encountered an instruction
       * which can only operate on memory locations in certain address spaces
       * (global, shared, or local), but was supplied a memory address not
       * belonging to an allowed address space.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * The device encountered an invalid program counter.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * An exception occurred on the device while executing a kernel. Common
       * causes include dereferencing an invalid device pointer and accessing
       * out of bounds shared memory. Less common cases can be system specific - more
       * information about these cases can be found in the system specific user guide.
       * This leaves the process in an inconsistent state and any further CUDA work
       * will return the same error. To continue using CUDA, the process must be terminated
       * and relaunched.
        }
  {*
       * This error indicates that the number of blocks launched per grid for a kernel that was
       * launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
       * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
       * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
       * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
        }
  {*
       * This error indicates the attempted operation is not permitted.
        }
  {*
       * This error indicates the attempted operation is not supported
       * on the current system or device.
        }
  {*
       * This error indicates that the system is not yet ready to start any CUDA
       * work.  To continue using CUDA, verify the system configuration is in a
       * valid state and all required driver daemons are actively running.
       * More information about this error can be found in the system specific
       * user guide.
        }
  {*
       * This error indicates that there is a mismatch between the versions of
       * the display driver and the CUDA driver. Refer to the compatibility documentation
       * for supported versions.
        }
  {*
       * This error indicates that the system was upgraded to run with forward compatibility
       * but the visible hardware detected by CUDA does not support this configuration.
       * Refer to the compatibility documentation for the supported hardware matrix or ensure
       * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
       * environment variable.
        }
  {*
       * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
        }
  {*
       * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
        }
  {*
       * This error indicates that the MPS server is not ready to accept new MPS client requests.
       * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
        }
  {*
       * This error indicates that the hardware resources required to create MPS client have been exhausted.
        }
  {*
       * This error indicates the the hardware resources required to device connections have been exhausted.
        }
  {*
       * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
        }
  {*
       * This error indicates, that the program is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
        }
  {*
       * This error indicates, that the program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
        }
  {*
       * The operation is not permitted when the stream is capturing.
        }
  {*
       * The current capture sequence on the stream has been invalidated due to
       * a previous error.
        }
  {*
       * The operation would have resulted in a merge of two independent capture
       * sequences.
        }
  {*
       * The capture was not initiated in this stream.
        }
  {*
       * The capture sequence contains a fork that was not joined to the primary
       * stream.
        }
  {*
       * A dependency would have been created which crosses the capture sequence
       * boundary. Only implicit in-stream ordering dependencies are allowed to
       * cross the boundary.
        }
  {*
       * The operation would have resulted in a disallowed implicit dependency on
       * a current capture sequence from cudaStreamLegacy.
        }
  {*
       * The operation is not permitted on an event which was last recorded in a
       * capturing stream.
        }
  {*
       * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
       * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
       * different thread.
        }
  {*
       * This indicates that the wait operation has timed out.
        }
  {*
       * This error indicates that the graph update was not performed because it included 
       * changes which violated constraints specific to instantiated graph update.
        }
  {*
       * This indicates that an async error has occurred in a device outside of CUDA.
       * If CUDA was waiting for an external device's signal before consuming shared data,
       * the external device signaled an error indicating that the data is not valid for
       * consumption. This leaves the process in an inconsistent state and any further CUDA
       * work will return the same error. To continue using CUDA, the process must be
       * terminated and relaunched.
        }
  {*
       * This indicates that a kernel launch error has occurred due to cluster
       * misconfiguration.
        }
  {*
       * This indicates that an unknown internal error has occurred.
        }
  {*
       * Any unhandled CUDA driver error is added to this value and returned via
       * the runtime. Production releases of CUDA should not return such errors.
       * \deprecated
       * This error return is deprecated as of CUDA 4.1.
        }

  type
    cudaError = (cudaSuccess = 0,cudaErrorInvalidValue = 1,
      cudaErrorMemoryAllocation = 2,cudaErrorInitializationError = 3,
      cudaErrorCudartUnloading = 4,cudaErrorProfilerDisabled = 5,
      cudaErrorProfilerNotInitialized = 6,
      cudaErrorProfilerAlreadyStarted = 7,
      cudaErrorProfilerAlreadyStopped = 8,
      cudaErrorInvalidConfiguration = 9,
      cudaErrorInvalidPitchValue = 12,cudaErrorInvalidSymbol = 13,
      cudaErrorInvalidHostPointer = 16,cudaErrorInvalidDevicePointer = 17,
      cudaErrorInvalidTexture = 18,cudaErrorInvalidTextureBinding = 19,
      cudaErrorInvalidChannelDescriptor = 20,
      cudaErrorInvalidMemcpyDirection = 21,
      cudaErrorAddressOfConstant = 22,cudaErrorTextureFetchFailed = 23,
      cudaErrorTextureNotBound = 24,cudaErrorSynchronizationError = 25,
      cudaErrorInvalidFilterSetting = 26,
      cudaErrorInvalidNormSetting = 27,cudaErrorMixedDeviceExecution = 28,
      cudaErrorNotYetImplemented = 31,cudaErrorMemoryValueTooLarge = 32,
      cudaErrorStubLibrary = 34,cudaErrorInsufficientDriver = 35,
      cudaErrorCallRequiresNewerDriver = 36,
      cudaErrorInvalidSurface = 37,cudaErrorDuplicateVariableName = 43,
      cudaErrorDuplicateTextureName = 44,
      cudaErrorDuplicateSurfaceName = 45,
      cudaErrorDevicesUnavailable = 46,cudaErrorIncompatibleDriverContext = 49,
      cudaErrorMissingConfiguration = 52,
      cudaErrorPriorLaunchFailure = 53,cudaErrorLaunchMaxDepthExceeded = 65,
      cudaErrorLaunchFileScopedTex = 66,cudaErrorLaunchFileScopedSurf = 67,
      cudaErrorSyncDepthExceeded = 68,cudaErrorLaunchPendingCountExceeded = 69,
      cudaErrorInvalidDeviceFunction = 98,
      cudaErrorNoDevice = 100,cudaErrorInvalidDevice = 101,
      cudaErrorDeviceNotLicensed = 102,cudaErrorSoftwareValidityNotEstablished = 103,
      cudaErrorStartupFailure = 127,cudaErrorInvalidKernelImage = 200,
      cudaErrorDeviceUninitialized = 201,cudaErrorMapBufferObjectFailed = 205,
      cudaErrorUnmapBufferObjectFailed = 206,
      cudaErrorArrayIsMapped = 207,cudaErrorAlreadyMapped = 208,
      cudaErrorNoKernelImageForDevice = 209,
      cudaErrorAlreadyAcquired = 210,cudaErrorNotMapped = 211,
      cudaErrorNotMappedAsArray = 212,cudaErrorNotMappedAsPointer = 213,
      cudaErrorECCUncorrectable = 214,cudaErrorUnsupportedLimit = 215,
      cudaErrorDeviceAlreadyInUse = 216,cudaErrorPeerAccessUnsupported = 217,
      cudaErrorInvalidPtx = 218,cudaErrorInvalidGraphicsContext = 219,
      cudaErrorNvlinkUncorrectable = 220,cudaErrorJitCompilerNotFound = 221,
      cudaErrorUnsupportedPtxVersion = 222,
      cudaErrorJitCompilationDisabled = 223,
      cudaErrorUnsupportedExecAffinity = 224,
      cudaErrorUnsupportedDevSideSync = 225,
      cudaErrorInvalidSource = 300,cudaErrorFileNotFound = 301,
      cudaErrorSharedObjectSymbolNotFound = 302,
      cudaErrorSharedObjectInitFailed = 303,
      cudaErrorOperatingSystem = 304,cudaErrorInvalidResourceHandle = 400,
      cudaErrorIllegalState = 401,cudaErrorSymbolNotFound = 500,
      cudaErrorNotReady = 600,cudaErrorIllegalAddress = 700,
      cudaErrorLaunchOutOfResources = 701,
      cudaErrorLaunchTimeout = 702,cudaErrorLaunchIncompatibleTexturing = 703,
      cudaErrorPeerAccessAlreadyEnabled = 704,
      cudaErrorPeerAccessNotEnabled = 705,
      cudaErrorSetOnActiveProcess = 708,cudaErrorContextIsDestroyed = 709,
      cudaErrorAssert = 710,cudaErrorTooManyPeers = 711,
      cudaErrorHostMemoryAlreadyRegistered = 712,
      cudaErrorHostMemoryNotRegistered = 713,
      cudaErrorHardwareStackError = 714,cudaErrorIllegalInstruction = 715,
      cudaErrorMisalignedAddress = 716,cudaErrorInvalidAddressSpace = 717,
      cudaErrorInvalidPc = 718,cudaErrorLaunchFailure = 719,
      cudaErrorCooperativeLaunchTooLarge = 720,
      cudaErrorNotPermitted = 800,cudaErrorNotSupported = 801,
      cudaErrorSystemNotReady = 802,cudaErrorSystemDriverMismatch = 803,
      cudaErrorCompatNotSupportedOnDevice = 804,
      cudaErrorMpsConnectionFailed = 805,cudaErrorMpsRpcFailure = 806,
      cudaErrorMpsServerNotReady = 807,cudaErrorMpsMaxClientsReached = 808,
      cudaErrorMpsMaxConnectionsReached = 809,
      cudaErrorMpsClientTerminated = 810,cudaErrorCdpNotSupported = 811,
      cudaErrorCdpVersionMismatch = 812,cudaErrorStreamCaptureUnsupported = 900,
      cudaErrorStreamCaptureInvalidated = 901,
      cudaErrorStreamCaptureMerge = 902,cudaErrorStreamCaptureUnmatched = 903,
      cudaErrorStreamCaptureUnjoined = 904,
      cudaErrorStreamCaptureIsolation = 905,
      cudaErrorStreamCaptureImplicit = 906,
      cudaErrorCapturedEvent = 907,cudaErrorStreamCaptureWrongThread = 908,
      cudaErrorTimeout = 909,cudaErrorGraphExecUpdateFailure = 910,
      cudaErrorExternalDevice = 911,cudaErrorInvalidClusterSize = 912,
      cudaErrorUnknown = 999,cudaErrorApiFailureBase = 10000
      );

  {*
   * Channel format kind
    }
  {*< Signed channel format  }
  {*< Unsigned channel format  }
  {*< Float channel format  }
  {*< No channel format  }
  {*< Unsigned 8-bit integers, planar 4:2:0 YUV format  }
  {*< 1 channel unsigned 8-bit normalized integer  }
  {*< 2 channel unsigned 8-bit normalized integer  }
  {*< 4 channel unsigned 8-bit normalized integer  }
  {*< 1 channel unsigned 16-bit normalized integer  }
  {*< 2 channel unsigned 16-bit normalized integer  }
  {*< 4 channel unsigned 16-bit normalized integer  }
  {*< 1 channel signed 8-bit normalized integer  }
  {*< 2 channel signed 8-bit normalized integer  }
  {*< 4 channel signed 8-bit normalized integer  }
  {*< 1 channel signed 16-bit normalized integer  }
  {*< 2 channel signed 16-bit normalized integer  }
  {*< 4 channel signed 16-bit normalized integer  }
  {*< 4 channel unsigned normalized block-compressed (BC1 compression) format  }
  {*< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding }
  {*< 4 channel unsigned normalized block-compressed (BC2 compression) format  }
  {*< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding  }
  {*< 4 channel unsigned normalized block-compressed (BC3 compression) format  }
  {*< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding  }
  {*< 1 channel unsigned normalized block-compressed (BC4 compression) format  }
  {*< 1 channel signed normalized block-compressed (BC4 compression) format  }
  {*< 2 channel unsigned normalized block-compressed (BC5 compression) format  }
  {*< 2 channel signed normalized block-compressed (BC5 compression) format  }
  {*< 3 channel unsigned half-float block-compressed (BC6H compression) format  }
  {*< 3 channel signed half-float block-compressed (BC6H compression) format  }
  {*< 4 channel unsigned normalized block-compressed (BC7 compression) format  }
  {*< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding  }
    cudaChannelFormatKind = (cudaChannelFormatKindSigned = 0,cudaChannelFormatKindUnsigned = 1,
      cudaChannelFormatKindFloat = 2,cudaChannelFormatKindNone = 3,
      cudaChannelFormatKindNV12 = 4,cudaChannelFormatKindUnsignedNormalized8X1 = 5,
      cudaChannelFormatKindUnsignedNormalized8X2 = 6,
      cudaChannelFormatKindUnsignedNormalized8X4 = 7,
      cudaChannelFormatKindUnsignedNormalized16X1 = 8,
      cudaChannelFormatKindUnsignedNormalized16X2 = 9,
      cudaChannelFormatKindUnsignedNormalized16X4 = 10,
      cudaChannelFormatKindSignedNormalized8X1 = 11,
      cudaChannelFormatKindSignedNormalized8X2 = 12,
      cudaChannelFormatKindSignedNormalized8X4 = 13,
      cudaChannelFormatKindSignedNormalized16X1 = 14,
      cudaChannelFormatKindSignedNormalized16X2 = 15,
      cudaChannelFormatKindSignedNormalized16X4 = 16,
      cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
      cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
      cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
      cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
      cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
      cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
      cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
      cudaChannelFormatKindSignedBlockCompressed4 = 24,
      cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
      cudaChannelFormatKindSignedBlockCompressed5 = 26,
      cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
      cudaChannelFormatKindSignedBlockCompressed6H = 28,
      cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
      cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30
      );

  {*
   * CUDA Channel format descriptor
    }
  {*< x  }
  {*< y  }
  {*< z  }
  {*< w  }
  {*< Channel format kind  }
    cudaChannelFormatDesc = record
        x : longint;
        y : longint;
        z : longint;
        w : longint;
        f : cudaChannelFormatKind;
      end;

  {*
   * CUDA array
    }

    cudaArray_t = ^cudaArray;
  {*
   * CUDA array (as source copy argument)
    }
(* Const before type ignored *)

    cudaArray_const_t = ^cudaArray;
    cudaArray = record
        {undefined structure}
      end;

  {*
   * CUDA mipmapped array
    }

    cudaMipmappedArray_t = ^cudaMipmappedArray;
  {*
   * CUDA mipmapped array (as source argument)
    }
(* Const before type ignored *)

    cudaMipmappedArray_const_t = ^cudaMipmappedArray;
    cudaMipmappedArray = record
        {undefined structure}
      end;

  {*
   * Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers
    }

  const
    cudaArraySparsePropertiesSingleMipTail = $1;    
  {*
   * Sparse CUDA array and CUDA mipmapped array properties
    }
  {*< Tile width in elements  }
  {*< Tile height in elements  }
  {*< Tile depth in elements  }
  {*< First mip level at which the mip tail begins  }  {*< Total size of the mip tail.  }
  {*< Flags will either be zero or ::cudaArraySparsePropertiesSingleMipTail  }

  type
    cudaArraySparseProperties = record
        tileExtent : record
            width : dword;
            height : dword;
            depth : dword;
          end;
        miptailFirstLevel : dword;
        miptailSize : qword;
        flags : dword;
        reserved : array[0..3] of dword;
      end;

  {*
   * CUDA array and CUDA mipmapped array memory requirements
    }
  {*< Total size of the array.  }
  {*< Alignment necessary for mapping the array.  }
    cudaArrayMemoryRequirements = record
        size : size_t;
        alignment : size_t;
        reserved : array[0..3] of dword;
      end;

  {*
   * CUDA memory types
    }
  {*< Unregistered memory  }
  {*< Host memory  }
  {*< Device memory  }
  {*< Managed memory  }
    cudaMemoryType = (cudaMemoryTypeUnregistered = 0,cudaMemoryTypeHost = 1,
      cudaMemoryTypeDevice = 2,cudaMemoryTypeManaged = 3
      );

  {*
   * CUDA memory copy types
    }
  {*< Host   -> Host  }
  {*< Host   -> Device  }
  {*< Device -> Host  }
  {*< Device -> Device  }
  {*< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing  }
    cudaMemcpyKind = (cudaMemcpyHostToHost = 0,cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,cudaMemcpyDeviceToDevice = 3,
      cudaMemcpyDefault = 4);

  {*
   * CUDA Pitched memory pointer
   *
   * \sa ::make_cudaPitchedPtr
    }
  {*< Pointer to allocated memory  }
  {*< Pitch of allocated memory in bytes  }
  {*< Logical width of allocation in elements  }
  {*< Logical height of allocation in elements  }
    cudaPitchedPtr = record
        ptr : pointer;
        pitch : size_t;
        xsize : size_t;
        ysize : size_t;
      end;

  {*
   * CUDA extent
   *
   * \sa ::make_cudaExtent
    }
  {*< Width in elements when referring to array memory, in bytes when referring to linear memory  }
  {*< Height in elements  }
  {*< Depth in elements  }
    cudaExtent = record
        width : size_t;
        height : size_t;
        depth : size_t;
      end;

  {*
   * CUDA 3D position
   *
   * \sa ::make_cudaPos
    }
  {*< x  }
  {*< y  }
  {*< z  }
    cudaPos = record
        x : size_t;
        y : size_t;
        z : size_t;
      end;

  {*
   * CUDA 3D memory copying parameters
    }
  {*< Source memory address  }
  {*< Source position offset  }
  {*< Pitched source memory address  }
  {*< Destination memory address  }
  {*< Destination position offset  }
  {*< Pitched destination memory address  }
  {*< Requested memory copy size  }
  {*< Type of transfer  }
    cudaMemcpy3DParms = record
        srcArray : cudaArray_t;
        srcPos : cudaPos;
        srcPtr : cudaPitchedPtr;
        dstArray : cudaArray_t;
        dstPos : cudaPos;
        dstPtr : cudaPitchedPtr;
        extent : cudaExtent;
        kind : cudaMemcpyKind;
      end;

  {*
   * CUDA 3D cross-device memory copying parameters
    }
  {*< Source memory address  }
  {*< Source position offset  }
  {*< Pitched source memory address  }
  {*< Source device  }
  {*< Destination memory address  }
  {*< Destination position offset  }
  {*< Pitched destination memory address  }
  {*< Destination device  }
  {*< Requested memory copy size  }
    cudaMemcpy3DPeerParms = record
        srcArray : cudaArray_t;
        srcPos : cudaPos;
        srcPtr : cudaPitchedPtr;
        srcDevice : longint;
        dstArray : cudaArray_t;
        dstPos : cudaPos;
        dstPtr : cudaPitchedPtr;
        dstDevice : longint;
        extent : cudaExtent;
      end;

  {*
   * CUDA Memset node parameters
    }
  {*< Destination device pointer  }
  {*< Pitch of destination device pointer. Unused if height is 1  }
  {*< Value to be set  }
  {*< Size of each element in bytes. Must be 1, 2, or 4.  }
  {*< Width of the row in elements  }
  {*< Number of rows  }
    cudaMemsetParams = record
        dst : pointer;
        pitch : size_t;
        value : dword;
        elementSize : dword;
        width : size_t;
        height : size_t;
      end;

  {*
   * Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
    }
  {*< Normal cache persistence.  }
  {*< Streaming access is less likely to persit from cache.  }
  {*< Persisting access is more likely to persist in cache. }
    cudaAccessProperty = (cudaAccessPropertyNormal = 0,cudaAccessPropertyStreaming = 1,
      cudaAccessPropertyPersisting = 2);

  {*
   * Specifies an access policy for a window, a contiguous extent of memory
   * beginning at base_ptr and ending at base_ptr + num_bytes.
   * Partition into many segments and assign segments such that.
   * sum of "hit segments" / window == approx. ratio.
   * sum of "miss segments" / window == approx 1-ratio.
   * Segments and ratio specifications are fitted to the capabilities of
   * the architecture.
   * Accesses in a hit segment apply the hitProp access policy.
   * Accesses in a miss segment apply the missProp access policy.
    }
  {*< Starting address of the access policy window. CUDA driver may align it.  }
  {*< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment.  }
  {*< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp.  }
  {*< ::CUaccessProperty set for hit.  }
  {*< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING.  }
    cudaAccessPolicyWindow = record
        base_ptr : pointer;
        num_bytes : size_t;
        hitRatio : single;
        hitProp : cudaAccessProperty;
        missProp : cudaAccessProperty;
      end;

  {*
   * CUDA host function
   * \param userData Argument value passed to the function
    }

    cudaHostFn_t = procedure (userData:pointer);WINAPI;
  {*
   * CUDA host node parameters
    }
  {*< The function to call when the node executes  }
  {*< Argument to pass to the function  }
    cudaHostNodeParams = record
        fn : cudaHostFn_t;
        userData : pointer;
      end;

  {*
   * Possible stream capture statuses returned by ::cudaStreamIsCapturing
    }
  {*< Stream is not capturing  }
  {*< Stream is actively capturing  }
  {*< Stream is part of a capture sequence that
                                                     has been invalidated, but not terminated  }
    cudaStreamCaptureStatus = (cudaStreamCaptureStatusNone = 0,cudaStreamCaptureStatusActive = 1,
      cudaStreamCaptureStatusInvalidated = 2
      );

  {*
   * Possible modes for stream capture thread interactions. For more details see
   * ::cudaStreamBeginCapture and ::cudaThreadExchangeStreamCaptureMode
    }
    cudaStreamCaptureMode = (cudaStreamCaptureModeGlobal = 0,cudaStreamCaptureModeThreadLocal = 1,
      cudaStreamCaptureModeRelaxed = 2);

    cudaSynchronizationPolicy = (cudaSyncPolicyAuto = 1,cudaSyncPolicySpin = 2,
      cudaSyncPolicyYield = 3,cudaSyncPolicyBlockingSync = 4
      );

  {*
   * Cluster scheduling policies. These may be passed to ::cudaFuncSetAttribute
    }
  {*< the default policy  }
  {*< spread the blocks within a cluster to the SMs  }
  {*< allow the hardware to load-balance the blocks in a cluster to the SMs  }
    cudaClusterSchedulingPolicy = (cudaClusterSchedulingPolicyDefault = 0,
      cudaClusterSchedulingPolicySpread = 1,
      cudaClusterSchedulingPolicyLoadBalancing = 2
      );

  {*
   * Flags for ::cudaStreamUpdateCaptureDependencies
    }
  {*< Add new nodes to the dependency set  }
  {*< Replace the dependency set with the new nodes  }
    cudaStreamUpdateCaptureDependenciesFlags = (cudaStreamAddCaptureDependencies = $0,
      cudaStreamSetCaptureDependencies = $1
      );

  {*
   * Flags for user objects for graphs
    }
  {*< Indicates the destructor execution is not synchronized by any CUDA handle.  }
    cudaUserObjectFlags = (cudaUserObjectNoDestructorSync = $1
      );

  {*
   * Flags for retaining user object references for graphs
    }
  {*< Transfer references from the caller rather than creating new references.  }
    cudaUserObjectRetainFlags = (cudaGraphUserObjectMove = $1);

  {*
   * CUDA graphics interop resource
    }
    cudaGraphicsResource = record
        {undefined structure}
      end;

  {*
   * CUDA graphics interop register flags
    }
  {*< Default  }
  {*< CUDA will not write to this resource  }  {*< CUDA will only write to and will not read from this resource  }
  {*< CUDA will bind this resource to a surface reference  }
  {*< CUDA will perform texture gather operations on this resource  }
    cudaGraphicsRegisterFlags = (cudaGraphicsRegisterFlagsNone = 0,
      cudaGraphicsRegisterFlagsReadOnly = 1,
      cudaGraphicsRegisterFlagsWriteDiscard = 2,
      cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
      cudaGraphicsRegisterFlagsTextureGather = 8
      );

  {*
   * CUDA graphics interop map flags
    }
  {*< Default; Assume resource can be read/written  }
  {*< CUDA will not write to this resource  }
  {*< CUDA will only write to and will not read from this resource  }
    cudaGraphicsMapFlags = (cudaGraphicsMapFlagsNone = 0,cudaGraphicsMapFlagsReadOnly = 1,
      cudaGraphicsMapFlagsWriteDiscard = 2
      );

  {*
   * CUDA graphics interop array indices for cube maps
    }
  {*< Positive X face of cubemap  }
  {*< Negative X face of cubemap  }
  {*< Positive Y face of cubemap  }
  {*< Negative Y face of cubemap  }
  {*< Positive Z face of cubemap  }
  {*< Negative Z face of cubemap  }
    cudaGraphicsCubeFace = (cudaGraphicsCubeFacePositiveX = $00,
      cudaGraphicsCubeFaceNegativeX = $01,
      cudaGraphicsCubeFacePositiveY = $02,
      cudaGraphicsCubeFaceNegativeY = $03,
      cudaGraphicsCubeFacePositiveZ = $04,
      cudaGraphicsCubeFaceNegativeZ = $05
      );

  {*
   * CUDA resource types
    }
  {*< Array resource  }
  {*< Mipmapped array resource  }
  {*< Linear resource  }
  {*< Pitch 2D resource  }
    cudaResourceType = (cudaResourceTypeArray = $00,cudaResourceTypeMipmappedArray = $01,
      cudaResourceTypeLinear = $02,cudaResourceTypePitch2D = $03
      );

  {*
   * CUDA texture resource view formats
    }
  {*< No resource view format (use underlying resource format)  }
  {*< 1 channel unsigned 8-bit integers  }
  {*< 2 channel unsigned 8-bit integers  }
  {*< 4 channel unsigned 8-bit integers  }
  {*< 1 channel signed 8-bit integers  }
  {*< 2 channel signed 8-bit integers  }
  {*< 4 channel signed 8-bit integers  }
  {*< 1 channel unsigned 16-bit integers  }
  {*< 2 channel unsigned 16-bit integers  }
  {*< 4 channel unsigned 16-bit integers  }
  {*< 1 channel signed 16-bit integers  }
  {*< 2 channel signed 16-bit integers  }
  {*< 4 channel signed 16-bit integers  }
  {*< 1 channel unsigned 32-bit integers  }
  {*< 2 channel unsigned 32-bit integers  }
  {*< 4 channel unsigned 32-bit integers  }
  {*< 1 channel signed 32-bit integers  }
  {*< 2 channel signed 32-bit integers  }
  {*< 4 channel signed 32-bit integers  }
  {*< 1 channel 16-bit floating point  }
  {*< 2 channel 16-bit floating point  }
  {*< 4 channel 16-bit floating point  }
  {*< 1 channel 32-bit floating point  }
  {*< 2 channel 32-bit floating point  }
  {*< 4 channel 32-bit floating point  }
  {*< Block compressed 1  }
  {*< Block compressed 2  }
  {*< Block compressed 3  }
  {*< Block compressed 4 unsigned  }
  {*< Block compressed 4 signed  }
  {*< Block compressed 5 unsigned  }
  {*< Block compressed 5 signed  }
  {*< Block compressed 6 unsigned half-float  }
  {*< Block compressed 6 signed half-float  }
  {*< Block compressed 7  }
    cudaResourceViewFormat = (cudaResViewFormatNone = $00,cudaResViewFormatUnsignedChar1 = $01,
      cudaResViewFormatUnsignedChar2 = $02,
      cudaResViewFormatUnsignedChar4 = $03,
      cudaResViewFormatSignedChar1 = $04,cudaResViewFormatSignedChar2 = $05,
      cudaResViewFormatSignedChar4 = $06,cudaResViewFormatUnsignedShort1 = $07,
      cudaResViewFormatUnsignedShort2 = $08,
      cudaResViewFormatUnsignedShort4 = $09,
      cudaResViewFormatSignedShort1 = $0a,
      cudaResViewFormatSignedShort2 = $0b,
      cudaResViewFormatSignedShort4 = $0c,
      cudaResViewFormatUnsignedInt1 = $0d,
      cudaResViewFormatUnsignedInt2 = $0e,
      cudaResViewFormatUnsignedInt4 = $0f,
      cudaResViewFormatSignedInt1 = $10,cudaResViewFormatSignedInt2 = $11,
      cudaResViewFormatSignedInt4 = $12,cudaResViewFormatHalf1 = $13,
      cudaResViewFormatHalf2 = $14,cudaResViewFormatHalf4 = $15,
      cudaResViewFormatFloat1 = $16,cudaResViewFormatFloat2 = $17,
      cudaResViewFormatFloat4 = $18,cudaResViewFormatUnsignedBlockCompressed1 = $19,
      cudaResViewFormatUnsignedBlockCompressed2 = $1a,
      cudaResViewFormatUnsignedBlockCompressed3 = $1b,
      cudaResViewFormatUnsignedBlockCompressed4 = $1c,
      cudaResViewFormatSignedBlockCompressed4 = $1d,
      cudaResViewFormatUnsignedBlockCompressed5 = $1e,
      cudaResViewFormatSignedBlockCompressed5 = $1f,
      cudaResViewFormatUnsignedBlockCompressed6H = $20,
      cudaResViewFormatSignedBlockCompressed6H = $21,
      cudaResViewFormatUnsignedBlockCompressed7 = $22
      );

  {*
   * CUDA resource descriptor
    }
  {*< Resource type  }
  {*< CUDA array  }
  {*< CUDA mipmapped array  }
  {*< Device pointer  }
  {*< Channel descriptor  }
  {*< Size in bytes  }
  {*< Device pointer  }
  {*< Channel descriptor  }
  {*< Width of the array in elements  }
  {*< Height of the array in elements  }
  {*< Pitch between two rows in bytes  }
    cudaResourceDesc = record
        resType : cudaResourceType;
        res : record
            case longint of
              0 : ( &array : record
                  &array : cudaArray_t;
                end );
              1 : ( mipmap : record
                  mipmap : cudaMipmappedArray_t;
                end );
              2 : ( linear : record
                  devPtr : pointer;
                  desc : cudaChannelFormatDesc;
                  sizeInBytes : size_t;
                end );
              3 : ( pitch2D : record
                  devPtr : pointer;
                  desc : cudaChannelFormatDesc;
                  width : size_t;
                  height : size_t;
                  pitchInBytes : size_t;
                end );
            end;
      end;

  {*
   * CUDA resource view descriptor
    }
  {*< Resource view format  }
  {*< Width of the resource view  }
  {*< Height of the resource view  }
  {*< Depth of the resource view  }
  {*< First defined mipmap level  }
  {*< Last defined mipmap level  }
  {*< First layer index  }
  {*< Last layer index  }
    cudaResourceViewDesc = record
        format : cudaResourceViewFormat;
        width : size_t;
        height : size_t;
        depth : size_t;
        firstMipmapLevel : dword;
        lastMipmapLevel : dword;
        firstLayer : dword;
        lastLayer : dword;
      end;

  {*
   * CUDA pointer attributes
    }
  {*
       * The type of memory - ::cudaMemoryTypeUnregistered, ::cudaMemoryTypeHost,
       * ::cudaMemoryTypeDevice or ::cudaMemoryTypeManaged.
        }
  {* 
       * The device against which the memory was allocated or registered.
       * If the memory type is ::cudaMemoryTypeDevice then this identifies 
       * the device on which the memory referred physically resides.  If
       * the memory type is ::cudaMemoryTypeHost or::cudaMemoryTypeManaged then
       * this identifies the device which was current when the memory was allocated
       * or registered (and if that device is deinitialized then this allocation
       * will vanish with that device's state).
        }
  {*
       * The address which may be dereferenced on the current device to access 
       * the memory or NULL if no such address exists.
        }
  {*
       * The address which may be dereferenced on the host to access the
       * memory or NULL if no such address exists.
       *
       * \note CUDA doesn't check if unregistered memory is allocated so this field
       * may contain invalid pointer if an invalid pointer has been passed to CUDA.
        }
    cudaPointerAttributes = record
        _type : cudaMemoryType;
        device : longint;
        devicePointer : pointer;
        hostPointer : pointer;
      end;

  {*
   * CUDA function attributes
    }
  {*
      * The size in bytes of statically-allocated shared memory per block
      * required by this function. This does not include dynamically-allocated
      * shared memory requested by the user at runtime.
       }
  {*
      * The size in bytes of user-allocated constant memory required by this
      * function.
       }
  {*
      * The size in bytes of local memory used by each thread of this function.
       }
  {*
      * The maximum number of threads per block, beyond which a launch of the
      * function would fail. This number depends on both the function and the
      * device on which the function is currently loaded.
       }
  {*
      * The number of registers used by each thread of this function.
       }
  {*
      * The PTX virtual architecture version for which the function was
      * compiled. This value is the major PTX version * 10 + the minor PTX
      * version, so a PTX version 1.3 function would return the value 13.
       }
  {*
      * The binary architecture version for which the function was compiled.
      * This value is the major binary version * 10 + the minor binary version,
      * so a binary version 1.3 function would return the value 13.
       }
  {*
      * The attribute to indicate whether the function has been compiled with 
      * user specified option "-Xptxas --dlcm=ca" set.
       }
  {*
      * The maximum size in bytes of dynamic shared memory per block for 
      * this function. Any launch must have a dynamic shared memory size
      * smaller than this value.
       }
  {*
      * On devices where the L1 cache and shared memory use the same hardware resources, 
      * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
      * Refer to ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
      * This is only a hint, and the driver can choose a different ratio if required to execute the function.
      * See ::cudaFuncSetAttribute
       }
  {*
      * If this attribute is set, the kernel must launch with a valid cluster dimension
      * specified.
       }
  {*
      * The required cluster width/height/depth in blocks. The values must either
      * all be 0 or all be positive. The validity of the cluster dimensions is
      * otherwise checked at launch time.
      *
      * If the value is set during compile time, it cannot be set at runtime.
      * Setting it at runtime should return cudaErrorNotPermitted.
      * See ::cudaFuncSetAttribute
       }
  {*
      * The block scheduling policy of a function.
      * See ::cudaFuncSetAttribute
       }
  {*
      * Whether the function can be launched with non-portable cluster size. 1 is
      * allowed, 0 is disallowed. A non-portable cluster size may only function
      * on the specific SKUs the program is tested on. The launch might fail if
      * the program is run on a different hardware platform.
      *
      * CUDA API provides ::cudaOccupancyMaxActiveClusters to assist with checking
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
      * See ::cudaFuncSetAttribute
       }
  {*
      * Reserved for future use.
       }
    cudaFuncAttributes = record
        sharedSizeBytes : size_t;
        constSizeBytes : size_t;
        localSizeBytes : size_t;
        maxThreadsPerBlock : longint;
        numRegs : longint;
        ptxVersion : longint;
        binaryVersion : longint;
        cacheModeCA : longint;
        maxDynamicSharedSizeBytes : longint;
        preferredShmemCarveout : longint;
        clusterDimMustBeSet : longint;
        requiredClusterWidth : longint;
        requiredClusterHeight : longint;
        requiredClusterDepth : longint;
        clusterSchedulingPolicyPreference : longint;
        nonPortableClusterSizeAllowed : longint;
        reserved : array[0..15] of longint;
      end;

  {*
   * CUDA function attributes that can be set using ::cudaFuncSetAttribute
    }
  {*< Maximum dynamic shared memory size  }
  {*< Preferred shared memory-L1 cache split  }
  {*< Indicator to enforce valid cluster dimension specification on kernel launch  }
  {*< Required cluster width  }
  {*< Required cluster height  }
  {*< Required cluster depth  }
  {*< Whether non-portable cluster scheduling policy is supported  }
  {*< Required cluster scheduling policy preference  }
    cudaFuncAttribute = (cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
      cudaFuncAttributePreferredSharedMemoryCarveout = 9,
      cudaFuncAttributeClusterDimMustBeSet = 10,
      cudaFuncAttributeRequiredClusterWidth = 11,
      cudaFuncAttributeRequiredClusterHeight = 12,
      cudaFuncAttributeRequiredClusterDepth = 13,
      cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
      cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
      cudaFuncAttributeMax);

  {*
   * CUDA function cache configurations
    }
  {*< Default function cache configuration, no preference  }
  {*< Prefer larger shared memory and smaller L1 cache   }
  {*< Prefer larger L1 cache and smaller shared memory  }
  {*< Prefer equal size L1 cache and shared memory  }
    cudaFuncCache = (cudaFuncCachePreferNone = 0,cudaFuncCachePreferShared = 1,
      cudaFuncCachePreferL1 = 2,cudaFuncCachePreferEqual = 3
      );

  {*
   * CUDA shared memory configuration
    }
    cudaSharedMemConfig = (cudaSharedMemBankSizeDefault = 0,cudaSharedMemBankSizeFourByte = 1,
      cudaSharedMemBankSizeEightByte = 2
      );

  {* 
   * Shared memory carveout configurations. These may be passed to cudaFuncSetAttribute
    }
  {*< No preference for shared memory or L1 (default)  }
  {*< Prefer maximum available shared memory, minimum L1 cache  }
  {*< Prefer maximum available L1 cache, minimum shared memory  }
    cudaSharedCarveout = (cudaSharedmemCarveoutDefault = -(1),cudaSharedmemCarveoutMaxShared = 100,
      cudaSharedmemCarveoutMaxL1 = 0);

  {*
   * CUDA device compute modes
    }
  {*< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device)  }
  {*< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device)  }
  {*< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device)  }
  {*< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device)  }
    cudaComputeMode = (cudaComputeModeDefault = 0,cudaComputeModeExclusive = 1,
      cudaComputeModeProhibited = 2,cudaComputeModeExclusiveProcess = 3
      );

  {*
   * CUDA Limits
    }
  {*< GPU thread stack size  }
  {*< GPU printf FIFO size  }
  {*< GPU malloc heap size  }
  {*< GPU device runtime synchronize depth  }
  {*< GPU device runtime pending launch count  }
  {*< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint  }
  {*< A size in bytes for L2 persisting lines cache size  }
    cudaLimit = (cudaLimitStackSize = $00,cudaLimitPrintfFifoSize = $01,
      cudaLimitMallocHeapSize = $02,cudaLimitDevRuntimeSyncDepth = $03,
      cudaLimitDevRuntimePendingLaunchCount = $04,
      cudaLimitMaxL2FetchGranularity = $05,
      cudaLimitPersistingL2CacheSize = $06
      );

  {*
   * CUDA Memory Advise values
    }
  {*< Data will mostly be read and only occassionally be written to  }
  {*< Undo the effect of ::cudaMemAdviseSetReadMostly  }
  {*< Set the preferred location for the data as the specified device  }
  {*< Clear the preferred location for the data  }
  {*< Data will be accessed by the specified device, so prevent page faults as much as possible  }
  {*< Let the Unified Memory subsystem decide on the page faulting policy for the specified device  }
    cudaMemoryAdvise = (cudaMemAdviseSetReadMostly = 1,cudaMemAdviseUnsetReadMostly = 2,
      cudaMemAdviseSetPreferredLocation = 3,
      cudaMemAdviseUnsetPreferredLocation = 4,
      cudaMemAdviseSetAccessedBy = 5,cudaMemAdviseUnsetAccessedBy = 6
      );

  {*
   * CUDA range attributes
    }
  {*< Whether the range will mostly be read and only occassionally be written to  }
  {*< The preferred location of the range  }
  {*< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device  }
  {*< The last location to which the range was prefetched  }
    cudaMemRangeAttribute = (cudaMemRangeAttributeReadMostly = 1,
      cudaMemRangeAttributePreferredLocation = 2,
      cudaMemRangeAttributeAccessedBy = 3,
      cudaMemRangeAttributeLastPrefetchLocation = 4
      );

  {*
   * CUDA GPUDirect RDMA flush writes APIs supported on the device
    }
  {*< ::cudaDeviceFlushGPUDirectRDMAWrites() and its CUDA Driver API counterpart are supported on the device.  }
  {*< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the CUDA device.  }
    cudaFlushGPUDirectRDMAWritesOptions = (cudaFlushGPUDirectRDMAWritesOptionHost = 1 shl 0,
      cudaFlushGPUDirectRDMAWritesOptionMemOps = 1 shl 1
      );

  {*
   * CUDA GPUDirect RDMA flush writes ordering features of the device
    }
  {*< The device does not natively support ordering of GPUDirect RDMA writes. ::cudaFlushGPUDirectRDMAWrites() can be leveraged if supported.  }
  {*< Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA devices may not.  }
  {*< Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device.  }
    cudaGPUDirectRDMAWritesOrdering = (cudaGPUDirectRDMAWritesOrderingNone = 0,
      cudaGPUDirectRDMAWritesOrderingOwner = 100,
      cudaGPUDirectRDMAWritesOrderingAllDevices = 200
      );

  {*
   * CUDA GPUDirect RDMA flush writes scopes
    }
  {*< Blocks until remote writes are visible to the CUDA device context owning the data.  }
  {*< Blocks until remote writes are visible to all CUDA device contexts.  }
    cudaFlushGPUDirectRDMAWritesScope = (cudaFlushGPUDirectRDMAWritesToOwner = 100,
      cudaFlushGPUDirectRDMAWritesToAllDevices = 200
      );

  {*
   * CUDA GPUDirect RDMA flush writes targets
    }
  {*< Sets the target for ::cudaDeviceFlushGPUDirectRDMAWrites() to the currently active CUDA device context.  }
    cudaFlushGPUDirectRDMAWritesTarget = (cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
      );

  {*
   * CUDA device attributes
    }
  {*< Maximum number of threads per block  }
  {*< Maximum block dimension X  }
  {*< Maximum block dimension Y  }
  {*< Maximum block dimension Z  }
  {*< Maximum grid dimension X  }
  {*< Maximum grid dimension Y  }
  {*< Maximum grid dimension Z  }
  {*< Maximum shared memory available per block in bytes  }
  {*< Memory available on device for __constant__ variables in a CUDA C kernel in bytes  }
  {*< Warp size in threads  }
  {*< Maximum pitch in bytes allowed by memory copies  }
  {*< Maximum number of 32-bit registers available per block  }
  {*< Peak clock frequency in kilohertz  }
  {*< Alignment requirement for textures  }
  {*< Device can possibly copy memory and execute a kernel concurrently  }
  {*< Number of multiprocessors on device  }
  {*< Specifies whether there is a run time limit on kernels  }
  {*< Device is integrated with host memory  }
  {*< Device can map host memory into CUDA address space  }
  {*< Compute mode (See ::cudaComputeMode for details)  }
  {*< Maximum 1D texture width  }
  {*< Maximum 2D texture width  }
  {*< Maximum 2D texture height  }
  {*< Maximum 3D texture width  }
  {*< Maximum 3D texture height  }
  {*< Maximum 3D texture depth  }
  {*< Maximum 2D layered texture width  }
  {*< Maximum 2D layered texture height  }
  {*< Maximum layers in a 2D layered texture  }
  {*< Alignment requirement for surfaces  }
  {*< Device can possibly execute multiple kernels concurrently  }
  {*< Device has ECC support enabled  }
  {*< PCI bus ID of the device  }
  {*< PCI device ID of the device  }
  {*< Device is using TCC driver model  }
  {*< Peak memory clock frequency in kilohertz  }
  {*< Global memory bus width in bits  }
  {*< Size of L2 cache in bytes  }
  {*< Maximum resident threads per multiprocessor  }
  {*< Number of asynchronous engines  }
  {*< Device shares a unified address space with the host  }  {*< Maximum 1D layered texture width  }
  {*< Maximum layers in a 1D layered texture  }
  {*< Maximum 2D texture width if cudaArrayTextureGather is set  }
  {*< Maximum 2D texture height if cudaArrayTextureGather is set  }
  {*< Alternate maximum 3D texture width  }
  {*< Alternate maximum 3D texture height  }
  {*< Alternate maximum 3D texture depth  }
  {*< PCI domain ID of the device  }
  {*< Pitch alignment requirement for textures  }
  {*< Maximum cubemap texture width/height  }
  {*< Maximum cubemap layered texture width/height  }
  {*< Maximum layers in a cubemap layered texture  }
  {*< Maximum 1D surface width  }
  {*< Maximum 2D surface width  }
  {*< Maximum 2D surface height  }
  {*< Maximum 3D surface width  }
  {*< Maximum 3D surface height  }
  {*< Maximum 3D surface depth  }
  {*< Maximum 1D layered surface width  }
  {*< Maximum layers in a 1D layered surface  }
  {*< Maximum 2D layered surface width  }
  {*< Maximum 2D layered surface height  }
  {*< Maximum layers in a 2D layered surface  }
  {*< Maximum cubemap surface width  }
  {*< Maximum cubemap layered surface width  }
  {*< Maximum layers in a cubemap layered surface  }
  {*< Maximum 1D linear texture width  }
  {*< Maximum 2D linear texture width  }
  {*< Maximum 2D linear texture height  }
  {*< Maximum 2D linear texture pitch in bytes  }
  {*< Maximum mipmapped 2D texture width  }
  {*< Maximum mipmapped 2D texture height  }
  {*< Major compute capability version number  }  {*< Minor compute capability version number  }
  {*< Maximum mipmapped 1D texture width  }
  {*< Device supports stream priorities  }
  {*< Device supports caching globals in L1  }
  {*< Device supports caching locals in L1  }
  {*< Maximum shared memory available per multiprocessor in bytes  }
  {*< Maximum number of 32-bit registers available per multiprocessor  }
  {*< Device can allocate managed memory on this system  }
  {*< Device is on a multi-GPU board  }
  {*< Unique identifier for a group of devices on the same multi-GPU board  }
  {*< Link between the device and the host supports native atomic operations  }
  {*< Ratio of single precision performance (in floating-point operations per second) to double precision performance  }
  {*< Device supports coherently accessing pageable memory without calling cudaHostRegister on it  }
  {*< Device can coherently access managed memory concurrently with the CPU  }
  {*< Device supports Compute Preemption  }
  {*< Device can access host registered memory at the same virtual address as the CPU  }
  {*< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel }
  {*< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.  }
  {*< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute  }
  {*< Device supports flushing of outstanding remote writes.  }
  {*< Device supports host memory registration via ::cudaHostRegister.  }
  {*< Device accesses pageable memory via the host's page tables.  }
  {*< Host can directly access managed memory on the device without migration.  }
  {*< Maximum number of blocks per multiprocessor  }
  {*< Maximum L2 persisting lines capacity setting in bytes.  }
  {*< Maximum value of cudaAccessPolicyWindow::num_bytes.  }
  {*< Shared memory reserved by CUDA driver per block in bytes  }
  {*< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays  }
  {*< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU  }
  {*< External timeline semaphore interop is supported on the device  }
  {*< Deprecated, External timeline semaphore interop is supported on the device  }
  {*< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs  }
  {*< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)  }
  {*< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum  }
  {*< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here.  }
  {*< Handle types supported with mempool based IPC  }
  {*< Indicates device supports cluster launch  }
  {*< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays  }
  {*< Device supports IPC Events.  }  {*< Number of memory synchronization domains the device supports.  }
    cudaDeviceAttr = (cudaDevAttrMaxThreadsPerBlock = 1,
      cudaDevAttrMaxBlockDimX = 2,cudaDevAttrMaxBlockDimY = 3,
      cudaDevAttrMaxBlockDimZ = 4,cudaDevAttrMaxGridDimX = 5,
      cudaDevAttrMaxGridDimY = 6,cudaDevAttrMaxGridDimZ = 7,
      cudaDevAttrMaxSharedMemoryPerBlock = 8,
      cudaDevAttrTotalConstantMemory = 9,
      cudaDevAttrWarpSize = 10,cudaDevAttrMaxPitch = 11,
      cudaDevAttrMaxRegistersPerBlock = 12,
      cudaDevAttrClockRate = 13,cudaDevAttrTextureAlignment = 14,
      cudaDevAttrGpuOverlap = 15,cudaDevAttrMultiProcessorCount = 16,
      cudaDevAttrKernelExecTimeout = 17,cudaDevAttrIntegrated = 18,
      cudaDevAttrCanMapHostMemory = 19,cudaDevAttrComputeMode = 20,
      cudaDevAttrMaxTexture1DWidth = 21,cudaDevAttrMaxTexture2DWidth = 22,
      cudaDevAttrMaxTexture2DHeight = 23,
      cudaDevAttrMaxTexture3DWidth = 24,cudaDevAttrMaxTexture3DHeight = 25,
      cudaDevAttrMaxTexture3DDepth = 26,cudaDevAttrMaxTexture2DLayeredWidth = 27,
      cudaDevAttrMaxTexture2DLayeredHeight = 28,
      cudaDevAttrMaxTexture2DLayeredLayers = 29,
      cudaDevAttrSurfaceAlignment = 30,cudaDevAttrConcurrentKernels = 31,
      cudaDevAttrEccEnabled = 32,cudaDevAttrPciBusId = 33,
      cudaDevAttrPciDeviceId = 34,cudaDevAttrTccDriver = 35,
      cudaDevAttrMemoryClockRate = 36,cudaDevAttrGlobalMemoryBusWidth = 37,
      cudaDevAttrL2CacheSize = 38,cudaDevAttrMaxThreadsPerMultiProcessor = 39,
      cudaDevAttrAsyncEngineCount = 40,cudaDevAttrUnifiedAddressing = 41,
      cudaDevAttrMaxTexture1DLayeredWidth = 42,
      cudaDevAttrMaxTexture1DLayeredLayers = 43,
      cudaDevAttrMaxTexture2DGatherWidth = 45,
      cudaDevAttrMaxTexture2DGatherHeight = 46,
      cudaDevAttrMaxTexture3DWidthAlt = 47,
      cudaDevAttrMaxTexture3DHeightAlt = 48,
      cudaDevAttrMaxTexture3DDepthAlt = 49,
      cudaDevAttrPciDomainId = 50,cudaDevAttrTexturePitchAlignment = 51,
      cudaDevAttrMaxTextureCubemapWidth = 52,
      cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
      cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
      cudaDevAttrMaxSurface1DWidth = 55,cudaDevAttrMaxSurface2DWidth = 56,
      cudaDevAttrMaxSurface2DHeight = 57,
      cudaDevAttrMaxSurface3DWidth = 58,cudaDevAttrMaxSurface3DHeight = 59,
      cudaDevAttrMaxSurface3DDepth = 60,cudaDevAttrMaxSurface1DLayeredWidth = 61,
      cudaDevAttrMaxSurface1DLayeredLayers = 62,
      cudaDevAttrMaxSurface2DLayeredWidth = 63,
      cudaDevAttrMaxSurface2DLayeredHeight = 64,
      cudaDevAttrMaxSurface2DLayeredLayers = 65,
      cudaDevAttrMaxSurfaceCubemapWidth = 66,
      cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
      cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
      cudaDevAttrMaxTexture1DLinearWidth = 69,
      cudaDevAttrMaxTexture2DLinearWidth = 70,
      cudaDevAttrMaxTexture2DLinearHeight = 71,
      cudaDevAttrMaxTexture2DLinearPitch = 72,
      cudaDevAttrMaxTexture2DMipmappedWidth = 73,
      cudaDevAttrMaxTexture2DMipmappedHeight = 74,
      cudaDevAttrComputeCapabilityMajor = 75,
      cudaDevAttrComputeCapabilityMinor = 76,
      cudaDevAttrMaxTexture1DMipmappedWidth = 77,
      cudaDevAttrStreamPrioritiesSupported = 78,
      cudaDevAttrGlobalL1CacheSupported = 79,
      cudaDevAttrLocalL1CacheSupported = 80,
      cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
      cudaDevAttrMaxRegistersPerMultiprocessor = 82,
      cudaDevAttrManagedMemory = 83,cudaDevAttrIsMultiGpuBoard = 84,
      cudaDevAttrMultiGpuBoardGroupID = 85,
      cudaDevAttrHostNativeAtomicSupported = 86,
      cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
      cudaDevAttrPageableMemoryAccess = 88,
      cudaDevAttrConcurrentManagedAccess = 89,
      cudaDevAttrComputePreemptionSupported = 90,
      cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
      cudaDevAttrReserved92 = 92,cudaDevAttrReserved93 = 93,
      cudaDevAttrReserved94 = 94,cudaDevAttrCooperativeLaunch = 95,
      cudaDevAttrCooperativeMultiDeviceLaunch = 96,
      cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
      cudaDevAttrCanFlushRemoteWrites = 98,
      cudaDevAttrHostRegisterSupported = 99,
      cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
      cudaDevAttrDirectManagedMemAccessFromHost = 101,
      cudaDevAttrMaxBlocksPerMultiprocessor = 106,
      cudaDevAttrMaxPersistingL2CacheSize = 108,
      cudaDevAttrMaxAccessPolicyWindowSize = 109,
      cudaDevAttrReservedSharedMemoryPerBlock = 111,
      cudaDevAttrSparseCudaArraySupported = 112,
      cudaDevAttrHostRegisterReadOnlySupported = 113,
      cudaDevAttrTimelineSemaphoreInteropSupported = 114,
      cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,
      cudaDevAttrMemoryPoolsSupported = 115,
      cudaDevAttrGPUDirectRDMASupported = 116,
      cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
      cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
      cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
      cudaDevAttrClusterLaunch = 120,cudaDevAttrDeferredMappingCudaArraySupported = 121,
      cudaDevAttrReserved122 = 122,cudaDevAttrReserved123 = 123,
      cudaDevAttrReserved124 = 124,cudaDevAttrIpcEventSupport = 125,
      cudaDevAttrMemSyncDomainCount = 126,
      cudaDevAttrReserved127 = 127,cudaDevAttrReserved128 = 128,
      cudaDevAttrReserved129 = 129,cudaDevAttrReserved132 = 132,
      cudaDevAttrMax);

  {*
   * CUDA memory pool attributes
    }
  {*
       * (value type = int)
       * Allow cuMemAllocAsync to use memory asynchronously freed
       * in another streams as long as a stream ordering dependency
       * of the allocating stream on the free action exists.
       * Cuda events and null stream interactions can create the required
       * stream ordered dependencies. (default enabled)
        }
  {*
       * (value type = int)
       * Allow reuse of already completed frees when there is no dependency
       * between the free and allocation. (default enabled)
        }
  {*
       * (value type = int)
       * Allow cuMemAllocAsync to insert new stream dependencies
       * in order to establish the stream ordering required to reuse
       * a piece of memory released by cuFreeAsync (default enabled).
        }
  {*
       * (value type = cuuint64_t)
       * Amount of reserved memory in bytes to hold onto before trying
       * to release memory back to the OS. When more than the release
       * threshold bytes of memory are held by the memory pool, the
       * allocator will try to release memory back to the OS on the
       * next call to stream, event or context synchronize. (default 0)
        }
  {*
       * (value type = cuuint64_t)
       * Amount of backing memory currently allocated for the mempool.
        }
  {*
       * (value type = cuuint64_t)
       * High watermark of backing memory allocated for the mempool since the
       * last time it was reset. High watermark can only be reset to zero.
        }
  {*
       * (value type = cuuint64_t)
       * Amount of memory from the pool that is currently in use by the application.
        }
  {*
       * (value type = cuuint64_t)
       * High watermark of the amount of memory from the pool that was in use by the application since
       * the last time it was reset. High watermark can only be reset to zero.
        }
    cudaMemPoolAttr = (cudaMemPoolReuseFollowEventDependencies = $1,
      cudaMemPoolReuseAllowOpportunistic = $2,
      cudaMemPoolReuseAllowInternalDependencies = $3,
      cudaMemPoolAttrReleaseThreshold = $4,
      cudaMemPoolAttrReservedMemCurrent = $5,
      cudaMemPoolAttrReservedMemHigh = $6,
      cudaMemPoolAttrUsedMemCurrent = $7,
      cudaMemPoolAttrUsedMemHigh = $8);

  {*
   * Specifies the type of location 
    }
  {*< Location is a device location, thus id is a device ordinal  }
    cudaMemLocationType = (cudaMemLocationTypeInvalid = 0,cudaMemLocationTypeDevice = 1
      );

  {*
   * Specifies a memory location.
   *
   * To specify a gpu, set type = ::cudaMemLocationTypeDevice and set id = the gpu's device ordinal.
    }
  {*< Specifies the location type, which modifies the meaning of id.  }
  {*< identifier for a given this location's ::CUmemLocationType.  }
    cudaMemLocation = record
        _type : cudaMemLocationType;
        id : longint;
      end;

  {*
   * Specifies the memory protection flags for mapping.
    }
  {*< Default, make the address range not accessible  }
  {*< Make the address range read accessible  }
  {*< Make the address range read-write accessible  }
    cudaMemAccessFlags = (cudaMemAccessFlagsProtNone = 0,cudaMemAccessFlagsProtRead = 1,
      cudaMemAccessFlagsProtReadWrite = 3
      );

  {*
   * Memory access descriptor
    }
  {*< Location on which the request is to change it's accessibility  }
  {*< ::CUmemProt accessibility flags to set on the request  }
    cudaMemAccessDesc = record
        location : cudaMemLocation;
        flags : cudaMemAccessFlags;
      end;

  {*
   * Defines the allocation types available
    }
  {* This allocation type is 'pinned', i.e. cannot migrate from its current
        * location while the application is actively using it
         }
    cudaMemAllocationType = (cudaMemAllocationTypeInvalid = $0,cudaMemAllocationTypePinned = $1,
      cudaMemAllocationTypeMax = $7FFFFFFF);

  {*
   * Flags for specifying particular handle types
    }
  {*< Does not allow any export mechanism. >  }
  {*< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)  }
  {*< Allows a Win32 NT handle to be used for exporting. (HANDLE)  }
  {*< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)  }
    cudaMemAllocationHandleType = (cudaMemHandleTypeNone = $0,cudaMemHandleTypePosixFileDescriptor = $1,
      cudaMemHandleTypeWin32 = $2,cudaMemHandleTypeWin32Kmt = $4
      );

  {*
   * Specifies the properties of allocations made from the pool.
    }
  {*< Allocation type. Currently must be specified as cudaMemAllocationTypePinned  }
  {*< Handle types that will be supported by allocations from the pool.  }
  {*< Location allocations should reside.  }
  {*
       * Windows-specific LPSECURITYATTRIBUTES required when
       * ::cudaMemHandleTypeWin32 is specified.  This security attribute defines
       * the scope of which exported allocations may be tranferred to other
       * processes.  In all other cases, this field is required to be zero.
        }
  {*< reserved for future use, must be 0  }
    cudaMemPoolProps = record
        allocType : cudaMemAllocationType;
        handleTypes : cudaMemAllocationHandleType;
        location : cudaMemLocation;
        win32SecurityAttributes : pointer;
        reserved : array[0..63] of byte;
      end;

  {*
   * Opaque data for exporting a pool allocation
    }
    cudaMemPoolPtrExportData = record
        reserved : array[0..63] of byte;
      end;

  {*
   * Memory allocation node parameters
    }
  {*
      * in: location where the allocation should reside (specified in ::location).
      * ::handleTypes must be ::cudaMemHandleTypeNone. IPC is not supported.
       }
  {*< in: array of memory access descriptors. Used to describe peer GPU access  }
(* Const before type ignored *)
  {*< in: number of memory access descriptors.  Must not exceed the number of GPUs.  }
  {*< in: Number of `accessDescs`s  }
  {*< in: size in bytes of the requested allocation  }
  {*< out: address of the allocation returned by CUDA  }
    cudaMemAllocNodeParams = record
        poolProps : cudaMemPoolProps;
        accessDescs : ^cudaMemAccessDesc;
        accessDescCount : size_t;
        bytesize : size_t;
        dptr : pointer;
      end;

  {*
   * Graph memory attributes
    }
  {*
       * (value type = cuuint64_t)
       * Amount of memory, in bytes, currently associated with graphs.
        }
  {*
       * (value type = cuuint64_t)
       * High watermark of memory, in bytes, associated with graphs since the
       * last time it was reset.  High watermark can only be reset to zero.
        }
  {*
       * (value type = cuuint64_t)
       * Amount of memory, in bytes, currently allocated for use by
       * the CUDA graphs asynchronous allocator.
        }
  {*
       * (value type = cuuint64_t)
       * High watermark of memory, in bytes, currently allocated for use by
       * the CUDA graphs asynchronous allocator.
        }
    cudaGraphMemAttributeType = (cudaGraphMemAttrUsedMemCurrent = $0,
      cudaGraphMemAttrUsedMemHigh = $1,cudaGraphMemAttrReservedMemCurrent = $2,
      cudaGraphMemAttrReservedMemHigh = $3
      );

  {*
   * CUDA device P2P attributes
    }
  {*< A relative value indicating the performance of the link between two devices  }
  {*< Peer access is enabled  }
  {*< Native atomic operation over the link supported  }
  {*< Accessing CUDA arrays over the link supported  }
    cudaDeviceP2PAttr = (cudaDevP2PAttrPerformanceRank = 1,
      cudaDevP2PAttrAccessSupported = 2,
      cudaDevP2PAttrNativeAtomicSupported = 3,
      cudaDevP2PAttrCudaArrayAccessSupported = 4
      );

  {*
   * CUDA UUID types
    }
{$if not defined(CUuuid_st)}
  {*< CUDA definition of UUID  }

    CUuuid_st = record
        bytes : array[0..15] of char;
      end;

   CUuuid = CUuuid_st;
{$endif}

   cudaUUID_t  = CUuuid_st;
  {*
   * CUDA device properties
    }
  {*< ASCII string identifying device  }
  {*< 16-byte unique identifier  }
  {*< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms  }
  {*< LUID device node mask. Value is undefined on TCC and non-Windows platforms  }
  {*< Global memory available on device in bytes  }
  {*< Shared memory available per block in bytes  }
  {*< 32-bit registers available per block  }
  {*< Warp size in threads  }
  {*< Maximum pitch in bytes allowed by memory copies  }
  {*< Maximum number of threads per block  }
  {*< Maximum size of each dimension of a block  }
  {*< Maximum size of each dimension of a grid  }
  {*< Deprecated, Clock frequency in kilohertz  }
  {*< Constant memory available on device in bytes  }
  {*< Major compute capability  }
  {*< Minor compute capability  }
  {*< Alignment requirement for textures  }
  {*< Pitch alignment requirement for texture references bound to pitched memory  }
  {*< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.  }
  {*< Number of multiprocessors on device  }
  {*< Deprecated, Specified whether there is a run time limit on kernels  }
  {*< Device is integrated as opposed to discrete  }
  {*< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer  }
  {*< Deprecated, Compute mode (See ::cudaComputeMode)  }
  {*< Maximum 1D texture size  }
  {*< Maximum 1D mipmapped texture size  }
  {*< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.  }
  {*< Maximum 2D texture dimensions  }
  {*< Maximum 2D mipmapped texture dimensions  }
  {*< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory  }
  {*< Maximum 2D texture dimensions if texture gather operations have to be performed  }
  {*< Maximum 3D texture dimensions  }
  {*< Maximum alternate 3D texture dimensions  }
  {*< Maximum Cubemap texture dimensions  }
  {*< Maximum 1D layered texture dimensions  }
  {*< Maximum 2D layered texture dimensions  }
  {*< Maximum Cubemap layered texture dimensions  }
  {*< Maximum 1D surface size  }
  {*< Maximum 2D surface dimensions  }
  {*< Maximum 3D surface dimensions  }
  {*< Maximum 1D layered surface dimensions  }
  {*< Maximum 2D layered surface dimensions  }
  {*< Maximum Cubemap surface dimensions  }
  {*< Maximum Cubemap layered surface dimensions  }
  {*< Alignment requirements for surfaces  }
  {*< Device can possibly execute multiple kernels concurrently  }
  {*< Device has ECC support enabled  }
  {*< PCI bus ID of the device  }
  {*< PCI device ID of the device  }
  {*< PCI domain ID of the device  }
  {*< 1 if device is a Tesla device using TCC driver, 0 otherwise  }
  {*< Number of asynchronous engines  }
  {*< Device shares a unified address space with the host  }
  {*< Deprecated, Peak memory clock frequency in kilohertz  }
  {*< Global memory bus width in bits  }
  {*< Size of L2 cache in bytes  }
  {*< Device's maximum l2 persisting lines capacity setting in bytes  }
  {*< Maximum resident threads per multiprocessor  }
  {*< Device supports stream priorities  }
  {*< Device supports caching globals in L1  }
  {*< Device supports caching locals in L1  }
  {*< Shared memory available per multiprocessor in bytes  }
  {*< 32-bit registers available per multiprocessor  }
  {*< Device supports allocating managed memory on this system  }
  {*< Device is on a multi-GPU board  }
  {*< Unique identifier for a group of devices on the same multi-GPU board  }
  {*< Link between the device and the host supports native atomic operations  }
  {*< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance  }
  {*< Device supports coherently accessing pageable memory without calling cudaHostRegister on it  }
  {*< Device can coherently access managed memory concurrently with the CPU  }
  {*< Device supports Compute Preemption  }
  {*< Device can access host registered memory at the same virtual address as the CPU  }
  {*< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel  }
  {*< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.  }
  {*< Per device maximum shared memory per block usable by special opt in  }
  {*< Device accesses pageable memory via the host's page tables  }
  {*< Host can directly access managed memory on the device without migration.  }
  {*< Maximum number of resident blocks per multiprocessor  }
  {*< The maximum value of ::cudaAccessPolicyWindow::num_bytes.  }
  {*< Shared memory reserved by CUDA driver per block in bytes  }
  {*< Device supports host memory registration via ::cudaHostRegister.  }
  {*< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise  }
  {*< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU  }
  {*< External timeline semaphore interop is supported on the device  }
  {*< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise  }
  {*< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise  }
  {*< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum  }
  {*< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values  }
  {*< Bitmask of handle types supported with mempool-based IPC  }
  {*< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays  }
  {*< Device supports IPC Events.  }
  {*< Indicates device supports cluster launch  }
  {*< Indicates device supports unified pointers  }
  {*< Reserved for future use  }
    cudaDeviceProp = record
        name : array[0..255] of char;
        uuid : cudaUUID_t;
        luid : array[0..7] of char;
        luidDeviceNodeMask : dword;
        totalGlobalMem : size_t;
        sharedMemPerBlock : size_t;
        regsPerBlock : longint;
        warpSize : longint;
        memPitch : size_t;
        maxThreadsPerBlock : longint;
        maxThreadsDim : array[0..2] of longint;
        maxGridSize : array[0..2] of longint;
        clockRate : longint;
        totalConstMem : size_t;
        major : longint;
        minor : longint;
        textureAlignment : size_t;
        texturePitchAlignment : size_t;
        deviceOverlap : longint;
        multiProcessorCount : longint;
        kernelExecTimeoutEnabled : longint;
        integrated : longint;
        canMapHostMemory : longint;
        computeMode : longint;
        maxTexture1D : longint;
        maxTexture1DMipmap : longint;
        maxTexture1DLinear : longint;
        maxTexture2D : array[0..1] of longint;
        maxTexture2DMipmap : array[0..1] of longint;
        maxTexture2DLinear : array[0..2] of longint;
        maxTexture2DGather : array[0..1] of longint;
        maxTexture3D : array[0..2] of longint;
        maxTexture3DAlt : array[0..2] of longint;
        maxTextureCubemap : longint;
        maxTexture1DLayered : array[0..1] of longint;
        maxTexture2DLayered : array[0..2] of longint;
        maxTextureCubemapLayered : array[0..1] of longint;
        maxSurface1D : longint;
        maxSurface2D : array[0..1] of longint;
        maxSurface3D : array[0..2] of longint;
        maxSurface1DLayered : array[0..1] of longint;
        maxSurface2DLayered : array[0..2] of longint;
        maxSurfaceCubemap : longint;
        maxSurfaceCubemapLayered : array[0..1] of longint;
        surfaceAlignment : size_t;
        concurrentKernels : longint;
        ECCEnabled : longint;
        pciBusID : longint;
        pciDeviceID : longint;
        pciDomainID : longint;
        tccDriver : longint;
        asyncEngineCount : longint;
        unifiedAddressing : longint;
        memoryClockRate : longint;
        memoryBusWidth : longint;
        l2CacheSize : longint;
        persistingL2CacheMaxSize : longint;
        maxThreadsPerMultiProcessor : longint;
        streamPrioritiesSupported : longint;
        globalL1CacheSupported : longint;
        localL1CacheSupported : longint;
        sharedMemPerMultiprocessor : size_t;
        regsPerMultiprocessor : longint;
        managedMemory : longint;
        isMultiGpuBoard : longint;
        multiGpuBoardGroupID : longint;
        hostNativeAtomicSupported : longint;
        singleToDoublePrecisionPerfRatio : longint;
        pageableMemoryAccess : longint;
        concurrentManagedAccess : longint;
        computePreemptionSupported : longint;
        canUseHostPointerForRegisteredMem : longint;
        cooperativeLaunch : longint;
        cooperativeMultiDeviceLaunch : longint;
        sharedMemPerBlockOptin : size_t;
        pageableMemoryAccessUsesHostPageTables : longint;
        directManagedMemAccessFromHost : longint;
        maxBlocksPerMultiProcessor : longint;
        accessPolicyMaxWindowSize : longint;
        reservedSharedMemPerBlock : size_t;
        hostRegisterSupported : longint;
        sparseCudaArraySupported : longint;
        hostRegisterReadOnlySupported : longint;
        timelineSemaphoreInteropSupported : longint;
        memoryPoolsSupported : longint;
        gpuDirectRDMASupported : longint;
        gpuDirectRDMAFlushWritesOptions : dword;
        gpuDirectRDMAWritesOrdering : longint;
        memoryPoolSupportedHandleTypes : dword;
        deferredMappingCudaArraySupported : longint;
        ipcEventSupported : longint;
        clusterLaunch : longint;
        unifiedFunctionPointers : longint;
        reserved2 : array[0..1] of longint;
        reserved : array[0..60] of longint;
      end;

  {*
   * CUDA IPC Handle Size
    }

  const
    CUDA_IPC_HANDLE_SIZE = 64;    
  {*
   * CUDA IPC event handle
    }

  type
    cudaIpcEventHandle_st = record
        reserved : array[0..(CUDA_IPC_HANDLE_SIZE)-1] of char;
      end;
    cudaIpcEventHandle_t = cudaIpcEventHandle_st;
  {*
   * CUDA IPC memory handle
    }

    cudaIpcMemHandle_st = record
        reserved : array[0..(CUDA_IPC_HANDLE_SIZE)-1] of char;
      end;
    cudaIpcMemHandle_t = cudaIpcMemHandle_st;
  {*
   * External memory handle types
    }
  {*
       * Handle is an opaque file descriptor
        }
  {*
       * Handle is an opaque shared NT handle
        }
  {*
       * Handle is an opaque, globally shared handle
        }
  {*
       * Handle is a D3D12 heap object
        }
  {*
       * Handle is a D3D12 committed resource
        }
  {*
      *  Handle is a shared NT handle to a D3D11 resource
       }
  {*
      *  Handle is a globally shared handle to a D3D11 resource
       }
  {*
      *  Handle is an NvSciBuf object
       }
    cudaExternalMemoryHandleType = (cudaExternalMemoryHandleTypeOpaqueFd = 1,
      cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
      cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
      cudaExternalMemoryHandleTypeD3D12Heap = 4,
      cudaExternalMemoryHandleTypeD3D12Resource = 5,
      cudaExternalMemoryHandleTypeD3D11Resource = 6,
      cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
      cudaExternalMemoryHandleTypeNvSciBuf = 8
      );

  {*
   * Indicates that the external memory object is a dedicated resource
    }

  const
    cudaExternalMemoryDedicated = $1;    
  {* When the /p flags parameter of ::cudaExternalSemaphoreSignalParams
   * contains this flag, it indicates that signaling an external semaphore object
   * should skip performing appropriate memory synchronization operations over all
   * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
   * which otherwise are performed by default to ensure data coherency with other
   * importers of the same NvSciBuf memory objects.
    }
    cudaExternalSemaphoreSignalSkipNvSciBufMemSync = $01;    
  {* When the /p flags parameter of ::cudaExternalSemaphoreWaitParams
   * contains this flag, it indicates that waiting an external semaphore object
   * should skip performing appropriate memory synchronization operations over all
   * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
   * which otherwise are performed by default to ensure data coherency with other
   * importers of the same NvSciBuf memory objects.
    }
    cudaExternalSemaphoreWaitSkipNvSciBufMemSync = $02;    
  {*
   * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
   * it indicates that application need signaler specific NvSciSyncAttr
   * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
    }
    cudaNvSciSyncAttrSignal = $1;    
  {*
   * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
   * it indicates that application need waiter specific NvSciSyncAttr
   * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
    }
    cudaNvSciSyncAttrWait = $2;    
  {*
   * External memory handle descriptor
    }
  {*
       * Type of the handle
        }
  {*
           * File descriptor referencing the memory object. Valid
           * when type is
           * ::cudaExternalMemoryHandleTypeOpaqueFd
            }
  {*
           * Win32 handle referencing the semaphore object. Valid when
           * type is one of the following:
           * - ::cudaExternalMemoryHandleTypeOpaqueWin32
           * - ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
           * - ::cudaExternalMemoryHandleTypeD3D12Heap 
           * - ::cudaExternalMemoryHandleTypeD3D12Resource
  		 * - ::cudaExternalMemoryHandleTypeD3D11Resource
  		 * - ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
           * Exactly one of 'handle' and 'name' must be non-NULL. If
           * type is one of the following: 
           * ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
           * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
           * then 'name' must be NULL.
            }
  {*
               * Valid NT handle. Must be NULL if 'name' is non-NULL
                }
  {*
               * Name of a valid memory object.
               * Must be NULL if 'handle' is non-NULL.
                }
(* Const before type ignored *)
  {*
           * A handle representing NvSciBuf Object. Valid when type
           * is ::cudaExternalMemoryHandleTypeNvSciBuf
            }
(* Const before type ignored *)
  {*
       * Size of the memory allocation
        }
  {*
       * Flags must either be zero or ::cudaExternalMemoryDedicated
        }

  type
    cudaExternalMemoryHandleDesc = record
        _type : cudaExternalMemoryHandleType;
        handle : record
            case longint of
              0 : ( fd : longint );
              1 : ( win32 : record
                  handle : pointer;
                  name : pointer;
                end );
              2 : ( nvSciBufObject : pointer );
            end;
        size : qword;
        flags : dword;
      end;

  {*
   * External memory buffer descriptor
    }
  {*
       * Offset into the memory object where the buffer's base is
        }
  {*
       * Size of the buffer
        }
  {*
       * Flags reserved for future use. Must be zero.
        }
    cudaExternalMemoryBufferDesc = record
        offset : qword;
        size : qword;
        flags : dword;
      end;

  {*
   * External memory mipmap descriptor
    }
  {*
       * Offset into the memory object where the base level of the
       * mipmap chain is.
        }
  {*
       * Format of base level of the mipmap chain
        }
  {*
       * Dimensions of base level of the mipmap chain
        }
  {*
       * Flags associated with CUDA mipmapped arrays.
       * See ::cudaMallocMipmappedArray
        }
  {*
       * Total number of levels in the mipmap chain
        }
    cudaExternalMemoryMipmappedArrayDesc = record
        offset : qword;
        formatDesc : cudaChannelFormatDesc;
        extent : cudaExtent;
        flags : dword;
        numLevels : dword;
      end;

  {*
   * External semaphore handle types
    }
  {*
       * Handle is an opaque file descriptor
        }
  {*
       * Handle is an opaque shared NT handle
        }
  {*
       * Handle is an opaque, globally shared handle
        }
  {*
       * Handle is a shared NT handle referencing a D3D12 fence object
        }
  {*
       * Handle is a shared NT handle referencing a D3D11 fence object
        }
  {*
       * Opaque handle to NvSciSync Object
        }
  {*
       * Handle is a shared NT handle referencing a D3D11 keyed mutex object
        }
  {*
       * Handle is a shared KMT handle referencing a D3D11 keyed mutex object
        }
  {*
       * Handle is an opaque handle file descriptor referencing a timeline semaphore
        }
  {*
       * Handle is an opaque handle file descriptor referencing a timeline semaphore
        }
    cudaExternalSemaphoreHandleType = (cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
      cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
      cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
      cudaExternalSemaphoreHandleTypeD3D12Fence = 4,
      cudaExternalSemaphoreHandleTypeD3D11Fence = 5,
      cudaExternalSemaphoreHandleTypeNvSciSync = 6,
      cudaExternalSemaphoreHandleTypeKeyedMutex = 7,
      cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
      cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
      cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10
      );

  {*
   * External semaphore handle descriptor
    }
  {*
       * Type of the handle
        }
  {*
           * File descriptor referencing the semaphore object. Valid when
           * type is one of the following:
           * - ::cudaExternalSemaphoreHandleTypeOpaqueFd
           * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
            }
  {*
           * Win32 handle referencing the semaphore object. Valid when
           * type is one of the following:
           * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32
           * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
           * - ::cudaExternalSemaphoreHandleTypeD3D12Fence
           * - ::cudaExternalSemaphoreHandleTypeD3D11Fence
           * - ::cudaExternalSemaphoreHandleTypeKeyedMutex
           * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
           * Exactly one of 'handle' and 'name' must be non-NULL. If
           * type is one of the following:
           * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
           * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt
           * then 'name' must be NULL.
            }
  {*
               * Valid NT handle. Must be NULL if 'name' is non-NULL
                }
  {*
               * Name of a valid synchronization primitive.
               * Must be NULL if 'handle' is non-NULL.
                }
(* Const before type ignored *)
  {*
           * Valid NvSciSyncObj. Must be non NULL
            }
(* Const before type ignored *)
  {*
       * Flags reserved for the future. Must be zero.
        }
    cudaExternalSemaphoreHandleDesc = record
        _type : cudaExternalSemaphoreHandleType;
        handle : record
            case longint of
              0 : ( fd : longint );
              1 : ( win32 : record
                  handle : pointer;
                  name : pointer;
                end );
              2 : ( nvSciSyncObj : pointer );
            end;
        flags : dword;
      end;

  {*
   * External semaphore signal parameters(deprecated)
    }
  {*
           * Parameters for fence objects
            }
  {*
               * Value of fence to be signaled
                }
  {*
               * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
               * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
                }
  {*
           * Parameters for keyed mutex objects
            }
  {
               * Value of key to release the mutex with
                }
  {*
       * Only when ::cudaExternalSemaphoreSignalParams is used to
       * signal a ::cudaExternalSemaphore_t of type
       * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
       * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
       * that while signaling the ::cudaExternalSemaphore_t, no memory
       * synchronization operations should be performed for any external memory
       * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
       * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
        }
    cudaExternalSemaphoreSignalParams_v1 = record
        params : record
            fence : record
                value : qword;
              end;
            nvSciSync : record
                case longint of
                  0 : ( fence : pointer );
                  1 : ( reserved : qword );
                end;
            keyedMutex : record
                key : qword;
              end;
          end;
        flags : dword;
      end;

  {*
  * External semaphore wait parameters(deprecated)
   }
  {*
          * Parameters for fence objects
           }
  {*
              * Value of fence to be waited on
               }
  {*
               * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
               * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
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
       * Only when ::cudaExternalSemaphoreSignalParams is used to
       * signal a ::cudaExternalSemaphore_t of type
       * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
       * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
       * that while waiting for the ::cudaExternalSemaphore_t, no memory
       * synchronization operations should be performed for any external memory
       * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
       * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
        }
    cudaExternalSemaphoreWaitParams_v1 = record
        params : record
            fence : record
                value : qword;
              end;
            nvSciSync : record
                case longint of
                  0 : ( fence : pointer );
                  1 : ( reserved : qword );
                end;
            keyedMutex : record
                key : qword;
                timeoutMs : dword;
              end;
          end;
        flags : dword;
      end;

  {*
   * External semaphore signal parameters, compatible with driver type
    }
  {*
           * Parameters for fence objects
            }
  {*
               * Value of fence to be signaled
                }
  {*
               * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
               * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
                }
  {*
           * Parameters for keyed mutex objects
            }
  {
               * Value of key to release the mutex with
                }
  {*
       * Only when ::cudaExternalSemaphoreSignalParams is used to
       * signal a ::cudaExternalSemaphore_t of type
       * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
       * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
       * that while signaling the ::cudaExternalSemaphore_t, no memory
       * synchronization operations should be performed for any external memory
       * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
       * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
        }
    cudaExternalSemaphoreSignalParams = record
        params : record
            fence : record
                value : qword;
              end;
            nvSciSync : record
                case longint of
                  0 : ( fence : pointer );
                  1 : ( reserved : qword );
                end;
            keyedMutex : record
                key : qword;
              end;
            reserved : array[0..11] of dword;
          end;
        flags : dword;
        reserved : array[0..15] of dword;
      end;

  {*
   * External semaphore wait parameters, compatible with driver type
    }
  {*
          * Parameters for fence objects
           }
  {*
              * Value of fence to be waited on
               }
  {*
               * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
               * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
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
       * Only when ::cudaExternalSemaphoreSignalParams is used to
       * signal a ::cudaExternalSemaphore_t of type
       * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
       * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
       * that while waiting for the ::cudaExternalSemaphore_t, no memory
       * synchronization operations should be performed for any external memory
       * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
       * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
        }
    cudaExternalSemaphoreWaitParams = record
        params : record
            fence : record
                value : qword;
              end;
            nvSciSync : record
                case longint of
                  0 : ( fence : pointer );
                  1 : ( reserved : qword );
                end;
            keyedMutex : record
                key : qword;
                timeoutMs : dword;
              end;
            reserved : array[0..9] of dword;
          end;
        flags : dword;
        reserved : array[0..15] of dword;
      end;

  {******************************************************************************
  *                                                                              *
  *  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
  *                                                                              *
  ****************************************************************************** }
  {*
   * CUDA Error types
    }

    cudaError_t = cudaError;
  {*
   * CUDA stream
    }

    cudaStream_t = ^CUstream_st;
    CUstream_st = record end;
  {*
   * CUDA event types
    }

    cudaEvent_t = ^CUevent_st;
    CUevent_st = record end;
  {*
   * CUDA graphics resource types
    }

    cudaGraphicsResource_t = ^cudaGraphicsResource;
  {*
   * CUDA external memory
    }

    cudaExternalMemory_t = ^CUexternalMemory_st;
    CUexternalMemory_st = record end;
  {*
   * CUDA external semaphore
    }

    cudaExternalSemaphore_t = ^CUexternalSemaphore_st;
    CUexternalSemaphore_st = record end;
  {*
   * CUDA graph
    }

    cudaGraph_t = ^CUgraph_st;
    CUgraph_st = record end;
  {*
   * CUDA graph node.
    }

    cudaGraphNode_t = ^CUgraphNode_st;
    CUgraphNode_st = record end;
  {*
   * CUDA user object for graphs
    }

    cudaUserObject_t = ^CUuserObject_st;
    CUuserObject_st = record end;
  {*
   * CUDA function
    }

    cudaFunction_t = ^CUfunc_st;
    CUfunc_st = record end;
  {*
   * CUDA kernel
    }

    cudaKernel_t = ^CUkern_st;
    CUkern_st = record end;
  {*
   * CUDA memory pool
    }

    cudaMemPool_t = ^CUmemPoolHandle_st;
    CUmemPoolHandle_st = record end;
  {*
   * CUDA cooperative group scope
    }
  {*< Invalid cooperative group scope  }
  {*< Scope represented by a grid_group  }
  {*< Scope represented by a multi_grid_group  }
    cudaCGScope = (cudaCGScopeInvalid = 0,cudaCGScopeGrid = 1,
      cudaCGScopeMultiGrid = 2);


(*    struct __device_builtin__ dim3
    {
        unsigned int x, y, z;
    #if defined(__cplusplus)
    #if __cplusplus >= 201103L
        __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
        __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
        __host__ __device__ constexpr operator uint3(void) const { return uint3{x, y, z}; }
    #else
        __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
        __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
        __host__ __device__ operator uint3(void) const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
    #endif
    };
*)

    { dim3 }

    dim3 = record
      x, y, z : longword;
      constructor Create(const aX:longword ; const aY:longword =1; const aZ:longword =1);
      class operator add(const a, b :dim3):dim3;
    end;

  {*
   * CUDA launch parameters
    }
  {*< Device function symbol  }
  {*< Grid dimentions  }
  {*< Block dimentions  }
  {*< Arguments  }
  {*< Shared memory  }
  {*< Stream identifier  }
    cudaLaunchParams = record
        func : pointer;
        gridDim : dim3;
        blockDim : dim3;
        args : ^pointer;
        sharedMem : size_t;
        stream : cudaStream_t;
      end;

  {*
   * CUDA GPU kernel node parameters
    }
  {*< Kernel to launch  }
  {*< Grid dimensions  }
  {*< Block dimensions  }
  {*< Dynamic shared-memory size per thread block in bytes  }
  {*< Array of pointers to individual kernel arguments }
  {*< Pointer to kernel arguments in the "extra" format  }
    cudaKernelNodeParams = record
        func : pointer;
        gridDim : dim3;
        blockDim : dim3;
        sharedMemBytes : dword;
        kernelParams : ^pointer;
        extra : ^pointer;
      end;

  {*
   * External semaphore signal node parameters
    }
  {*< Array of external semaphore handles.  }
(* Const before type ignored *)
  {*< Array of external semaphore signal parameters.  }
  {*< Number of handles and parameters supplied in extSemArray and paramsArray.  }
    cudaExternalSemaphoreSignalNodeParams = record
        extSemArray : ^cudaExternalSemaphore_t;
        paramsArray : ^cudaExternalSemaphoreSignalParams;
        numExtSems : dword;
      end;

  {*
   * External semaphore wait node parameters
    }
  {*< Array of external semaphore handles.  }
(* Const before type ignored *)
  {*< Array of external semaphore wait parameters.  }
  {*< Number of handles and parameters supplied in extSemArray and paramsArray.  }
    cudaExternalSemaphoreWaitNodeParams = record
        extSemArray : ^cudaExternalSemaphore_t;
        paramsArray : ^cudaExternalSemaphoreWaitParams;
        numExtSems : dword;
      end;

  {*
  * CUDA Graph node types
   }
  {*< GPU kernel node  }
  {*< Memcpy node  }
  {*< Memset node  }
  {*< Host (executable) node  }
  {*< Node which executes an embedded graph  }
  {*< Empty (no-op) node  }
  {*< External event wait node  }
  {*< External event record node  }
  {*< External semaphore signal node  }
  {*< External semaphore wait node  }
  {*< Memory allocation node  }
  {*< Memory free node  }
    cudaGraphNodeType = (cudaGraphNodeTypeKernel = $00,cudaGraphNodeTypeMemcpy = $01,
      cudaGraphNodeTypeMemset = $02,cudaGraphNodeTypeHost = $03,
      cudaGraphNodeTypeGraph = $04,cudaGraphNodeTypeEmpty = $05,
      cudaGraphNodeTypeWaitEvent = $06,cudaGraphNodeTypeEventRecord = $07,
      cudaGraphNodeTypeExtSemaphoreSignal = $08,
      cudaGraphNodeTypeExtSemaphoreWait = $09,
      cudaGraphNodeTypeMemAlloc = $0a,cudaGraphNodeTypeMemFree = $0b,
      cudaGraphNodeTypeCount);

  {*
   * CUDA executable (launchable) graph
    }

    cudaGraphExec_t = ^CUgraphExec_st;
    CUgraphExec_st = record end;
  {*
  * CUDA Graph Update error types
   }
  {*< The update succeeded  }
  {*< The update failed for an unexpected reason which is described in the return value of the function  }
  {*< The update failed because the topology changed  }
  {*< The update failed because a node type changed  }
  {*< The update failed because the function of a kernel node changed (CUDA driver < 11.2)  }
  {*< The update failed because the parameters changed in a way that is not supported  }
  {*< The update failed because something about the node is not supported  }
  {*< The update failed because the function of a kernel node changed in an unsupported way  }
  {*< The update failed because the node attributes changed in a way that is not supported  }
    cudaGraphExecUpdateResult = (cudaGraphExecUpdateSuccess = $0,cudaGraphExecUpdateError = $1,
      cudaGraphExecUpdateErrorTopologyChanged = $2,
      cudaGraphExecUpdateErrorNodeTypeChanged = $3,
      cudaGraphExecUpdateErrorFunctionChanged = $4,
      cudaGraphExecUpdateErrorParametersChanged = $5,
      cudaGraphExecUpdateErrorNotSupported = $6,
      cudaGraphExecUpdateErrorUnsupportedFunctionChange = $7,
      cudaGraphExecUpdateErrorAttributesChanged = $8
      );

  {*
   * Graph instantiation results
   }
  {*< Instantiation succeeded  }
  {*< Instantiation failed for an unexpected reason which is described in the return value of the function  }
  {*< Instantiation failed due to invalid structure, such as cycles  }
  {*< Instantiation for device launch failed because the graph contained an unsupported operation  }
  {*< Instantiation for device launch failed due to the nodes belonging to different contexts  }

    cudaGraphInstantiateResult = (cudaGraphInstantiateSuccess = 0,cudaGraphInstantiateError = 1,
      cudaGraphInstantiateInvalidStructure = 2,
      cudaGraphInstantiateNodeOperationNotSupported = 3,
      cudaGraphInstantiateMultipleDevicesNotSupported = 4
      );
  {*
   * Graph instantiation parameters
    }
  {*< Instantiation flags  }
  {*< Upload stream  }
  {*< The node which caused instantiation to fail, if any  }
  {*< Whether instantiation was successful.  If it failed, the reason why  }

    cudaGraphInstantiateParams_st = record
        flags : qword;
        uploadStream : cudaStream_t;
        errNode_out : cudaGraphNode_t;
        result_out : cudaGraphInstantiateResult;
      end;
    cudaGraphInstantiateParams = cudaGraphInstantiateParams_st;
  {*
   * Result information returned by cudaGraphExecUpdate
    }
  {*
       * Gives more specific detail when a cuda graph update fails. 
        }
  {*
       * The "to node" of the error edge when the topologies do not match.
       * The error node when the error is associated with a specific node.
       * NULL when the error is generic.
        }
  {*
       * The from node of error edge when the topologies do not match. Otherwise NULL.
        }

    cudaGraphExecUpdateResultInfo_st = record
        result : cudaGraphExecUpdateResult;
        errorNode : cudaGraphNode_t;
        errorFromNode : cudaGraphNode_t;
      end;
    cudaGraphExecUpdateResultInfo = cudaGraphExecUpdateResultInfo_st;
  {*
   * Flags to specify search options to be used with ::cudaGetDriverEntryPoint
   * For more details see ::cuGetProcAddress
    }  {*< Default search mode for driver symbols.  }
  {*< Search for legacy versions of driver symbols.  }
  {*< Search for per-thread versions of driver symbols.  }
    cudaGetDriverEntryPointFlags = (cudaEnableDefault = $0,cudaEnableLegacyStream = $1,
      cudaEnablePerThreadDefaultStream = $2
      );

  {*
   * Enum for status from obtaining driver entry points, used with ::cudaApiGetDriverEntryPoint
    }
  {*< Search for symbol found a match  }
  {*< Search for symbol was not found  }
  {*< Search for symbol was found but version wasn't great enough  }
    cudaDriverEntryPointQueryResult = (cudaDriverEntryPointSuccess = 0,cudaDriverEntryPointSymbolNotFound = 1,
      cudaDriverEntryPointVersionNotSufficent = 2
      );

  {*
   * CUDA Graph debug write options
    }
  {*< Output all debug data as if every debug flag is enabled  }
  {*< Adds cudaKernelNodeParams to output  }
  {*< Adds cudaMemcpy3DParms to output  }
  {*< Adds cudaMemsetParams to output  }
  {*< Adds cudaHostNodeParams to output  }
  {*< Adds cudaEvent_t handle from record and wait nodes to output  }
  {*< Adds cudaExternalSemaphoreSignalNodeParams values to output  }
  {*< Adds cudaExternalSemaphoreWaitNodeParams to output  }
  {*< Adds cudaKernelNodeAttrID values to output  }
  {*< Adds node handles and every kernel function handle to output  }
    cudaGraphDebugDotFlags = (cudaGraphDebugDotFlagsVerbose = 1 shl 0,
      cudaGraphDebugDotFlagsKernelNodeParams = 1 shl 2,
      cudaGraphDebugDotFlagsMemcpyNodeParams = 1 shl 3,
      cudaGraphDebugDotFlagsMemsetNodeParams = 1 shl 4,
      cudaGraphDebugDotFlagsHostNodeParams = 1 shl 5,
      cudaGraphDebugDotFlagsEventNodeParams = 1 shl 6,
      cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1 shl 7,
      cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 1 shl 8,
      cudaGraphDebugDotFlagsKernelNodeAttributes = 1 shl 9,
      cudaGraphDebugDotFlagsHandles = 1 shl 10
      );

  {*
   * Flags for instantiating a graph
    }
  {*< Automatically free memory allocated in a graph before relaunching.  }
  {*< Automatically upload the graph after instantiaton.  }
  {*< Instantiate the graph to be launchable from the device.  }
  {*< Run the graph using the per-node priority attributes rather than the
                                                        priority of the stream it is launched into.  }
    cudaGraphInstantiateFlags = (cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
      cudaGraphInstantiateFlagUpload = 2,
      cudaGraphInstantiateFlagDeviceLaunch = 4,
      cudaGraphInstantiateFlagUseNodePriority = 8
      );


    cudaLaunchMemSyncDomain = (cudaLaunchMemSyncDomainDefault = 0,
      cudaLaunchMemSyncDomainRemote = 1
      );

    cudaLaunchMemSyncDomainMap_st = record
        default_ : byte;
        remote : byte;
      end;
    cudaLaunchMemSyncDomainMap = cudaLaunchMemSyncDomainMap_st;
  {*
   * Launch attributes enum; used as id field of ::cudaLaunchAttribute
    }
  {*< Ignored entry, for convenient composition  }
  {*< Valid for streams, graph nodes, launches.  }
  {*< Valid for graph nodes, launches.  }
  {*< Valid for streams.  }
  {*< Valid for graph nodes, launches.  }
  {*< Valid for graph nodes, launches.  }
  {*< Valid for launches. Setting
                                                                    programmaticStreamSerializationAllowed to non-0
                                                                    signals that the kernel will use programmatic
                                                                    means to resolve its stream dependency, so that
                                                                    the CUDA runtime should opportunistically allow
                                                                    the grid's execution to overlap with the previous
                                                                    kernel in the stream, if that kernel requests the
                                                                    overlap. The dependent launches can choose to wait on
                                                                    the dependency using the programmatic sync
                                                                    (cudaGridDependencySynchronize() or equivalent PTX
                                                                    instructions).  }
  {*< Valid for launches. Event recorded through this
                                                                    launch attribute is guaranteed to only trigger after
                                                                    all block in the associated kernel trigger the event.
                                                                    A block can trigger the event programmatically in a
                                                                    future CUDA release. A trigger can also be inserted at
                                                                    the beginning of each block's execution if
                                                                    triggerAtBlockStart is set to non-0. The dependent
                                                                    launches can choose to wait on the dependency using
                                                                    the programmatic sync (cudaGridDependencySynchronize()
                                                                    or equivalent PTX instructions). Note that dependents
                                                                    (including the CPU thread calling
                                                                    cudaEventSynchronize()) are not guaranteed to observe
                                                                    the release precisely when it is released. For
                                                                    example, cudaEventSynchronize() may only observe the
                                                                    event trigger long after the associated kernel has
                                                                    completed. This recording type is primarily meant for
                                                                    establishing programmatic dependency between device
                                                                    tasks. The event supplied must not be an interprocess
                                                                    or interop event. The event must disable timing (i.e.
                                                                    created with ::cudaEventDisableTiming flag set).  }
  {*< Valid for streams, graph nodes, launches.  }

    cudaLaunchAttributeID = (cudaLaunchAttributeIgnore = 0,cudaLaunchAttributeAccessPolicyWindow = 1,
      cudaLaunchAttributeCooperative = 2,
      cudaLaunchAttributeSynchronizationPolicy = 3,
      cudaLaunchAttributeClusterDimension = 4,
      cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
      cudaLaunchAttributeProgrammaticStreamSerialization = 6,
      cudaLaunchAttributeProgrammaticEvent = 7,
      cudaLaunchAttributePriority = 8,cudaLaunchAttributeMemSyncDomainMap = 9,
      cudaLaunchAttributeMemSyncDomain = 10
      );
    cudaStreamAttrID = cudaLaunchAttributeID;
    cudaKernelNodeAttrID = cudaLaunchAttributeID;
  {*
   * Launch attributes union; used as value field of ::cudaLaunchAttribute
    }
  { Pad to 64 bytes  }
  {*< Attribute ::cudaAccessPolicyWindow.  }
  {*< Nonzero indicates a cooperative kernel (see ::cudaLaunchCooperativeKernel).  }
  {*< ::cudaSynchronizationPolicy for work queued up in this stream  }
  {*< Cluster dimensions for the kernel node.  }
  {*< Cluster scheduling policy preference for the kernel node.  }
  {*< Execution priority of the kernel.  }

    cudaLaunchAttributeValue = record
        case longint of
          0 : ( pad : array[0..63] of char );
          1 : ( accessPolicyWindow : cudaAccessPolicyWindow );
          2 : ( cooperative : longint );
          3 : ( syncPolicy : cudaSynchronizationPolicy );
          4 : ( clusterDim : record
              x : dword;
              y : dword;
              z : dword;
            end );
          5 : ( clusterSchedulingPolicyPreference : cudaClusterSchedulingPolicy );
          6 : ( programmaticStreamSerializationAllowed : longint );
          7 : ( programmaticEvent : record
              event : cudaEvent_t;
              flags : longint;
              triggerAtBlockStart : longint;
            end );
          8 : ( priority : longint );
          9 : ( memSyncDomainMap : cudaLaunchMemSyncDomainMap );
          10 : ( memSyncDomain : cudaLaunchMemSyncDomain );
        end;
  {*
   * Launch attribute
    }

    cudaLaunchAttribute_st = record
        id : cudaLaunchAttributeID;
        pad : array[0..(8-(sizeof(cudaLaunchAttributeID)))-1] of char;
        val : cudaLaunchAttributeValue;
      end;
    cudaLaunchAttribute = cudaLaunchAttribute_st;
  {*
   * CUDA extensible launch configuration
    }
  {*< Grid dimensions  }
  {*< Block dimensions  }
  {*< Dynamic shared-memory size per thread block in bytes  }
  {*< Stream identifier  }
  {*< nullable if numAttrs == 0  }
  {*< Number of attributes populated in attrs  }

    cudaLaunchConfig_st = record
        gridDim : dim3;
        blockDim : dim3;
        dynamicSmemBytes : size_t;
        stream : cudaStream_t;
        attrs : ^cudaLaunchAttribute;
        numAttrs : dword;
      end;
    cudaLaunchConfig_t = cudaLaunchConfig_st;

    pcudaJitOption = ^cudaJitOption;
    cudaJitOption = (
        (**
         * Max number of registers that a thread may use.\n
         * Option type: unsigned int\n
         * Applies to: compiler only
         *)
        cudaJitMaxRegisters = 0,

        (**
         * IN: Specifies minimum number of threads per block to target compilation
         * for\n
         * OUT: Returns the number of threads the compiler actually targeted.
         * This restricts the resource utilization of the compiler (e.g. max
         * registers) such that a block with the given number of threads should be
         * able to launch based on register limitations. Note, this option does not
         * currently take into account any other resource limitations, such as
         * shared memory utilization.\n
         * Option type: unsigned int\n
         * Applies to: compiler only
         *)
        cudaJitThreadsPerBlock = 1,

        (**
         * Overwrites the option value with the total wall clock time, in
         * milliseconds, spent in the compiler and linker\n
         * Option type: float\n
         * Applies to: compiler and linker
         *)
        cudaJitWallTime = 2,

        (**
         * Pointer to a buffer in which to print any log messages
         * that are informational in nature (the buffer size is specified via
         * option ::cudaJitInfoLogBufferSizeBytes)\n
         * Option type: char *\n
         * Applies to: compiler and linker
         *)
        cudaJitInfoLogBuffer = 3,

        (**
         * IN: Log buffer size in bytes.  Log messages will be capped at this size
         * (including null terminator)\n
         * OUT: Amount of log buffer filled with messages\n
         * Option type: unsigned int\n
         * Applies to: compiler and linker
         *)
        cudaJitInfoLogBufferSizeBytes = 4,

        (**
         * Pointer to a buffer in which to print any log messages that
         * reflect errors (the buffer size is specified via option
         * ::cudaJitErrorLogBufferSizeBytes)\n
         * Option type: char *\n
         * Applies to: compiler and linker
         *)
        cudaJitErrorLogBuffer = 5,

        (**
         * IN: Log buffer size in bytes.  Log messages will be capped at this size
         * (including null terminator)\n
         * OUT: Amount of log buffer filled with messages\n
         * Option type: unsigned int\n
         * Applies to: compiler and linker
         *)
        cudaJitErrorLogBufferSizeBytes = 6,

        (**
         * Level of optimizations to apply to generated code (0 - 4), with 4
         * being the default and highest level of optimizations.\n
         * Option type: unsigned int\n
         * Applies to: compiler only
         *)
        cudaJitOptimizationLevel = 7,

        (**
         * Specifies choice of fallback strategy if matching cubin is not found.
         * Choice is based on supplied ::cudaJit_Fallback.
         * Option type: unsigned int for enumerated type ::cudaJit_Fallback\n
         * Applies to: compiler only
         *)
        cudaJitFallbackStrategy = 10,

        (**
         * Specifies whether to create debug information in output (-g)
         * (0: false, default)\n
         * Option type: int\n
         * Applies to: compiler and linker
         *)
        cudaJitGenerateDebugInfo = 11,

        (**
         * Generate verbose log messages (0: false, default)\n
         * Option type: int\n
         * Applies to: compiler and linker
         *)
        cudaJitLogVerbose = 12,

        (**
         * Generate line number information (-lineinfo) (0: false, default)\n
         * Option type: int\n
         * Applies to: compiler only
         *)
        cudaJitGenerateLineInfo = 13,

        (**
         * Specifies whether to enable caching explicitly (-dlcm) \n
         * Choice is based on supplied ::cudaJit_CacheMode.\n
         * Option type: unsigned int for enumerated type ::cudaJit_CacheMode\n
         * Applies to: compiler only
         *)
        cudaJitCacheMode = 14,

        (**
         * Generate position independent code (0: false)\n
         * Option type: int\n
         * Applies to: compiler only
         *)
        cudaJitPositionIndependentCode = 30,

        (**
         * This option hints to the JIT compiler the minimum number of CTAs from the
         * kernel’s grid to be mapped to a SM. This option is ignored when used together
         * with ::cudaJitMaxRegisters or ::cudaJitThreadsPerBlock.
         * Optimizations based on this option need ::cudaJitMaxThreadsPerBlock to
         * be specified as well. For kernels already using PTX directive .minnctapersm,
         * this option will be ignored by default. Use ::cudaJitOverrideDirectiveValues
         * to let this option take precedence over the PTX directive.
         * Option type: unsigned int\n
         * Applies to: compiler only
        *)
        cudaJitMinCtaPerSm = 31,

         (**
         * Maximum number threads in a thread block, computed as the product of
         * the maximum extent specifed for each dimension of the block. This limit
         * is guaranteed not to be exeeded in any invocation of the kernel. Exceeding
         * the the maximum number of threads results in runtime error or kernel launch
         * failure. For kernels already using PTX directive .maxntid, this option will
         * be ignored by default. Use ::cudaJitOverrideDirectiveValues to let this
         * option take precedence over the PTX directive.
         * Option type: int\n
         * Applies to: compiler only
        *)
        cudaJitMaxThreadsPerBlock = 32,

        (**
         * This option lets the values specified using ::cudaJitMaxRegisters,
         * ::cudaJitThreadsPerBlock, ::cudaJitMaxThreadsPerBlock and
         * ::cudaJitMinCtaPerSm take precedence over any PTX directives.
         * (0: Disable, default; 1: Enable)
         * Option type: int\n
         * Applies to: compiler only
        *)
        cudaJitOverrideDirectiveValues = 33
    );


    (**
     * Library options to be specified with ::cudaLibraryLoadData() or ::cudaLibraryLoadFromFile()
     *)
    pcudaLibraryOption = ^cudaLibraryOption;
    cudaLibraryOption =(
        cudaLibraryHostUniversalFunctionAndDataTable = 0,

        (**
         * Specifes that the argument \p code passed to ::cudaLibraryLoadData() will be preserved.
         * Specifying this option will let the driver know that \p code can be accessed at any point
         * until ::cudaLibraryUnload(). The default behavior is for the driver to allocate and
         * maintain its own copy of \p code. Note that this is only a memory usage optimization
         * hint and the driver can choose to ignore it if required.
         * Specifying this option with ::cudaLibraryLoadFromFile() is invalid and
         * will return ::cudaErrorInvalidValue.
         *)
        cudaLibraryBinaryIsPreserved = 1
    );
    //pcudalibraryHostUniversalFunctionAndDataTable = ^cudalibraryHostUniversalFunctionAndDataTable;
    //cudalibraryHostUniversalFunctionAndDataTable = record
    //    functionTable       : pointer             ;
    //    functionWindowSize  : size_t              ;
    //    dataTable           : pointer             ;
    //    dataWindowSize      : size_t              ;
    //end;

    (**
     * Caching modes for dlcm
     *)
    cudaJit_CacheMode = (
        cudaJitCacheOptionNone = 0,   (**< Compile with no -dlcm flag specified *)
        cudaJitCacheOptionCG,         (**< Compile with L1 cache disabled *)
        cudaJitCacheOptionCA          (**< Compile with L1 cache enabled *)
    );

    (**
     * Cubin matching fallback strategies
     *)
    cudaJit_Fallback  = (
        cudaPreferPtx = 0,  (**< Prefer to compile ptx if exact binary match not found *)

        cudaPreferBinary    (**< Prefer to fall back to compatible binary code if exact match not found *)
    );

    (**
     * CUDA library
     *)
    pcudaLibrary_t = ^cudaLibrary_t;
    cudaLibrary_t = ^CUlib_st;
    CUlib_st =record end;


implementation


{ dim3 }

constructor dim3.Create(const aX: longword; const aY: longword;
  const aZ: longword);
begin
  x := ax; y := ay; z := az
end;

class operator dim3.add(const a, b: dim3): dim3;
begin
  result := dim3.create(a.x+b.x, a.y+b.y, a.z+b.z)
end;

end.
