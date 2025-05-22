(*
 26/04/2025   converted to pascal by Haitham Shatti <haitham.shatti@gmail.com>
 * Copyright 1993-2019 NVIDIA Corporation. All rights reserved.
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

(*
 * This is the public header file for the CUBLAS library, defining the API
 *
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines)
 * on top of the CUDA runtime.
 *)

(* CUBLAS data types *)
{$ifdef FPC}
{$packrecords C}
{$mode delphi}
{$endif}
unit cublas;
interface
uses cudaTypes, cublas_api;
{$define CUBLAS}
{$i cublas.inc}

type
  cublasStatus = cublasStatus_t;

var
  cublasInit: function():cublasStatus; WINAPI;
  cublasShutdown: function():cublasStatus;  WINAPI;
  cublasGetError: function():cublasStatus; WINAPI;

  cublasGetVersion: function( version: Plongint):cublasStatus; WINAPI;
  cublasAlloc: function( n: longint; elemSize: longint; devicePtr: PPointer):cublasStatus; WINAPI;

  cublasFree: function( devicePtr: Pointer):cublasStatus; WINAPI;

  cublasSetKernelStream: function( stream: cudaStream_t):cublasStatus; WINAPI;

  (* ---------------- CUBLAS BLAS1 functions ---------------- *)
  (* NRM2 *)
  cublasSnrm2: function(n: longint; const  x: Psingle; incx: longint):single; WINAPI;
  cublasDnrm2: function(n: longint; const  x: Pdouble; incx: longint):double; WINAPI;
  cublasScnrm2: function(n: longint; const  x: PcuComplex; incx: longint):single; WINAPI;
  cublasDznrm2: function(n: longint; const  x: PcuDoubleComplex; incx: longint):double; WINAPI;
  (*------------------------------------------------------------------------*)
  (* DOT *)
  cublasSdot: function(n: longint; const  x: Psingle; incx: longint; const  y: Psingle; incy: longint):single; WINAPI;
  cublasDdot: function(n: longint; const  x: Pdouble; incx: longint; const  y: Pdouble; incy: longint):double; WINAPI;
  cublasCdotu: function(n: longint; const  x: PcuComplex; incx: longint; const  y: PcuComplex; incy: longint):cuComplex; WINAPI;
  cublasCdotc: function(n: longint; const  x: PcuComplex; incx: longint; const  y: PcuComplex; incy: longint):cuComplex; WINAPI;
  cublasZdotu: function(n: longint; const  x: PcuDoubleComplex; incx: longint; const  y: PcuDoubleComplex; incy: longint):cuDoubleComplex; WINAPI;
  cublasZdotc: function(n: longint; const  x: PcuDoubleComplex; incx: longint; const  y: PcuDoubleComplex; incy: longint):cuDoubleComplex; WINAPI;
  (*------------------------------------------------------------------------*)
  (* SCAL *)
  cublasSscal: procedure(n: longint; alpha: single; x: Psingle; incx: longint); WINAPI;
  cublasDscal: procedure(n: longint; alpha: double; x: Pdouble; incx: longint); WINAPI;
  cublasCscal: procedure(n: longint; alpha: cuComplex; x: PcuComplex; incx: longint); WINAPI;
  cublasZscal: procedure(n: longint; alpha: cuDoubleComplex; x: PcuDoubleComplex; incx: longint); WINAPI;

  cublasCsscal: procedure(n: longint; alpha: single; x: PcuComplex; incx: longint); WINAPI;
  cublasZdscal: procedure(n: longint; alpha: double; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* AXPY *)
  cublasSaxpy: procedure(n: longint; alpha: single; const  x: Psingle; incx: longint; y: Psingle; incy: longint); WINAPI;
  cublasDaxpy: procedure(n: longint; alpha: double; const  x: Pdouble; incx: longint; y: Pdouble; incy: longint); WINAPI;
  cublasCaxpy: procedure(n: longint; alpha: cuComplex; const  x: PcuComplex; incx: longint; y: PcuComplex; incy: longint); WINAPI;
  cublasZaxpy: procedure(n: longint; alpha: cuDoubleComplex; const  x: PcuDoubleComplex; incx: longint; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* COPY *)
  cublasScopy: procedure(n: longint; const  x: Psingle; incx: longint; y: Psingle; incy: longint); WINAPI;
  cublasDcopy: procedure(n: longint; const  x: Pdouble; incx: longint; y: Pdouble; incy: longint); WINAPI;
  cublasCcopy: procedure(n: longint; const  x: PcuComplex; incx: longint; y: PcuComplex; incy: longint); WINAPI;
  cublasZcopy: procedure(n: longint; const  x: PcuDoubleComplex; incx: longint; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* SWAP *)
  cublasSswap: procedure(n: longint; x: Psingle; incx: longint; y: Psingle; incy: longint); WINAPI;
  cublasDswap: procedure(n: longint; x: Pdouble; incx: longint; y: Pdouble; incy: longint); WINAPI;
  cublasCswap: procedure(n: longint; x: PcuComplex; incx: longint; y: PcuComplex; incy: longint); WINAPI;
  cublasZswap: procedure(n: longint; x: PcuDoubleComplex; incx: longint; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* AMAX *)
  cublasIsamax: function(n: longint; const  x: Psingle; incx: longint):longint; WINAPI;
  cublasIdamax: function(n: longint; const  x: Pdouble; incx: longint):longint; WINAPI;
  cublasIcamax: function(n: longint; const  x: PcuComplex; incx: longint):longint; WINAPI;
  cublasIzamax: function(n: longint; const  x: PcuDoubleComplex; incx: longint):longint; WINAPI;
  (*------------------------------------------------------------------------*)
  (* AMIN *)
  cublasIsamin: function(n: longint; const  x: Psingle; incx: longint):longint; WINAPI;
  cublasIdamin: function(n: longint; const  x: Pdouble; incx: longint):longint; WINAPI;

  cublasIcamin: function(n: longint; const  x: PcuComplex; incx: longint):longint; WINAPI;
  cublasIzamin: function(n: longint; const  x: PcuDoubleComplex; incx: longint):longint; WINAPI;
  (*------------------------------------------------------------------------*)
  (* ASUM *)
  cublasSasum: function(n: longint; const  x: Psingle; incx: longint):single; WINAPI;
  cublasDasum: function(n: longint; const  x: Pdouble; incx: longint):double; WINAPI;
  cublasScasum: function(n: longint; const  x: PcuComplex; incx: longint):single; WINAPI;
  cublasDzasum: function(n: longint; const  x: PcuDoubleComplex; incx: longint):double; WINAPI;
  (*------------------------------------------------------------------------*)
  (* ROT *)
  cublasSrot: procedure(n: longint; x: Psingle; incx: longint; y: Psingle; incy: longint; sc: single; ss: single); WINAPI;
  cublasDrot: procedure(n: longint; x: Pdouble; incx: longint; y: Pdouble; incy: longint; sc: double; ss: double); WINAPI;
  cublasCrot: procedure(n: longint; x: PcuComplex; incx: longint; y: PcuComplex; incy: longint; c: single; s: cuComplex); WINAPI;
  cublasZrot: procedure(n: longint; x: PcuDoubleComplex; incx: longint; y: PcuDoubleComplex; incy: longint; sc: double; cs: cuDoubleComplex); WINAPI;
  cublasCsrot: procedure(n: longint; x: PcuComplex; incx: longint; y: PcuComplex; incy: longint; c: single; s: single); WINAPI;
  cublasZdrot: procedure(n: longint; x: PcuDoubleComplex; incx: longint; y: PcuDoubleComplex; incy: longint; c: double; s: double); WINAPI;
  (*------------------------------------------------------------------------*)
  (* ROTG *)
  cublasSrotg: procedure(sa: Psingle; sb: Psingle; sc: Psingle; ss: Psingle); WINAPI;
  cublasDrotg: procedure(sa: Pdouble; sb: Pdouble; sc: Pdouble; ss: Pdouble); WINAPI;
  cublasCrotg: procedure(ca: PcuComplex; cb: cuComplex; sc: Psingle; cs: PcuComplex); WINAPI;
  cublasZrotg: procedure(ca: PcuDoubleComplex; cb: cuDoubleComplex; sc: Pdouble; cs: PcuDoubleComplex); WINAPI;
  (*------------------------------------------------------------------------*)
  (* ROTM *)
  cublasSrotm: procedure(n: longint; x: Psingle; incx: longint; y: Psingle; incy: longint; const  sparam: Psingle); WINAPI;
  cublasDrotm: procedure(n: longint; x: Pdouble; incx: longint; y: Pdouble; incy: longint; const  sparam: Pdouble); WINAPI;
  (*------------------------------------------------------------------------*)
  (* ROTMG *)
  cublasSrotmg: procedure(sd1: Psingle; sd2: Psingle; sx1: Psingle; const  sy1: Psingle; sparam: Psingle); WINAPI;
  cublasDrotmg: procedure(sd1: Pdouble; sd2: Pdouble; sx1: Pdouble; const  sy1: Pdouble; sparam: Pdouble); WINAPI;

  (* --------------- CUBLAS BLAS2 functions  ---------------- *)
  (* GEMV *)
  cublasSgemv: procedure(trans: char; m: longint; n: longint; alpha: single; const  A: Psingle; lda: longint; const  x: Psingle; incx: longint; beta: single; y: Psingle; incy: longint); WINAPI;
  cublasDgemv: procedure(trans: char; m: longint; n: longint; alpha: double; const  A: Pdouble; lda: longint; const  x: Pdouble; incx: longint; beta: double; y: Pdouble; incy: longint); WINAPI;
  cublasCgemv: procedure(trans: char; m: longint; n: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  x: PcuComplex; incx: longint; beta: cuComplex; y: PcuComplex; incy: longint); WINAPI;
  cublasZgemv: procedure(trans: char; m: longint; n: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  x: PcuDoubleComplex; incx: longint; beta: cuDoubleComplex; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* GBMV *)
  cublasSgbmv: procedure(trans: char; m: longint; n: longint; kl: longint; ku: longint; alpha: single; const  A: Psingle; lda: longint; const  x: Psingle; incx: longint; beta: single; y: Psingle; incy: longint); WINAPI;
  cublasDgbmv: procedure(trans: char; m: longint; n: longint; kl: longint; ku: longint; alpha: double; const  A: Pdouble; lda: longint; const  x: Pdouble; incx: longint; beta: double; y: Pdouble; incy: longint); WINAPI;
  cublasCgbmv: procedure(trans: char; m: longint; n: longint; kl: longint; ku: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  x: PcuComplex; incx: longint; beta: cuComplex; y: PcuComplex; incy: longint); WINAPI;
  cublasZgbmv: procedure(trans: char; m: longint; n: longint; kl: longint; ku: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  x: PcuDoubleComplex; incx: longint; beta: cuDoubleComplex; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TRMV *)
  cublasStrmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: Psingle; lda: longint; x: Psingle; incx: longint); WINAPI;
  cublasDtrmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: Pdouble; lda: longint; x: Pdouble; incx: longint); WINAPI;
  cublasCtrmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: PcuComplex; lda: longint; x: PcuComplex; incx: longint); WINAPI;
  cublasZtrmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: PcuDoubleComplex; lda: longint; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TBMV *)
  cublasStbmv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: Psingle; lda: longint; x: Psingle; incx: longint); WINAPI;
  cublasDtbmv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: Pdouble; lda: longint; x: Pdouble; incx: longint); WINAPI;
  cublasCtbmv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: PcuComplex; lda: longint; x: PcuComplex; incx: longint); WINAPI;
  cublasZtbmv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: PcuDoubleComplex; lda: longint; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TPMV *)
  cublasStpmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: Psingle; x: Psingle; incx: longint); WINAPI;

  cublasDtpmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: Pdouble; x: Pdouble; incx: longint); WINAPI;

  cublasCtpmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: PcuComplex; x: PcuComplex; incx: longint); WINAPI;

  cublasZtpmv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: PcuDoubleComplex; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TRSV *)
  cublasStrsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: Psingle; lda: longint; x: Psingle; incx: longint); WINAPI;

  cublasDtrsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: Pdouble; lda: longint; x: Pdouble; incx: longint); WINAPI;

  cublasCtrsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: PcuComplex; lda: longint; x: PcuComplex; incx: longint); WINAPI;

  cublasZtrsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  A: PcuDoubleComplex; lda: longint; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TPSV *)
  cublasStpsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: Psingle; x: Psingle; incx: longint); WINAPI;

  cublasDtpsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: Pdouble; x: Pdouble; incx: longint); WINAPI;

  cublasCtpsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: PcuComplex; x: PcuComplex; incx: longint); WINAPI;

  cublasZtpsv: procedure(uplo: char; trans: char; diag: char; n: longint; const  AP: PcuDoubleComplex; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TBSV *)
  cublasStbsv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: Psingle; lda: longint; x: Psingle; incx: longint); WINAPI;

  cublasDtbsv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: Pdouble; lda: longint; x: Pdouble; incx: longint); WINAPI;
  cublasCtbsv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: PcuComplex; lda: longint; x: PcuComplex; incx: longint); WINAPI;

  cublasZtbsv: procedure(uplo: char; trans: char; diag: char; n: longint; k: longint; const  A: PcuDoubleComplex; lda: longint; x: PcuDoubleComplex; incx: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* SYMV/HEMV *)
  cublasSsymv: procedure(uplo: char; n: longint; alpha: single; const  A: Psingle; lda: longint; const  x: Psingle; incx: longint; beta: single; y: Psingle; incy: longint); WINAPI;
  cublasDsymv: procedure(uplo: char; n: longint; alpha: double; const  A: Pdouble; lda: longint; const  x: Pdouble; incx: longint; beta: double; y: Pdouble; incy: longint); WINAPI;
  cublasChemv: procedure(uplo: char; n: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  x: PcuComplex; incx: longint; beta: cuComplex; y: PcuComplex; incy: longint); WINAPI;
  cublasZhemv: procedure(uplo: char; n: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  x: PcuDoubleComplex; incx: longint; beta: cuDoubleComplex; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* SBMV/HBMV *)
  cublasSsbmv: procedure(uplo: char; n: longint; k: longint; alpha: single; const  A: Psingle; lda: longint; const  x: Psingle; incx: longint; beta: single; y: Psingle; incy: longint); WINAPI;
  cublasDsbmv: procedure(uplo: char; n: longint; k: longint; alpha: double; const  A: Pdouble; lda: longint; const  x: Pdouble; incx: longint; beta: double; y: Pdouble; incy: longint); WINAPI;
  cublasChbmv: procedure(uplo: char; n: longint; k: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  x: PcuComplex; incx: longint; beta: cuComplex; y: PcuComplex; incy: longint); WINAPI;
  cublasZhbmv: procedure(uplo: char; n: longint; k: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  x: PcuDoubleComplex; incx: longint; beta: cuDoubleComplex; y: PcuDoubleComplex; incy: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* SPMV/HPMV *)
  cublasSspmv: procedure(uplo: char; n: longint; alpha: single; const  AP: Psingle; const  x: Psingle; incx: longint; beta: single; y: Psingle; incy: longint); WINAPI;
  cublasDspmv: procedure(uplo: char; n: longint; alpha: double; const  AP: Pdouble; const  x: Pdouble; incx: longint; beta: double; y: Pdouble; incy: longint); WINAPI;
  cublasChpmv: procedure(uplo: char; n: longint; alpha: cuComplex; const  AP: PcuComplex; const  x: PcuComplex; incx: longint; beta: cuComplex; y: PcuComplex; incy: longint); WINAPI;
  cublasZhpmv: procedure(uplo: char; n: longint; alpha: cuDoubleComplex; const  AP: PcuDoubleComplex; const  x: PcuDoubleComplex; incx: longint; beta: cuDoubleComplex; y: PcuDoubleComplex; incy: longint); WINAPI;

  (*------------------------------------------------------------------------*)
  (* GER *)
  cublasSger: procedure(m: longint; n: longint; alpha: single; const  x: Psingle; incx: longint; const  y: Psingle; incy: longint; A: Psingle; lda: longint); WINAPI;
  cublasDger: procedure(m: longint; n: longint; alpha: double; const  x: Pdouble; incx: longint; const  y: Pdouble; incy: longint; A: Pdouble; lda: longint); WINAPI;

  cublasCgeru: procedure(m: longint; n: longint; alpha: cuComplex; const  x: PcuComplex; incx: longint; const  y: PcuComplex; incy: longint; A: PcuComplex; lda: longint); WINAPI;
  cublasCgerc: procedure(m: longint; n: longint; alpha: cuComplex; const  x: PcuComplex; incx: longint; const  y: PcuComplex; incy: longint; A: PcuComplex; lda: longint); WINAPI;
  cublasZgeru: procedure(m: longint; n: longint; alpha: cuDoubleComplex; const  x: PcuDoubleComplex; incx: longint; const  y: PcuDoubleComplex; incy: longint; A: PcuDoubleComplex; lda: longint); WINAPI;
  cublasZgerc: procedure(m: longint; n: longint; alpha: cuDoubleComplex; const  x: PcuDoubleComplex; incx: longint; const  y: PcuDoubleComplex; incy: longint; A: PcuDoubleComplex; lda: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* SYR/HER *)
  cublasSsyr: procedure(uplo: char; n: longint; alpha: single; const  x: Psingle; incx: longint; A: Psingle; lda: longint); WINAPI;
  cublasDsyr: procedure(uplo: char; n: longint; alpha: double; const  x: Pdouble; incx: longint; A: Pdouble; lda: longint); WINAPI;

  cublasCher: procedure(uplo: char; n: longint; alpha: single; const  x: PcuComplex; incx: longint; A: PcuComplex; lda: longint); WINAPI;
  cublasZher: procedure(uplo: char; n: longint; alpha: double; const  x: PcuDoubleComplex; incx: longint; A: PcuDoubleComplex; lda: longint); WINAPI;

  (*------------------------------------------------------------------------*)
  (* SPR/HPR *)
  cublasSspr: procedure(uplo: char; n: longint; alpha: single; const  x: Psingle; incx: longint; AP: Psingle); WINAPI;
  cublasDspr: procedure(uplo: char; n: longint; alpha: double; const  x: Pdouble; incx: longint; AP: Pdouble); WINAPI;
  cublasChpr: procedure(uplo: char; n: longint; alpha: single; const  x: PcuComplex; incx: longint; AP: PcuComplex); WINAPI;
  cublasZhpr: procedure(uplo: char; n: longint; alpha: double; const  x: PcuDoubleComplex; incx: longint; AP: PcuDoubleComplex); WINAPI;
  (*------------------------------------------------------------------------*)
  (* SYR2/HER2 *)
  cublasSsyr2: procedure(uplo: char; n: longint; alpha: single; const  x: Psingle; incx: longint; const  y: Psingle; incy: longint; A: Psingle; lda: longint); WINAPI;
  cublasDsyr2: procedure(uplo: char; n: longint; alpha: double; const  x: Pdouble; incx: longint; const  y: Pdouble; incy: longint; A: Pdouble; lda: longint); WINAPI;
  cublasCher2: procedure(uplo: char; n: longint; alpha: cuComplex; const  x: PcuComplex; incx: longint; const  y: PcuComplex; incy: longint; A: PcuComplex; lda: longint); WINAPI;
  cublasZher2: procedure(uplo: char; n: longint; alpha: cuDoubleComplex; const  x: PcuDoubleComplex; incx: longint; const  y: PcuDoubleComplex; incy: longint; A: PcuDoubleComplex; lda: longint); WINAPI;

  (*------------------------------------------------------------------------*)
  (* SPR2/HPR2 *)
  cublasSspr2: procedure(uplo: char; n: longint; alpha: single; const  x: Psingle; incx: longint; const  y: Psingle; incy: longint; AP: Psingle); WINAPI;
  cublasDspr2: procedure(uplo: char; n: longint; alpha: double; const  x: Pdouble; incx: longint; const  y: Pdouble; incy: longint; AP: Pdouble); WINAPI;
  cublasChpr2: procedure(uplo: char; n: longint; alpha: cuComplex; const  x: PcuComplex; incx: longint; const  y: PcuComplex; incy: longint; AP: PcuComplex); WINAPI;
  cublasZhpr2: procedure(uplo: char; n: longint; alpha: cuDoubleComplex; const  x: PcuDoubleComplex; incx: longint; const  y: PcuDoubleComplex; incy: longint; AP: PcuDoubleComplex); WINAPI;
  (* ------------------------BLAS3 Functions ------------------------------- *)
  (* GEMM *)
  cublasSgemm: procedure(transa: char; transb: char; m: longint; n: longint; k: longint; alpha: single; const  A: Psingle; lda: longint; const  B: Psingle; ldb: longint; beta: single; C: Psingle; ldc: longint); WINAPI;
  cublasDgemm: procedure(transa: char; transb: char; m: longint; n: longint; k: longint; alpha: double; const  A: Pdouble; lda: longint; const  B: Pdouble; ldb: longint; beta: double; C: Pdouble; ldc: longint); WINAPI;
  cublasCgemm: procedure(transa: char; transb: char; m: longint; n: longint; k: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  B: PcuComplex; ldb: longint; beta: cuComplex; C: PcuComplex; ldc: longint); WINAPI;
  cublasZgemm: procedure(transa: char; transb: char; m: longint; n: longint; k: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  B: PcuDoubleComplex; ldb: longint; beta: cuDoubleComplex; C: PcuDoubleComplex; ldc: longint); WINAPI;
  (* -------------------------------------------------------*)
  (* SYRK *)
  cublasSsyrk: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: single; const  A: Psingle; lda: longint; beta: single; C: Psingle; ldc: longint); WINAPI;
  cublasDsyrk: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: double; const  A: Pdouble; lda: longint; beta: double; C: Pdouble; ldc: longint); WINAPI;

  cublasCsyrk: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; beta: cuComplex; C: PcuComplex; ldc: longint); WINAPI;
  cublasZsyrk: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; beta: cuDoubleComplex; C: PcuDoubleComplex; ldc: longint); WINAPI;
  (* ------------------------------------------------------- *)
  (* HERK *)
  cublasCherk: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: single; const  A: PcuComplex; lda: longint; beta: single; C: PcuComplex; ldc: longint); WINAPI;
  cublasZherk: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: double; const  A: PcuDoubleComplex; lda: longint; beta: double; C: PcuDoubleComplex; ldc: longint); WINAPI;
  (* ------------------------------------------------------- *)
  (* SYR2K *)
  cublasSsyr2k: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: single; const  A: Psingle; lda: longint; const  B: Psingle; ldb: longint; beta: single; C: Psingle; ldc: longint); WINAPI;

  cublasDsyr2k: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: double; const  A: Pdouble; lda: longint; const  B: Pdouble; ldb: longint; beta: double; C: Pdouble; ldc: longint); WINAPI;
  cublasCsyr2k: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  B: PcuComplex; ldb: longint; beta: cuComplex; C: PcuComplex; ldc: longint); WINAPI;

  cublasZsyr2k: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  B: PcuDoubleComplex; ldb: longint; beta: cuDoubleComplex; C: PcuDoubleComplex; ldc: longint); WINAPI;
  (* ------------------------------------------------------- *)
  (* HER2K *)
  cublasCher2k: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  B: PcuComplex; ldb: longint; beta: single; C: PcuComplex; ldc: longint); WINAPI;

  cublasZher2k: procedure(uplo: char; trans: char; n: longint; k: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  B: PcuDoubleComplex; ldb: longint; beta: double; C: PcuDoubleComplex; ldc: longint); WINAPI;

  (*------------------------------------------------------------------------*)
  (* SYMM*)
  cublasSsymm: procedure(side: char; uplo: char; m: longint; n: longint; alpha: single; const  A: Psingle; lda: longint; const  B: Psingle; ldb: longint; beta: single; C: Psingle; ldc: longint); WINAPI;
  cublasDsymm: procedure(side: char; uplo: char; m: longint; n: longint; alpha: double; const  A: Pdouble; lda: longint; const  B: Pdouble; ldb: longint; beta: double; C: Pdouble; ldc: longint); WINAPI;

  cublasCsymm: procedure(side: char; uplo: char; m: longint; n: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  B: PcuComplex; ldb: longint; beta: cuComplex; C: PcuComplex; ldc: longint); WINAPI;

  cublasZsymm: procedure(side: char; uplo: char; m: longint; n: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  B: PcuDoubleComplex; ldb: longint; beta: cuDoubleComplex; C: PcuDoubleComplex; ldc: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* HEMM*)
  cublasChemm: procedure(side: char; uplo: char; m: longint; n: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; const  B: PcuComplex; ldb: longint; beta: cuComplex; C: PcuComplex; ldc: longint); WINAPI;
  cublasZhemm: procedure(side: char; uplo: char; m: longint; n: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; const  B: PcuDoubleComplex; ldb: longint; beta: cuDoubleComplex; C: PcuDoubleComplex; ldc: longint); WINAPI;

  (*------------------------------------------------------------------------*)
  (* TRSM*)
  cublasStrsm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: single; const  A: Psingle; lda: longint; B: Psingle; ldb: longint); WINAPI;

  cublasDtrsm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: double; const  A: Pdouble; lda: longint; B: Pdouble; ldb: longint); WINAPI;

  cublasCtrsm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; B: PcuComplex; ldb: longint); WINAPI;

  cublasZtrsm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; B: PcuDoubleComplex; ldb: longint); WINAPI;
  (*------------------------------------------------------------------------*)
  (* TRMM*)
  cublasStrmm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: single; const  A: Psingle; lda: longint; B: Psingle; ldb: longint); WINAPI;
  cublasDtrmm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: double; const  A: Pdouble; lda: longint; B: Pdouble; ldb: longint); WINAPI;
  cublasCtrmm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: cuComplex; const  A: PcuComplex; lda: longint; B: PcuComplex; ldb: longint); WINAPI;
  cublasZtrmm: procedure(side: char; uplo: char; transa: char; diag: char; m: longint; n: longint; alpha: cuDoubleComplex; const  A: PcuDoubleComplex; lda: longint; B: PcuDoubleComplex; ldb: longint); WINAPI;

implementation
uses sysutils;
var hlib : TLibHandle;

procedure freelibcublas();
begin
  FreeLibrary(hlib);
  cublasInit := nil;
  cublasShutdown := nil;
  cublasGetError := nil;

  cublasGetVersion := nil;
  cublasAlloc := nil;

  cublasFree := nil;

  cublasSetKernelStream := nil;

  (* ---------------- CUBLAS BLAS1 functions ---------------- *)
  (* NRM2 *)
  cublasSnrm2 := nil;
  cublasDnrm2 := nil;
  cublasScnrm2 := nil;
  cublasDznrm2 := nil;
  (*------------------------------------------------------------------------*)
  (* DOT *)
  cublasSdot := nil;
  cublasDdot := nil;
  cublasCdotu := nil;
  cublasCdotc := nil;
  cublasZdotu := nil;
  cublasZdotc := nil;
  (*------------------------------------------------------------------------*)
  (* SCAL *)
  cublasSscal := nil;
  cublasDscal := nil;
  cublasCscal := nil;
  cublasZscal := nil;

  cublasCsscal := nil;
  cublasZdscal := nil;
  (*------------------------------------------------------------------------*)
  (* AXPY *)
  cublasSaxpy := nil;
  cublasDaxpy := nil;
  cublasCaxpy := nil;
  cublasZaxpy := nil;
  (*------------------------------------------------------------------------*)
  (* COPY *)
  cublasScopy := nil;
  cublasDcopy := nil;
  cublasCcopy := nil;
  cublasZcopy := nil;
  (*------------------------------------------------------------------------*)
  (* SWAP *)
  cublasSswap := nil;
  cublasDswap := nil;
  cublasCswap := nil;
  cublasZswap := nil;
  (*------------------------------------------------------------------------*)
  (* AMAX *)
  cublasIsamax := nil;
  cublasIdamax := nil;
  cublasIcamax := nil;
  cublasIzamax := nil;
  (*------------------------------------------------------------------------*)
  (* AMIN *)
  cublasIsamin := nil;
  cublasIdamin := nil;

  cublasIcamin := nil;
  cublasIzamin := nil;
  (*------------------------------------------------------------------------*)
  (* ASUM *)
  cublasSasum := nil;
  cublasDasum := nil;
  cublasScasum := nil;
  cublasDzasum := nil;
  (*------------------------------------------------------------------------*)
  (* ROT *)
  cublasSrot := nil;
  cublasDrot := nil;
  cublasCrot := nil;
  cublasZrot := nil;
  cublasCsrot := nil;
  cublasZdrot := nil;
  (*------------------------------------------------------------------------*)
  (* ROTG *)
  cublasSrotg := nil;
  cublasDrotg := nil;
  cublasCrotg := nil;
  cublasZrotg := nil;
  (*------------------------------------------------------------------------*)
  (* ROTM *)
  cublasSrotm := nil;
  cublasDrotm := nil;
  (*------------------------------------------------------------------------*)
  (* ROTMG *)
  cublasSrotmg := nil;
  cublasDrotmg := nil;

  (* --------------- CUBLAS BLAS2 functions  ---------------- *)
  (* GEMV *)
  cublasSgemv := nil;
  cublasDgemv := nil;
  cublasCgemv := nil;
  cublasZgemv := nil;
  (*------------------------------------------------------------------------*)
  (* GBMV *)
  cublasSgbmv := nil;
  cublasDgbmv := nil;
  cublasCgbmv := nil;
  cublasZgbmv := nil;
  (*------------------------------------------------------------------------*)
  (* TRMV *)
  cublasStrmv := nil;
  cublasDtrmv := nil;
  cublasCtrmv := nil;
  cublasZtrmv := nil;
  (*------------------------------------------------------------------------*)
  (* TBMV *)
  cublasStbmv := nil;
  cublasDtbmv := nil;
  cublasCtbmv := nil;
  cublasZtbmv := nil;
  (*------------------------------------------------------------------------*)
  (* TPMV *)
  cublasStpmv := nil;

  cublasDtpmv := nil;

  cublasCtpmv := nil;

  cublasZtpmv := nil;
  (*------------------------------------------------------------------------*)
  (* TRSV *)
  cublasStrsv := nil;

  cublasDtrsv := nil;

  cublasCtrsv := nil;

  cublasZtrsv := nil;
  (*------------------------------------------------------------------------*)
  (* TPSV *)
  cublasStpsv := nil;

  cublasDtpsv := nil;

  cublasCtpsv := nil;

  cublasZtpsv := nil;
  (*------------------------------------------------------------------------*)
  (* TBSV *)
  cublasStbsv := nil;

  cublasDtbsv := nil;
  cublasCtbsv := nil;

  cublasZtbsv := nil;
  (*------------------------------------------------------------------------*)
  (* SYMV/HEMV *)
  cublasSsymv := nil;
  cublasDsymv := nil;
  cublasChemv := nil;
  cublasZhemv := nil;
  (*------------------------------------------------------------------------*)
  (* SBMV/HBMV *)
  cublasSsbmv := nil;
  cublasDsbmv := nil;
  cublasChbmv := nil;
  cublasZhbmv := nil;
  (*------------------------------------------------------------------------*)
  (* SPMV/HPMV *)
  cublasSspmv := nil;
  cublasDspmv := nil;
  cublasChpmv := nil;
  cublasZhpmv := nil;

  (*------------------------------------------------------------------------*)
  (* GER *)
  cublasSger := nil;
  cublasDger := nil;

  cublasCgeru := nil;
  cublasCgerc := nil;
  cublasZgeru := nil;
  cublasZgerc := nil;
  (*------------------------------------------------------------------------*)
  (* SYR/HER *)
  cublasSsyr := nil;
  cublasDsyr := nil;

  cublasCher := nil;
  cublasZher := nil;

  (*------------------------------------------------------------------------*)
  (* SPR/HPR *)
  cublasSspr := nil;
  cublasDspr := nil;
  cublasChpr := nil;
  cublasZhpr := nil;
  (*------------------------------------------------------------------------*)
  (* SYR2/HER2 *)
  cublasSsyr2 := nil;
  cublasDsyr2 := nil;
  cublasCher2 := nil;
  cublasZher2 := nil;

  (*------------------------------------------------------------------------*)
  (* SPR2/HPR2 *)
  cublasSspr2 := nil;
  cublasDspr2 := nil;
  cublasChpr2 := nil;
  cublasZhpr2 := nil;
  (* ------------------------BLAS3 Functions ------------------------------- *)
  (* GEMM *)
  cublasSgemm := nil;
  cublasDgemm := nil;
  cublasCgemm := nil;
  cublasZgemm := nil;
  (* -------------------------------------------------------*)
  (* SYRK *)
  cublasSsyrk := nil;
  cublasDsyrk := nil;

  cublasCsyrk := nil;
  cublasZsyrk := nil;
  (* ------------------------------------------------------- *)
  (* HERK *)
  cublasCherk := nil;
  cublasZherk := nil;
  (* ------------------------------------------------------- *)
  (* SYR2K *)
  cublasSsyr2k := nil;

  cublasDsyr2k := nil;
  cublasCsyr2k := nil;

  cublasZsyr2k := nil;
  (* ------------------------------------------------------- *)
  (* HER2K *)
  cublasCher2k := nil;

  cublasZher2k := nil;

  (*------------------------------------------------------------------------*)
  (* SYMM*)
  cublasSsymm := nil;
  cublasDsymm := nil;

  cublasCsymm := nil;

  cublasZsymm := nil;
  (*------------------------------------------------------------------------*)
  (* HEMM*)
  cublasChemm := nil;
  cublasZhemm := nil;

  (*------------------------------------------------------------------------*)
  (* TRSM*)
  cublasStrsm := nil;

  cublasDtrsm := nil;

  cublasCtrsm := nil;

  cublasZtrsm := nil;
  (*------------------------------------------------------------------------*)
  (* TRMM*)
  cublasStrmm := nil;
  cublasDtrmm := nil;
  cublasCtrmm := nil;

end;

procedure loadlibcublas(const lib:pchar);
begin

  hlib := LoadLibrary(lib);
  if hlib = 0 then
    raise Exception.create(Format('Could not load library: %s', [lib]));

  cublasInit := GetProcAddress(hlib, 'cublasInit');
  cublasShutdown := GetProcAddress(hlib, 'cublasShutdown');
  cublasGetError := GetProcAddress(hlib, 'cublasGetError');

  cublasGetVersion := GetProcAddress(hlib, 'cublasGetVersion');
  cublasAlloc := GetProcAddress(hlib, 'cublasAlloc');

  cublasFree := GetProcAddress(hlib, 'cublasFree');

  cublasSetKernelStream := GetProcAddress(hlib, 'cublasSetKernelStream');

  (* ---------------- CUBLAS BLAS1 functions ---------------- *)
  (* NRM2 *)
  cublasSnrm2 := GetProcAddress(hlib, 'cublasSnrm2');
  cublasDnrm2 := GetProcAddress(hlib, 'cublasDnrm2');
  cublasScnrm2 := GetProcAddress(hlib, 'cublasScnrm2');
  cublasDznrm2 := GetProcAddress(hlib, 'cublasDznrm2');
  (*------------------------------------------------------------------------*)
  (* DOT *)
  cublasSdot := GetProcAddress(hlib, 'cublasSdot');
  cublasDdot := GetProcAddress(hlib, 'cublasDdot');
  cublasCdotu := GetProcAddress(hlib, 'cublasCdotu');
  cublasCdotc := GetProcAddress(hlib, 'cublasCdotc');
  cublasZdotu := GetProcAddress(hlib, 'cublasZdotu');
  cublasZdotc := GetProcAddress(hlib, 'cublasZdotc');
  (*------------------------------------------------------------------------*)
  (* SCAL *)
  cublasSscal := GetProcAddress(hlib, 'cublasSscal');
  cublasDscal := GetProcAddress(hlib, 'cublasDscal');
  cublasCscal := GetProcAddress(hlib, 'cublasCscal');
  cublasZscal := GetProcAddress(hlib, 'cublasZscal');

  cublasCsscal := GetProcAddress(hlib, 'cublasCsscal');
  cublasZdscal := GetProcAddress(hlib, 'cublasZdscal');
  (*------------------------------------------------------------------------*)
  (* AXPY *)
  cublasSaxpy := GetProcAddress(hlib, 'cublasSaxpy');
  cublasDaxpy := GetProcAddress(hlib, 'cublasDaxpy');
  cublasCaxpy := GetProcAddress(hlib, 'cublasCaxpy');
  cublasZaxpy := GetProcAddress(hlib, 'cublasZaxpy');
  (*------------------------------------------------------------------------*)
  (* COPY *)
  cublasScopy := GetProcAddress(hlib, 'cublasScopy');
  cublasDcopy := GetProcAddress(hlib, 'cublasDcopy');
  cublasCcopy := GetProcAddress(hlib, 'cublasCcopy');
  cublasZcopy := GetProcAddress(hlib, 'cublasZcopy');
  (*------------------------------------------------------------------------*)
  (* SWAP *)
  cublasSswap := GetProcAddress(hlib, 'cublasSswap');
  cublasDswap := GetProcAddress(hlib, 'cublasDswap');
  cublasCswap := GetProcAddress(hlib, 'cublasCswap');
  cublasZswap := GetProcAddress(hlib, 'cublasZswap');
  (*------------------------------------------------------------------------*)
  (* AMAX *)
  cublasIsamax := GetProcAddress(hlib, 'cublasIsamax');
  cublasIdamax := GetProcAddress(hlib, 'cublasIdamax');
  cublasIcamax := GetProcAddress(hlib, 'cublasIcamax');
  cublasIzamax := GetProcAddress(hlib, 'cublasIzamax');
  (*------------------------------------------------------------------------*)
  (* AMIN *)
  cublasIsamin := GetProcAddress(hlib, 'cublasIsamin');
  cublasIdamin := GetProcAddress(hlib, 'cublasIdamin');

  cublasIcamin := GetProcAddress(hlib, 'cublasIcamin');
  cublasIzamin := GetProcAddress(hlib, 'cublasIzamin');
  (*------------------------------------------------------------------------*)
  (* ASUM *)
  cublasSasum := GetProcAddress(hlib, 'cublasSasum');
  cublasDasum := GetProcAddress(hlib, 'cublasDasum');
  cublasScasum := GetProcAddress(hlib, 'cublasScasum');
  cublasDzasum := GetProcAddress(hlib, 'cublasDzasum');
  (*------------------------------------------------------------------------*)
  (* ROT *)
  cublasSrot := GetProcAddress(hlib, 'cublasSrot');
  cublasDrot := GetProcAddress(hlib, 'cublasDrot');
  cublasCrot := GetProcAddress(hlib, 'cublasCrot');
  cublasZrot := GetProcAddress(hlib, 'cublasZrot');
  cublasCsrot := GetProcAddress(hlib, 'cublasCsrot');
  cublasZdrot := GetProcAddress(hlib, 'cublasZdrot');
  (*------------------------------------------------------------------------*)
  (* ROTG *)
  cublasSrotg := GetProcAddress(hlib, 'cublasSrotg');
  cublasDrotg := GetProcAddress(hlib, 'cublasDrotg');
  cublasCrotg := GetProcAddress(hlib, 'cublasCrotg');
  cublasZrotg := GetProcAddress(hlib, 'cublasZrotg');
  (*------------------------------------------------------------------------*)
  (* ROTM *)
  cublasSrotm := GetProcAddress(hlib, 'cublasSrotm');
  cublasDrotm := GetProcAddress(hlib, 'cublasDrotm');
  (*------------------------------------------------------------------------*)
  (* ROTMG *)
  cublasSrotmg := GetProcAddress(hlib, 'cublasSrotmg');
  cublasDrotmg := GetProcAddress(hlib, 'cublasDrotmg');

  (* --------------- CUBLAS BLAS2 functions  ---------------- *)
  (* GEMV *)
  cublasSgemv := GetProcAddress(hlib, 'cublasSgemv');
  cublasDgemv := GetProcAddress(hlib, 'cublasDgemv');
  cublasCgemv := GetProcAddress(hlib, 'cublasCgemv');
  cublasZgemv := GetProcAddress(hlib, 'cublasZgemv');
  (*------------------------------------------------------------------------*)
  (* GBMV *)
  cublasSgbmv := GetProcAddress(hlib, 'cublasSgbmv');
  cublasDgbmv := GetProcAddress(hlib, 'cublasDgbmv');
  cublasCgbmv := GetProcAddress(hlib, 'cublasCgbmv');
  cublasZgbmv := GetProcAddress(hlib, 'cublasZgbmv');
  (*------------------------------------------------------------------------*)
  (* TRMV *)
  cublasStrmv := GetProcAddress(hlib, 'cublasStrmv');
  cublasDtrmv := GetProcAddress(hlib, 'cublasDtrmv');
  cublasCtrmv := GetProcAddress(hlib, 'cublasCtrmv');
  cublasZtrmv := GetProcAddress(hlib, 'cublasZtrmv');
  (*------------------------------------------------------------------------*)
  (* TBMV *)
  cublasStbmv := GetProcAddress(hlib, 'cublasStbmv');
  cublasDtbmv := GetProcAddress(hlib, 'cublasDtbmv');
  cublasCtbmv := GetProcAddress(hlib, 'cublasCtbmv');
  cublasZtbmv := GetProcAddress(hlib, 'cublasZtbmv');
  (*------------------------------------------------------------------------*)
  (* TPMV *)
  cublasStpmv := GetProcAddress(hlib, 'cublasStpmv');

  cublasDtpmv := GetProcAddress(hlib, 'cublasDtpmv');

  cublasCtpmv := GetProcAddress(hlib, 'cublasCtpmv');

  cublasZtpmv := GetProcAddress(hlib, 'cublasZtpmv');
  (*------------------------------------------------------------------------*)
  (* TRSV *)
  cublasStrsv := GetProcAddress(hlib, 'cublasStrsv');

  cublasDtrsv := GetProcAddress(hlib, 'cublasDtrsv');

  cublasCtrsv := GetProcAddress(hlib, 'cublasCtrsv');

  cublasZtrsv := GetProcAddress(hlib, 'cublasZtrsv');
  (*------------------------------------------------------------------------*)
  (* TPSV *)
  cublasStpsv := GetProcAddress(hlib, 'cublasStpsv');

  cublasDtpsv := GetProcAddress(hlib, 'cublasDtpsv');

  cublasCtpsv := GetProcAddress(hlib, 'cublasCtpsv');

  cublasZtpsv := GetProcAddress(hlib, 'cublasZtpsv');
  (*------------------------------------------------------------------------*)
  (* TBSV *)
  cublasStbsv := GetProcAddress(hlib, 'cublasStbsv');

  cublasDtbsv := GetProcAddress(hlib, 'cublasDtbsv');
  cublasCtbsv := GetProcAddress(hlib, 'cublasCtbsv');

  cublasZtbsv := GetProcAddress(hlib, 'cublasZtbsv');
  (*------------------------------------------------------------------------*)
  (* SYMV/HEMV *)
  cublasSsymv := GetProcAddress(hlib, 'cublasSsymv');
  cublasDsymv := GetProcAddress(hlib, 'cublasDsymv');
  cublasChemv := GetProcAddress(hlib, 'cublasChemv');
  cublasZhemv := GetProcAddress(hlib, 'cublasZhemv');
  (*------------------------------------------------------------------------*)
  (* SBMV/HBMV *)
  cublasSsbmv := GetProcAddress(hlib, 'cublasSsbmv');
  cublasDsbmv := GetProcAddress(hlib, 'cublasDsbmv');
  cublasChbmv := GetProcAddress(hlib, 'cublasChbmv');
  cublasZhbmv := GetProcAddress(hlib, 'cublasZhbmv');
  (*------------------------------------------------------------------------*)
  (* SPMV/HPMV *)
  cublasSspmv := GetProcAddress(hlib, 'cublasSspmv');
  cublasDspmv := GetProcAddress(hlib, 'cublasDspmv');
  cublasChpmv := GetProcAddress(hlib, 'cublasChpmv');
  cublasZhpmv := GetProcAddress(hlib, 'cublasZhpmv');

  (*------------------------------------------------------------------------*)
  (* GER *)
  cublasSger := GetProcAddress(hlib, 'cublasSger');
  cublasDger := GetProcAddress(hlib, 'cublasDger');

  cublasCgeru := GetProcAddress(hlib, 'cublasCgeru');
  cublasCgerc := GetProcAddress(hlib, 'cublasCgerc');
  cublasZgeru := GetProcAddress(hlib, 'cublasZgeru');
  cublasZgerc := GetProcAddress(hlib, 'cublasZgerc');
  (*------------------------------------------------------------------------*)
  (* SYR/HER *)
  cublasSsyr := GetProcAddress(hlib, 'cublasSsyr');
  cublasDsyr := GetProcAddress(hlib, 'cublasDsyr');

  cublasCher := GetProcAddress(hlib, 'cublasCher');
  cublasZher := GetProcAddress(hlib, 'cublasZher');

  (*------------------------------------------------------------------------*)
  (* SPR/HPR *)
  cublasSspr := GetProcAddress(hlib, 'cublasSspr');
  cublasDspr := GetProcAddress(hlib, 'cublasDspr');
  cublasChpr := GetProcAddress(hlib, 'cublasChpr');
  cublasZhpr := GetProcAddress(hlib, 'cublasZhpr');
  (*------------------------------------------------------------------------*)
  (* SYR2/HER2 *)
  cublasSsyr2 := GetProcAddress(hlib, 'cublasSsyr2');
  cublasDsyr2 := GetProcAddress(hlib, 'cublasDsyr2');
  cublasCher2 := GetProcAddress(hlib, 'cublasCher2');
  cublasZher2 := GetProcAddress(hlib, 'cublasZher2');

  (*------------------------------------------------------------------------*)
  (* SPR2/HPR2 *)
  cublasSspr2 := GetProcAddress(hlib, 'cublasSspr2');
  cublasDspr2 := GetProcAddress(hlib, 'cublasDspr2');
  cublasChpr2 := GetProcAddress(hlib, 'cublasChpr2');
  cublasZhpr2 := GetProcAddress(hlib, 'cublasZhpr2');
  (* ------------------------BLAS3 Functions ------------------------------- *)
  (* GEMM *)
  cublasSgemm := GetProcAddress(hlib, 'cublasSgemm');
  cublasDgemm := GetProcAddress(hlib, 'cublasDgemm');
  cublasCgemm := GetProcAddress(hlib, 'cublasCgemm');
  cublasZgemm := GetProcAddress(hlib, 'cublasZgemm');
  (* -------------------------------------------------------*)
  (* SYRK *)
  cublasSsyrk := GetProcAddress(hlib, 'cublasSsyrk');
  cublasDsyrk := GetProcAddress(hlib, 'cublasDsyrk');

  cublasCsyrk := GetProcAddress(hlib, 'cublasCsyrk');
  cublasZsyrk := GetProcAddress(hlib, 'cublasZsyrk');
  (* ------------------------------------------------------- *)
  (* HERK *)
  cublasCherk := GetProcAddress(hlib, 'cublasCherk');
  cublasZherk := GetProcAddress(hlib, 'cublasZherk');
  (* ------------------------------------------------------- *)
  (* SYR2K *)
  cublasSsyr2k := GetProcAddress(hlib, 'cublasSsyr2k');

  cublasDsyr2k := GetProcAddress(hlib, 'cublasDsyr2k');
  cublasCsyr2k := GetProcAddress(hlib, 'cublasCsyr2k');

  cublasZsyr2k := GetProcAddress(hlib, 'cublasZsyr2k');
  (* ------------------------------------------------------- *)
  (* HER2K *)
  cublasCher2k := GetProcAddress(hlib, 'cublasCher2k');

  cublasZher2k := GetProcAddress(hlib, 'cublasZher2k');

  (*------------------------------------------------------------------------*)
  (* SYMM*)
  cublasSsymm := GetProcAddress(hlib, 'cublasSsymm');
  cublasDsymm := GetProcAddress(hlib, 'cublasDsymm');

  cublasCsymm := GetProcAddress(hlib, 'cublasCsymm');

  cublasZsymm := GetProcAddress(hlib, 'cublasZsymm');
  (*------------------------------------------------------------------------*)
  (* HEMM*)
  cublasChemm := GetProcAddress(hlib, 'cublasChemm');
  cublasZhemm := GetProcAddress(hlib, 'cublasZhemm');

  (*------------------------------------------------------------------------*)
  (* TRSM*)
  cublasStrsm := GetProcAddress(hlib, 'cublasStrsm');

  cublasDtrsm := GetProcAddress(hlib, 'cublasDtrsm');

  cublasCtrsm := GetProcAddress(hlib, 'cublasCtrsm');

  cublasZtrsm := GetProcAddress(hlib, 'cublasZtrsm');
  (*------------------------------------------------------------------------*)
  (* TRMM*)
  cublasStrmm := GetProcAddress(hlib, 'cublasStrmm');
  cublasDtrmm := GetProcAddress(hlib, 'cublasDtrmm');
  cublasCtrmm := GetProcAddress(hlib, 'cublasCtrmm');

end;

initialization
  loadlibcublas(libcublas);

finalization
  freelibcublas();

end.
