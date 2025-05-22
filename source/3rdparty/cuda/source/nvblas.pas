{$IFDEF FPC}
  {$mode delphi}
  {$PACKRECORDS C}
{$ENDIF}

unit nvblas;
interface
uses cudatypes;

{
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
  }
{ GEMM  }


{$if defined(MSWINDOWS)}
const libnvblas = 'nvblas64_12.dll';
{$elseif defined(DARWIN) or defined(MACOS)}
const libnvblas = 'libnvblas.dylib';
{$else}
const libnvblas = 'libnvblas.so';
{$endif}

var
sgemm_ : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:Psingle; a:Psingle; lda:Plongint; b:Psingle; ldb:Plongint; 
    beta:Psingle; c:Psingle; ldc:Plongint);WINAPI;

dgemm_ : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:Pdouble; a:Pdouble; lda:Plongint; b:Pdouble; ldb:Plongint; 
    beta:Pdouble; c:Pdouble; ldc:Plongint);WINAPI;

cgemm_ : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:PcuComplex; a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; 
    beta:PcuComplex; c:PcuComplex; ldc:Plongint);WINAPI;

zgemm_ : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:PcuDoubleComplex; a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; 
    beta:PcuDoubleComplex; c:PcuDoubleComplex; ldc:Plongint);WINAPI;

sgemm : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:Psingle; a:Psingle; lda:Plongint; b:Psingle; ldb:Plongint; 
    beta:Psingle; c:Psingle; ldc:Plongint);WINAPI;

dgemm : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:Pdouble; a:Pdouble; lda:Plongint; b:Pdouble; ldb:Plongint; 
    beta:Pdouble; c:Pdouble; ldc:Plongint);WINAPI;

cgemm : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:PcuComplex; a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; 
    beta:PcuComplex; c:PcuComplex; ldc:Plongint);WINAPI;

zgemm : procedure(transa:Pchar; transb:Pchar; m:Plongint; n:Plongint; k:Plongint; 
    alpha:PcuDoubleComplex; a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; 
    beta:PcuDoubleComplex; c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ SYRK  }

ssyrk_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Psingle; 
    a:Psingle; lda:Plongint; beta:Psingle; c:Psingle; ldc:Plongint);WINAPI;

dsyrk_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Pdouble; 
    a:Pdouble; lda:Plongint; beta:Pdouble; c:Pdouble; ldc:Plongint);WINAPI;

csyrk_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; beta:PcuComplex; c:PcuComplex; ldc:Plongint);WINAPI;

zsyrk_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; beta:PcuDoubleComplex; c:PcuDoubleComplex; ldc:Plongint);WINAPI;

ssyrk : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Psingle; 
    a:Psingle; lda:Plongint; beta:Psingle; c:Psingle; ldc:Plongint);WINAPI;

dsyrk : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Pdouble; 
    a:Pdouble; lda:Plongint; beta:Pdouble; c:Pdouble; ldc:Plongint);WINAPI;

csyrk : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; beta:PcuComplex; c:PcuComplex; ldc:Plongint);WINAPI;

zsyrk : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; beta:PcuDoubleComplex; c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ HERK  }

cherk_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Psingle; 
    a:PcuComplex; lda:Plongint; beta:Psingle; c:PcuComplex; ldc:Plongint);WINAPI;

zherk_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Pdouble; 
    a:PcuDoubleComplex; lda:Plongint; beta:Pdouble; c:PcuDoubleComplex; ldc:Plongint);WINAPI;

cherk : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Psingle; 
    a:PcuComplex; lda:Plongint; beta:Psingle; c:PcuComplex; ldc:Plongint);WINAPI;

zherk : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Pdouble; 
    a:PcuDoubleComplex; lda:Plongint; beta:Pdouble; c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ TRSM  }

strsm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Psingle; a:Psingle; lda:Plongint; b:Psingle; 
    ldb:Plongint);WINAPI;

dtrsm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Pdouble; a:Pdouble; lda:Plongint; b:Pdouble; 
    ldb:Plongint);WINAPI;

ctrsm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuComplex; a:PcuComplex; lda:Plongint; b:PcuComplex; 
    ldb:Plongint);WINAPI;

ztrsm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuDoubleComplex; a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; 
    ldb:Plongint);WINAPI;

strsm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Psingle; a:Psingle; lda:Plongint; b:Psingle; 
    ldb:Plongint);WINAPI;

dtrsm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Pdouble; a:Pdouble; lda:Plongint; b:Pdouble; 
    ldb:Plongint);WINAPI;

ctrsm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuComplex; a:PcuComplex; lda:Plongint; b:PcuComplex; 
    ldb:Plongint);WINAPI;

ztrsm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuDoubleComplex; a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; 
    ldb:Plongint);WINAPI;
{ SYMM  }

ssymm_ : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:Psingle; 
    a:Psingle; lda:Plongint; b:Psingle; ldb:Plongint; beta:Psingle; 
    c:Psingle; ldc:Plongint);WINAPI;

dsymm_ : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:Pdouble; 
    a:Pdouble; lda:Plongint; b:Pdouble; ldb:Plongint; beta:Pdouble; 
    c:Pdouble; ldc:Plongint);WINAPI;

csymm_ : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:PcuComplex; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zsymm_ : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:PcuDoubleComplex; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;

ssymm : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:Psingle; 
    a:Psingle; lda:Plongint; b:Psingle; ldb:Plongint; beta:Psingle; 
    c:Psingle; ldc:Plongint);WINAPI;

dsymm : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:Pdouble; 
    a:Pdouble; lda:Plongint; b:Pdouble; ldb:Plongint; beta:Pdouble; 
    c:Pdouble; ldc:Plongint);WINAPI;

csymm : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:PcuComplex; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zsymm : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:PcuDoubleComplex; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ HEMM  }

chemm_ : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:PcuComplex; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zhemm_ : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:PcuDoubleComplex; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ HEMM with no underscore }

chemm : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:PcuComplex; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zhemm : procedure(side:Pchar; uplo:Pchar; m:Plongint; n:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:PcuDoubleComplex; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ SYR2K  }

ssyr2k_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Psingle; 
    a:Psingle; lda:Plongint; b:Psingle; ldb:Plongint; beta:Psingle; 
    c:Psingle; ldc:Plongint);WINAPI;

dsyr2k_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Pdouble; 
    a:Pdouble; lda:Plongint; b:Pdouble; ldb:Plongint; beta:Pdouble; 
    c:Pdouble; ldc:Plongint);WINAPI;

csyr2k_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:PcuComplex; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zsyr2k_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:PcuDoubleComplex; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ SYR2K no_underscore }

ssyr2k : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Psingle; 
    a:Psingle; lda:Plongint; b:Psingle; ldb:Plongint; beta:Psingle; 
    c:Psingle; ldc:Plongint);WINAPI;

dsyr2k : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:Pdouble; 
    a:Pdouble; lda:Plongint; b:Pdouble; ldb:Plongint; beta:Pdouble; 
    c:Pdouble; ldc:Plongint);WINAPI;

csyr2k : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:PcuComplex; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zsyr2k : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:PcuDoubleComplex; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ HERK  }

cher2k_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:Psingle; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zher2k_ : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:Pdouble; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ HER2K with no underscore  }

cher2k : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuComplex; 
    a:PcuComplex; lda:Plongint; b:PcuComplex; ldb:Plongint; beta:Psingle; 
    c:PcuComplex; ldc:Plongint);WINAPI;

zher2k : procedure(uplo:Pchar; trans:Pchar; n:Plongint; k:Plongint; alpha:PcuDoubleComplex; 
    a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; ldb:Plongint; beta:Pdouble; 
    c:PcuDoubleComplex; ldc:Plongint);WINAPI;
{ TRMM  }

strmm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Psingle; a:Psingle; lda:Plongint; b:Psingle; 
    ldb:Plongint);WINAPI;

dtrmm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Pdouble; a:Pdouble; lda:Plongint; b:Pdouble; 
    ldb:Plongint);WINAPI;

ctrmm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuComplex; a:PcuComplex; lda:Plongint; b:PcuComplex; 
    ldb:Plongint);WINAPI;

ztrmm_ : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuDoubleComplex; a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; 
    ldb:Plongint);WINAPI;

strmm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Psingle; a:Psingle; lda:Plongint; b:Psingle; 
    ldb:Plongint);WINAPI;

dtrmm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:Pdouble; a:Pdouble; lda:Plongint; b:Pdouble; 
    ldb:Plongint);WINAPI;

ctrmm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuComplex; a:PcuComplex; lda:Plongint; b:PcuComplex; 
    ldb:Plongint);WINAPI;

ztrmm : procedure(side:Pchar; uplo:Pchar; transa:Pchar; diag:Pchar; m:Plongint; 
    n:Plongint; alpha:PcuDoubleComplex; a:PcuDoubleComplex; lda:Plongint; b:PcuDoubleComplex; 
    ldb:Plongint);WINAPI;

implementation

  uses
    SysUtils, dynlibs;

  var
    hlib : tlibhandle;


  procedure Freenvblas;
    begin
      FreeLibrary(hlib);
      sgemm_:=nil;
      dgemm_:=nil;
      cgemm_:=nil;
      zgemm_:=nil;
      sgemm:=nil;
      dgemm:=nil;
      cgemm:=nil;
      zgemm:=nil;
      ssyrk_:=nil;
      dsyrk_:=nil;
      csyrk_:=nil;
      zsyrk_:=nil;
      ssyrk:=nil;
      dsyrk:=nil;
      csyrk:=nil;
      zsyrk:=nil;
      cherk_:=nil;
      zherk_:=nil;
      cherk:=nil;
      zherk:=nil;
      strsm_:=nil;
      dtrsm_:=nil;
      ctrsm_:=nil;
      ztrsm_:=nil;
      strsm:=nil;
      dtrsm:=nil;
      ctrsm:=nil;
      ztrsm:=nil;
      ssymm_:=nil;
      dsymm_:=nil;
      csymm_:=nil;
      zsymm_:=nil;
      ssymm:=nil;
      dsymm:=nil;
      csymm:=nil;
      zsymm:=nil;
      chemm_:=nil;
      zhemm_:=nil;
      chemm:=nil;
      zhemm:=nil;
      ssyr2k_:=nil;
      dsyr2k_:=nil;
      csyr2k_:=nil;
      zsyr2k_:=nil;
      ssyr2k:=nil;
      dsyr2k:=nil;
      csyr2k:=nil;
      zsyr2k:=nil;
      cher2k_:=nil;
      zher2k_:=nil;
      cher2k:=nil;
      zher2k:=nil;
      strmm_:=nil;
      dtrmm_:=nil;
      ctrmm_:=nil;
      ztrmm_:=nil;
      strmm:=nil;
      dtrmm:=nil;
      ctrmm:=nil;
      ztrmm:=nil;
    end;


  procedure Loadnvblas(lib : pchar);
    begin
      Freenvblas;
      hlib:=LoadLibrary(lib);
      if hlib=0 then
        raise Exception.Create(format('Could not load library: %s',[lib]));

      sgemm_ := GetProcAddress(hlib,'sgemm_');
      dgemm_ := GetProcAddress(hlib,'dgemm_');
      cgemm_ := GetProcAddress(hlib,'cgemm_');
      zgemm_ := GetProcAddress(hlib,'zgemm_');
      sgemm := GetProcAddress(hlib,'sgemm');
      dgemm := GetProcAddress(hlib,'dgemm');
      cgemm := GetProcAddress(hlib,'cgemm');
      zgemm := GetProcAddress(hlib,'zgemm');
      ssyrk_ := GetProcAddress(hlib,'ssyrk_');
      dsyrk_ := GetProcAddress(hlib,'dsyrk_');
      csyrk_ := GetProcAddress(hlib,'csyrk_');
      zsyrk_ := GetProcAddress(hlib,'zsyrk_');
      ssyrk := GetProcAddress(hlib,'ssyrk');
      dsyrk := GetProcAddress(hlib,'dsyrk');
      csyrk := GetProcAddress(hlib,'csyrk');
      zsyrk := GetProcAddress(hlib,'zsyrk');
      cherk_ := GetProcAddress(hlib,'cherk_');
      zherk_ := GetProcAddress(hlib,'zherk_');
      cherk := GetProcAddress(hlib,'cherk');
      zherk := GetProcAddress(hlib,'zherk');
      strsm_ := GetProcAddress(hlib,'strsm_');
      dtrsm_ := GetProcAddress(hlib,'dtrsm_');
      ctrsm_ := GetProcAddress(hlib,'ctrsm_');
      ztrsm_ := GetProcAddress(hlib,'ztrsm_');
      strsm := GetProcAddress(hlib,'strsm');
      dtrsm := GetProcAddress(hlib,'dtrsm');
      ctrsm := GetProcAddress(hlib,'ctrsm');
      ztrsm := GetProcAddress(hlib,'ztrsm');
      ssymm_ := GetProcAddress(hlib,'ssymm_');
      dsymm_ := GetProcAddress(hlib,'dsymm_');
      csymm_ := GetProcAddress(hlib,'csymm_');
      zsymm_ := GetProcAddress(hlib,'zsymm_');
      ssymm := GetProcAddress(hlib,'ssymm');
      dsymm := GetProcAddress(hlib,'dsymm');
      csymm := GetProcAddress(hlib,'csymm');
      zsymm := GetProcAddress(hlib,'zsymm');
      chemm_ := GetProcAddress(hlib,'chemm_');
      zhemm_ := GetProcAddress(hlib,'zhemm_');
      chemm := GetProcAddress(hlib,'chemm');
      zhemm := GetProcAddress(hlib,'zhemm');
      ssyr2k_ := GetProcAddress(hlib,'ssyr2k_');
      dsyr2k_ := GetProcAddress(hlib,'dsyr2k_');
      csyr2k_ := GetProcAddress(hlib,'csyr2k_');
      zsyr2k_ := GetProcAddress(hlib,'zsyr2k_');
      ssyr2k := GetProcAddress(hlib,'ssyr2k');
      dsyr2k := GetProcAddress(hlib,'dsyr2k');
      csyr2k := GetProcAddress(hlib,'csyr2k');
      zsyr2k := GetProcAddress(hlib,'zsyr2k');
      cher2k_ := GetProcAddress(hlib,'cher2k_');
      zher2k_ := GetProcAddress(hlib,'zher2k_');
      cher2k := GetProcAddress(hlib,'cher2k');
      zher2k := GetProcAddress(hlib,'zher2k');
      strmm_ := GetProcAddress(hlib,'strmm_');
      dtrmm_ := GetProcAddress(hlib,'dtrmm_');
      ctrmm_ := GetProcAddress(hlib,'ctrmm_');
      ztrmm_ := GetProcAddress(hlib,'ztrmm_');
      strmm := GetProcAddress(hlib,'strmm');
      dtrmm := GetProcAddress(hlib,'dtrmm');
      ctrmm := GetProcAddress(hlib,'ctrmm');
      ztrmm := GetProcAddress(hlib,'ztrmm');
    end;


initialization
  Loadnvblas(libnvblas);
finalization
  Freenvblas;

end.
