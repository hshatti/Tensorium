unit nOpMetrics;

{$mode Delphi}

interface

type

  TMeasureOps = (
    opIncFill, opFill, opCopy, opNorm, opBatchAddvs, opAddvs, opBatchMulvs,
    opMulvs, opBatchSubvs, opSubvs, opBatchDivvs, opDivvs, opPow, opAxpy, opMulvv, opAddvv, opSubvv, opDivvv, opFmavv, opDot, opBatchFmavss, opFmavss, opGemm, opIm2col, opCol2im,
    opConv2D, opConcat, opAddConcat, opHostToDevice, opDeviceToHost, opMemAllocate, opMemRelease, opGPUAllocate,
    opGPURelease, opIm2ColExt, opCol2ImExt, opMeans, opVariances,
    opMeansVars, opNormalize, opMeansVarsDelta, opNormalizeDelta, opAddDots, opForwardBias, opBackwardBias, opForwardScale, opForwardScaleAdd,
    opReduce , opFwDropout, opBwDropout, opL2,
    opInit
  );

  { TTensorOps }
  PTensorMetrics = ^TTensorMetrics;

  { TTensorMetrics }

  TTensorMetrics = record
  private
      m:array[0..999] of int64;
      stack: longint;
      function GetItem(i: TMeasureOps): int64;
  public
      elapsed: array[low(TMeasureOps)..high(TMeasureOps)] of int64;
      counts: array[low(TMeasureOps)..high(TMeasureOps)] of SizeInt;
      procedure start(const a:TMeasureOps);
      procedure finish(const a:TMeasureOps);
      function total():int64;
      property Item[i:TMeasureOps]:int64 read GetItem ;default;
  end;

var
  tensorMetrics : TTensorMetrics;

implementation
uses nChrono, sysutils;

{ TTensorMetric }

function TTensorMetrics.GetItem(i: TMeasureOps): int64;
begin
  result := elapsed[i]
end;

procedure TTensorMetrics.start(const a: TMeasureOps);
begin
  m[stack]:=clock;
  inc(counts[a]);
  inc(stack)
end;

procedure TTensorMetrics.finish(const a: TMeasureOps);
begin
  dec(stack);
  dec(counts[a]);
  elapsed[a] := elapsed[a] + clock()- m[stack]
end;

function TTensorMetrics.total(): int64;
var
  i: TMeasureOps;
begin
  result := 0;
  for i:=low(TMeasureOps) to high(TMeasureOps) do
    inc(result, elapsed[i])
end;

end.

