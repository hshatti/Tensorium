{$if defined(MSWINDOWS)}
  const libcublas='cublas64_12.dll';
{$elseif defined(LINUX)}
  const libcublas='libcublas.so';
{$endif}

