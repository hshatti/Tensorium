program shakespeare_infer_rnn;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, math, ntensors, ntypes, nDatasets, nBaseLayer, nConnectedlayer
  , nLogisticLayer, nSoftmaxLayer, nCostLayer, nnet, nChrono, nConvolutionLayer, nUpSampleLayer, nRNNLayer
  , nParser, Keyboard, nNormalizationLayer, termesc, steroids
  {$if defined(MSWINDOWS)}
  , ShellApi, cudnn_graph, cudnn_adv, cudnn_ops, cudnn_cnn
  {$endif}
  {$if defined(USE_OPENCL)}
  , OpenCLHelper, OpenCL, nnOpenCL, nOpMetrics//
  {$endif}
  { you can add units after this };

const
  READ_BATCH   : SizeInt = 32;
  READ_MEASURE : SizeInt = 4;
  READ_TEST    : SizeInt = 3;
  OUTPUT_CHARS = 10000;
  CFG_FILE     = '../../../../../cfg/rnn.cfg';

var
  parser : TDarknetParser;
  net:TNNet;
  i, j, k, l :SizeInt;

  tokens: TArray<string>;
  n, inputs, c, len: SizeInt;
  input, truth: TSingleTensor;
  outchar: PSingleTensor;
  seed: String;
  temp: single;
{$ifdef USE_OPENCL}
  dev : TArray<cl_device_id>;

{$endif}

function sample_array(a:PSingle; N:SizeInt):sizeInt;
var sm, r :single;
  i:SizeInt;
begin
    sm := sum(a, N);
    for i:=0 to N-1 do
        a[i]:=a[i] / sm;
    r := rnd()/$7FFFFFFF;
    for i := 0 to N-1 do begin
        r := r - a[i];
        if r <= 0 then exit(i);
    end;
    result := n - 1;
end;

function top_max_index(a: PSingle; n: SizeInt; k: SizeInt):longint;
var
    values: TArray<Single>;
    indexes: TArray<SizeInt>;
    i: SizeInt;
    j: SizeInt;
    count: SizeInt;
    get_index: SizeInt;
    val: SizeInt;
begin
    if n <= 0 then
        exit(-1);
    setLength(values  , k);
    setLength(indexes , k);
    for i := 0 to n -1 do
        for j := 0 to k -1 do
            if a[i] > values[j] then
                begin
                    values[j] := a[i];
                    indexes[j] := i;
                    break
                end;
    count := 0;
    for j := 0 to k -1 do
        if values[j] > 0 then
            inc(count);
    get_index := rnd() mod (count);
    val := indexes[get_index];
    exit(val)
end;


procedure RunGenerator;
begin
    srnd(2000);
    parser := TDarknetParser.Create(CFG_FILE);
    Parser.loadWeights('shakespeare.weights');
    net := parser.Neural;
    inputs := net.input.groupSize();
    writeln('inputs : ', inputs);
    input := TSingleTensor.Create([net.batch, inputs], net.batch);
    temp := 0.7;
    for i := 0 to net.layerCount() -1 do
        if net.layers[i].layerType=ltSOFTMAX then
          TSoftmaxLayer(net.layers[i]).temperature := temp;
    seed := 'Hello';

    c := 0;
    for i := 1 to length(seed)-1 do begin
        c := SizeInt(seed[i]);
        input.data[c] := 1;
        net.predict(input);
        input.data[c] := 0;
        write(ansichar(c));
    end;

    if(seed<>'') then
      c := SizeInt(seed[length(seed)]);

    write(ansichar(c));
    for i := 0 to OUTPUT_CHARS -1 do
        begin
            input.data[c] := 1;
            outChar := net.predict(input);
            input.data[c] := 0;
            for j := 0 to inputs-1 do
              if outchar.data[j] < 0.0001 then outchar.data[j] := 0;
            c := outchar.sample(inputs);
            //c := sample_array(outchar.data, inputs);
            //c:=top_max_index(outchar.data, inputs, 2);
            write(format('%s', [ansichar(c)]));
        end;
    readln();
    input.free;
    parser.free;
end;

procedure runTrainer;
var f : TextFile;
  i, j, b, txtSize : SizeInt;
  line , txt: ansistring;
  rands : TArray<SizeInt>;
  data : TData;
begin
    parser := TDarknetParser.Create('../../../../../cfg/rnn.train.cfg');

    net := parser.Neural;
    net.setTraining(true);
    inputs := net.input.groupSize();
    txt := '';
    AssignFile(f, 'input.txt');
    reset(f);
    while not EOF(f) do begin
       readln(f, line);
       txt := txt + #13#10 + line
    end;
    closeFile(f);
    txtSize := length(txt);
    setLength(rands, net.batch);
    srnd(RandSeed);
    for i:=0 to high(rands) do
      rands[i]:= rnd() mod txtSize;

    input := TSingleTensor.Create([net.timeSteps, net.batch, inputs], net.batch, net.timeSteps);
    truth := TSingleTensor.Create([net.timeSteps, net.batch, inputs], net.batch, net.timeSteps);
    net.input.reSize([net.timeSteps, net.batch, inputs], net.batch, net.timeSteps);
    data.X := input;
    data.y := truth;
    for b:=0 to net.batch-1 do begin
      for j := 0 to net.timeSteps-1 do begin
        input.Value[[j, b, ord(txt[rands[b]   ]) ]] := 1;
        truth.Value[[j, b, ord(txt[rands[b] +1]) ]] := 1;
      end;
    end;

    net.trainEpoch(data);
    writeln('trained epoch... ');
    readln();
    input.free;
    truth.free;
    parser.free;
end;

var
  tensor1, tensor2, tensor3 : TSingleTensor;
function sine(const val:single; const index:SizeInt):single;
begin
  exit(sin(index*0.1))
end;

begin
  DeleteFile('heap.trc');
  setHeapTraceOutput('heap.trc');

  tensor1 := TSingleTensor.Create([100]); // one dimension tensor of size [100], will always be initialized with zero
  tensor1.fill(3.14159); //  filling a tensor with a number
  tensor1.printStat;

  tensor2 := TSingleTensor.Create([20, 20]);    // two dimensions tensor (100 X 100) filled with zeros
  tensor2.UniformDistribution(0, 100);  // fill the tensor with random numbers uniformly between 0 and 100 (but not 100)
  tensor2.printStat;
  writeln();
// you can also create a tensor by calling '.resize' method
  tensor3.resize([100]);  // three dimensional (100 X 100 X 100) tensor
  tensor3.map(sine, tensor3);  //
  tensor3.plot();
  readln();

  exit;
  //write(#$1B'[1J');
{$ifdef USE_OPENCL}
  i:=0;
  j:=0;
  //initOpenCL(i, j);

  TSingleTensor.defaultDevice := cdOpenCL;
  if TOpenCL.PlatformCount>1 then
  repeat
    writeln('Choose computing platform:');
    for i:=0 to TOpenCL.PlatformCount-1 do
      writeln(' ',i,' : ', TOpenCL.PlatformName(i));
    try
      readln(i);
      if (i<0) or (i > TOpenCL.PlatformCount-1) then
        writeln('invalid platform id, try again..');
    except on e:Exception do
      begin
        writeln(e.Message);
        i:=-1
      end
    end;
  until (i>=0) and (i<TOpenCL.PlatformCount)
  else
    raise Exception.create('No OpenCL platforms found!');
  writeln('Using : ',  TOpenCL.PlatformName(i), #13#10'select device :');
  dev := TOpenCL.getDevices(i);

  repeat
    for j:=0 to high(dev) do
      writeln(' ', j, ' [',TOpenCL.getDeviceTypeName(dev[j]),']: ', TOpenCL.getDeviceName(dev[j]));
    try
      readln(j);
    except on e:Exception do
      begin
        writeln(e.message);
        j:=-1
      end
    end;

  until (j>=0) and (j<length(dev)) ;
  initOpenCL(i, j);
  ocl.useBLAS := 2;
  //ocl.queueInOrder:=true;
  writeln('  - out of Order mode : ', not ocl.queueInOrder);

{$endif}
  sDigits := 3;

  //sleep(500);
  ////img.loadFromFile(['../../../../../data/dog.jpg', '../../../../../data/eagle.jpg'], 416, 416);
  //img.loadFromFile('../../../../../data/dog.jpg');
  //t1 := img.toTensor();
  //t1.printStat;
  //t1.im2Col(5, 5, 2, 2, 1, 1, 1, 1, t2);
  //t2.SaveToImage('tmp.bmp');
  //ShellExecute(0, 'open', 'tmp.bmp', '', '', 0);
  //readln;
  //t3.col2Im(5, 5, 2, 2, 1, 1, 1, 1, t2);
  //t3.printStat;
  //t3.SaveToImage('tmp.bmp');
  //ShellExecute(0, 'open', 'tmp.bmp', '', '', 0);
  //readln;
  //t2.pushToDevice;
  //ocl.fill(t3.size(), t3.devData, 0, 1);
  //ocl.col2im(3, t3.h, t3.w, 5, 5, 2, 2, 1, 1, 1, 1, t2.devData, 0, t3.devData, 0);
  //t3.pullFromDevice();
  //t3.printStat;
  //t3.SaveToImage('tmp.bmp');
  //ShellExecute(0, 'open', 'tmp.bmp', '', '', 0);
  //readln;
  //exit;

  {$ifdef USE_TELEMETRY}
  benchmark:=true;
  {$endif}
  Randomize;

  //RunGenerator;
  RunGenerator;
  //if assigned(token_file) then
  //    begin
  //        tokens := read_tokens(token_file,  and n)
  //    end;
  //srand(rseed);

end.

