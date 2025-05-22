program lstm_shakespeare_train;
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
  , OpenCLHelper, OpenCL, nnOpenCL//
  {$endif}
  { you can add units after this };

const
  READ_BATCH   : SizeInt = 32;
  READ_MEASURE : SizeInt = 4;
  READ_TEST    : SizeInt = 3;
  OUTPUT_CHARS = 10000;
  CFG_FILE     = '../../../../../cfg/lstm.cfg';
  //SAVE_FILE='lstm_shakespeare.weights';
  SAVE_FILE='lstm_shakespeare.train.weights';

var
  parser : TDarknetParser;
  net:TNNet;
  i, j, k, l :SizeInt;

  tokens: TArray<string>;
  n, inputs, c, len: SizeInt;
  input, truth, t: TSingleTensor;
  outchar: PSingleTensor;
  seed: String;
  temp: single;
  history : TSingleTensor;
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

function loadRNN():SizeInt;
begin
  srnd(2000);
  parser := TDarknetParser.Create(CFG_FILE);
  Parser.loadWeights(SAVE_FILE);
  //Parser.saveWeights('shakespeare.train.weights');
  //Parser.loadWeights('shakespeare.weights');
  net := parser.Neural;
  inputs := net.input.groupSize();
  result := inputs;
end;

procedure RunGenerator(paragraphLength:SizeInt=100);
begin
    writeln('inputs : ', inputs);
    input := TSingleTensor.Create([net.batch, inputs], net.batch);
    temp := 0.7;
    for i := 0 to net.layerCount() -1 do
        if net.layers[i].layerType=ltSOFTMAX then
          TSoftmaxLayer(net.layers[i]).temperature := temp;

    seed := 'Sir ';

    c := 0;
    for i := 1 to length(seed)-1 do begin
        c := SizeInt(seed[i]);
        input.data[c] := 1;
        input.pushToDevice;
        net.predict(input);
        input.data[c] := 0;
        write(ansichar(c));
    end;

    if(seed<>'') then
      c := SizeInt(seed[length(seed)]);

    write(ansichar(c));
    for i := 0 to paragraphLength -1 do
        begin
            input.data[c] := 1;
            input.pushToDevice;
            outChar := net.predict(input);
            input.data[c] := 0;
            outchar.pullFromDevice();
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

procedure AfterOptimization(const net:TNNet; const batchId:SizeInt);
var currBatch : SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  cursorClearScreen();
  write(metrics.print());
  metrics.reset;
  cursorAbsPos(40);
  {$else}
  cursorHome();
  {$endif}
  history.reSize([history.Size()+1]);
  history.DynData[history.Size()-1] := net.cost() / net.batch;
  history.plot();
  currBatch := net.getTrainedBatchs();
  if (currBatch>0) and (currBatch mod 8=0) then
  begin
    writeln('batch [', currBatch,'] saving ', SAVE_FILE);
    parser.saveWeights(SAVE_FILE);
  end else begin
    curserDown();
    writeln('batch [', currBatch,']');
  end;

end;

procedure resetRnnState();
var i, j:SizeInt; rnn:TRNNLayer;
begin
  for i:=0 to net.layerCount()-1 do
    if net.layers[i].layerType=ltRNN then begin
      rnn := TRNNLayer(net.layers[i]);
      //for j:=0 to rnn.batch -1 do
        rnn.state.Fill(0)
    end;
end;

{$ifdef USE_OPENCL}
procedure resetRnnStateGPU();
var i, j:SizeInt; rnn:TRNNLayer;
begin
  for i:=0 to net.layerCount()-1 do
    if net.layers[i].layerType=ltRNN then begin
      rnn := TRNNLayer(net.layers[i]);
      //for j:=0 to rnn.batch -1 do
       ocl.fill(rnn.state.size(), rnn.state.devData, 0, 0, 1)
    end;
end;
{$endif}

procedure runTrainer;
var f : TextFile;
  i, k, j, b, index, txtSize, streams , charpos: SizeInt;
  line , txt: ansistring;
  curr, next : ansichar;
  rands : TArray<SizeInt>;
  data : TData;
  s: ansistring;
begin
    parser := TDarknetParser.Create('../../../../../cfg/lstm.train.cfg');
     //if last saved weights exists load the last checkpoint
    if fileExists(SAVE_FILE) then begin
      writeln(' loading last check point [', SAVE_FILE,']...');
      cursorHome();
      parser.loadWeights(SAVE_FILE);
    end;
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
    streams := net.batch div net.timeSteps;
    setLength(rands, streams);
    srnd(RandSeed);

    TSingleTensor.noDeviceAllocation:=true;// do not memory allocate in GPU for these two tensors
    input := TSingleTensor.Create([txtSize div (net.batch), net.batch, inputs]);
    truth := TSingleTensor.Create([txtSize div (net.batch), net.batch, inputs]);
    TSingleTensor.noDeviceAllocation:=false;

    //net.input.reSize([net.batch, inputs], net.batch);
    data.X := input;
    data.y := truth;

    for k:=0 to 499 do begin
      input.fill(0);
      truth.fill(0);
      for i:=0 to txtSize div net.batch-1 do begin
        for b:=0 to high(rands) do begin
          //rands[i]:= i+1;
          // note : remember that "txt" is a pascal string type which starts at 1
          //rands[b]:= 1 + (rnd() mod (txtSize-1));
          rands[b]:= 1+random(txtSize - net.timeSteps -2);
          //rands[b]:= 20+b*600;
        end;

        for b:=0 to streams -1 do begin
          for j := 0 to net.timeSteps-1 do begin
            curr := txt[rands[b]];
            next := txt[rands[b]+1];
            index := (i*net.batch + j*streams + b)*inputs;
            input.dynData[index + SizeInt(curr)] := 1;
            truth.dynData[index + SizeInt(next)] := 1;
            // increment character
            rands[b] := (rands[b]+1) ;
          end;
        end;
      end;
      net.OnAfterNetOptimization := AfterOptimization;
      {$ifdef USE_OPENCL}
      //resetRnnStateGPU();
      {$else}
      //resetRnnState();
      {$endif}

      net.trainEpoch(data);
      cursorAbsPos(1, 30);
      writeln('================ epoch [', k:2,'] ===============)');
      cursorMove(cmUp,2);
      //readln
    end;
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

const BATCH = 9;
      STEPS = 567;
      W = 279;
      H = 177;
      IM = W*H;
      B = BATCH*IM;
      S = B*STEPS;
var t1, t2, t3, t4, t5, t6 :TSingleTensor;
begin
  DeleteFile('heap.trc');
  //setHeapTraceOutput('heap.trc');

//  tensor1 := TSingleTensor.Create([100]); // one dimension tensor of size [100], will always be initialized with zero
//  tensor1.fill(3.14159); //  filling a tensor with a number
//  tensor1.printStat;
//
//  tensor2 := TSingleTensor.Create([20, 20]);    // two dimensions tensor (100 X 100) filled with zeros
//  tensor2.UniformDistribution(0, 100);  // fill the tensor with random numbers uniformly between 0 and 100 (but not 100)
//  tensor2.printStat;
//  writeln();
//// you can also create a tensor by calling '.resize' method
//  tensor3.resize([100]);  // three dimensional (100 X 100 X 100) tensor
//  tensor3.map(sine, tensor3);  //
//  tensor3.plot();
//  readln();
//
//  exit;
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
  //
  runTrainer;
  //loadRNN();
  //RunGenerator(OUTPUT_CHARS);
  exit;

{$ifdef USE_OPENCL}
  t1.resize([STEPS, BATCH, IM]);
  t4.resize([STEPS, BATCH, IM]);
  t5.resize([STEPS, BATCH, IM]);
  t1.UniformDistribution(-1, 1);
  t5.UniformDistribution(-1, 1);
  t1.pushToDevice;
  t5.pushToDevice;
  t1.Groups:= BATCH;
  t5.Groups:= BATCH;
  writeln('ADD : ');
  for i:=0 to STEPS -1 do begin
    t5.add(t1, i*B, i*B, B);
    ocl.addvv(B, t1.devData, i*B, 1, t5.devData, i*B, 1, t5.devData, i*B, 1);
    t5.printGpuSumSqrDiff(B, i*B);
  end;
  t5.printGpuSumSqrDiff();
  t2.resize([IM]);
  t3.resize([IM]);

  writeln('meanVar, Normalize : ');
  for i:=0 to STEPS -1 do begin
    t1.MeansAndVars(t2, t3, i*B, B);
    ocl.meanAndVars({t1.size()} B, t2.Size, t1.groups, t1.devData, i*B, t2.devData, t3.devData );
    writeln(#13#10, '[', i, ']');
    //t3.printStat();
    //t4.printStat();
    ocl.normalize(t2.size, {t1.size()} B, t1.groups, t2.devData, 1, t3.devData, 1, t1.devData, i*B);
    t1.Normalize(t2, t3, i*B, B);
    t1.printGpuSumSqrDiff(B, i*B);
    t2.printGpuSumSqrDiff();
    t3.printGpuSumSqrDiff();
  end;
  t1.printGpuSumSqrDiff();
  readln
{$endif}

end.

