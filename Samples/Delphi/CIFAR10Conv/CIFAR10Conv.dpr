program CIFAR10Conv;
{$APPTYPE CONSOLE}

{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
{$pointermath on}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, ntensors, ntypes, nDatasets, nBaseLayer, nConnectedlayer
  , nLogisticLayer, nSoftmaxLayer, nCostLayer, nnet, nChrono, nConvolutionLayer, nUpSampleLayer, nDropOutLayer
  , nAttentionLayer, nModels, {Keyboard, nParser,} termesc, steroids
  {$if defined(MSWINDOWS)}
  , ShellApi, uTokenizer, sixel, SortedMap
  //, cudnn_graph
  //, cudnn_adv
  //, cudnn_ops
  //, cudnn_cnn
  {$endif}
  {$if defined(USE_OPENCL)}
  , OpenCLHelper, OpenCL, nnOpenCL, nOpMetrics//
  {$elseif defined(USE_CUDART)}
  , nnCuda
  {$endif}
  { you can add units after this };

const
  READ_BATCH   = 32;
  READ_MEASURE = 8;
  READ_TEST    = 3;
var
  Neural:TNNet;
  CF10 : TCIFAR10Data;
  i, j, k, l :SizeInt;
  Data : TData;
  cost :single;
  costDelta : single;
  s : clock_t;
  Predicted, Truth : TInt64Tensor;
  output  : PSingleTensor;
  sampled : TSingleTensor;
  c : shortstring;


procedure _conv2d(const src: PSingle; ker: PSingle; var dest: PSingle;
  const wSrc, hSrc, wKernel, hKernel, wPad, hPad, xStr, yStr, xDil,
  yDil: SizeInt);
var
  {kx, kw, }ky {,kh}, wp, hp, wDst, hDst, i, j: SizeInt;
  ker2, srcIM, dstIM:PSingle;
  acc:Single;
begin

  //kw := wKernel div 2;
  //kh := hKernel div 2;
  //kSize := wKernel * hKernel;
  wDst := wSrc div xStr + wPad*2 - wKernel + 1;
  hDst := hSrc div yStr + hPad*2 - hKernel + 1;
  wP := {kw} - wPad;
  hP := {kh} - hPad;
  ker := ker {+ kh*wKernel}{ + kw};
  for i := hPad to hDst - hPad -1 do begin
    dstIM := dest + i*wDst;
    for j := wPad to wDst - wPad-1 do begin
      acc := dstIM[j];
      for ky := 0{-kh} to hKernel-1{kh} do begin
        srcIM := src + (i*yStr + ky*yDil)*wSrc + j*xStr + hP*wSrc + wp;
        ker2 := ker + ky*wKernel;
        acc := acc + cblas_sdot(wKernel, ker2, 1, srcIm, xDil);
        //for kx := 0{-kw} to wKernel-1{kw} do
        //  acc :=  plus(acc , ker2[kx]*srcIM[kx*xDil]);
      end;
      dstIM[j] := acc
    end;
  end
end;

var
  //img : TImageData;
  coor : TArray<SizeInt>;
  trainingHistory , bmp: TSingleTensor;
  //drop: TDropoutLayer;
  //sn : TNNetState;

  {$ifdef USE_OPENCL}
  dev : TArray<cl_device_id>;
  res1, res2 : single;
  t : int64;
  {$endif}
begin
  //write(#$1B'[1J');
{$if defined(USE_OPENCL)}
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
    for j:=0 to length(dev)-1 do
      writeln(' ',j, ' : ', TOpenCL.getDeviceName(dev[j]));
//      writeln(' ', j, ' [',TOpenCL.getDeviceTypeName(dev[j]),']: '+ TOpenCL.getDeviceName(dev[j]));
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
  ocl.useBLAS := CL_LIB_BLAST;
  //ocl.queueInOrder:=true;
  writeln('  - out of Order mode : ', not ocl.queueInOrder);

{$elseif defined(USE_CUDART)}
  initCUDART(0);
  writeln(cuda.properties.name);
  cuda.useBLAS := 1;

{$endif}
  sDigits := 6;


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

 //testing pseudorandom gen for dropout
  //bmp := TSingleTensor.Create([80, 80]);
  //drop := TDropoutLayer.Create(1, 0.1, 80*80);
  //sn.input:=@bmp;
  //sn.isTraining:=true;
  //while true do begin
  //  bmp.fill(1);
  //  bmp.setCPU;
  //  drop.forwardGPU(sn);
  //  drop.output.pullFromDevice();
  //  cursorHome();
  //  drop.output.print(psGray);
  //  readln;
  //end;
  //
  //drop.free();
  //exit;

  CF10 := TCIFAR10Data.Create('');

  speedOverSize:=true;
  Neural:=TNNet.Create(deepCIFAR10);
  //Neural:=TNNet.Create(leNetCIFAR10);

  Neural.setTraining(true);
  Neural.batch       := READ_BATCH;
  Neural.learningRate:= 0.001;
  Neural.momentum    := 0.9;
  neural.decay       := 0.0001;
  neural.policy      := lrpCOST;

  CF10.load(Neural.batch);

  data.X.resize([Neural.batch, CF10.CHANNELS, CF10.IMAGE_RES, CF10.IMAGE_RES], Neural.batch);
  data.Y.reSize([Neural.batch, CF10.CLASS_COUNT], Neural.batch);
  //Data.X.reShape([Neural.batch, CF10.IMAGE_SIZE]);
  //Data.Y.reShape([Neural.batch, CF10.CLASS_COUNT]);
  sampled.reShape([Neural.batch, CF10.CLASS_COUNT], Neural.batch);
  //Neural.truth.resize(Data.Y.Shape);

  i         := 0;
  j         := 0;
  l         := 0;
  costDelta := 0;
  cost :=0 ;
  Randomize;

  s := clock();
  predicted.resize([READ_MEASURE * Neural.batch]);
  truth.resize([READ_MEASURE * Neural.batch]);
  termesc.cursorClearScreen();
//  InitKeyboard;
  //write(#27'[8;80;200t'); // resize terminal to (80 row, 200 col)?

  while true do begin

    //i := random(CF10.DATA_COUNT div Neural.batch)-1;
    if not CF10.read(j) then break;

    CF10.TrainingData.toSingles(pointer(data.X.Data));
    CF10.TrainingLabels.toSingles(pointer(Data.Y.Data));

    data.X.maxNormalize(1);//FusedMultiplyAdd(1/128, -1);

    cost := cost + Neural.trainEpoch(Data);
//    k := PollKeyEvent;
//    if keyPressed then
//      Break;

    output := Neural.output();
    sampled.ShallowCopy(Neural.truth);

    //writeln(#$1B'[4;0H', 'Press [ESC] to stop training...');

    output.pullFromDevice;
    output.argMax(pointer(Predicted.data + (j mod READ_MEASURE)*READ_BATCH));
    sampled.argMax(pointer(Truth.data + (j mod READ_MEASURE)*READ_BATCH));
    if j mod READ_MEASURE = READ_MEASURE-1 then begin
      cost := cost / READ_MEASURE ;
      costDelta := costDelta - cost;
      s :=  READ_MEASURE * trunc(CLOCKS_PER_SEC / (clock() - s));
      inc(l);


      trainingHistory.resize([l]);
      trainingHistory.Data[l-1] := cost;
      cursorAbsPos();
      //cursorClearDown();
      writeln('Batch [',j:4,'], epoch[',i {j*Neural.batch div CF10.DATA_COUNT}:5,'], Cost [',cost:1:8,']',widechar($2191 +2*ord(costDelta>0)),' speed [', s*Neural.batch :5,'] Sample per second, '
        ,'Accuracy [', 100*truth.similarity(predicted.Data):3:2,'%], learningRate [',Neural.computeCurrentLearningRate:1:3,']', sLineBreak);
      //writeln('Conv[1] ');
      //Neural.layers[0].weights.print(true, 18);
      //Neural.layers[0].biases.print(true);
      //Neural.layers[0].output.print(true, 4);
      //writeln('Conv[3] ');
      //Neural.layers[3].weights.print(true, 3);
      //Neural.layers[2].output.print(psGray24, 16);

      //writeln(sLineBreak,'prediction:');
      //output.print(psGray24);
      //writeln(sLineBreak, 'truth:');
      //Sampled.print(psGray24);

      //write('Predicted:',#$1B'[1J');

      write('Predicted:', cursorMove(cmBackward, 10), cursorMove(cmDown));
      coor := output.print(psGray24);
      write(cursorMove(cmUP, coor[1]+1), cursorMove(cmForward, 42));

      write('Actual:', cursorMove(cmBackward,7), cursorMove(cmDown));
      coor := sampled.print(psGray24);

      write(#13, cursorMove(cmForward, 42), cursorMove(cmDown, 1));

      coor := trainingHistory.plot;
      {$ifdef USE_TELEMETRY}
      termesc.cursorAbsPos(1, 24);
      writeln(sLineBreak, metrics.print(TELEMETRY_OPS {or TELEMETRY_FWD or TELEMETRY_BWD or TELEMETRY_UPD}));
      metrics.reset;
      {$endif}
      //writeln(sLineBreak, 'Predicted :');
      //Predicted.print();
      //writeln(sLineBreak, 'Truth :');
      //truth.print();
      //t1.print(true, -1, 1);
      costDelta := cost;
      cost := 0;
      truth.fill(0);
      predicted.fill(0);

      //if j > CF10.DATA_COUNT * Neural.batch then
      //begin
      //  readln(c);
      //  if c = 'b' then break;
      //end;
      //inc(i);
      s := clock();
      //inc(k)
    end;
    inc(j);
    if j>= CF10.DATA_COUNT div Neural.batch then begin
      inc(i);
      j:=0;
    end;
  end;
//  DoneKeyboard;
  // test prediction

  writeln(sLineBreak,' press [Enter] to test:');
  readln;

  CF10.load(0, READ_TEST);

  Data.X.reSize([READ_TEST, CF10.CHANNELS, CF10.IMAGE_RES, CF10.IMAGE_RES], READ_TEST);
  Data.Y.reSize([READ_TEST, CF10.CLASS_COUNT], READ_TEST);

  Predicted := TInt64Tensor.create([READ_TEST]);
  truth     := TInt64Tensor.create([READ_TEST]);
  Neural.setTraining(false);
  Neural.Batch := READ_TEST;

  cursorClearUp();
  while true do try
    cursorAbsPos();
    i := random(CF10.TEST_COUNT div READ_TEST);
    CF10.read(-1, i);
    CF10.TestingData.toSingles(pointer(Data.X.Data));
    CF10.TestingLabels.toSingles(pointer(Data.Y.Data));
    Data.X.maxNormalize(1);//t1.FusedMultiplyAdd(1/127, -1);

    Data.X.print(psColor24, READ_TEST);

    writeln('truth');
    Data.Y.argMax(pointer(truth.Data));
    Data.Y.print(psGray);
    truth.print;
    writeln(sLineBreak, 'Predicted');
    Neural.Input := Data.X;
    //Neural.input.reshape([READ_TEST, CF10.IMAGE_SIZE], READ_TEST);
    Neural.predict(Neural.input);
    Neural.output().argMax(pointer(predicted.Data));
    Neural.output().print(psGray);
    predicted.print();

    writeln('Press [Enter] for next random digit, [Ctrl-C] to exit ...');
    readln(c);
    if LowerCase(c) = 'q' then break;
  except on E:Exception do
    writeln(E.Message)
  end;

  CF10.free;
  Neural.free;

  //freeandnil(mp); freeandnil(mp2); freeandnil(mp3)
end.
