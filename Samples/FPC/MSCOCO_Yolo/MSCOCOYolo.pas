program MSCOCOYolo;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$ModeSwitch typehelpers}
{$endif}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, math, ntensors, ntypes, nDatasets, nBaseLayer
  , nConnectedlayer
  , nLogisticLayer
  , nSoftmaxLayer
  , nCostLayer
  , nnet
  , nChrono
  , nConvolutionLayer
  , nUpSampleLayer
  , nAddLayer
  , nMaxPoolLayer
  , nConcatLayer
  , nModels, Keyboard, nparser, nXML, termesc, sixel
  {$ifdef MSWINDOWS}, ShellApi, nHttp, nRegionLayer {$endif}
  { you can add units after this };


const
    //cfgFile = '../../../../../cfg/yolov7.cfg';
    //weightFile = '../../../../../yolov7.weights';
    cfgFileVOC = '../../../../../cfg/yolov3-voc.cfg';
    //cfgFile = '../../../../../cfg/yolov3.cfg';
    //weightFile = '../../../../../yolov3.weights';
    cfgFile = '../../../../../cfg/yolo9000.cfg';
    weightFile = '../../../../../yolo9000.weights';

    images :TStringArray = ['eagle.jpg', 'kite.jpg', 'person.jpg', 'dog.jpg', 'giraffe.jpg', 'horses.jpg', 'startrek1.jpg'];
    imageRoot = '../../../../../data/';
    classNamesFile = '../cfg/9k.names';
    //classNamesFile = '../cfg/coco.names';
    scaleDownSteps = 4.0 ;

    colors: array [0..5,0..2] of single = ( (1,0,1), (0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0) );

type
    TDetectionWithClass = record
        det:TDetection;
        // The most probable class id: the best class index in this->prob.
        // Is filled temporary when processing results, otherwise not initialized
        best_class: longint;
    end;

var

  darknet : TDarknetParser;
  i: SizeInt;
  cfg, c : string;
  t : clock_t;
  img , sized: TImageData;
  ImageTensor : TSingleTensor;
  detections : TDetections;
  classNames :  TArray<string>;
  l: TBaseLayer;
  //gDriver, gMode: SmallInt;

function get_color(const c, x, _max: SizeInt):single;
var
    ratio: single;
    i,j: SizeInt;
begin
    ratio := (x / _max) * 5;
    i := floor(ratio);
    j := ceil(ratio);
    ratio := ratio - i;
    result := (1-ratio) * colors[i][c]+ratio * colors[j][c];
end;

function get_actual_detections(const dets: TArray<TDetection>; const dets_num: SizeInt; const thresh: single; const selected_detections_num: PSizeInt; const names: TArray<string>):TArray<TDetectionWithClass>;
var
    selected_num: SizeInt;
    i: SizeInt;
    best_class: SizeInt;
    best_class_prob: single;
    j: SizeInt;
    //show: boolean;
begin
    selected_num := 0;
    setLength(result, dets_num);
    for i := 0 to dets_num -1 do
        begin
            best_class := -1;
            best_class_prob := thresh;
            for j := 0 to dets[i].classes -1 do
                begin
                    //show := names[j] <> 'dont_show';
                    if (dets[i].prob[j] > best_class_prob) {and show} then
                        begin
                            best_class := j;
                            best_class_prob := dets[i].prob[j]
                        end
                end;
            if best_class >= 0 then
                begin
                    result[selected_num].det := dets[i];
                    result[selected_num].best_class := best_class;
                    inc(selected_num)
                end
        end;
    if assigned(selected_detections_num) then
        selected_detections_num[0] := selected_num;
end;

function compare_by_lefts(const a, b: TDetectionWithClass):SizeInt; WINAPI;
var delta: single;
begin
    delta := (a.det.bbox.x-a.det.bbox.w / 2)-(b.det.bbox.x-b.det.bbox.w / 2);
    //exit(ifthen(delta < 0, -1, ifthen(delta > 0, 1, 0)))
end;

function compare_by_probs(const a, b: TDetectionWithClass):SizeInt; WINAPI;
var
    delta: single;
begin
    delta := a.det.prob[a.best_class]-b.det.prob[b.best_class];
    exit(ifthen(delta < 0, -1, ifthen(delta > 0, 1, 0)))
end;

procedure draw_detections_v3(const im: TImageData; const dets: TDetections; const num: SizeInt; const thresh: single; const names: TArray<string>; const alphabet: TArray<TArray<TImageData>>; const classes: SizeInt; labelAlpha: single; const aBatch:sizeInt = 0);
var
    frame_id, selected_detections_num, i, best_class, j, width, offset: SizeInt;
    red, green, blue: single;
    rgb : array[0..2] of single;
    b: TBox;
    left, right, top, bot: SizeInt;
    labelstr, prob_str: string;
    &label, mask, resized_mask, tmask: TImageData;
    selected_detections : TArray<TDetectionWithClass>;
begin
    frame_id := 0;
    inc(frame_id);
    selected_detections := get_actual_detections(dets, num, thresh, @selected_detections_num, names);
    if selected_detections_num=0 then exit();
    //TTools<TDetectionWithClass>.QuickSort(pointer(selected_detections), 0, selected_detections_num-1, compare_by_lefts);
    //for i := 0 to selected_detections_num -1 do
    //    begin
    //        best_class := selected_detections[i].best_class;
    //        write(format('%s: %.0f%%', [names[best_class], selected_detections[i].det.prob[best_class] * 100]));
    //        if ext_output then
    //            writeln(format(#9'(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)', [
    //               (selected_detections[i].det.bbox.x-selected_detections[i].det.bbox.w / 2) * im.w,
    //               (selected_detections[i].det.bbox.y-selected_detections[i].det.bbox.h / 2) * im.h,
    //               selected_detections[i].det.bbox.w * im.w,
    //               selected_detections[i].det.bbox.h * im.h]))
    //        else
    //            writeln('');
    //        for j := 0 to classes -1 do
    //            if (selected_detections[i].det.prob[j] > thresh) and (j <> best_class) then
    //                begin
    //                    write(format('%s: %.0f%%', [names[j], selected_detections[i].det.prob[j] * 100]));
    //                    if ext_output then
    //                        writeln(format(#9'(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)',[
    //                             (selected_detections[i].det.bbox.x-selected_detections[i].det.bbox.w / 2) * im.w,
    //                             (selected_detections[i].det.bbox.y-selected_detections[i].det.bbox.h / 2) * im.h,
    //                             selected_detections[i].det.bbox.w * im.w,
    //                             selected_detections[i].det.bbox.h * im.h]))
    //                    else
    //                        writeln('')
    //                end
    //    end;
    TTools<TDetectionWithClass>.QuickSort(pointer(selected_detections), 0, selected_detections_num-1, compare_by_probs);
    for i := 0 to selected_detections_num -1 do
        begin
            width := trunc(im.h * 0.004);
            if width < 1 then
                width := 1;
            offset := selected_detections[i].best_class * 123457 mod classes;
            red := get_color(2, offset, classes);
            green := get_color(1, offset, classes);
            blue := get_color(0, offset, classes);
            rgb[0] := red;
            rgb[1] := green;
            rgb[2] := blue;
            b := selected_detections[i].det.bbox;
            left  := trunc((b.x-b.w / 2) * im.w);
            right := trunc((b.x+b.w / 2) * im.w);
            top   := trunc((b.y-b.h / 2) * im.h);
            bot   := trunc((b.y+b.h / 2) * im.h);
            if left < 0 then
                left := 0;
            if right > im.w-1 then
                right := im.w-1;
            if top < 0 then
                top := 0;
            if bot > im.h-1 then
                bot := im.h-1;
            if im.c = 1 then
                //draw_box_width_bw(im, left, top, right, bot, 2{width}, 0.8)
                im.rect(left, top, right, bot, 6{width}, 0.8, aBatch)
            else
                //draw_box_width(im, left, top, right, bot, 2{width}, red, green, blue, labelAlpha);
                im.rect(left, top, right, bot, 6{width}, red, green, blue, labelAlpha, aBatch);

            //if assigned(alphabet) then
            //    begin
            //        labelstr:='';
            //        labelstr:= labelstr + names[selected_detections[i].best_class];
            //        prob_str := format(': %.2f', [selected_detections[i].det.prob[selected_detections[i].best_class]]);
            //        labelstr := labelstr + prob_str;
            //        for j := 0 to classes -1 do
            //            if (selected_detections[i].det.prob[j] > thresh) and (j <> selected_detections[i].best_class) then
            //                    labelstr := labelstr + ', ' + names[j];
            //        &label := get_label_v3(alphabet, labelstr, trunc(im.h * 0.02));
            //        draw_weighted_label(im, top + width, left, &label, @rgb[0], labelAlpha);
            //        free_image(&label)
            //    end;
            //if assigned(selected_detections[i].det.mask) then
            //    begin
            //        mask := float_to_image(14, 14, 1, @selected_detections[i].det.mask[0]);
            //        resized_mask := resize_image(mask, trunc(b.w * im.w), trunc(b.h * im.h));
            //        tmask := threshold_image(resized_mask, 0.5);
            //        embed_image(tmask, im, left, top);
            //        free_image(mask);
            //        free_image(resized_mask);
            //        free_image(tmask)
            //    end
        end;
    //free(selected_detections)
end;


//procedure OnForward(var state :TNNetState);
//var
//  img: TSingleTensor;
//  l:TBaseLayer;
//  c:string;
//  i:sizeInt;
//begin
//  //write(#$1B'[1J'#$1B'[1H');
//  l := TNNet(state.net).layers[state.index];
//  writeln(state.index:3, ' : ', 100*state.index/darknet.Neural.layerCount():1:1,'%', ' ', l.LayerTypeStr);
//  //l.output.printStat();
//  //repeat
//  //  writeln('Enter index to Interrogate Output, or [Enter] for next layer:');
//  //  readln(C);
//  //  if TryStrToInt64(c, i) then
//  //      writeln(l.output.Data[i]:1:4);
//  //
//  //
//  //until C='';
//
//end;

const thresh = 0.45;
    NMS =0.45;
    //M = 10;
    //N = 20;
    //K =30;
var
  //ocl  : TOpenCL;
  a, b: TSingleTensor;
  //kernel : cl_kernel;
  off, gws, lws : TArray<SizeInt>;
  //AA, BB, CC: cl_mem;
  NN ,R, j: Integer;

  conv : TConvolutionalLayer;
  state:TNNetState;

procedure inferYOLO;
begin

  //write(#$1B'[1J');
  //TSingleTensor.computingDevice := cdOpenCL;
  //ocl.ActivePlatformId:=1;
  //img.loadFromFile(imageRoot+images[0]);
  //a := img.toTensor();


  //a.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //readln;


  //conv:= TConvolutionalLayer.Create(1, a.h, a.w, a.c, 6, 1, 3);
  //conv.biases.fill(0);
  //state.input := a;
  //conv.ActivationType:=acLINEAR;
  //state.workspace.resize([conv.workspaceSize*10]);
  //
  //conv.forward(state);
  ////writeln('A');a.printStat;
  //DeleteFile(GetCurrentDir+PathDelim+'tmp.jpg');
  //conv.output.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //writeln('outputs :'); conv.output.printStat();
  //readln;
  //
  //b.reSize([1, 9*3, a.h(), a.w()]);
  //a.Conv2D(conv.weights, b);
  //writeLn('outputs 2');b.printStat();
  //DeleteFile(GetCurrentDir+PathDelim+'tmp.jpg');
  //b.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //readln;

  //a.im2Col(conv.weights.w(), conv.weights.w(), 1, 1, 1, 1, 1, 1, b.data);
  //b.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //readln;
  //exit;

  cfg := GetCurrentDir + PathDelim + cfgfile;
  if not FileExists(cfg) then begin
    writeln('File [',cfg,'] doesn''t exist!');
    readln();
  end;


  {$if defined(USE_OPENCL)}
  initOpenCL(0, 0);
  ocl.useBLAS := 0;
  {$elseif defined(USE_CUDART)}
  initCUDART(0);
  cuda.useBLAS := 1;
  {$endif}

  t := clock;
  darknet := TDarknetParser.Create(cfg, 1, 1);
  writeln('Model : ',cfg,' [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');

  t := clock;
  darknet.loadWeights(weightFile);
  writeln('Weights : ', weightFile,' [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
  writeln('press [enter] to start...');
  readln();
  classNames := fromFile(imageRoot+classNamesFile);

  sDigits := 6;

  //darknet.Neural.OnForward := OnForward();
  darknet.Neural.fuseBatchNorm;
  i:=0;
  {$ifdef USE_TELEMETRY}
  benchmark:= true;
  {$endif}
  repeat
    //write(#$1B'[1J');

    img.loadFromFile(imageRoot+images[i]);
    //sized := img.letterBox(darknet.Neural.input.w(), darknet.Neural.input.h());
    sized := img.resize(darknet.Neural.input.w(), darknet.Neural.input.h());
    //sized.toTensor().printStat();
    //readln;

    {$ifdef USE_TELEMETRY}
    metrics.reset;
    {$endif}
    t := clock;
    darknet.Neural.predict(sized.toTensor);
    writeln('Inference : [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
    t := clock;
    //for j:=0 to darknet.Neural.layerCount()-1 do
    //  if darknet.Neural.layers[j].layerType=ltYOLO then
    //      darknet.Neural.layers[j].output.printStat;
    detections := darknet.Neural.Detections(img.w, img.h, thresh, true, false);
    writeln('Detection : [', length(detections),'] took [', (clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
    t := clock;
    detections.doNMSSort(darknet.Neural.classCount(), NMS);
    writeln('Sorting : [', length(detections), '] took [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
    //for j:=0 to high(detections) do begin
    //    writeln(format('id: %d, bestclass : %d, (x, y, w, h) : (%.3f, %.3f, %.3f, %.3f)\n', [detections[j].id, detections[j].best_class_idx
    //    , detections[j].bbox.x, detections[j].bbox.y, detections[j].bbox.w, detections[j].bbox.h]));
    //end;
    //readln;
    //writeln('thresh ', thresh:1:3);
    //readln;
    t := clock;
    draw_detections_v3(img, detections, length(detections), thresh, classNames, nil, length(classNames), 0.5);
    writeln('Drawing : [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');

    {$ifdef USE_TELEMETRY}
    writeln('Metrics :', sLineBreak, metrics.print);
    {$endif}

    //img.toTensor.print(0.3);
    if not DeleteFile(GetCurrentDir+PathDelim+'tmp.jpg') then
        writeln('No result image, Saving...');
    img.saveToFile(GetCurrentDir+PathDelim+'tmp.jpg');
    ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
    inc(i);
    if i>high(images) then i:=0;
    writeln('press [Enter] for next image...');
    readLn(c)
  until lowerCase(c) ='Q';

  //writeln('workspace : [', length(darknet.Neural.workspace)*4/1000000:1:1,'] MB');
  writeln(sLineBreak);
  //ImageTensor.print(0.5);
  //canv := TFPImageCanvas.create(bmp);



(*  //*********************************
  gDriver  := vga;
  gMode    := VGAHi;
  SetDirectVideo(true);
  InitGraph(gDriver, gMode, '');
  //PutImage(1,1,ImageTensor.Data, 0);
  for i:=0 to $ff do
    for j :=0 to $fff do begin
      //SetColor(j div 4);
      //DirectPutPixel(j, i);
      PutPixel(j, i, j div 4);
    end;
  Closegraph;

*) //************************************
  darknet.free;

end;

function darkHash(const str:ansistring):longword;
var i:integer;
begin
  result := 5381;
  for i:=1 to length(str) do
    result := longword((result shl 5) + result) + ord(str[i])
end;

const VOC_NAMES : TArray<string> = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',  'tvmonitor'];


function loadVOCBatch(const imgList:TArray<string>; const resizeWidth, resizeHeight, offset, batchSize, maxBoxes:SizeInt):TData;
const MAX_OBJ = 4000;
    TRUTH_SIZE = 4+2;
var
  img:TImageData;
  i, j, k, left, top, right, bottom: integer;
  xml : TNXml;
  xmls : TArray<TNXml>;
  xmlDir, xmlFile, name:string;
  box : TBox;
  id, trackId : integer;
  dy, dx: SizeInt;
  ratio: single;
  t1: TSingleTensor;
  t2 : TByteTensor;
begin
  result.X.reSize([batchSize, 3, resizeHeight, resizeWidth]);
  result.Y.reSize([batchSize, maxBoxes, TRUTH_SIZE]);
  for i:=offset to min(High(imgList), offset + batchSize-1) do begin
    k:= i- offset;
    xmlDir := ExtractFilePath(imgList[i])+'../Annotations/';
    img.loadFromFile(imgList[i]);
    img := img.letterBoxEx(resizeWidth, resizeHeight, ratio, dx, dy);
    move(img.data[0], result.X.data[k*length(img.data)], length(img.data)*sizeof(Single));
    xmlFile := ChangeFileExt(ExtractFileName(imgList[i]), '.xml');
    trackId := (darkHash(xmlFile) mod MAX_OBJ)*MAX_OBJ;
    xmlFile := xmlDir + xmlFile;
    xml := TNXml.LoadFromFile(xmlFile);
    xmls := xml['annotation'].querySelectorAll('object');
    for j:=0 to min(high(xmls), maxBoxes) do begin
      if (not boolean(xmls[j]['difficult'])) {and (not boolean(xmls[j]['truncated']))} then
      begin
        name := xmls[j]['name'];
        id := TTools<string>.IndexOf(pointer(VOC_NAMES), length(VOC_NAMES), name);
        assert(id>=0, 'VOC Parse : cannot find class "'+name+'"');
        left   := dx + trunc(single(xmls[j]['bndbox']['xmin'])*ratio);
        top    := dy + trunc(single(xmls[j]['bndbox']['ymin'])*ratio);
        right  := dx + trunc(single(xmls[j]['bndbox']['xmax'])*ratio);
        bottom := dy + trunc(single(xmls[j]['bndbox']['ymax'])*ratio);

        box.x  := (left + right) / (2*img.w);
        box.y  := (top + bottom) / (2*img.h);
        box.w  := (right - left) / img.w;
        box.h  := (bottom - top) / img.h;
        result.Y.dyndata[(k*maxBoxes + j)*TRUTH_SIZE + 0] := box.x;
        result.Y.dyndata[(k*maxBoxes + j)*TRUTH_SIZE + 1] := box.y;
        result.Y.dyndata[(k*maxBoxes + j)*TRUTH_SIZE + 2] := box.w;
        result.Y.dyndata[(k*maxBoxes + j)*TRUTH_SIZE + 3] := box.h;
        result.Y.dyndata[(k*maxBoxes + j)*TRUTH_SIZE + 4] := id;
        //result.Y.dyndata[(k*maxBoxes + j)*TRUTH_SIZE + 5] := trackId+j;
        //img.rect(left, top, right, bottom, 4, 0.0, 1.0, 0.0, 0.4);
      end;
    end;
//
//    t1 := img.toTensor();
//    t1.reshape([t1.c, t1.h, t1.w]);
//    t1.Multiply($FF);
//
//    t2.resize(t1.Shape);
//    t1.toBytes(t2.Data);
//    writeln(i);
//    printSixel(t2.Data, img.w, img.h, true, poCHW);
//    readln

 end;
end;

var history : TSingleTensor;

procedure OnAfterOptimize(const net:TNNet; const batchId: SizeInt);
var i:SizeInt;
  detections : TDetections;
  img : TImageData;
begin
  if batchId <2 then exit;
  cursorClearUp();
  cursorHome();
  history.reSize([history.size+1]);
  history.DynData[high(history.dynData)]:=net.cost()/(net.batch*net.subDivisions);
  writeln('batchId : ', batchId, ', Seen : ', net.seen);
  history.plot();
  img.fromTensor(net.input);
  if batchId mod 10 = 0 then begin
    for i:=0 to net.batch-1 do begin
      detections := darknet.Neural.Detections(img.w, img.h, 0.1, true, false, i);
      if assigned(detections) then begin
        detections.doNMSSort(darknet.Neural.classCount(), NMS);
        draw_detections_v3(img, detections, length(detections), thresh, classNames, nil, length(classNames), 0.5, i);
        img.toTensor.print(0.4, false, 2);
      end;
    end;
  end;

end;

procedure OnForward(var state:TNNetState);
begin
  write('Forward [',state.index:3,'][',TNNet(state.net).layers[state.index].LayerTypeStr:20,'] : ', 100*state.index/TNNet(state.net).layerCount():1:2, '%', #13)
end;

procedure OnBackward(var state:TNNetState);
begin
  write('Backward [',state.index:3,'][',TNNet(state.net).layers[state.index].LayerTypeStr:20,']: ', 100*state.index/TNNet(state.net).layerCount():1:2, '%', #13)
end;

procedure OnAfterPropagation(var state:TNNetState);
begin
  write(setClearLineEnd, 100*TNNet(state.net).currentSubDivision/TNNet(state.net).subDivisions:1:2 , '%'#13);
end;

procedure trainYOLO;
const
    BATCH_SIZE=64;
var path, imgDir, xmlFile, xmlDir : string;
  sr : TSearchRec;
  imgList : TArray<string>;
  train: TData;
begin

  path := 'C:/development/Projects/VOCdevkit/VOC2012';
  imgDir := path+'/JPEGImages/';
  if FindFirst(imgDir+'*.jp*', faNormal, sr)=0 then begin
    insert(imgDir + sr.Name, imgList, length(imgList));
    try
      while FindNext(sr)=0 do
        insert(imgDir + sr.Name, imgList, length(imgList));
    finally
      FindClose(sr)
    end;
  end;

  cfg := GetCurrentDir + PathDelim + cfgfileVOC;
  if not FileExists(cfg) then begin
    writeln('File [',cfg,'] doesn''t exist!');
    readln();
  end;


  {$if defined(USE_OPENCL)}
  initOpenCL(0, 0);
  ocl.useBLAS :=0;
  {$elseif defined(USE_CUDART)}
  initCUDART(0);
  cuda.useBLAS := 1;
  {$endif}

  t := clock;
  //speedOverSize:=true;
  darknet := TDarknetParser.Create(cfg, 1, 1);
  darknet.neural.subDivisions:=32;
  darknet.Neural.setTraining(True);
  darknet.Neural.setBatch(BATCH_SIZE);
  writeln('Model : ',cfg,' [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.', #13#10, 'Heap :', GetHeapStatus.TotalAllocated div 1000000, 'MB');
  //readln;
  darknet.Neural.OnAfterNetOptimization := OnAfterOptimize;
  //darknet.Neural.OnForward := OnForward;
  //darknet.Neural.OnBackward := OnBackward;
  darknet.Neural.OnAfterPropagation:=OnAfterPropagation;
  for i:=0 to length(imgList) div BATCH_SIZE-1 do begin
    TSingleTensor.noDeviceAllocation := true;
    train := loadVOCBatch(imgList, 416, 416, i*BATCH_SIZE, BATCH_SIZE, darknet.Neural.maxBoxes);
    TSingleTensor.noDeviceAllocation := false;
    if not assigned(darknet.Neural.truth.Data) then
      darknet.Neural.truth.resize([darknet.Neural.batch, train.y.h, train.y.w], darknet.Neural.batch);
    darknet.Neural.trainEpoch(train);
  end;
end;

begin
  //trainYOLO;
  inferYOLO;
end.
