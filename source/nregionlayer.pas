unit nRegionLayer;
{$ifdef FPC}
{$mode Delphi}
{$else}
  {$POINTERMATH ON}
  {$T+}
{$endif}

interface
uses nTypes, nBaseLayer, nTensors, nActivation;

type

  { TRegionLayer }

  TRegionLayer = class(TBaseImageLayer)
    N         : SizeInt;
    truthSize : SizeInt;
    truths    : SizeInt;
    maxBoxes  : SizeInt;
    classes   : SizeInt;
    coords    : SizeInt;
    classFix  : SizeInt;
    map       : TArray<SizeInt>;
    classScale, noObjectScale, objectScale, coordScale, thresh: single;
    softmaxTree: TArray<TTree>;
    isSoftmax, focalLoss, biasMatch, reScore : boolean;
    constructor Create(const aBatch, aWidth, aHeight, aN, aClasses, aCoords, aMaxBoxes: SizeInt);
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    function getBox(const n, index, col, row, netWidth, netHeight: SizeInt):TBox;
    function deltaBox(const truth: TBox; const n, index, col, row, netWeight, netHeight: SizeInt; const scale: single):single;
    procedure getBoxes(const aWidth, aHeight: SizeInt; const thresh: single; var probs: TArray<TArray<Single>>; var boxes: TArray<TBox>; const only_objectness: boolean; const map: TArray<SizeInt>);

    procedure getDetections(const width, height, net_w, net_h: SizeInt; const thresh: single; const map: TArray<SizeInt>; const hier: single; const relative: boolean; const dets: PDetection; const letter: boolean);
  end;

implementation

{ TRegionLayer }

constructor TRegionLayer.Create(const aBatch, aWidth, aHeight, aN, aClasses, aCoords, aMaxBoxes: SizeInt);
begin

  layerType := ltREGION;
  n := aN;
  batch := abatch;
  classes := aClasses;
  coords := aCoords;
  h := aHeight;
  w := aWidth;
  c := aN * (classes+coords+1);
  outW := w;
  outH := h;
  outC := c;
  cost := [0];//TSingles.Create(1);
  biases.resize([n * 2]);
  bias_updates.resize([n * 2]);
  biases.Fill(0.5);
  outputs := h * w * n * (classes+coords+1);
  inputs := outputs;
  inputShape := [batch , h , w , n , (classes+coords+1)];
  maxBoxes := aMaxBoxes;
  truthSize := 4+2;
  truths := maxBoxes * truthSize;
  output.resize([batch , h , w , n , (classes+coords+1)]);
  //delta.resize([batch , h , w , n , (classes+coords+1)]);

end;


procedure TRegionLayer.setTrain(ATrain: boolean);
begin
  if aTrain=train then exit;
  train := aTrain;
  if train then
      delta.resize([batch , h , w , n , (classes+coords+1)])
  else
    delta.free
end;

procedure TRegionLayer.setBatch(ABatch: SizeInt);
begin
  if aBatch = batch then exit;
  batch := Abatch;
  inputShape[0] := batch;
  output.resize([batch , h , w , n , (classes+coords+1)]);
  if train then
      delta.resize([batch , h , w , n , (classes+coords+1)])
  else
    delta.free
end;


procedure flatten(const x: PSingle; const size, layers, batch: SizeInt; const forward: boolean);
var
  swap: TArray<single>;//TSingles;
  i, c, b, i1, i2: SizeInt;
begin
  setLength(swap, size*layers*batch);
  //swap:=TSingles.Create(size*layers*batch);
  for b := 0 to  batch-1 do
      for c := 0 to layers-1do
          for i := 0 to size-1 do begin
              i1 := b*layers*size + c*size + i;
              i2 := b*layers*size + i*layers + c;
              if (forward) then
                swap[i2] := x[i1]
              else
                swap[i1] := x[i2];
           end;
  move(swap[0], x[0], size * layers * batch *sizeof(single));
  //swap.free
end;

function logistic_activate(const x:single):single;inline;
begin
  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))
  result := 1/(1 + exp(-x))
end;

function logistic_gradient(const x:single):single;inline;
begin
  result := (1-x)*x;
end;

procedure softmax(const input: PSingle; const n: SizeInt; const temp: single; const output: PSingle; const stride: SizeInt);
var i:SizeInt;
    sum, largest, e : single;
    o:PSingle;
begin
  // todo [Softmax] SIMDIfy & GPU
  if n=0 then exit;
  sum := 0;
  largest := input[0];
  for i := 1 to n-1 do
      if input[i*stride] > largest then
          largest := input[i*stride];
  for i := 0 to n-1 do  begin
      //e := exp(ensureRange((input[i*stride] - largest)/temp, minSingleExp, maxSingleExp));
      e := exp((input[i*stride] - largest)/temp);
      sum := sum + e;
      output[i*stride] := e;
  end;
  for i := 0 to n-1 do  begin
      o:=@output[i*stride];
      o^ := o^ / sum;
  end;

end;

procedure softmax_tree(const input: Psingle; const batch, inputs: SizeInt; const temp: single; const hierarchy: PTree; const output: PSingle);
var
    b, i, count, group_size: SizeInt;
begin
  for b := 0 to batch -1 do
    begin
      count := 0;
      for i := 0 to hierarchy.groups -1 do
        begin
          group_size := hierarchy.group_size[i];
          softmax(input+b * inputs + count, group_size, temp, output+b * inputs+count, 1);
          inc(count , group_size)
        end
    end
end;

function get_hierarchy_probability(const x: PSingle; const hier: TTree; c: SizeInt; const stride: SizeInt = 1): single;
begin
    result := 1;
    while (c >= 0) do
        begin
            result := result * x[c * stride];
            c := hier.parent[c]
        end;
    //exit(result)
end;


function ifthen(const cond:boolean; const ifTrue:single; const ifFalse:single):single;inline;
begin
  if cond then exit(ifTrue);
  result := ifFalse
end;

procedure delta_region_class(const output, delta: Psingle; const index:SizeInt; class_id:SizeInt; const classes: SizeInt; const hier: PTree; const scale: single; const avg_cat: Psingle; const focal_loss: boolean);
var
    i, n, g, offset, ti: SizeInt;
    pred, alpha, pt, grad: single;
begin
    if assigned(hier) then
        begin
            pred := 1;
            while (class_id >= 0) do
                begin
                    pred := pred * output[index+class_id];
                    g := hier.group[class_id];
                    offset := hier.group_offset[g];
                    for i := 0 to hier.group_size[g] -1 do
                        delta[index+offset+i] := scale * (-output[index+offset+i]);
                    delta[index+class_id] := scale * (1-output[index+class_id]);
                    class_id := hier.parent[class_id]
                end;
            avg_cat[0] := avg_cat[0] + pred
        end
    else
        begin
            if focal_loss then
                begin
                    alpha := 0.5;
                    ti := index+class_id;
                    pt := output[ti]+0.000000000000001;
                    grad := -(1-pt) * (2 * pt * ln(pt)+pt-1);
                    for n := 0 to classes -1 do
                        begin
                            delta[index+n] := scale * ((ifthen((n = class_id), 1, 0))-output[index+n]);
                            delta[index+n] := delta[index+n] * (alpha * grad);
                            if n = class_id then
                                avg_cat[0] := avg_cat[0] + output[index+n]
                        end
                end
            else
                for n := 0 to classes -1 do
                    begin
                        delta[index+n] := scale * ((ifthen((n = class_id), 1, 0))-output[index+n]);
                        if n = class_id then
                            avg_cat[0] := avg_cat[0] + output[index+n]
                    end
        end
end;

procedure hierarchy_predictions(const predictions: PSingle; const n: SizeInt;const hier: TTree; const only_leaves: boolean; const stride: SizeInt = 1);
var
    j, parent: SizeInt;
begin
    for j := 0 to n -1 do
        begin
            parent := hier.parent[j];
            if parent >= 0 then
                predictions[j * stride] := predictions[j * stride] * predictions[parent * stride]
        end;
    if only_leaves then
        for j := 0 to n -1 do
            if hier.leaf[j]=0 then
                predictions[j * stride] := 0
end;

function hierarchy_top_prediction(const predictions: PSingle; const hier: TTree; const thresh: single; const stride: SizeInt): SizeInt;
var
    group, i, max_i, index: SizeInt;
    p, _max, val: single;
begin
    p := 1;
    group := 0;
    while true do
        begin
            _max := 0;
            max_i := 0;
            for i := 0 to hier.group_size[group] -1 do
                begin
                    index := i+hier.group_offset[group];
                    val := predictions[(i+hier.group_offset[group]) * stride];
                    if val > _max then
                        begin
                            max_i := index;
                            _max := val
                        end
                end;
            if p * _max > thresh then
                begin
                    p := p * _max;
                    group := hier.child[max_i];
                    if hier.child[max_i] < 0 then
                        exit(max_i)
                end
            else
                if group = 0 then
                    exit(max_i)
            else
                exit(hier.parent[hier.group_offset[group]])
        end;
    result := 0
end;


procedure TRegionLayer.forward(var state: TNNetState);
var
    i, j, b, t, _n, size, index, count, class_count, class_id, onlyclass_id, maxi, best_class_id, best_index, best_n: SizeInt;
    avg_iou, recall, avg_cat, avg_obj, avg_anyobj, maxp, scale, p, best_iou, iou: single;
    truth, pred, truth_shift: TBox;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layertype);
  {$endif}
  size := coords+classes+1;
  state.input.copyTo(output);
  //move(state.input[0], output[0], outputs * batch * sizeof(Single));
{$ifndef GPU}
  flatten(output, w * h, size * n, batch, true);
{$endif}
  for b := 0 to batch -1 do
      for i := 0 to h * w * n -1 do
          begin
              index := size * i+b * outputs;
              output[index+4] := logistic_activate(output[index+4])
          end;
{$ifndef GPU}
  if assigned(softmaxTree) then
      for b := 0 to batch -1 do
          for i := 0 to h * w * n -1 do
              begin
                  index := size * i+b * outputs;
                  softmax_tree(output.data+index+5, 1, 0, 1, @softmaxTree[0], output.data+index+5)
              end
  else
      if isSoftmax then
          for b := 0 to batch -1 do
              for i := 0 to h * w * n -1 do
                  begin
                      index := size * i+b * outputs;
                      softmax(output.data+index+5, classes, 1, output.data+index+5, 1)
                  end;
{$endif}
  if not state.isTraining then
      exit();
  delta.fill(0);//;
  avg_iou := 0;
  recall := 0;
  avg_cat := 0;
  avg_obj := 0;
  avg_anyobj := 0;
  count := 0;
  class_count := 0;
  cost[0] := 0;
  for b := 0 to batch -1 do  begin
          if assigned(softmaxTree) then begin
              onlyclass_id := 0;
              for t := 0 to maxBoxes -1 do begin
                  truth.fromFloat(state.truth.Data + t*truthSize+b * truths);
                  if truth.x=0 then
                      break;
                  class_id := trunc(state.truth.data[t * truthSize+b * truths+4]);
                  maxp := 0;
                  maxi := 0;
                  if (truth.x > 100000) and (truth.y > 100000) then begin
                      for _n := 0 to n * w * h -1 do begin
                          index := size * _n+b * outputs+5;
                          scale := output[index-1];
                          p := scale * get_hierarchy_probability(output.data+index, softmaxTree[0], class_id);
                          if p > maxp then begin
                              maxp := p;
                              maxi := _n
                          end
                      end;
                      index := size * maxi+b * outputs+5;
                      delta_region_class(output.Data, delta.Data, index, class_id, classes, @softmaxTree[0], classScale, @avg_cat, focalLoss);
                      inc(class_count);
                      onlyclass_id := 1;
                      break
                  end
              end;
              if onlyclass_id<>0 then
                  continue
          end;
          for j := 0 to h -1 do
              for i := 0 to w -1 do
                  for _n := 0 to n -1 do
                      begin
                          index := size * (j*w*n + i*n + _n)+b * outputs;
                          pred := getBox(_n, index, i, j, w, h);
                          best_iou := 0;
                          best_class_id := -1;
                          for t := 0 to maxBoxes -1 do
                              begin
                                  truth.fromFloat(state.truth.Data + t*truthSize + b*truths);
                                  class_id := trunc(state.truth[t * truthSize+b * truths+4]);
                                  if class_id >= classes then
                                      continue;
                                  if truth.x=0 then
                                      break;
                                  iou := pred.iou(truth);
                                  if iou > best_iou then
                                      begin
                                          best_class_id := trunc(state.truth[t * truthSize+b * truths+4]);
                                          best_iou := iou
                                      end
                              end;
                          avg_anyobj := avg_anyobj + output[index+4];
                          delta[index+4] := noObjectScale * ((-output[index+4]) * logistic_gradient(output[index+4]));
                          if classfix = -1 then
                              delta[index+4] := noObjectScale * ((best_iou-output[index+4]) * logistic_gradient(output[index+4]))
                          else
                              if best_iou > thresh then
                                  begin
                                      delta[index+4] := 0;
                                      if classfix > 0 then
                                          begin
                                              delta_region_class(output.Data, delta.Data, index+5, best_class_id, classes, @softmaxTree[0], classScale * (ifthen(classfix = 2, output[index+4], 1)),  @avg_cat, focalLoss);
                                              inc(class_count)
                                          end
                                  end;
                          if  state.seen^ < 12800 then
                              begin
                                  truth := default(TBox);
                                  truth.x := (i+0.5) / w;
                                  truth.y := (j+0.5) / h;
                                  //truth.w := biases[2 * _n];
                                  //truth.h := biases[2 * _n+1];
                                  //if DOABS then
                                      begin
                                          truth.w := biases[2 * _n] / w;
                                          truth.h := biases[2 * _n+1] / h
                                      end;
                                  deltaBox(truth, _n, index, i, j, w, h, 0.01)
                              end
                      end;
          for t := 0 to maxBoxes -1 do
              begin
                  truth.fromFloat(state.truth.data + t*truthSize + b*truths);
                  class_id := trunc(state.truth.data[t*truthSize + b*truths + 4]);
                  if class_id >= classes then
                      begin
                          //writeln(format(#10' Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] ',[ class_id, classes, classes-1]));
                          continue
                      end;
                  if truth.x=0 then
                      break;
                  best_iou := 0;
                  best_index := 0;
                  best_n := 0;
                  i := trunc(truth.x * w);
                  j := trunc(truth.y * h);
                  truth_shift := truth;
                  truth_shift.x := 0;
                  truth_shift.y := 0;
                  for _n := 0 to n -1 do
                      begin
                          index := size * (j*w*n + i*n + _n)+b * outputs;
                          pred := getBox(_n, index, i, j, w, h);
                          if biasMatch then
                              begin
                                  //pred.w := biases[2 * _n];
                                  //pred.h := biases[2 * _n+1];
                                  //if DOABS then
                                      begin
                                          pred.w := biases.data[2 * _n] / w;
                                          pred.h := biases.data[2 * _n+1] / h
                                      end
                              end;
                          pred.x := 0;
                          pred.y := 0;
                          iou := pred.iou(truth_shift);
                          if iou > best_iou then
                              begin
                                  best_index := index;
                                  best_iou := iou;
                                  best_n := _n
                              end
                      end;
                  iou := deltaBox(truth, best_n, best_index, i, j, w, h, coordScale);
                  if iou > 0.5 then
                      recall := recall + 1;
                  avg_iou := avg_iou + iou;
                  avg_obj := avg_obj + output[best_index+4];
                  delta[best_index+4] := objectScale * (1-output.data[best_index+4]) * logistic_gradient(output.data[best_index+4]);
                  if reScore then
                      delta[best_index+4] := objectScale * (iou-output.data[best_index+4]) * logistic_gradient(output.data[best_index+4]);
                  if assigned(map) then
                      class_id := map[class_id];
                  delta_region_class(output.Data, delta.Data, best_index+5, class_id, classes, @softmaxTree[0], classScale,  @avg_cat, focalLoss);
                  inc(count);
                  inc(class_count)
              end
      end;
  {$ifndef GPU}
  flatten(delta, w * h, size * n, batch, false);
  {$endif}
  cost[0] := sqr(delta.sumSquares());

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layertype);
  {$endif}
end;

procedure TRegionLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layertype);
  {$endif}
  state.delta.add(delta);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layertype);
  {$endif}
end;

function TRegionLayer.getBox(const n, index, col, row, netWidth, netHeight: SizeInt): TBox;
begin
    result.x := (col+logistic_activate(output.data[index+0])) / netWidth;
    result.y := (row+logistic_activate(output.data[index+1])) / netHeight;
    //result.w := exp(output.data[index+2]) * biases[2 * n];
    //result.h := exp(output.data[index+3]) * biases[2 * n+1];
    //if DOABS then
        begin
            result.w := exp(output.data[index+2]) * biases.data[2 * n] / netWidth;
            result.h := exp(output.data[index+3]) * biases.data[2 * n+1] / netHeight
        end;
end;

function TRegionLayer.deltaBox(const truth: TBox; const n, index, col, row,
  netWeight, netHeight: SizeInt; const scale: single): single;
var
    pred: TBox;
    tx, ty, tw, th: single;
begin
    pred := getBox(n, index, col, row, w, h);
    result := pred.iou(truth);
    tx := (truth.x * w-col);
    ty := (truth.y * h-row);
    //tw := ln(truth.w / biases[2 * n]);
    //th := ln(truth.h / biases[2 * n+1]);
    //if DOABS then
        begin
            tw := ln(truth.w * w / biases[2 * n]);
            th := ln(truth.h * h / biases[2 * n+1])
        end;
    delta[index+0] := scale * (tx-logistic_activate(output.data[index+0])) * logistic_gradient(logistic_activate(output.data[index+0]));
    delta[index+1] := scale * (ty-logistic_activate(output.data[index+1])) * logistic_gradient(logistic_activate(output.data[index+1]));
    delta[index+2] := scale * (tw-output.data[index+2]);
    delta[index+3] := scale * (th-output.data[index+3]);
end;

procedure TRegionLayer.getBoxes(const aWidth, aHeight: SizeInt;
  const thresh: single; var probs: TArray<TArray<Single>>; var boxes: TArray<
  TBox>; const only_objectness: boolean; const map: TArray<SizeInt>);
var
    i, j, _n, row, col, index, p_index, box_index, class_index: SizeInt;
    found :boolean;
    predictions: PSingle;
    scale, prob: single;
begin
    predictions := output.Data;
    //output.printStat;
    // todo [get_region_boxes] Parallelize
    for i := 0 to w * h -1 do
        begin
            row := i div w;
            col := i mod w;
            for _n := 0 to n -1 do
                begin
                    index := i*n + _n;
                    p_index := index * (classes+5)+4;
                    scale := predictions[p_index];
                    if (classfix = -1) and (scale < 0.5) then
                        scale := 0;
                    box_index := index * (classes+5);
                    boxes[index] := getBox(_n, box_index, col, row, w, h);
                    boxes[index].x := boxes[index].x * aWidth;
                    boxes[index].y := boxes[index].y * aHeight;
                    boxes[index].w := boxes[index].w * aWidth;
                    boxes[index].h := boxes[index].h * aHeight;
                    class_index := index * (classes+5)+5;
                    if assigned(softmaxTree) then
                        begin
                            hierarchy_predictions(predictions+class_index, classes, softmaxTree[0], false);
                            found := false;
                            if assigned(map) then
                                for j := 0 to 200 -1 do
                                    begin
                                        prob := scale * predictions[class_index+map[j]];
                                        if (prob > thresh) then
                                            probs[index][j] := prob
                                        else
                                            probs[index][j] := 0
                                    end
                            else
                                j := classes-1;
                                while j >= 0 do begin
                                    if not found and (predictions[class_index+j] > 0.5) then
                                        found := true
                                    else
                                        predictions[class_index+j] := 0;
                                    prob := predictions[class_index+j];
                                    if (scale > thresh) then
                                        probs[index][j] := prob
                                    else
                                        probs[index][j] := 0;
                                    dec(j)
                                end
                        end
                    else
                        for j := 0 to classes -1 do
                            begin
                                prob := scale * predictions[class_index+j];
                                if (prob > thresh) then
                                    probs[index][j] := prob
                                else
                                    probs[index][j] := 0
                            end;
                    if only_objectness then
                        probs[index][0] := scale
                end
        end
end;

procedure correct_yolo_boxes(const dets: PDetection; const n, w, h, netw, neth: SizeInt; const relative, letter: boolean);
var
    i, new_w, new_h: SizeInt;
    deltaw, deltah, ratiow, ratioh: single;
    b: TBox;
begin
    new_w := 0;
    new_h := 0;
    if letter then
        begin
            if (netw / w) < (neth / h) then
                begin
                    new_w := netw;
                    new_h := (h * netw) div w
                end
            else
                begin
                    new_h := neth;
                    new_w := (w * neth) div h
                end
        end
    else
        begin
            new_w := netw;
            new_h := neth
        end;
    deltaw := netw-new_w;
    deltah := neth-new_h;
    ratiow := new_w / netw;
    ratioh := new_h / neth;
    for i := 0 to n -1 do
        begin
            b := dets[i].bbox;
            b.x := (b.x-deltaw / 2.0 / netw) / ratiow;
            b.y := (b.y-deltah / 2.0 / neth) / ratioh;
            b.w := b.w * (1 / ratiow);
            b.h := b.h * (1 / ratioh);
            if not relative then
                begin
                    b.x := b.x * w;
                    b.w := b.w * w;
                    b.y := b.y * h;
                    b.h := b.h * h
                end;
            dets[i].bbox := b
        end
end;

procedure TRegionLayer.getDetections(const width, height, net_w,
  net_h: SizeInt; const thresh: single; const map: TArray<SizeInt>;
  const hier: single; const relative: boolean; const dets: PDetection;
  const letter: boolean);
var
    probs: TArray<TArray<Single>>;
    i, j: SizeInt;
    highest_prob: single;
    boxes :TArray<TBox>;
begin
    setLength(boxes,w * h * n);
    setLength(probs, w * h * n, classes);
    //for j := 0 to l.w * l.h * l.n -1 do
    //    probs[j] := single(xcalloc(l.classes, sizeof(float)));
    getBoxes(1, 1, thresh, probs, boxes, false, map);
    for j := 0 to w * h * n -1 do
        begin
            dets[j].classes := classes;
            dets[j].bbox := boxes[j];
            dets[j].objectness := 1;
            highest_prob := 0;
            dets[j].best_class_idx := -1;
            for i := 0 to classes -1 do
                begin
                    if probs[j][i] > highest_prob then
                        begin
                            highest_prob := probs[j][i];
                            dets[j].best_class_idx := i
                        end;
                    dets[j].prob[i] := probs[j][i]
                end
        end;
    //free(boxes);
    //free_ptrs(PPointer(probs), l.w * l.h * l.n);
    correct_yolo_boxes(dets, w * h * n, w, h, net_w, net_h, relative, letter)
end;

end.

