unit nLSTMLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  ntensors, NTypes, nBaseLayer, nConnectedlayer, nActivation
  , termesc
  ;

type

  { TLSTMLayer }

  TLSTMLayer = class(TbaseLayer)
    uf, ui, ug, uo : TConnectedLayer;
    wf, wi, wg, wo : TConnectedLayer;
    //State,
    prevState, Cell, prevCell, temp, temp2, temp3, _f, _i, _g, _o, _c, _h, dc{, dh} : TSingleTensor;
    constructor Create(const aBatch: SizeInt ;const aInputs, aOutputs, aSteps:SizeInt; const aBatchNormalized: boolean=false);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
    destructor destroy; override;

    {$ifdef USE_OPENCL}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}
  end;

implementation

{ TLSTMLayer }

constructor TLSTMLayer.Create(const aBatch: SizeInt; const aInputs, aOutputs, aSteps: SizeInt; const aBatchNormalized: boolean);
begin
  batch     := aBatch div aSteps;

  layerType := ltLSTM;
  Steps     := aSteps;
  inputs    := aInputs;

  outputs           := aOutputs;
  isBatchNormalized := aBatchNormalized;

  uf        := TConnectedLayer.Create(Batch , Steps, inputs, outputs, acLINEAR, isBatchNormalized);
  ui        := TConnectedLayer.Create(Batch , Steps, inputs, outputs, acLINEAR, isBatchNormalized);
  ug        := TConnectedLayer.Create(Batch , Steps, inputs, outputs, acLINEAR, isBatchNormalized);
  uo        := TConnectedLayer.Create(Batch , Steps, inputs, outputs, acLINEAR, isBatchNormalized);
  wf        := TConnectedLayer.Create(Batch , Steps, outputs, outputs, acLINEAR, isBatchNormalized);
  wi        := TConnectedLayer.Create(Batch , Steps, outputs, outputs, acLINEAR, isBatchNormalized);
  wg        := TConnectedLayer.Create(Batch , Steps, outputs, outputs, acLINEAR, isBatchNormalized);
  wo        := TConnectedLayer.Create(Batch , Steps, outputs, outputs, acLINEAR, isBatchNormalized);

  output    := TSingleTensor.Create([batch*steps, outputs], batch*steps);
  Cell      := TSingleTensor.Create([batch*steps, outputs], batch*steps);

  //state     := TSingleTensor.Create([batch, outputs], batch);
  prevState := TSingleTensor.Create([batch, outputs], batch);
  prevCell  := TSingleTensor.Create([batch, outputs], batch);

  _f        := TSingleTensor.Create([batch, outputs], batch);
  _i        := TSingleTensor.Create([batch, outputs], batch);
  _g        := TSingleTensor.Create([batch, outputs], batch);
  _o        := TSingleTensor.Create([batch, outputs], batch);
  _c        := TSingleTensor.Create([batch, outputs], batch);
  _h        := TSingleTensor.Create([batch, outputs], batch);

  temp      := TSingleTensor.Create([batch, outputs], batch);
  temp2     := TSingleTensor.Create([batch, outputs], batch);
  temp3     := TSingleTensor.Create([batch, outputs], batch);
  dc        := TSingleTensor.Create([batch, outputs], batch);
  //dh        := TSingleTensor.Create([batch, outputs], batch);

end;

procedure TLSTMLayer.setBatch(ABatch: SizeInt);
begin

  if ABatch=Batch then exit();
  Batch := ABatch div steps;
  inputShape[0] := ABatch*steps;


  uf.setBatch(batch);
  ui.setBatch(batch);
  ug.setBatch(batch);
  uo.setBatch(batch);
  wf.setBatch(batch);
  wi.setBatch(batch);
  wg.setBatch(batch);
  wo.setBatch(batch);

  output    .reSize([batch*steps, outputs], batch*steps);
  Cell      .reSize([batch*steps, outputs], batch*steps);

  //state     := TSingleTensor.Create([batch, outputs], batch);
  prevState .reSize([batch, outputs], batch);
  prevCell  .reSize([batch, outputs], batch);
  _f        .reSize([batch, outputs], batch);
  _i        .reSize([batch, outputs], batch);
  _g        .reSize([batch, outputs], batch);
  _o        .reSize([batch, outputs], batch);
  _c        .reSize([batch, outputs], batch);
  _h        .reSize([batch, outputs], batch);

  temp      .reSize([batch, outputs], batch);
  temp2     .reSize([batch, outputs], batch);
  temp3     .reSize([batch, outputs], batch);
  dc        .reSize([batch, outputs], batch);
end;

procedure TLSTMLayer.setTrain(ATrain: boolean);
begin
  if ATrain = FTrain then exit;
  FTrain := ATrain;
  wf.setTrain(ATrain);
  wi.setTrain(ATrain);
  wg.setTrain(ATrain);
  wo.setTrain(ATrain);
  uf.setTrain(ATrain);
  ui.setTrain(ATrain);
  ug.setTrain(ATrain);
  uo.setTrain(ATrain);
  if FTrain  then
    delta := TSingleTensor.Create([batch*steps, outputs], batch*steps)
  else
    delta.free
end;

procedure TLSTMLayer.forward(var state: TNNetState);
var s : TNNetState;
  i, outputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  s := Default(TNNetState);
  s.isTraining := state.isTraining;
  s.workspace := state.workspace;
  s.index := state.index;
  s.net := state.net;
  outputStep := batch*outputs;

  wf.reGroup(batch);
  wi.reGroup(batch);
  wg.reGroup(batch);
  wo.reGroup(batch);
  uf.reGroup(batch);
  ui.reGroup(batch);
  ug.reGroup(batch);
  uo.reGroup(batch);

  if state.isTraining then begin
    wf.delta.fill(0);
    wi.delta.fill(0);
    wg.delta.fill(0);
    wo.delta.fill(0);

    uf.delta.fill(0);
    ui.delta.fill(0);
    ug.delta.fill(0);
    uo.delta.fill(0);

    delta.fill(0);

  end;

  for i := 0 to steps -1 do
      begin
          offset := i* outputStep;

          s.step      := i;
          s.inputStep := 0;
          s.input     := @_h;
          wf.forward(s);
          wi.forward(s);
          wg.forward(s);
          wo.forward(s);

          s.inputStep := i;
          s.input := state.input;
          uf.forward(s);
          ui.forward(s);
          ug.forward(s);
          uo.forward(s);

          TSingleTensor.addvv(outputStep, wf.output.data+offset, 1, uf.output.data+offset, 1, _f.Data, 1);
          TSingleTensor.addvv(outputStep, wi.output.data+offset, 1, ui.output.data+offset, 1, _i.Data, 1);
          TSingleTensor.addvv(outputStep, wg.output.data+offset, 1, ug.output.data+offset, 1, _g.Data, 1);
          TSingleTensor.addvv(outputStep, wo.output.data+offset, 1, uo.output.data+offset, 1, _o.Data, 1);

          activate_array(_f.data, outputStep, acLOGISTIC);
          activate_array(_i.data, outputStep, acLOGISTIC);
          activate_array(_g.data, outputStep, acTANH);
          activate_array(_o.data, outputStep, acLOGISTIC);

          TSingleTensor.mulvv(outputStep, _i.data, 1, _g.data, 1, temp.Data, 1);

          //temp.printStat;
          //readln;

          _c.Multiply(_f);
          _c.add(temp);

          _c.copyTo(_h);
          activate_array(_h, outputStep, acTANH);
          _h.Multiply(_o);

          _c.CopyTo(Cell, offset);
          _h.copyTo(output, offset);
          if FTrain then
            write(#13'LSTM FW : layer [', state.index,'], ', 100*i/steps:3:1);

      end;
  wf.reGroup(steps*batch);
  wi.reGroup(steps*batch);
  wg.reGroup(steps*batch);
  wo.reGroup(steps*batch);
  uf.reGroup(steps*batch);
  ui.reGroup(steps*batch);
  ug.reGroup(steps*batch);
  uo.reGroup(steps*batch);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TLSTMLayer.backward(var state: TNNetState);
var s : TNNetState;
  i, outputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}

  s := Default(TNNetState);
  s.isTraining := state.isTraining;
  s.workspace := state.workspace;
  s.index := state.index;
  s.net := state.net;
  outputStep := batch*outputs;

  //state.input := state.input + (l.inputs * l.batch * (l.steps-1));
  //if assigned(state.delta) then
      //state.delta := state.delta + (l.inputs * l.batch * (l.steps-1));

  //l.output := l.output + (l.outputs * l.batch * (l.steps-1));
  //l.cell_cpu := l.cell_cpu + (l.outputs * l.batch * (l.steps-1));
  //l.delta := l.delta + (l.outputs * l.batch * (l.steps-1));
  wf.reGroup(batch);
  wi.reGroup(batch);
  wg.reGroup(batch);
  wo.reGroup(batch);
  uf.reGroup(batch);
  ui.reGroup(batch);
  ug.reGroup(batch);
  uo.reGroup(batch);


  for i := steps-1 downto 0 do begin
      offset      := i*outputStep;
      s.step      :=i;
      s.deltaStep :=i;
      if i <> 0 then
          cell.CopyTo(prevCell, 0, 1, offset-outputStep, 1, outputStep);
      cell.copyTo(_c, 0, 1, offset, 1, outputStep);
      if i <> 0 then
          output.copyTo(prevState, 0, 1, offset-outputStep, 1, outputStep);

      output.copyTo(_h, 0, 1, offset, 1, outputStep);

      TSingleTensor.addvv(outputStep, wf.output.data+offset, 1, uf.output.data+offset, 1, _f.data, 1);
      TSingleTensor.addvv(outputStep, wi.output.data+offset, 1, ui.output.data+offset, 1, _i.data, 1);
      TSingleTensor.addvv(outputStep, wg.output.data+offset, 1, ug.output.data+offset, 1, _g.data, 1);
      TSingleTensor.addvv(outputStep, wo.output.data+offset, 1, uo.output.data+offset, 1, _o.data, 1);

      activate_array(_f.data, outputStep, acLOGISTIC);
      activate_array(_i.data, outputStep, acLOGISTIC);
      activate_array(_g.data, outputStep, acTANH);
      activate_array(_o.data, outputStep, acLOGISTIC);

      delta.copyTo(temp3, 0, 1, offset, 1, outputStep);

      _c.copyTo(temp);
      activate_array(temp.data, outputStep, acTANH);

      TSingleTensor.mulvv(outputStep, temp3.data, 1, _o.data, 1, temp2.data, 1);

      gradient_array(temp.data, outputStep, acTANH, temp2.data);
      temp2.add(dc);

      _c.CopyTo(temp);
      activate_array(temp.data, outputStep, acTANH);
      temp.Multiply(temp3);
      gradient_array(_o.data, outputStep, acLOGISTIC, temp.data);
      temp.copyTo(wo.delta, offset);

      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.input := @prevState;
      //s.delta := @dh;
      s.inputStep:=0;
      wo.backward(s);
      //backward_connected_layer(wo, @s);

      temp.CopyTo(uo.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uo.delta, 1);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep:=i;
      s.deltaStep:=i;
      uo.backward(s);
      //backward_connected_layer(uo, @s);

      TSingleTensor.mulvv(outputStep, temp2.data, 1, _i.data, 1, temp.data, 1);
      gradient_array(_g, outputStep, acTANH, temp.Data);
      temp.copyTo(wg.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wg.delta, 1);

      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.input := @prevState;
      //s.delta := dh_cpu;
      s.inputstep :=0;
      wg.backward(s);
      //backward_connected_layer(wg, @s);

      temp.copyTo(ug.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ug.delta, 1);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep :=i;
      s.deltaStep :=i;
      ug.backward(s);
      //backward_connected_layer(ug, @s);

      TSingleTensor.mulvv(outputStep, temp2.data, 1, _g.Data, 1, temp.data, 1);
      //copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
      //mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
      gradient_array(_i.data, outputStep, acLOGISTIC, temp.Data);
      temp.copyTo(wi.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wi.delta, 1);

      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.input := @prevState;
      //s.delta := l.dh_cpu;
      s.inputStep := 0;
      wi.backward(s);
      //backward_connected_layer(wi, @s);

      temp.copyTo(ui.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ui.delta, 1);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep:=i;
      s.deltaStep:=i;
      ui.backward(s);
      //backward_connected_layer(ui, @s);

      TSingleTensor.mulvv(outputstep, temp2.data, 1, prevCell.data, 1, temp.data, 1);
      //copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
      //mul_cpu(l.outputs * l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
      gradient_array(_f.data, outputStep, acLOGISTIC, temp.Data);
      temp.CopyTo(wf.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wf.delta, 1);
      s.input := @prevState;
      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.inputStep := 0;
      wf.backward(s);

      temp.CopyTo(uf.delta, offset);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uf.delta, 1);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep:=i;
      s.deltaStep:=i;
      uf.backward(s);
      //backward_connected_layer(uf, @s);

      TSingleTensor.mulvv(outputStep, temp2.data, 1, _f.data, 1, temp.data, 1);
      //copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
      //mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.temp_cpu, 1);
      temp.CopyTo(dc);
      //copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, l.dc_cpu, 1);
      write(#13'LSTM BW : layer [', state.index,'], ', 100*i/steps:3:1);

      //state.input := state.input - (l.inputs * l.batch);
      //if assigned(state.delta) then
      //    state.delta := state.delta - (l.inputs * l.batch);
      //l.output := l.output - (l.outputs * l.batch);
      //l.cell_cpu := l.cell_cpu - (l.outputs * l.batch);
      //l.delta := l.delta - (l.outputs * l.batch);
      //
      //increment_layer( wf, -1);
      //increment_layer( wi, -1);
      //increment_layer( wg, -1);
      //increment_layer( wo, -1);
      //
      //increment_layer( uf, -1);
      //increment_layer( ui, -1);
      //increment_layer( ug, -1);
      //increment_layer( uo, -1);
  end;
  wf.reGroup(steps*batch);
  wi.reGroup(steps*batch);
  wg.reGroup(steps*batch);
  wo.reGroup(steps*batch);
  uf.reGroup(steps*batch);
  ui.reGroup(steps*batch);
  ug.reGroup(steps*batch);
  uo.reGroup(steps*batch);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TLSTMLayer.update(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}
  wf.update(args);
  wi.update(args);
  wg.update(args);
  wo.update(args);
  uf.update(args);
  ui.update(args);
  ug.update(args);
  uo.update(args);
  inherited update(args);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

destructor TLSTMLayer.destroy;
begin
  uf.free;
  ui.free;
  ug.free;
  uo.free;
  wf.free;
  wi.free;
  wg.free;
  wo.free;
  inherited destroy;
end;

{$ifdef USE_OPENCL}
procedure TLSTMLayer.forwardGPU(var state: TNNetState);
var
  outputStep, i, offset: SizeInt;
  s: TNNetState;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  s := Default(TNNetState);
  s.isTraining := state.isTraining;
  s.workspace := state.workspace;
  s.index := state.index;
  s.net := state.net;
  outputStep := batch*outputs;

  wf.reGroup(batch);
  wi.reGroup(batch);
  wg.reGroup(batch);
  wo.reGroup(batch);
  uf.reGroup(batch);
  ui.reGroup(batch);
  ug.reGroup(batch);
  uo.reGroup(batch);

  if state.isTraining then begin
    ocl.fill(wf.Delta.size(), wf.delta.devData, 0, 0, 1);
    ocl.fill(wi.Delta.size(), wi.delta.devData, 0, 0, 1);
    ocl.fill(wg.Delta.size(), wg.delta.devData, 0, 0, 1);
    ocl.fill(wo.Delta.size(), wo.delta.devData, 0, 0, 1);
    ocl.fill(uf.Delta.size(), uf.delta.devData, 0, 0, 1);
    ocl.fill(ui.Delta.size(), ui.delta.devData, 0, 0, 1);
    ocl.fill(ug.Delta.size(), ug.delta.devData, 0, 0, 1);
    ocl.fill(uo.Delta.size(), uo.delta.devData, 0, 0, 1);
    ocl.fill(Delta.size()   , delta.devData   , 0, 0, 1);
  end;

  for i := 0 to steps -1 do
      begin
          offset := i* outputStep;

          s.step      := i;
          s.inputStep := 0;
          s.input     := @_h;
          wf.forwardGPU(s);
          wi.forwardGPU(s);
          wg.forwardGPU(s);
          wo.forwardGPU(s);

          s.inputStep := i;
          s.input := state.input;
          uf.forwardGPU(s);
          ui.forwardGPU(s);
          ug.forwardGPU(s);
          uo.forwardGPU(s);


          ocl.addvv(outputStep, wf.output.devdata, offset, 1, uf.output.devData, offset, 1, _f.devData, 0, 1);
          ocl.addvv(outputStep, wi.output.devdata, offset, 1, ui.output.devData, offset, 1, _i.devData, 0, 1);
          ocl.addvv(outputStep, wg.output.devdata, offset, 1, ug.output.devData, offset, 1, _g.devData, 0, 1);
          ocl.addvv(outputStep, wo.output.devdata, offset, 1, uo.output.devData, offset, 1, _o.devData, 0, 1);

          {$ifdef USE_TELEMETRY}
          if benchmark then metrics.act.start(acLOGISTIC);
          {$endif}
          ocl.activateArray(outputStep, _f.devData, 0, longint(acLOGISTIC));
          ocl.activateArray(outputStep, _i.devData, 0, longint(acLOGISTIC));
          ocl.activateArray(outputStep, _o.devData, 0, longint(acLOGISTIC));
          {$ifdef USE_TELEMETRY}
          ocl.finish();
          if benchmark then metrics.act.finish(acLOGISTIC);
          if benchmark then metrics.act.start(acTANH);
          {$endif}
          ocl.activateArray(outputStep, _g.devData, 0, longint(acTANH));
          {$ifdef USE_TELEMETRY}
          ocl.finish();
          if benchmark then metrics.act.finish(acTANH);
          {$endif}

          ocl.mulvv(outputStep, _i.devData, 0, 1, _g.devData, 0, 1, temp.devData, 0, 1);
          ocl.fmavv(outputstep, _c.devData, 0, 1, _f.devData, 0, 1, temp.devData, 0, 1, _c.devData, 0, 1);
          //_c.Multiply(_f);
          //_c.add(temp);
          // c := c*f + temp

          ocl.copy(outputStep, _c.devData, 0,1, _h.devData, 0, 1);
          {$ifdef USE_TELEMETRY}
          if benchmark then metrics.act.start(acTANH);
          {$endif}
          ocl.ActivateArray(outputStep, _h.devData, 0, longint(acTANH));
          {$ifdef USE_TELEMETRY}
          ocl.finish();
          if benchmark then metrics.act.finish(acTANH);
          {$endif}
          ocl.mulvv(outputStep, _o.devData, 0, 1, _h.devData, 0, 1, _h.devData, 0, 1);

          ocl.copy(outputStep, _c.devData, 0, 1, cell.devData, offset, 1);
          ocl.copy(outputStep, _h.devData, 0, 1, output.devData, offset, 1);

          if FTrain then begin
            cursorAbsPos(40, 30);
            write('LSTM FW : layer [', state.index,'], ', 100*i/steps:3:1);
          end;

      end;

  wf.reGroup(steps*batch);
  wi.reGroup(steps*batch);
  wg.reGroup(steps*batch);
  wo.reGroup(steps*batch);
  uf.reGroup(steps*batch);
  ui.reGroup(steps*batch);
  ug.reGroup(steps*batch);
  uo.reGroup(steps*batch);
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.forward.finish(layerType);
  {$endif}

end;

procedure TLSTMLayer.backwardGPU(var state: TNNetState);
var s : TNNetState;
  i, outputStep, offset:SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}

  s := Default(TNNetState);
  s.isTraining := state.isTraining;
  s.workspace := state.workspace;
  s.index := state.index;
  s.net := state.net;
  outputStep := batch*outputs;

  //state.input := state.input + (l.inputs * l.batch * (l.steps-1));
  //if assigned(state.delta) then
      //state.delta := state.delta + (l.inputs * l.batch * (l.steps-1));

  //l.output := l.output + (l.outputs * l.batch * (l.steps-1));
  //l.cell_cpu := l.cell_cpu + (l.outputs * l.batch * (l.steps-1));
  //l.delta := l.delta + (l.outputs * l.batch * (l.steps-1));
  wf.reGroup(batch);
  wi.reGroup(batch);
  wg.reGroup(batch);
  wo.reGroup(batch);
  uf.reGroup(batch);
  ui.reGroup(batch);
  ug.reGroup(batch);
  uo.reGroup(batch);


  for i := steps-1 downto 0 do begin
      offset      := i*outputStep;
      s.step      :=i;
      s.deltaStep :=i;
      if i <> 0 then
          ocl.copy(outputStep, cell.devData, offset-outputStep, 1, prevCell.devData, 0, 1);
          //cell.CopyTo(prevCell, 0, 1, offset-outputStep, 1, outputStep);
      ocl.copy(outputStep, cell.devData, offset, 1, _c.devData, 0, 1);
      //cell.copyTo(_c, 0, 1, offset, 1, outputStep);
      if i <> 0 then
          ocl.copy(outputStep, output.devData, offset-outputStep, 1, prevState.devData, 0, 1);
          //output.copyTo(prevState, 0, 1, offset-outputStep, 1, outputStep);

      ocl.copy(outputStep, output.devData, offset, 1, _h.devData, 0, 1);
      //output.copyTo(_h, 0, 1, offset, 1, outputStep);

      ocl.addvv(outputStep, wf.output.devdata, offset, 1, uf.output.devData, offset, 1, _f.devData, 0, 1);
      ocl.addvv(outputStep, wi.output.devdata, offset, 1, ui.output.devData, offset, 1, _i.devData, 0, 1);
      ocl.addvv(outputStep, wg.output.devdata, offset, 1, ug.output.devData, offset, 1, _g.devData, 0, 1);
      ocl.addvv(outputStep, wo.output.devdata, offset, 1, uo.output.devData, offset, 1, _o.devData, 0, 1);
      //TSingleTensor.addvv(outputStep, wf.output.data+offset, 1, uf.output.data+offset, 1, _f.data, 1);
      //TSingleTensor.addvv(outputStep, wi.output.data+offset, 1, ui.output.data+offset, 1, _i.data, 1);
      //TSingleTensor.addvv(outputStep, wg.output.data+offset, 1, ug.output.data+offset, 1, _g.data, 1);
      //TSingleTensor.addvv(outputStep, wo.output.data+offset, 1, uo.output.data+offset, 1, _o.data, 1);

      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.act.start(acLOGISTIC);
      {$endif}
      ocl.activateArray(outputStep, _f.devData, 0, longInt(acLOGISTIC));
      ocl.activateArray(outputStep, _i.devData, 0, longInt(acLOGISTIC));
      ocl.activateArray(outputStep, _o.devData, 0, longInt(acLOGISTIC));
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.act.finish(acLOGISTIC);
      if benchmark then metrics.act.start(acTANH);
      {$endif}
      ocl.activateArray(outputStep, _g.devData, 0, longInt(acTANH));
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.act.finish(acTANH);
      {$endif}

      ocl.copy(outputStep, delta.devData, offset, 1, temp3.devData, 0, 1);
      //delta.copyTo(temp3, 0, 1, offset, 1, outputStep);

      ocl.copy(outputStep, _c.devData, 0, 1, temp.devData, 0, 1);
      //_c.copyTo(temp);
      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.act.start(acTANH);
      {$endif}
      ocl.activateArray(outputStep, temp.devData, 0, longInt(acTANH));
      //activate_array(temp.data, outputStep, acTANH);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.act.finish(acTANH);
      {$endif}

      ocl.mulvv(outputStep, temp3.devData, 0, 1, _o.devData, 0, 1, temp2.devData, 0, 1);
      //TSingleTensor.mulvv(outputStep, temp3.data, 1, _o.data, 1, temp2.data, 1);

      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.grad.start(acTANH);
      {$endif}
      ocl.DeriveArray(outputStep, temp.devData, 0, longInt(acTANH), temp2.devData);
      //gradient_array(temp.data, outputStep, acTANH, temp2.data);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.grad.finish(acTANH);
      {$endif}

      ocl.addvv(outputStep, dc.devData, 0, 1, temp2.devData, 0, 1, temp2.devData, 0, 1);
      //temp2.add(dc);

      ocl.copy(outputStep, _c.devData, 0, 1, temp.devData, 0, 1);
      //_c.CopyTo(temp);

      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.act.start(acTANH);
      {$endif}
      ocl.activateArray(outputStep, temp.devData, 0, longint(acTANH));
      //activate_array(temp.data, outputStep, acTANH);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.act.finish(acTANH);
      {$endif}

      ocl.mulvv(outputStep, temp3.devData, 0, 1, temp.devData, 0, 1, temp.devData, 0, 1);
      //temp.Multiply(temp3);
      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.grad.start(acLOGISTIC);
      {$endif}
      ocl.DeriveArray(outputStep, _o.devData, 0, longint(acLOGISTIC), temp.devData);
      //gradient_array(_o.data, outputStep, acLOGISTIC, temp.data);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.grad.finish(acLOGISTIC);
      {$endif}
      ocl.copy(outputStep, temp.devData, 0, 1, wo.delta.devData, offset, 1);
      //temp.copyTo(wo.delta, offset);
      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.input := @prevState;
      //s.delta := @dh;
      s.inputStep:=0;
      wo.backwardGPU(s);

      ocl.copy(outputStep, temp.devData, 0, 1, uo.delta.devData, offset, 1);
      //temp.CopyTo(uo.delta, offset);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep:=i;
      s.deltaStep:=i;
      uo.backwardGPU(s);

      ocl.mulvv(outputStep, temp2.devData, 0, 1, _i.devData, 0, 1, temp.devData, 0, 1);
      //TSingleTensor.mulvv(outputStep, temp2.data, 1, _i.data, 1, temp.data, 1);
      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.grad.start(acTANH);
      {$endif}
      ocl.DeriveArray(outputStep, _g.devData, 0, longInt(acTANH), temp.devData);
      //gradient_array(_g, outputStep, acTANH, temp.Data);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.grad.finish(acTANH);
      {$endif}
      ocl.copy(outputStep, temp.devData, 0, 1, wg.delta.devData, offset, 1);
      //temp.copyTo(wg.delta, offset);
      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.input := @prevState;
      //s.delta := dh_cpu;
      s.inputstep :=0;
      wg.backwardGPU(s);

      ocl.copy(outputStep, temp.devData, 0, 1, ug.delta.devData, offset, 1);
      //temp.copyTo(ug.delta, offset);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep :=i;
      s.deltaStep :=i;
      ug.backwardGPU(s);

      ocl.mulvv(outputStep, temp2.devData, 0, 1, _g.devData, 0, 1, temp.devData, 0, 1);
      //TSingleTensor.mulvv(outputStep, temp2.data, 1, _g.Data, 1, temp.data, 1);
      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.grad.start(acLOGISTIC);
      {$endif}
      ocl.DeriveArray(outputStep, _i.devData, 0, longInt(acLOGISTIC), temp.devData);
      //gradient_array(_i.data, outputStep, acLOGISTIC, temp.Data);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.grad.finish(acLOGISTIC);
      {$endif}
      ocl.copy(outputStep, temp.devData, 0, 1, wi.delta.devData, offset, 1);
      //temp.copyTo(wi.delta, offset);
      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.input := @prevState;
      //s.delta := l.dh_cpu;
      s.inputStep := 0;
      wi.backwardGPU(s);

      ocl.copy(outputStep, temp.devData, 0, 1, ui.delta.devData, offset, 1);
      //temp.copyTo(ui.delta, offset);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep:=i;
      s.deltaStep:=i;
      ui.backwardGPU(s);

      ocl.mulvv(outputstep, temp2.devData, 0, 1, prevCell.devData, 0, 1, temp.devData, 0, 1);
      //TSingleTensor.mulvv(outputstep, temp2.data, 1, prevCell.data, 1, temp.data, 1);
      {$ifdef USE_TELEMETRY}
      if benchmark then metrics.grad.start(acLOGISTIC);
      {$endif}
      ocl.DeriveArray(outputStep, _f.devData, 0, longInt(acLOGISTIC), temp.devData);
      //gradient_array(_f.data, outputStep, acLOGISTIC, temp.Data);
      {$ifdef USE_TELEMETRY}
      ocl.finish();
      if benchmark then metrics.grad.finish(acLOGISTIC);
      {$endif}
      ocl.copy(outputStep, temp.devData, 0, 1, wf.delta.devData, offset, 1);
      //temp.CopyTo(wf.delta, offset);
      s.input := @prevState;
      if (i = 0) then
        s.delta :=  nil
      else begin
        s.delta     := @delta;
        s.deltaStep := i-1;
        //dh := l.delta-l.outputs * l.batch;
      end;
      s.inputStep := 0;
      wf.backwardGPU(s);

      ocl.copy(outputStep, temp.devData, 0, 1, uf.delta.devData, offset, 1);
      //temp.CopyTo(uf.delta, offset);
      s.input := state.input;
      s.delta := state.delta;
      s.inputStep:=i;
      s.deltaStep:=i;
      uf.backwardGPU(s);

      ocl.mulvv(outputStep, temp2.devData, 0, 1, _f.devData, 0, 1, temp.devData, 0, 1);
      //TSingleTensor.mulvv(outputStep, temp2.data, 1, _f.data, 1, temp.data, 1);

      ocl.copy(outputStep, temp.devData, 0, 1, dc.devData, 0, 1);
      //temp.CopyTo(dc);
      cursorAbsPos(40, 30);
      write('LSTM BW : layer [', state.index,'], ', 100*i/steps:3:1);

  end;
  wf.reGroup(steps*batch);
  wi.reGroup(steps*batch);
  wg.reGroup(steps*batch);
  wo.reGroup(steps*batch);
  uf.reGroup(steps*batch);
  ui.reGroup(steps*batch);
  ug.reGroup(steps*batch);
  uo.reGroup(steps*batch);

  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TLSTMLayer.updateGPU(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}
  wf.updateGPU(args);
  wi.updateGPU(args);
  wg.updateGPU(args);
  wo.updateGPU(args);
  uf.updateGPU(args);
  ui.updateGPU(args);
  ug.updateGPU(args);
  uo.updateGPU(args);
  inherited updateGPU(args);
  {$ifdef USE_TELEMETRY}
  ocl.finish();
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;
{$endif}

end.

