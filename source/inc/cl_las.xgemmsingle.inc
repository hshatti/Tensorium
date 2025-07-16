with XgemmSingle.init('Xgemm', kSingle, ['GEMMK', 'KREG', 'KWG', 'KWI', 'MDIMA', 'MDIMC', 'MWG', 'NDIMB', 'NDIMC', 'NWG', 'SA', 'SB', 'STRM', 'STRN', 'VWM', 'VWN'])^ do begin
  with addVendor(kDeviceTypeGPU, 'AMD')^ do begin
    with addArch('Ellesmere')^ do begin
       addDevice( 'AMD Radeon RX 480', [ 0, 1, 32, 2, 8, 8, 16, 16, 16, 64, 1, 1, 0, 0, 1, 2 ]);
       addDevice('AMD Radeon RX 580 2048SP', [ 0, 1, 16, 2, 32, 32, 128, 16, 8, 128, 1, 1, 1, 1, 2, 4 ]);
       addDevice('AMD Radeon RX590 GME', [ 0, 1, 32, 2, 16, 16, 64, 32, 16, 128, 1, 1, 0, 1, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 ]);
    end;
    with addArch('Fiji')^ do begin
       addDevice('AMD Radeon 500 Series', [ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 ]);
       addDevice('AMD Radeon R9 Fury X', [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 ]);
       addDevice('AMD Radeon R9 M370X Compute Engine', [ 0, 1, 32, 2, 16, 16, 64, 8, 16, 128, 0, 0, 0, 0, 2, 8 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 2, 4 ]);
    end;
    with addArch('Hawaii')^ do begin
       addDevice('AMD FirePro W8100', [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 ]);
       addDevice('AMD Radeon R9 290X', [ 0, 1, 16, 2, 16, 32, 128, 32, 8, 64, 1, 1, 1, 1, 4, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 16, 2, 16, 16, 64, 16, 8, 64, 1, 1, 0, 0, 4, 2 ]);
    end;
    with addArch('Oland')^ do begin
       addDevice('Oland', [ 0, 1, 16, 2, 32, 16, 64, 32, 16, 128, 1, 1, 1, 0, 2, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 16, 2, 32, 16, 64, 32, 16, 128, 1, 1, 1, 0, 2, 4 ]);
    end;
    with addArch('Pitcairn')^ do begin
       addDevice('AMD Radeon R9 270X', [ 0, 1, 16, 2, 16, 8, 32, 16, 16, 128, 0, 0, 1, 0, 1, 1 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 16, 2, 16, 8, 32, 16, 16, 128, 0, 0, 1, 0, 1, 1 ]);
    end;
    with addArch('Tahiti')^ do begin
       addDevice('AMD Radeon HD 7970', [ 0, 1, 32, 2, 16, 32, 128, 16, 8, 64, 0, 0, 0, 0, 4, 1 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 32, 128, 16, 8, 64, 0, 0, 0, 0, 4, 1 ]);
    end;
    with addArch('Tonga')^ do begin
       addDevice('AMD Radeon R9 380', [ 0, 1, 16, 2, 16, 32, 64, 16, 8, 128, 1, 1, 0, 0, 2, 8 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 16, 2, 16, 32, 64, 16, 8, 128, 1, 1, 0, 0, 2, 8 ]);
    end;
    with addArch('Turks')^ do begin
       addDevice('AMD Radeon HD 6770M', [ 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4 ]);
    end;
    with addArch('Vancouver')^ do begin
       addDevice('ATI Radeon HD 6750M', [ 0, 1, 32, 2, 8, 16, 128, 8, 8, 128, 0, 0, 1, 1, 8, 8 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 8, 16, 128, 8, 8, 128, 0, 0, 1, 1, 8, 8 ]);
    end;
    with addArch('Vega')^ do begin
       addDevice('Radeon RX Vega', [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 2, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 2, 4 ]);
    end;
    with addArch('default')^ do begin
       addDevice('AMD Radeon Pro 450 Compute Engine', [ 0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 ]);
       addDevice('AMD Radeon Pro 580 Compute Engine', [ 0, 1, 32, 2, 8, 8, 32, 8, 8, 64, 0, 0, 0, 0, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 ]);
    end;
    with addArch('gfx1010:xnack-')^ do begin
       addDevice('AMD Radeon RX 5700', [ 0, 1, 32, 2, 16, 16, 128, 16, 8, 64, 1, 1, 1, 0, 4, 1 ]);
       addDevice('AMD Radeon RX 5700 XT', [ 0, 1, 16, 2, 16, 8, 64, 8, 8, 64, 1, 1, 0, 1, 2, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 0, 0, 0, 0, 4, 4 ]);
    end;
    with addArch('gfx1030')^ do begin
       addDevice('AMD Radeon RX 6800 XT', [ 0, 1, 16, 2, 8, 8, 128, 32, 8, 64, 1, 0, 1, 1, 4, 1 ]);
       addDevice('AMD Radeon RX 6900 XT', [ 0, 1, 32, 2, 16, 16, 128, 16, 8, 64, 1, 1, 1, 0, 4, 1 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 ]);
    end;
    with addArch('gfx1031')^ do begin
       addDevice('AMD Radeon RX 6700 XT', [ 0, 1, 32, 2, 8, 8, 128, 32, 16, 128, 1, 1, 1, 1, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 8, 8, 128, 32, 16, 128, 1, 1, 1, 1, 4, 4 ]);
    end;
    with addArch('gfx1032')^ do begin
       addDevice('AMD Radeon RX 6600 XT', [ 0, 1, 32, 2, 8, 8, 128, 32, 16, 128, 1, 1, 1, 1, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 8, 8, 128, 32, 16, 128, 1, 1, 1, 1, 4, 4 ]);
    end;
    with addArch('gfx1034')^ do begin
       addDevice('AMD Radeon RX 6500 XT', [ 0, 1, 32, 2, 16, 16, 128, 16, 8, 64, 1, 1, 1, 0, 4, 1 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 128, 16, 8, 64, 1, 1, 1, 0, 4, 1 ]);
    end;
    with addArch('gfx1035')^ do begin
       addDevice('AMD Radeon Graphics', [ 0, 1, 16, 2, 8, 16, 32, 32, 8, 128, 1, 1, 0, 0, 2, 1 ]);
       addDevice('AMD Radeon(TM) Graphics', [ 0, 1, 32, 2, 16, 8, 128, 8, 8, 64, 0, 1, 1, 0, 4, 1 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 32, 32, 64, 8, 8, 64, 0, 0, 0, 0, 1, 2 ]);
    end;
    with addArch('gfx1100')^ do begin
       addDevice('Radeon RX 7900 XTX', [ 0, 1, 32, 2, 16, 16, 64, 8, 8, 32, 0, 1, 1, 0, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 8, 8, 32, 0, 1, 1, 0, 4, 4 ]);
    end;
    with addArch('gfx1101')^ do begin
       addDevice('AMD Radeon RX 7800 XT', [ 0, 1, 32, 2, 16, 16, 64, 32, 8, 128, 0, 1, 1, 0, 1, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 32, 8, 128, 0, 1, 1, 0, 1, 2 ]);
    end;
    with addArch('gfx1102')^ do begin
       addDevice('AMD Radeon RX 7600', [ 0, 1, 32, 2, 16, 16, 64, 32, 8, 128, 0, 1, 1, 0, 1, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 32, 8, 128, 0, 1, 1, 0, 1, 2 ]);
    end;
    with addArch('gfx1103')^ do begin
       addDevice('AMD Radeon 780M Graphics', [ 0, 1, 32, 2, 16, 8, 128, 8, 8, 64, 1, 0, 1, 0, 1, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 8, 128, 8, 8, 64, 1, 0, 1, 0, 1, 2 ]);
    end;
    with addArch('gfx1201')^ do begin
       addDevice('AMD Radeon RX 9070 XT', [ 0, 1, 32, 2, 16, 8, 64, 16, 8, 128, 1, 1, 0, 1, 4, 8 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 8, 64, 16, 8, 128, 1, 1, 0, 1, 4, 8 ]);
    end;
    with addArch('gfx902')^ do begin
       addDevice('AMD Radeon(TM) Graphics', [ 0, 1, 16, 2, 32, 32, 128, 16, 8, 128, 1, 1, 1, 1, 2, 4 ]);
       addDevice('AMD Radeon(TM) RX Vega 10 Graphics', [ 0, 1, 32, 2, 8, 8, 128, 16, 32, 128, 1, 1, 1, 1, 4, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4 ]);
    end;
    with addArch( 'gfx906:sramecc+:xnack-')^ do begin
       addDevice('AMD Instinct MI50/MI60', [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 ]);
       addDevice('AMD Radeon VII', [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 ]);
    end;
    with addArch('gfx90c')^ do begin
       addDevice('AMD Radeon(TM) Graphics', [ 0, 1, 16, 2, 32, 16, 128, 8, 8, 128, 1, 1, 0, 1, 2, 2 ]);
       addDevice( kDeviceNameDefault, [ 0, 1, 16, 2, 32, 16, 128, 8, 8, 128, 1, 1, 0, 1, 2, 2 ]);
    end;
  end;
  with addVendor(kDeviceTypeGPU, 'NVIDIA')^ do begin
    with addArch('SM12.0')^ do begin
      addDevice('NVIDIA GeForce RTX 5080', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
    end;
    with addArch('SM2.0')^ do begin
      addDevice('GeForce GTX 480', [0, 1, 16, 2, 16, 8, 64, 32, 16, 64, 1, 1, 1, 1, 2, 2]);
      addDevice('GeForce GTX 580', [0, 1, 16, 2, 32, 8, 128, 16, 32, 64, 1, 1, 1, 0, 4, 2]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 32, 32, 32, 64, 0, 0, 0, 0, 1, 2]);
    end;
    with addArch('SM3.0')^ do begin
      addDevice('GRID K520', [0, 1, 16, 2, 16, 8, 32, 8, 16, 64, 1, 1, 1, 1, 2, 4]);
      addDevice('GeForce GT 650M', [0, 1, 32, 2, 8, 8, 32, 32, 32, 64, 1, 1, 0, 0, 4, 2]);
      addDevice('GeForce GTX 670', [0, 1, 16, 2, 8, 8, 64, 16, 16, 64, 1, 1, 1, 0, 2, 4]);
      addDevice('GeForce GTX 680', [0, 1, 32, 8, 8, 16, 64, 32, 16, 128, 1, 1, 0, 0, 4, 2]);
      addDevice('GeForce GTX 760 Ti OEM', [0, 1, 32, 2, 8, 8, 32, 8, 8, 64, 0, 0, 0, 0, 2, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 32, 32, 32, 64, 1, 1, 0, 0, 4, 2]);
    end;
    with addArch('SM3.5')^ do begin
      addDevice('GeForce 920A', [0, 1, 16, 2, 16, 8, 128, 16, 32, 128, 1, 1, 1, 0, 4, 1]);
      addDevice('GeForce GTX TITAN', [0, 1, 16, 8, 32, 16, 64, 8, 8, 64, 1, 1, 1, 0, 2, 2]);
      addDevice('GeForce GTX TITAN Black', [0, 1, 16, 2, 16, 8, 64, 16, 16, 64, 1, 1, 1, 0, 4, 1]);
      addDevice('NVIDIA GeForce GT 730', [0, 1, 32, 2, 16, 16, 64, 16, 8, 64, 0, 1, 1, 0, 4, 2]);
      addDevice('Tesla K20m', [0, 1, 16, 2, 32, 16, 64, 16, 8, 64, 1, 1, 1, 0, 2, 4]);
      addDevice('Tesla K40m', [0, 1, 16, 8, 16, 8, 64, 16, 16, 128, 1, 1, 1, 0, 2, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4]);
    end;
    with addArch('SM5.0')^ do begin
      addDevice('GeForce 920MX', [0, 1, 16, 2, 32, 8, 128, 8, 16, 128, 1, 1, 1, 0, 4, 1]);
      addDevice('GeForce GTX 750', [0, 1, 16, 2, 16, 16, 64, 32, 8, 128, 1, 1, 1, 1, 1, 2]);
      addDevice('GeForce GTX 750 Ti', [0, 1, 16, 2, 16, 16, 128, 32, 8, 64, 1, 1, 0, 1, 8, 2]);
      addDevice('Quadro M2000M', [0, 1, 16, 2, 16, 16, 128, 32, 16, 128, 1, 1, 1, 1, 1, 2]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 64, 32, 32, 64, 0, 0, 0, 0, 2, 1]);
    end;
    with addArch('SM5.2')^ do begin
      addDevice('GeForce GTX 970', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 2]);
      addDevice('GeForce GTX 980', [0, 1, 16, 2, 16, 16, 64, 16, 8, 128, 1, 1, 1, 0, 4, 8]);
      addDevice('GeForce GTX TITAN X', [0, 1, 16, 2, 8, 16, 128, 8, 8, 128, 1, 1, 1, 1, 4, 8]);
      addDevice(kDeviceNameDefault, [0, 1, 16, 2, 16, 16, 32, 16, 8, 128, 1, 1, 1, 0, 1, 2]);
    end;
    with addArch('SM6.0')^ do begin
      addDevice('Tesla P100-PCIE-16GB', [0, 1, 32, 2, 8, 8, 32, 8, 8, 32, 1, 1, 0, 0, 4, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 32, 8, 8, 32, 1, 1, 0, 0, 4, 4]);
    end;
    with addArch('SM6.1')^ do begin
      addDevice('GeForce GTX 1050 Ti', [0, 1, 16, 2, 32, 8, 128, 32, 16, 128, 1, 1, 1, 0, 4, 1]);
      addDevice('GeForce GTX 1070', [0, 1, 16, 2, 32, 16, 128, 32, 8, 128, 1, 1, 1, 0, 4, 1]);
      addDevice('GeForce GTX 1070 Ti', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 1]);
      addDevice('GeForce GTX 1080', [0, 1, 32, 2, 16, 8, 64, 8, 8, 64, 1, 1, 1, 1, 4, 8]);
      addDevice('GeForce GTX 1080 Ti', [0, 1, 16, 2, 32, 16, 64, 16, 8, 128, 1, 1, 0, 1, 2, 8]);
      addDevice('NVIDIA GeForce GTX 1060 6GB', [0, 1, 32, 2, 16, 16, 128, 32, 8, 128, 1, 1, 1, 1, 4, 4]);
      addDevice('NVIDIA GeForce MX150', [0, 1, 32, 2, 16, 16, 128, 32, 8, 128, 1, 1, 1, 1, 4, 4]);
      addDevice('TITAN X (Pascal)', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 1]);
      addDevice('Tesla P4', [0, 1, 32, 2, 16, 16, 64, 16, 8, 64, 0, 1, 1, 0, 4, 2]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 1]);
    end;
    with addArch('SM7.0')^ do begin
      addDevice('Quadro GV100', [0, 1, 32, 2, 16, 16, 64, 8, 8, 32, 1, 1, 0, 0, 4, 4]);
      addDevice('Tesla V100-PCIE-16GB', [0, 1, 32, 2, 16, 16, 64, 8, 8, 32, 1, 1, 0, 0, 4, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 16, 16, 64, 8, 8, 32, 1, 1, 0, 0, 4, 4]);
    end;
    with addArch('SM7.5')^ do begin
      addDevice('GeForce GTX 1650', [0, 1, 16, 2, 32, 8, 128, 32, 16, 128, 1, 1, 1, 0, 4, 1]);
      addDevice('NVIDIA GeForce GTX 1650 SUPER', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce GTX 1650 Ti', [0, 1, 16, 2, 32, 8, 128, 32, 16, 128, 1, 1, 1, 0, 4, 1]);
      addDevice('NVIDIA GeForce MX450', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 2060', [1, 8, 1, 1, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 8]);
      addDevice('NVIDIA GeForce RTX 2070 SUPER', [0, 1, 16, 2, 8, 8, 32, 32, 16, 128, 0, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 2070 Super', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice('NVIDIA GeForce RTX 2070 with Max-Q Design', [1, 8, 1, 1, 16, 16, 64, 4, 4, 128, 0, 0, 0, 0, 1, 1]);
      addDevice('NVIDIA GeForce RTX 2080 Ti', [0, 1, 32, 2, 8, 32, 128, 16, 8, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 2080 with Max-Q Design', [1, 16, 1, 1, 16, 16, 64, 8, 8, 128, 0, 0, 0, 0, 2, 2]);
      addDevice('Quadro T2000 ', [0, 1, 16, 2, 16, 16, 128, 32, 16, 128, 1, 1, 1, 1, 1, 2]);
      addDevice('TITAN RTX', [0, 1, 32, 2, 8, 32, 128, 16, 8, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('Tesla T4', [0, 1, 16, 2, 16, 16, 128, 32, 16, 128, 1, 1, 1, 1, 1, 2]);
      addDevice(kDeviceNameDefault, [1, 4, 1, 1, 4, 4, 64, 8, 8, 32, 0, 0, 0, 0, 4, 4]);
    end;
    with addArch('SM8.0')^ do begin
      addDevice('A100-PCIE-40GB', [0, 1, 32, 2, 8, 8, 32, 16, 16, 64, 1, 1, 0, 0, 4, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 32, 16, 16, 64, 1, 1, 0, 0, 4, 4]);
    end;
    with addArch('SM8.6')^ do begin
      addDevice('NVIDIA GeForce RTX 2050', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 3050 Laptop GPU', [1, 8, 1, 1, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 8]);
      addDevice('NVIDIA GeForce RTX 3050 Ti Laptop GPU', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice('NVIDIA GeForce RTX 3060', [0, 1, 32, 2, 8, 8, 32, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice('NVIDIA GeForce RTX 3060 Laptop GPU', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 3070', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 3070 Ti Laptop GPU', [0, 1, 16, 2, 8, 8, 32, 8, 16, 128, 1, 0, 1, 1, 4, 8]);
      addDevice('NVIDIA GeForce RTX 3080', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 3080 Laptop GPU', [0, 1, 16, 2, 8, 8, 32, 8, 16, 128, 1, 0, 1, 1, 4, 8]);
      addDevice('NVIDIA GeForce RTX 3080 Ti', [0, 1, 16, 2, 8, 8, 32, 32, 16, 128, 0, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 3090', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice('NVIDIA RTX A6000', [0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 16, 8, 8, 32, 0, 0, 0, 0, 1, 4]);
    end;
    with addArch('SM8.9')^ do begin
      addDevice('NVIDIA GeForce RTX 4050 Laptop GPU', [0, 1, 32, 2, 8, 8, 64, 32, 16, 128, 1, 1, 1, 1, 2, 2]);
      addDevice('NVIDIA GeForce RTX 4060', [0, 1, 32, 2, 8, 8, 32, 8, 8, 64, 1, 1, 0, 0, 4, 4]);
      addDevice('NVIDIA GeForce RTX 4060 Laptop GPU', [0, 1, 32, 2, 32, 32, 128, 8, 8, 64, 0, 1, 1, 1, 4, 4]);
      addDevice('NVIDIA GeForce RTX 4060 Ti', [0, 1, 32, 2, 32, 16, 128, 8, 8, 64, 0, 0, 0, 0, 4, 8]);
      addDevice('NVIDIA GeForce RTX 4070 Laptop GPU', [1, 8, 1, 1, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 8]);
      addDevice('NVIDIA GeForce RTX 4070 Ti', [0, 1, 16, 2, 8, 8, 32, 8, 16, 128, 1, 0, 1, 1, 4, 8]);
      addDevice('NVIDIA GeForce RTX 4080', [0, 1, 16, 2, 8, 8, 32, 8, 16, 128, 1, 0, 1, 1, 4, 8]);
      addDevice('NVIDIA GeForce RTX 4090', [0, 1, 32, 2, 32, 32, 128, 8, 8, 64, 0, 1, 1, 1, 4, 4]);
      addDevice(kDeviceNameDefault, [1, 8, 1, 1, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 8]);
    end;
    with addArch('default')^ do begin
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 16, 8, 8, 32, 1, 1, 0, 0, 2, 4]);
    end;
  end;
  with addVendor(kDeviceTypeAll, 'default')^ do
    with addArch('default')^ do
      addDevice(kDeviceNameDefault, [0, 1, 32, 2, 8, 8, 32, 8, 8, 32, 1, 1, 0, 0, 4, 4]);
end;


