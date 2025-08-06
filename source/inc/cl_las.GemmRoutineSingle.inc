
with GemmRoutineSingle.init('GemmRoutine', kSingle, ['XGEMM_MIN_INDIRECT_SIZE'])^ do begin
       // AMD GPUs
    with addVendor(kDeviceTypeGPU, 'AMD')^ do begin
      with addArch('Ellesmere')^ do begin
          addDevice('AMD Radeon RX 580 2048SP'                       ,[512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice('AMD Radeon RX590 GME'                           ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[704, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('Fiji')^ do begin
          addDevice('AMD Radeon 500 Series'                          ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('Hawaii')^ do begin
          addDevice('AMD FirePro W8100'                              ,[512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('default')^ do begin
          addDevice('AMD Radeon Pro 450 Compute Engine'              ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1010:xnack-')^ do begin
          addDevice('AMD Radeon RX 5700'                             ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice('AMD Radeon RX 5700 XT'                          ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1030')^ do begin
          addDevice('AMD Radeon RX 6800 XT'                          ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice('AMD Radeon RX 6900 XT'                          ,[512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1031')^ do begin
          addDevice('AMD Radeon RX 6700 XT'                          ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1032')^ do begin
          addDevice('AMD Radeon RX 6600 XT'                          ,[832, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[832, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1034')^ do begin
          addDevice('AMD Radeon RX 6500 XT'                          ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1035')^ do begin
          addDevice('AMD Radeon Graphics'                            ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1100')^ do begin
          addDevice('Radeon RX 7900 XTX'                             ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1101')^ do begin
          addDevice('AMD Radeon RX 7800 XT'                          ,[704, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[704, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1102')^ do begin
          addDevice('AMD Radeon RX 7600'                             ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1103')^ do begin
          addDevice('AMD Radeon 780M Graphics'                       ,[320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx1201')^ do begin
          addDevice('AMD Radeon RX 9070 XT'                          ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx902')^ do begin
          addDevice('AMD Radeon(TM) Graphics'                        ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice('AMD Radeon(TM) RX Vega 10 Graphics'             ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx906:sramecc+:xnack-')^ do begin
          addDevice('AMD Instinct MI50/MI60'                         ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice('AMD Radeon VII'                                 ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('gfx90c')^ do begin
          addDevice('AMD Radeon(TM) Graphics'                        ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
          addDevice(kDeviceNameDefault                               ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
   end;

     // ARM GPUs
    with addVendor(kDeviceTypeGPU, 'ARM')^ do begin
      with addArch('default')^ do begin
        addDevice('Mali-T628'                                      ,[128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                               ,[128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // Apple GPUs
    with addVendor(kDeviceTypeGPU, 'Apple')^ do begin
      with addArch('default')^ do begin
        addDevice('Apple M1'                                       ,[1152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Apple M2 Max'                                   ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                               ,[1600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // Imagination Technologies GPUs
    with addVendor(kDeviceTypeGPU, 'Imagination Technologies')^ do begin
      with addArch('default')^ do begin
        addDevice('PowerVR B-Series BXE-4-32'                      ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                               ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // Intel CPUs
    with addVendor(kDeviceTypeCPU, 'Intel')^ do begin
      with addArch('default')^ do begin
        addDevice('12th Gen Intel(R) Core(TM) i7-12700H'           ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Core(TM) i5-4590S CPU @ 3.00GHz'       ,[320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz'       ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Core(TM) i9-9980HK CPU @ 2.40GH'       ,[1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz'      ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz'      ,[768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('cpu-penryn-Intel(R) Core(TM)2 Duo CPU     T6670  @',[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // Intel GPUs
    with addVendor(kDeviceTypeGPU, 'Intel')^ do begin
      with addArch('default')^ do begin
        addDevice('Intel(R) Arc(TM) A750 Graphics'                    ,[1216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Arc(TM) A770 Graphics'                    ,[768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) HD Graphics 6000 BroadWell U-Processor GT',[128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) HD Graphics IvyBridge M GT2'              ,[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) HD Graphics Skylake ULT GT2'              ,[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) Iris(R) Xe Graphics'                      ,[576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) RaptorLake-S Mobile Graphics Controller'  ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) UHD Graphics 620'                         ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Intel(R) UHD Graphics 770'                         ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // Intel accelerators
    with addVendor(kDeviceTypeAccelerator, 'Intel')^ do begin
      with addArch('default')^ do begin
        addDevice('Intel(R) FPGA Emulation Device'                    ,[1344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[1344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // NVIDIA GPUs
    with addVendor(kDeviceTypeGPU, 'NVIDIA')^do  begin
      with addArch('SM12.0')^ do begin
        addDevice('NVIDIA GeForce RTX 5080'                           ,[1664, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[1664, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM3.5')^ do begin
        addDevice('GeForce 920A'                                      ,[640, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce GT 730'                             ,[384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM5.0')^ do begin
        addDevice('GeForce GTX 750 Ti'                                ,[768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Quadro M2000M'                                     ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                 ,[832, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM5.2')^ do begin
        addDevice('GeForce GTX 970'                                   ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM6.0')^ do begin
        addDevice('Tesla P100-PCIE-16GB'                              ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM6.1')^ do begin
        addDevice('GeForce GTX 1050 Ti'                               ,[1344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('GeForce GTX 1070 Ti'                               ,[1472, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('GeForce GTX 1080 Ti'                               ,[1792, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce GTX 1060 6GB'                       ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce MX150'                              ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('TITAN X (Pascal)'                                  ,[1664, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Tesla P4'                                          ,[1536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[1280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM7.0')^ do begin
        addDevice('Quadro GV100'                                      ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Tesla V100-PCIE-16GB'                              ,[1408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                  ,[1728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM7.5')^ do begin
        addDevice('GeForce GTX 1650'                                   ,[1280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce GTX 1650 SUPER'                      ,[1152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce GTX 1650 Ti'                         ,[1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce MX450'                               ,[512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 2060'                            ,[832, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 2070 SUPER'                      ,[1664, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 2070 Super'                      ,[1600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 2070 with Max-Q Design'          ,[1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 2080 Ti'                         ,[1728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 2080 with Max-Q Design'          ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Quadro T2000'                                       ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('TITAN RTX'                                          ,[1344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('Tesla T4'                                           ,[896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM8.0')^ do begin
        addDevice('A100-PCIE-40GB'                                     ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM8.6')^ do begin
        addDevice('NVIDIA GeForce RTX 2050'                            ,[448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3050 Ti Laptop GPU'              ,[1216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3060'                            ,[1536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3060 Laptop GPU'                 ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3070'                            ,[1920, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3070 Ti Laptop GPU'              ,[960, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3080'                            ,[1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3080 Laptop GPU'                 ,[1600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3080 Ti'                         ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 3090'                            ,[1600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA RTX A6000'                                   ,[1728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[1472, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('SM8.9')^ do begin
        addDevice('NVIDIA GeForce RTX 4050 Laptop GPU'                 ,[1664, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4060'                            ,[832, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4060 Laptop GPU'                 ,[1728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4060 Ti'                         ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4070 Laptop GPU'                 ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4070 Ti'                         ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4080'                            ,[1984, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice('NVIDIA GeForce RTX 4090'                            ,[1728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[1536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('default')^ do begin
        addDevice(kDeviceNameDefault                                   ,[1344, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // QUALCOMM GPUs
    with addVendor(kDeviceTypeGPU, 'QUALCOMM')^ do begin
      with addArch('default')^ do begin
        addDevice('QUALCOMM Adreno(TM)'                                ,[64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('OpenC)^ do C 2.0 Adreno(TM) 640')^ do begin
        addDevice('QUALCOMM Adreno(TM)'                                ,[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('OpenC)^ do C 2.0 Adreno(TM) 650')^ do begin
        addDevice('QUALCOMM Adreno(TM)'                                ,[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('OpenC)^ do C 3.0 Adreno(TM) 730')^ do begin
        addDevice('QUALCOMM Adreno(TM)'                                ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
      with addArch('OpenC)^ do C 3.0 Adreno(TM) 740')^ do begin
        addDevice('QUALCOMM Adreno(TM)'                                ,[320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        addDevice(kDeviceNameDefault                                   ,[320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
     // Default
    with addVendor(kDeviceTypeAll, 'default')^ do begin
      with addArch('default')^ do begin
        addDevice(kDeviceNameDefault                                   ,[960, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      end;
    end;
  end;


