  XgemmHalf.init('Xgemm', kHalf, ['GEMMK', 'KREG', 'KWG', 'KWI', 'MDIMA', 'MDIMC', 'MWG', 'NDIMB', 'NDIMC', 'NWG', 'SA', 'SB', 'STRM', 'STRN', 'VWM', 'VWN'])
   .addVendor(kDeviceTypeAll, 'default')
     .addArch('default')
       .addDevice(kDeviceNameDefault, [0, 1, 32, 2, 16, 16, 64, 8, 8, 32, 0, 0, 0, 0, 4, 4]);
