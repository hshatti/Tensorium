  XgemmDirectSingle.init('XgemmDirect', kSingle,
                 kDeviceTypeAll, 'default', 'default', kDeviceNameDefault,
                 ['KWID', 'MDIMAD', 'MDIMCD', 'NDIMBD', 'NDIMCD', 'PADA', 'PADB', 'VWMD', 'VWND', 'WGD'],
                 [2, 8, 8, 8, 8, 1, 1, 1, 2, 16, 0, 0, 0, 0, 0, 0])
                 .addVendor(kDeviceTypeAll, 'NVIDIA')
                 .addArch('default')
                 .addDevice(kDeviceNameDefault, [2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0]);
