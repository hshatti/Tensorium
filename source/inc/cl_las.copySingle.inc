  CopySingle.init('Copy', kSingle,
                 kDeviceTypeAll, 'default', 'default', kDeviceNameDefault,
                 ['COPY_DIMX', 'COPY_DIMY', 'COPY_VW', 'COPY_WPT'],
                 [8, 8, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
                 .addVendor(kDeviceTypeGPU, 'NVIDIA')
                 .addArch('default')
                 .addDevice(kDeviceNameDefault, [8, 8, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
