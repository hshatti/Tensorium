  TransposeSingle.init('Transpose', kSingle,
                 kDeviceTypeAll, 'default', 'default', kDeviceNameDefault,
                 ['TRA_DIM', 'TRA_PAD', 'TRA_SHUFFLE', 'TRA_WPT'],
                 [4, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                 .addVendor(kDeviceTypeGPU, 'NVIDIA')
                 .addArch('default')
                 .addDevice(kDeviceNameDefault, [16, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);


