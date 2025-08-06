  padTransposeSingle.init('Padtranspose', kSingle,
                 kDeviceTypeAll, 'default', 'default', kDeviceNameDefault,
                 ['PADTRA_PAD', 'PADTRA_TILE', 'PADTRA_WPT'],
                 [1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                 .addVendor(kDeviceTypeGPU, 'NVIDIA')
                 .addArch('default')
                 .addDevice(kDeviceNameDefault, [0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);


