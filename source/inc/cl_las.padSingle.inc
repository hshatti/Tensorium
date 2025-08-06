  padSingle.init('Pad', kSingle, kDeviceTypeAll,
                'default', 'default', kDeviceNameDefault,
                ['PAD_DIMX', 'PAD_DIMY', 'PAD_WPTX', 'PAD_WPTY'],
                [16, 8, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                .addVendor(kDeviceTypeGPU, 'NVIDIA')
                .addArch('default')
                .addDevice(kDeviceNameDefault, [8, 16, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

