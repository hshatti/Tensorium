    XgemmDirectHalf.init('XgemmDirect', kHalf, ['KWID', 'MDIMAD', 'MDIMCD', 'NDIMBD', 'NDIMCD', 'PADA', 'PADB', 'VWMD', 'VWND', 'WGD'])^
      .addVendor(kDeviceTypeAll, 'default')
        .addArch('default')^
          .addDevice(kDeviceNameDefault, [2, 8, 8, 8, 8, 1, 1, 2, 2, 16, 0, 0, 0, 0, 0, 0]);
