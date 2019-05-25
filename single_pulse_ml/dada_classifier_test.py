#!/usr/bin/env python3
import numpy as np
from psrdada import Reader
import time
import matplotlib.pylab as plt

import realtime_tools
import frbkeras

fn_model = 'model/20190125-17114-freqtimefreq_time_model.hdf5'
model = frbkeras.load_model(fn_model)

# Create a reader instace
reader = Reader()

# Connect to a running ringbuffer with key 'dada'
#reader.connect(0xdada)
reader.connect(0x1200)


#{'SCANLEN': '     448.512', 'NBIT': '        8', 'HDR_VERSION': ' 1.0', 'SCIENCE_MODE': '0', 'SOURCE': '      B0531+21', 'PADDED_SIZE': '12500', 'AZ_START': '    278.157723538', 'INSTRUMENT': '  ARTS', 'NPOL': '        2', 'HDR_SIZE': '    40960', 'FREQ': '        1399.60327148', 'BW': '          300', 'MJD_START': '   58576.6766217', 'FILE_NUMBER': ' 0                      ', 'HEADER': '      DADA', 'TSAMP': '       8.192e-05', 'ZA_START': '    68.1351904127', 'MIN_FREQUENCY': '1249.70092773', 'PARSET': '      425a6839314159265359a5f5b214001176df806132508ffff27fe7df9abffffff060108e92ee3eec2f0b3db97b4f0763090a0a000068daf98ab6dbdefbdbefb89ad57c633ad5361242104c49e469a2693d091a321ea1b280f5191e53ca3d401240991088529e9a83d4d0190001a03468c8002a9edea4a7a9b49b28832613200321934d34c8320006129faa92453f51a9b5308f4218d4c2030101a320c013045114f4d44c843d4c9ea69ea001a3d0d350c81a0c8000a9220104d27a4f489a0698430990000000301680c31587bf4df3b8975efd7f8d35c9e1deb8279b1264cc99933230c30da56986186186186fd9ad6aab084d70cdc1deb77deef0efbddf077a3bc3bc9dea77a154a48116333519ddeb7f192531a728b8b64947302634003b92648021a9292929298a5ac6a929ab359ac531accb56a98a6ab2e3ebeacdb163161673155f5842114610984a7e87d902fdfc6da1e7dd75f96d87c24fb64ed0f3879aa39529d91caf11ea8f54688e11ca8e546d2e62c31d4aaf550c841fc76772dfcddddddf47a2ce091f725fa6e9d263be87009244e907ed0e24167f9a4907c8219009fefebd77e52efd7a92a7ddeab2db333222234ed04c5c6666e2b3b5ec4c92617b74e2640324c0f36fd781b8c526495292a533b4aa9542a7327f524cf3ab460f4424798cccfab1767767430258a0884a1c604cca0b6850ccda17800c9c169c41f2c2f89b62d9a44e81328e304b1081e721430b196e013aa766713a64e6a1e14633a0942641b65646bf85dd9a530beaa69876a5cd0aee942c6704fa4e213898c7ad6ab4b7ad9a95ad330c2604d495534d348b2d2c80068a3c264cd5808f3bf5a4234ef66ba07066665fff06ee139737f340000000000000000000b7abcfc3cce0b8a980129801248012480376dbbddcdf9776b46fb7dbee8db5b5b5b496d6d6d41042e7b2b2307c4c8915199906d2e266d2a0a17932b2a285830e3946ca2cc24c346186126cd1839b3dfb6f5f1f55be75ce74de6bcaf5ac6dd6f236f7795b8baccdc65666f27511fd3997fd5f7fecf8b4e684f34fb35e328400000023446509ead1294279319188a0011111119e589becd556ac79adc31c72cb0c32366e7438810c6d9ad90499b0c2c30c30ca9b74f93a6000000000000000727a7cfcbd7ea73aa6f52700e2373b4c9bccceff2e1111133abcce5ebeae37417fdc62eacfbd4cdce58305294a52eb266949497bcbcd6975294a75eb75b6716d9a924999c3633162e3b6d64924b9da86307036891c466bb0612aaf7c05f584dd4fa33858730c6dae5cf8cdd972a9e124d44b6553822567b249664edf42a013f4fac7b3bf7cf73bd7a380791f9bb04ad9077409e01b5c99bd6efd085290b0141a660169c97381d3c333d38255c7dcd868f539ad82de45ee8157acea7f6ca05ce723506e66509a9af917685535b059e3f7a916ac1720053b7be4d6f2dee0429370e71c9051c88622d0a8b95d86c789bb154103592f7362d3aa7a3e9ade527949f79990b02c6ecc75496d5a2552549467dab294a54626997560c54a5298ad30b292e5e4b455539046f06f1668d59382aaf7c470dcef86525a8e3432eb25c96274e9b72c37cddd5b4d5868eeda413753f6cf292dc278e38dd15ad2e502d1684a3393bd735cd6d6b622d8161f83c7d2bdf7809eb7cea4f5b580af8fbf3a9f2b5dfd81ab93ae2d02dcf6412ce098fc90b45d566491d05a717920a733f2592de7a7be4f83bfc005ee4f01445fe69cd87e13ed27d5d7d9c74daa1dd97b473abc5273bd9547781b86ae6c1c655efbd8954565cddce86b3544e471de76efc84ad33e3a72cafac82d47db75fa84f6b4c858b3f00fd74f6288a261a1ded83d17f5b421c52a842ccf1e042a21dd2d25a56482ef4ec0437295f873539ca109a299982b5ba86cfb9f969f5cb77171380b9d85a8f25f72ced70baa5dc3ab649c136b50a2ba0537a331db3bb3d834b29267899e191a9d1d8cfb38f4d7ab8a4ece9832bf7e740496787ca678eca06606f92cd266cefe7db4cfc82ee30b2ba90b058e5026241433a96bf4f3a88e32be1f5684b5a96cf58b9a218451b8053323d2ea5c14d1b86b45c28d6f5a36e06ed4212796424c6d9ae696357e1429d4027b8eedb78a1886a199bfd9204757e80ef383bf3c04a2e1afa7902ab1c3634ec72dbb3899deb9b82ed5acabdf431436131c95a2f1c1fe33ceafad7a869ab940d0cecb1a99d950c6be87d0c402ea63edd370641a630e748355c7c966eafbbc92f409e755809b4d6ca3bcad9920278f92fd479bef609cd48c84ddf5c41ab7cfdf044d385febec53e9e521467231b636bebbf3ee629f0724e2688fb13753d8ed1dbc70fa611f00ae6dc178d12d9f41b353d6bd8e2de86c493439b67ad9a188d78fa6de371c8298db56cde7d937c9eb10b39a04c3d440ef5a789a89d828888a050ffa475ee7d6fe4a983b3842888b0ebc8f3cf653fa66c10b80b599a8fbf5a680597106ea416ea98acaef59625dd8e554dd186f7d69e63cfbb5ce9f9c66e4cefae5a130ba8ee5f087d6598cdb66f978015e6cb2b80b9ae3f5d41ccad3eb60b61a0ddf00c778a666e397f23fc47ce9fe0c3d16769a2301c12f720aa1fbcf7cd7bd4fa6055aaeaeb3d930fbb16e77ce02adf3201739bac424dc013e902d6f0624d582e7337ca89626e3735953992f55aa879ce4430d809842dee25b75e347b2ab9eba7f7f5cef861d9e2688e079d331d884f300ad34bb30999beac80514d1bf8a7f87d6efe6109c7cce482bec9dedaef955a2be3a301655b0589452142a654db87429b7c705a99791a90b38e6aa9da775aae696ca992322bca56850031b6974ca0e5e9a2310dc489256281c139e7ae6815636b46ec54324ecfaf887c59635a06c6a87d08f899ecd4e8c9ddeb99b2a766ddf776db0549b997557770da714241231a8766430b1f557cb6e59dd502dde9fac7d33908d5a6655214a3db23a0c981725ca199a1f1f671edeb009a5bcb001fd001864a9eea7b64df4fd5aa554ab79c986052ab16c1fdde3aab2c2f9457a7622a4a23011a3003b70ac904a325a42d1ad5e07e9cbabc3538b730cdd6c4d8f4d8ca8abb9f3b3a37404b64def8d17c6026b73eeda222222222223b26bb9740fdde38560a405472cf77816e9b3a757f063f85f2b72965196dc79ec6e5367d0ff7bf155675dbd7c95d6cceea0726066499819819806605d39feb49ef0057a58546f40881846cc34d7c0e160904798903d3d5ddf6828600dd110e2cdc7af6c0123571f1cf25bab672ecae40132dc52b90182408f05773e909f38ddcaa9002098048ec2543e1c2eb20c98160e98c1444bf25d88c7572177f446c2798c7f8e9faf09e3f157a3b24230bf077478b788e544eac42329eb083ebfebb5797fc6dbd6366ee684827bb99b23e005040bfccd3c070a01d61a842d5b7ef6934b63d56331ecbaa27d9aaa95e9aca1e4eb8bff205f71d67ac3e33e21cca18e959f5d8924aabf93b6caa7d3718cca832ab9335c577fc6427775ad3433bbbbb3bbbb9eff9e67e1bf9a2b573af99bfe8b8702f6c661b6a6ca869526639e80031a80031a80031a800350001a805294bff2d8d3124c2cb30c290526528930b2cc30a40c6a1ed9e67919cd3c11ae7308b9a635a8d53b98401d1f7fc0064ba50b35c647116ef7b6c31c4bf0e696389eb048e8066b5f9d710ed0a2ef379b2af26157a8edad5539e5508be2b136908e78fa4057ca80c4436b5b547f8213e3b7579ea640000eadca6b50ef4dd54ac2aace9e8a1d4025b11433c02872c2a199e797fc0d39b46734139dde553283cf9b62a97404d2708052829414a0bd7cf3b9fe4d5f5479fd6119eb5e860f45fb422e0384f6645807a5c05fbe48781d084da27515da3fb34fca60afaa97ee4a5dfe1361ace0e97068bd1b808b9922e1fa72fcfcbe4af1d9ed121ac4dcdc5a08d3bf742827626013e02ccd7cbc11169a8033977e4a40c8400ac92492a425720dee65b0342a617bec7726d6932a248c3a5904c203b6b62b7b19084b82ac063ee10a5684e29caff8e2c542291b90b61b06f272f3edc2bc183c5708e8d5f25993ede0ce5ca4444444444444444444444444444444444444444c71d58cd6666d4aa911f23bd3523054e185df719900280ab04674f7e6770c34007ad41374034455677f06c744d054977e6808ed37be50aa0c5641a02ad931d0133876f460a0916dd926ea81a6eb08c73aae957115c9d3c5ebddb46edb595211c5bf252cb561b2c4b3ca48672d13ace48507ecc36a4d33e2250010343a0c07bd37c9601e4d98d21f9f7c6832f95fe7e9a5dcfc3533924669cb9244d1dd08e9c65664aca26404bacba3550412ad8d0d3efa5556e50bc3f31558c1647bc77b7a4c941d31490e74c4c2d5c02181aa443194acaa95555b998508012050c5267b808bba7bfd2fb5b6d061c768956aaaaae169bdd4156b9921a8df6b61b6dac931de4fe2423d3bb140f506cc723824925184cba91540a88a8b4002ebdbb074e0a7c4ae1271bdbd199dfdb555ec9ff177245385090a5f5b2140', 'DEC': '         220052.2', 'SCIENCE_CASE': '4', 'TELESCOPE': '   WSRT', 'OBS_OFFSET': '  0                      ', 'RA': '          235615.8', 'LST_START': '   19654.7180198', 'UTC_START': '   2019-04-03-16:14:20', 'RA_HMS': '      23:56:15.8', 'RESOLUTION': '  230400000', 'NCHAN': '       1536', 'SAMPLES_PER_BATCH': '12500', 'CHANNEL_BANDWIDTH': '0.1953125', 'BEAM': '        0', 'BYTES_PER_SECOND': '225000000', 'FILE_SIZE': '   101376000000           ', 'IN_USE': '      1', 'DEC_HMS': '     22:00:52.2', 'NDIM': '        2'}

nfreq_plot = 32
ntime_plot = 64
ntab = 12
dt = 8.192e-5
RtProc = realtime_tools.RealtimeProc()

counter = -1

dm = 568.

# Leaving this as is. Need to think of how I will 
# actually read the data as it comes in. 
# Ask Leon again how the trigger headers actually work.
# Need to figure out how many triggers per minute the code 
# can keep up with. Also need to update the dedisperser for edge 
# effect. The frequency roll should be more like presto. .

for page in reader:
    counter += 1
    data = np.array(page)

    print("%d seconds" % counter)

    header = reader.getHeader()
    H = realtime_tools.DadaHeader(header)

    t_batch = H.ntime_batch*H.dt
    dshape = (ntab, H.nchan, H.ntime_batch)
    data = np.reshape(data, dshape)

    if len(data)==0:
        continue

    # This method will rfi clean, dedisperse, and downsample data.
    data_classify, data_dmtime = RtProc.proc_all(data, dm, nfreq_plot=nfreq_plot, ntime_plot=ntime_plot, 
                                    invert_spectrum=True, downsample=16)
    print('dtms', data_dmtime.shape)
    prob = model.predict(data_classify[..., None])

    indpmax = np.argmax(prob[:, 1])

    if prob[indpmax,1]>0.5:
        fig = plt.figure()
        plt.imshow(data_classify[indpmax], aspect='auto', vmax=3, vmin=-2.)
        plt.show()
    else:
        print('Nothing good')

reader.disconnect()