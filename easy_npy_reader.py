import numpy as np
import os
from apriltag_helper.tag import BoxWithoutLid
# x = np.load('/media/okemo/extraHDD31/samueljin/7_4/run3/0/data1.npy', allow_pickle=True, encoding= 'latin1').item()
# print(x)
moi = BoxWithoutLid(0.18529, [0.15, 0.15, 0.085], 0.003).moi_vector()
addr = '/media/okemo/extraHDD31/samueljin/7_4/run2/'
for file1 in os.listdir(addr):
    new_addr = os.path.join(addr, file1)
    for file in os.listdir(new_addr):
        final = os.path.join(new_addr, file)
        dict = np.load(final, allow_pickle=True, encoding= 'latin1').item()
        print(dict)
