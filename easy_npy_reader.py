import numpy as np
import os
from apriltag_helper.tag import BoxWithoutLid
# x = np.load('/media/okemo/extraHDD31/samueljin/7_4/run1/0/data1.npy', allow_pickle=True, encoding= 'latin1').item()
# print(x)
moi = BoxWithoutLid(0.18529, [0.15, 0.15, 0.085], 0.003).moi_vector()
addr = '/media/okemo/extraHDD31/samueljin/7_7/'
save_addr = '/media/okemo/extraHDD31/samueljin/haptic/'
# for file in os.listdir(save_addr):
#     final = os.path.join(save_addr, file)
#     dict = np.load(final, allow_pickle=True, encoding= 'latin1').item()
#     print(dict)
for file1 in os.listdir(addr):
    new_addr = os.path.join(addr, file1)
    for file2 in os.listdir(new_addr):
        final1 = os.path.join(new_addr, file2)
        for file3 in os.listdir(final1):
            final2 = os.path.join(final1, file3)
            dict = np.load(final2, allow_pickle=True, encoding= 'latin1').item()
            new_dict = {}
            new_dict['force'] = dict['force']
            new_dict['loc'] = dict['GT'][:3]
            num_file = len(os.listdir(save_addr))+1
            np.save(os.path.join(save_addr, str(num_file)+'.npy'), new_dict)
            # raise ValueError('Done')
