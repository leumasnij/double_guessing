import os
import numpy as np
import matplotlib.pyplot as plt

folder = '/media/okemo/extraHDD31/samueljin/data2'
data_array = []
CoM_array = []
for file in os.listdir(folder):
    if '.' in file:
        continue
    address = os.path.join(folder, file)
    for file2 in os.listdir(address):
        address2 = os.path.join(address, file2)
        data_dict = np.load(address2, allow_pickle=True).item()
        if data_dict['force'][-1] == 0 and data_dict['force'][-2] == 0:
            print(data_dict['force'])
        
            data_array.append(data_dict['force'])
            CoM_array.append(data_dict['GT'][:3]*100)

dict = {'force': data_array, 'CoM': CoM_array}
np.save('/media/okemo/extraHDD31/samueljin/data2.npy', dict)
            
        