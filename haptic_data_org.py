import os
addr = '/media/okemo/extraHDD31/samueljin/CoM_dataset'
new_addr = '/media/okemo/extraHDD31/samueljin/haptic'
if not os.path.exists(new_addr):
    os.mkdir(new_addr)
for file in os.listdir(addr):
    final = os.path.join(addr, file)
    final = os.path.join(final, 'data.npy')
    os.rename(final, os.path.join(new_addr, file + '.npy'))
    # print(final)
