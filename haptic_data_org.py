import os
import shutil, errno
import numpy as np
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise
        
# addr = '/media/okemo/extraHDD31/samueljin/7_15'
# for file in np.sort(os.listdir(addr)):
#     final = os.path.join(addr, file)
#     base = 0
#     file_directory = np.linspace(0, 9, 10)
#     now_list = file_directory
#     ref_file = str(int(file_directory[0]))
#     while ref_file in os.listdir(final):
#         for i in now_list:
#             if i%10 == 0:
#                 ref_addr = os.path.join(final, ref_file)
#                 continue
#             add = os.path.join(final, str(int(i)))
#             if not os.path.exists(add):
#                 break
#             for j in file_directory:
#                 if len(add) == 1:
#                     break
#                 len1 = len(os.listdir(ref_addr))
#                 old_addr = os.path.join(add, 'data' + str(int(j)) + '.npy')
#                 new_addr = os.path.join(ref_addr, 'data' + str(int(len1)) + '.npy')
#                 copyanything(old_addr, new_addr)
#         now_list = now_list+len(file_directory)
#         ref_file = str(int(now_list[0]))
        # print(now_list)
        # print(ref_file)
    


# addr = '/media/okemo/extraHDD31/samueljin/haptic'
# new_addr = '/media/okemo/extraHDD31/samueljin/haptic2'
# for file in np.sort(os.listdir(addr)):
#     final = os.path.join(addr, file)
#     dict1 = np.load(final, allow_pickle=True, encoding='latin1').item()
#     if dict1['force'][6] == 0 and dict1['force'][7] == 0:
#         np.save(os.path.join(new_addr, file), dict1)
# for file1 in np.sort(os.listdir(addr)):
#     add2 = os.path.join(addr, file1)
#     dict1 = np.load(add2, allow_pickle=True, encoding='latin1').item()
#     if np.sum(dict1['force'][:6]) == 0:
#         print(add2)
#         os.remove(add2)
    # shutil.rmtree(final)



# addr = '/media/okemo/extraHDD31/samueljin/7_10'
# new_addr = '/media/okemo/extraHDD31/samueljin/data2'
# for file1 in np.sort(os.listdir(addr)):
#     add2 = os.path.join(addr, file1)
#     for file2 in np.sort(os.listdir(add2)):
#         add3 = os.path.join(add2, file2)
#         if int(file2) % 10 == 0:
#             len1 = len(os.listdir(new_addr))
#             new_add = os.path.join(new_addr, str(int(len1)))
#             copyanything(add3, new_add)
            
            
# addr = '/media/okemo/extraHDD31/samueljin/7_11'
# for file1 in np.sort(os.listdir(addr)):
#     add2 = os.path.join(addr, file1)
#     for file2 in np.sort(os.listdir(add2)):
#         add3 = os.path.join(add2, file2)
#         if int(file2) % 10 == 0:
#             for file3 in np.sort(os.listdir(add3)):
#                 file_no = int(file3.split('.')[0].split('data')[1])
#                 if file_no > 10:
#                     os.remove(os.path.join(add3, file3))
#                     # print('removed')


addr = '/media/okemo/extraHDD31/samueljin/data2'
data_count = 0
for file1 in np.sort(os.listdir(addr)):
    add2 = os.path.join(addr, file1)
    data_count += len(os.listdir(add2))
    
    
print(data_count)