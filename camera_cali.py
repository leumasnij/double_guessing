import cv2
import os
# for i in range(11):
#     cap = cv2.VideoCapture(i)
#     ret, frame = cap.read()
#     print(i, ret)
#     if ret:
#         cv2.imwrite(str(i) + '.jpg', frame)
#         file = 'video' + str(i)
#         real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
#         with open(real_file, "rt") as name_file:
#             name = name_file.read()
#         print(name, real_file)
#     cap.release()

dir_list = []
dir = '/media/okemo/extraHDD31/samueljin/CoM_dataset_outlier'
dir2 = '/media/okemo/extraHDD31/samueljin/CoM_dataset'
if not os.path.exists(dir2):
    os.mkdir(dir2)
for file in os.listdir(dir):
    true_dir = os.path.join(dir, file)
    dir_list.append(true_dir)

num = len(os.listdir(dir2))+1
for i in range(len(dir_list)):
    os.rename(dir_list[i], os.path.join(dir2, str(num+i)))

