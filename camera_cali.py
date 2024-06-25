import cv2
import os
for i in range(11):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    print(i, ret)
    if ret:
        cv2.imwrite(str(i) + '.jpg', frame)
        file = 'video' + str(i)
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        with open(real_file, "rt") as name_file:
            name = name_file.read()
        print(name, real_file)
    cap.release()
