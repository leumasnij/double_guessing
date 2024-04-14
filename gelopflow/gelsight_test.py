import sys
sys.path.append('../')
import os 
# print(os.getcwd())
import cv2
import numpy as np
import matplotlib.pyplot as plt
saving_adr = '/media/okemo/extraHDD31/samueljin/'
def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0]  / 7), int(img.shape[1]  / 7)
    # print(border_size_x, border_size_y)
    # keep the ratio the same as the original image size
    img = img[border_size_x :img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img


def find_markers(frame):
    ''' detect markers '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (5, 5), 5)
    mask = cv2.inRange(gray, 0, 55)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # return mask
    num, labels = cv2.connectedComponents(mask)
    marker_center = []
    for i in range(1, num):
        mask = np.where(labels == i, 255, 0)
        center_x = int(np.mean(np.where(mask == 255)[1]))
        center_y = int(np.mean(np.where(mask == 255)[0]))
        area = np.sum(mask == 255)
        flag = False
        for j in range(len(marker_center)):
            if np.linalg.norm(np.array([center_x, center_y]) - np.array(marker_center[j][:2])) < 30:
                marker_center[j] = np.array([int((center_x + marker_center[j][0]) / 2), int((center_y + marker_center[j][1]) / 2), area + marker_center[j][2]])
                flag = True
                break
        if not flag:
            marker_center.append([center_x, center_y, area])
    for i in range(len(marker_center)):
        cv2.circle(frame, (marker_center[i][0], marker_center[i][1]), 2, (0, 0, 255), 2, 6)
    marker_center = np.array(marker_center)
    return marker_center



# track_marker()
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = resize_crop_mini(frame, 1280, 960)
        mask = find_markers(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break