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
def update_markerMotion(marker_present, marker_prev, marker_init):
    # No. of markers identified in the initial frame. This remains constant through all the frames.
    markerCount = len(marker_init)
    markerU = np.zeros(markerCount)  # X motion of all the markers. 0 if the marker is not found
    markerV = np.zeros(markerCount)  # Y motion of all the markers. 0 if the marker is not found
    # No. of markers in the present frame.
    Nt = len(marker_present)
    # Temporary variable used for analysis
    no_seq2 = np.zeros(Nt)
    # center_now is the variable that will be returned by the function that contains equal no. of centers as the initial frame.
    center_now = np.zeros([markerCount, 3])

    for i in range(Nt):
        # Calculating the motion of each marker in the present frame w.r.to all the markers in the previous frame.
        dif = np.abs(marker_present[i, 0] - marker_prev[:, 0]) + np.abs(marker_present[i, 1] - marker_prev[:, 1])
        # Multiplying the above variable with the difference of the contour area of the each marker w.r.to the contour areas of all the markers in the previous frame.
        no_seq2[i] = np.argmin(dif * (100 + np.abs(marker_present[i, 2] - marker_init[:, 2])))

    for i in range(markerCount):

        dif = np.abs(marker_present[:, 0] - marker_prev[i, 0]) + np.abs(marker_present[:, 1] - marker_prev[i, 1])
        t = dif * (100 + np.abs(marker_present[:, 2] - marker_init[i, 2]))

        # a is a threshold further used in the analysis to filter out markers that might not have significant marker motion.
        a = np.amin(t) / 100
        b = np.argmin(t)

        # If the contour area of a marker in the present frame is less than obtained 'a', set the x and y motion of that marker 0
        if marker_init[i, 2] < a:  # for small area
            markerU[i] = 0
            markerV[i] = 0
            center_now[i] = marker_prev[i]

        # When the index i matches the index of the variable b, x and y motion are calculated w.r.to the initial marker location.
        elif i == no_seq2[b]:
            markerU[i] = marker_present[b, 0] - marker_init[i, 0]
            markerV[i] = marker_present[b, 1] - marker_init[i, 1]
            center_now[i] = marker_present[b]
        else:
            markerU[i] = 0
            markerV[i] = 0
            center_now[i] = marker_prev[i]

    return center_now, markerU, markerV
def displaycentres(img, marker_init, markerU, markerV):
    # Just rounding markerCenters location
    markerCenter = np.around(marker_init[:, 0:2]).astype(np.int16)

    for i in range(marker_init.shape[0]):

        if markerU[i] != 0:
            cX = int(marker_init[i, 0])
            cY = int(marker_init[i, 1])
            cv2.arrowedLine(img, (cX, cY), (cX + int(markerU[i]), cY + int(markerV[i])), (0, 255, 255), 2)
    return img
def track_marker():
    cap = cv2.VideoCapture(0)
    # ret, frameI = cap.read()
    # markerI = mm.find_markers(frameI, frameI)
    print('start')
    ret, frame = cap.read()
    frameI = resize_crop_mini(frame, 640, 480)
    markersI = find_markers(frameI)
    old_markers = markersI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    index = len([name for name in os.listdir(saving_adr) if os.path.isfile(os.path.join(saving_adr, name))] ) + 1
    out = cv2.VideoWriter(saving_adr + 'output' + str(index) + '.avi', fourcc, 10.0, (640, 480))
    while True:
        ret, frame = cap.read()
        frame = resize_crop_mini(frame, 640, 480)
        markers = find_markers(frame)
        center_now, markerU, markerV = update_markerMotion(markers, old_markers, markersI)
        frame = displaycentres(frame, center_now, markerU, markerV)
        old_markers = center_now
        average_movement = np.mean(np.sqrt(markerU**2 + markerV**2))
        cv2.putText(frame, 'Average Movement: ' + str(average_movement), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        # print(display_frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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