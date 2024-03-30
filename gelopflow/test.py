import cv2
from markertracker import MarkerTracker
import threading
import numpy as np
import gelsight_test as gs
class CameraCapture1:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        # rospy.sleep(1)

    def update(self):
        while self.is_running:
            self.ret, self.frame = self.cap.read()
            # cv2.imshow('frame', self.frame)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.is_running = False
        self.cap.release()


if __name__ == '__main__':
    imgw = 640
    imgh = 480
    camera = CameraCapture1()
    ret, f0 = camera.read()
    f0 = cv2.resize(f0, (imgw, imgh))
    f0gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    img = np.float32(f0) / 255.0
    # mtracker = MarkerTracker(img)
    # tracker = MarkerTracker()
    marker_centers = gs.find_markers(f0)
    Ox = marker_centers[:, 1]
    Oy = marker_centers[:, 0]
    nct = len(marker_centers)
    old_gray = f0gray
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))

    # Existing p0 array
    p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
    for i in range(nct - 1):
        # New point to be added
        new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
        # Append new point to p0
        p0 = np.append(p0, new_point, axis=0)
    while True:
        ret, frame = camera.read()
        frame = cv2.resize(frame, (imgw, imgh))
        if ret:
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, cur_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            p0 = good_new.reshape(-1, 1, 2)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                ix  = int(Ox[i])
                iy  = int(Oy[i])
                offrame = cv2.arrowedLine(frame, (ix,iy), (int(a), int(b)), (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
                offrame = cv2.circle(offrame, (int(a), int(b)), 5, color[i].tolist(), -1)
            cv2.imshow('optical flow frame', cv2.resize(offrame, (2*offrame.shape[1], 2*offrame.shape[0])))
            # cv2.imshow('frame', offrame)
            old_gray = cur_gray.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()