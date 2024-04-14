import cv2
import threading
import numpy as np
import gelsight_test as gs

class CameraCapture1:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
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
    f0 = gs.resize_crop_mini(f0, imgw, imgh)
    f0gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    # mtracker = MarkerTracker(img)
    # tracker = MarkerTracker()
    marker_centers = gs.find_markers(f0)
    Ox = marker_centers[:, 0]
    Oy = marker_centers[:, 1]
    nct = len(marker_centers)

    lk_params = dict(winSize=(50, 50), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))
    # old_gray = f0gray.copy()
    old_gray = f0.copy()

    # Existing p0 array
    p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
    for i in range(nct - 1):
        # New point to be added
        new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
        # Append new point to p0
        p0 = np.append(p0, new_point, axis=0)
    while True:
        ret, frame = camera.read()
        frame = gs.resize_crop_mini(frame, imgw, imgh)
        # cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_gray = frame.copy()
        if ret:
       
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, cur_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            p0 = good_new.reshape(-1, 1, 2)

            A_mat = []
            b_mat = []
            Mid = []
            half_length = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                ix  = int(Ox[i])
                iy  = int(Oy[i])
                offrame = cv2.arrowedLine(frame, (ix,iy), (int(a), int(b)), (255,255,255), thickness=2, line_type=cv2.LINE_8, tipLength=.25)
                offrame = cv2.circle(offrame, (int(a), int(b)), 10, color[i].tolist(), -1)

            # if len(A_mat) != 0:
            #     A_mat = np.array(A_mat)
            #     b_mat = np.array(b_mat)
            #     ROC = np.linalg.inv(A_mat.T@A_mat)@A_mat.T@b_mat
            #     # print(ROC)
            #     offrame = cv2.circle(offrame, (int(ROC[0]), int(ROC[1])), 5, (0, 255, 0), -1)
            #     cv2.putText(offrame, 'Rotate', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            #     angle = 0
            #     for i in range(len(Mid)):
            #         angle += (np.arctan((half_length[i])/(np.linalg.norm(Mid[i] - ROC)))*180/np.pi)*2
            #         offrame = cv2.line(offrame, (int(Mid[i][0]), int(Mid[i][1])), (int(ROC[0]), int(ROC[1])), (0, 255, 0), 1)
            #     angle = angle/len(Mid)
            #     cv2.putText(offrame, 'Angle: ' + str(angle), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # offrame = cv2.circle(offrame, (int(ROC[0]), int(ROC[1])), 5, (0, 255, 0), -1)

            # cv2.putText(offrame, 'Average Movement: ' + str(movemnt), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(offrame, 'Rotation: ' + str(rotation), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('optical flow frame', cv2.resize(offrame, (2*offrame.shape[1], 2*offrame.shape[0])))
            # cv2.imshow('frame', offrame)
            f0 = frame.copy()
            old_gray = cur_gray.copy()
            # print('frame')
            if cv2.waitKey(1) & 0xFF == ord('s'):
                marker_centers = gs.find_markers(f0)
                Ox = marker_centers[:, 0]
                Oy = marker_centers[:, 1]
                nct = len(marker_centers)
                p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
                for i in range(nct - 1):
                    # New point to be added
                    new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
                    # Append new point to p0
                    p0 = np.append(p0, new_point, axis=0)
    camera.release()
    cv2.destroyAllWindows()