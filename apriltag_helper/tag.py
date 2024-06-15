import cv2
import apriltag
import numpy as np

def detect_tag():
# Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initialize the apriltag detector
    options = apriltag.DetectorOptions(families='tag25h9')
    detector = apriltag.Detector(options)
    tag_size = 2.5
    tag_center = []

    while len(tag_center) < 4 or np.sum(np.std(tag_center, axis=0)) > 0.5:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        tags = detector.detect(gray)
        
        # Draw detections on the frame
        for tag in tags:
            corners = tag.corners
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i+1) % 4][0]), int(corners[(i+1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, str(tag.tag_id), (int(corners[0][0]), int(corners[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if len(tag_center) < 4:
                tag_center.append([(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2])
            else:
                tag_center.pop(0)
                tag_center.append([(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2])
            return np.mean(tag_center, axis=0), tag.tag_id
# print(detect_tag())

def img2robot(img_pos):
    matrix = np.load('/home/okemo/samueljin/stablegrasp_ws/src/cali50.npy', allow_pickle=True).item()['img2robot'][0]
    robot_pos = np.dot(matrix, np.array([img_pos[0], img_pos[1], 1]))

    return robot_pos

def verification():
# Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initialize the apriltag detector
    options = apriltag.DetectorOptions(families='tag25h9')
    detector = apriltag.Detector(options)
    tag_size = 2.5
    tag_center = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        tags = detector.detect(gray)
        
        # Draw detections on the frame
        for tag in tags:
            corners = tag.corners
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i+1) % 4][0]), int(corners[(i+1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, str(tag.tag_id), (int(corners[0][0]), int(corners[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            center = [(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2]
            trans_mat = np.load('/home/okemo/samueljin/stablegrasp_ws/src/cali.npy', allow_pickle=True).item()['img2robot'][0]
            robot_pos = np.dot(trans_mat, np.array([center[0], center[1], 1]))
            robot_pos = ((robot_pos[:2]*10000).astype(int)).astype(float)/100
            cv2.putText(frame, str(robot_pos), (int(corners[0][0]), int(corners[0][1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# verification()

if __name__ == '__main__':
    print(detect_tag())
    print(img2robot(detect_tag()))
    verification()  