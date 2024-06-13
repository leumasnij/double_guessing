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
    return np.mean(tag_center, axis=0)

