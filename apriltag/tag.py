import cv2
import apriltag

def detect_tag():
# Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initialize the apriltag detector
    options = apriltag.DetectorOptions(families='tag25h9')
    detector = apriltag.Detector(options)
    tag_size = 2.5
    tag_center = [0, 0]

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
        
        # Display the resulting frame
        cv2.imshow('AprilTag Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
