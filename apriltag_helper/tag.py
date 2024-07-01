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
def CoM_calulation(cap, num_tag):
    
    enought_tag = False
    while not enought_tag:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families='tag25h9')
        detector = apriltag.Detector(options)
        tags = detector.detect(gray)
        if len(tags) == num_tag:
            enought_tag = True
    main_tag = None
    main_center = np.zeros(3)

    CoM = np.zeros(3)
    total_weight = 0
    tag1_lookfor2_flag = False
    
    for tag in tags:
        
        corners = tag.corners
        center = [(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2]
        center = np.array([center[0], center[1]])
        # print(tag.tag_id, center)
        center = img2robot(center)
        
        if tag1_lookfor2_flag and tag.tag_id == 2:
            CoM = np.array([(center[0]+main_center[0])/2, (center[1]+main_center[1])/2, 0.25])
            main_center = (center + main_center)/2
            tag1_lookfor2_flag = False
            total_weight += 185.29
            continue
        
        if main_tag is None and (tag.tag_id == 0 or tag.tag_id == 1):
            main_tag = tag.tag_id
            
            if tag.tag_id == 0:
                CoM = np.array([center[0]-0.075, center[1]-0.075, 0.234])
                total_weight = 115.62
                main_center = center - np.array([0.075, 0.075, 0])
            else:
                CoM = np.array([center[0]+0.075, center[1]-0.075, 0.25])
                main_center = center
                tag1_lookfor2_flag = True
            continue
        elif main_tag is not None and (tag.tag_id == 0 or tag.tag_id == 1):
            raise ValueError("Multiple main tags detected")
        
        if tag.tag_id == 4:
            Obj_CoM = np.array([center[0], center[1], 0.254])
            weight = 121.83
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
        if tag.tag_id == 6:
            Obj_CoM = np.array([center[0], center[1], 0.23])
            weight = 161.29
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
        if tag.tag_id == 8:
            Obj_CoM = np.array([center[0], center[1], 0.23])
            weight = 200
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
        if tag.tag_id == 9:
            Obj_CoM = np.array([center[0], center[1], 0.23])
            weight = 500
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
    
    cv2.imwrite('frame.jpg', frame)
    return main_tag, main_center, CoM



        


def img2robot(img_pos):
    matrix = np.load('/home/okemo/samueljin/stablegrasp_ws/src/cali50.npy', allow_pickle=True, encoding= 'latin1').item()['img2robot'][0]
    robot_pos = np.dot(matrix, np.array([img_pos[0], img_pos[1], 1]))

    return robot_pos

def robot2img(robot_pos):
    matrix = np.load('/home/okemo/samueljin/stablegrasp_ws/src/cali50.npy', allow_pickle=True, encoding= 'latin1').item()['robot2img'][0]
    img_pos = np.dot(matrix, np.array([robot_pos[0], robot_pos[1], 1]))

    return img_pos

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
        
        main_id, main_center, CoM = CoM_calulation(cap, 3)
        main_center = robot2img(main_center)[:2].astype(int)
        CoM = robot2img(CoM).astype(int)
        # print(main_center, CoM)
        for tag in tags:
            # print(tag.tag_id)
            corners = tag.corners
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i+1) % 4][0]), int(corners[(i+1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, str(tag.tag_id), (int(corners[0][0]), int(corners[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # center = [(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2]
            # trans_mat = np.load('/home/okemo/samueljin/stablegrasp_ws/src/cali.npy', allow_pickle=True, encoding= 'latin1').item()['img2robot'][0]
            # robot_pos = np.dot(trans_mat, np.array([center[0], center[1], 1]))
            # robot_pos = ((robot_pos[:2]*10000).astype(int)).astype(float)/100
            # cv2.putText(frame, str(robot_pos), (int(corners[0][0]), int(corners[0][1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'main id: ' + str(main_id), (int(main_center[0]), int(main_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, (int(CoM[0]), int(CoM[1])), 5, (0, 0, 255), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# verification()

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # cap.release()
    # print(CoM_calulation(frame))
    # print(img2robot(detect_tag()))
    verification()  