import cv2
import apriltag
import numpy as np

def combine_moi(mass1, mass2, moi_1, moi_2, com1, com2):
    # Calculate the combined center of mass
    total_mass = mass1 + mass2
    combined_com = (mass1 * com1 + mass2 * com2) / total_mass

    if len(moi_1) == 6:
        moi_1 = np.array([[moi_1[0], moi_1[3], moi_1[4]], [moi_1[3], moi_1[1], moi_1[5]], [moi_1[4], moi_1[5], moi_1[2]]])
    if len(moi_2) == 6:
        moi_2 = np.array([[moi_2[0], moi_2[3], moi_2[4]], [moi_2[3], moi_2[1], moi_2[5]], [moi_2[4], moi_2[5], moi_2[2]]])

    # Calculate the distance of each body's CoM from the combined CoM
    r1 = com1 - combined_com
    r2 = com2 - combined_com

    # Parallel axis theorem for each body
    moi1 = moi_1 + mass1 * (np.dot(r1, r1) * np.eye(3) - np.outer(r1, r1))
    moi2 = moi_2 + mass2 * (np.dot(r2, r2) * np.eye(3) - np.outer(r2, r2))

    # Combined moment of inertia tensor
    combined_moi_tensor = moi1 + moi2

    # Extracting the unique elements as a vector
    combined_moi_vector = [combined_moi_tensor[0, 0], combined_moi_tensor[1, 1], combined_moi_tensor[2, 2], 
                           combined_moi_tensor[0, 1], combined_moi_tensor[0, 2], combined_moi_tensor[1, 2]]
    return combined_moi_vector

class BoxWithoutLid:
    def __init__(self, mass, dimensions, thickness):
        self.mass = mass
        self.length, self.width, self.height = dimensions
        self.thickness = thickness
        self.density = self.mass / self.volume()
    
    def volume(self):
        # Volume of the 5 walls
        return 2 * self.thickness * self.height * (self.length + self.width) + self.thickness * self.length * self.width

    def plane_moi(self, m, l, w):
        # Moment of inertia for a thin rectangular plane about its center
        I_xx = (1/12) * m * (w**2)
        I_yy = (1/12) * m * (l**2)
        I_zz = (1/12) * m * (l**2 + w**2)
        return I_xx, I_yy, I_zz

    def moi(self):
        I = np.zeros((3, 3))
        
        # Wall masses
        wall_mass = self.density * self.thickness * self.height * self.length
        wall_mass_width = self.density * self.thickness * self.height * self.width
        bottom_mass = self.density * self.thickness * self.length * self.width

        # MoI for each wall about the box's center of mass (CoM)
        # Two long side walls
        I_xx_wall, I_yy_wall, I_zz_wall = self.plane_moi(wall_mass, self.length, self.height)
        # Parallel axis theorem to move MoI to the box CoM
        I_yy_wall += wall_mass * (self.width / 2) ** 2
        I_zz_wall += wall_mass * (self.width / 2) ** 2
        
        I += np.array([[I_xx_wall, 0, 0], [0, I_yy_wall, 0], [0, 0, I_zz_wall]])
        I += np.array([[I_xx_wall, 0, 0], [0, I_yy_wall, 0], [0, 0, I_zz_wall]])

        # Two short side walls
        I_xx_wall_w, I_yy_wall_w, I_zz_wall_w = self.plane_moi(wall_mass_width, self.width, self.height)
        # Parallel axis theorem to move MoI to the box CoM
        I_yy_wall_w += wall_mass_width * (self.length / 2) ** 2
        I_zz_wall_w += wall_mass_width * (self.length / 2) ** 2
        
        I += np.array([[I_xx_wall_w, 0, 0], [0, I_yy_wall_w, 0], [0, 0, I_zz_wall_w]])
        I += np.array([[I_xx_wall_w, 0, 0], [0, I_yy_wall_w, 0], [0, 0, I_zz_wall_w]])

        # Bottom
        I_xx_bottom, I_yy_bottom, I_zz_bottom = self.plane_moi(bottom_mass, self.length, self.width)
        # Parallel axis theorem to move MoI to the box CoM
        I_xx_bottom += bottom_mass * (self.height / 2) ** 2
        I_yy_bottom += bottom_mass * (self.height / 2) ** 2
        
        I += np.array([[I_xx_bottom, 0, 0], [0, I_yy_bottom, 0], [0, 0, I_zz_bottom]])
        
        return I

    def moi_vector(self):
        I = self.moi()
        I_vector = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        return I_vector
    
class Cylinder:
    def __init__(self, mass, radius, height):
        self.mass = mass
        self.radius = radius
        self.height = height

    def moi_tensor(self):
        I_xx = I_yy = (1/12) * self.mass * (3 * self.radius**2 + self.height**2)
        I_zz = 0.5 * self.mass * self.radius**2
        return np.array([[I_xx, 0, 0], [0, I_yy, 0], [0, 0, I_zz]])

    def moi_vector(self):
        I = self.moi_tensor()
        I_vector = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        return I_vector
    
class Weight_:
    def __init__(self, mass, radius1, radius2, radius3, height1, height2, height3):
        volume1 = np.pi * radius1**2 * height1
        volume2 = np.pi * radius2**2 * height2
        volume3 = np.pi * radius3**2 * height3
        mass1 = mass * volume1 / (volume1 + volume2 + volume3)
        mass2 = mass * volume2 / (volume1 + volume2 + volume3)
        mass3 = mass * volume3 / (volume1 + volume2 + volume3)
        self.cyc1 = Cylinder(mass1, radius1, height1)
        self.cyc2 = Cylinder(mass2, radius2, height2)
        self.cyc3 = Cylinder(mass3, radius3, height3)
        self.moi1 = self.cyc1.moi_vector()
        self.moi2 = self.cyc2.moi_vector()
        self.moi3 = self.cyc3.moi_vector()
        self.com1 = np.array([0, 0, height1/2])
        self.com2 = np.array([0, 0, height1 + height2/2])
        self.com3 = np.array([0, 0, height1 + height2 + height3/2])
        self.com = (self.com1 * mass1 + self.com2 * mass2 + self.com3 * mass3) / mass
        self.moi = combine_moi(mass1, mass2, self.moi1, self.moi2, self.com1, self.com2)
        self.moi = combine_moi(mass, mass3, self.moi, self.moi3, self.com, self.com3)

    def moi_(self):
        return self.moi

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
    no_grasp_zone = []
    moi = np.zeros(6)
    
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
                moi = BoxWithoutLid(0.18529, [0.15, 0.15, 0.085], 0.003).moi_vector()
            continue
        elif main_tag is not None and (tag.tag_id == 0 or tag.tag_id == 1):
            raise ValueError("Multiple main tags detected")
        
        if tag.tag_id == 4:
            Obj_CoM = np.array([center[0], center[1], 0.254])
            weight = 121.83
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
            no_grasp_zone.append([Obj_CoM[0] - 0.025, Obj_CoM[0] + 0.025, Obj_CoM[1] - 0.025, Obj_CoM[1] + 0.025, 0 , 0.272])
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
            no_grasp_zone.append([Obj_CoM[0] - 0.06, Obj_CoM[0] + 0.06, Obj_CoM[1] - 0.06, Obj_CoM[1] + 0.06, 0 , 0.272])
            obj_moi = Weight_(0.2, 0.018, 0.008, 0.013, 0.22, 0.04, 0.06).moi_()
            moi = combine_moi(total_weight, 0.2, moi, obj_moi, CoM, Obj_CoM)
        if tag.tag_id == 9:
            Obj_CoM = np.array([center[0], center[1], 0.23])
            weight = 500
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
        if tag.tag_id == 10 or tag.tag_id == 12:
            Obj_CoM = np.array([center[0], center[1], 0.23])
            weight = 100
            CoM = (CoM*total_weight + Obj_CoM*weight)/(total_weight + weight)
            total_weight += weight
            no_grasp_zone.append([Obj_CoM[0] - 0.06, Obj_CoM[0] + 0.06, Obj_CoM[1] - 0.06, Obj_CoM[1] + 0.06, 0 , 0.265])
            
    cv2.imwrite('frame.jpg', frame)
    
    return main_tag, main_center, CoM, no_grasp_zone, moi
        


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
        
        main_id, main_center, CoM = CoM_calulation(cap, 1)
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