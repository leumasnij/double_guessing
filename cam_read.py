import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import rospy

class RealSenseCam(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break
        if not self.found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
    
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def __del__(self):
        self.pipeline.stop()
# Configure depth and color streams

class CameraCapture(object):
    def __init__(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        self.record = False
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        while self.is_running:
            self.ret, self.frame = self.cap.read()
            while not self.ret:
               self.ret, self.frame = self.cap.read()
            if self.record:
                self.out.write(self.frame)

    def read(self):
        return self.ret, self.frame.copy()
    def start_record(self, out_adr):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(out_adr, fourcc, 10.0, self.frame.shape[1::-1])
        self.record = True

    def end_record(self):
        self.record = False
        rospy.sleep(1)
        if self.out.isOpened():
           self.out.release()

    def release(self):
        self.is_running = False
        rospy.sleep(1)
        self.cap.release()