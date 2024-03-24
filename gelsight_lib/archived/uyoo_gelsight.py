import rospy
from datetime import datetime
import csv
import rosbag
import os
import cv2
import time
import utils
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from skimage.io import imsave
from matplotlib import pyplot as plt
import math


from sensor_msgs.msg import Image

def getCurrentTimeStamp():
	now = datetime.now()
	timestamp = time.mktime(now.timetuple())
	return int(timestamp)


'''
Sensor: GelSight Tactile Sensor
Data: Tactile Images
Format: .jpg
'''
class GelSight:
	def __init__(self, object_dir, video=True):
		self.object_dir = object_dir
		self.bridge = CvBridge()

		self.gelsight_count = 0
		self.maxmovement = 0
		self.initialmove = 0
		self.movementratio = 0
		self.complete = False

		# self.gel_path = object_dir
		self.initial_markers = None
		self.cur_markers = None
		self.prev_markers = None
		self.edge = None
		self.f0 = None
		#elf.img = None
		gel_directory = 'gelsight'
		self.gel_path = os.path.join(self.object_dir, gel_directory)
		os.mkdir(self.gel_path)
		
		# First check the usb channel for the input video using 'ls -lrth /dev/video*'
		# Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
		# Launch the file: roslaunch usb_cam usb_cam-test.launch
		#if video:
		self.gel_sub = rospy.Subscriber('/gelsight/usb_cam/image_raw', Image,self.gelSightCallback)

	def gelSightCallback(self, img):
		try:
			self.img = self.bridge.imgmsg_to_cv2(img, 'bgr8')
			timestamp = utils.get_current_time_stamp()
			filename0 = 'unaltered_gelsight_'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
			
			cv2.imwrite(self.gel_path + '/' + filename0, self.img)
			center_coordinates = (320, 240)
			radius = 30
			color = (255, 255, 255)
			thickness = 2
			img_show = self.img.copy()
			#img_show = np.int16(cv2.GaussianBlur(self.img, (101,101), 50))
			# cv2.waitKey(1)

			# track markers
			if self.initial_markers is None:
				self.initial_markers = self.extractMarker(self.img)
				self.prev_markers = self.initial_markers
				self.f0o = np.int16(self.img)
				self.f0=np.int16(cv2.GaussianBlur(self.img, (101,101), 50))
				self.edge = self.f0
				#self.f0 = np.int16(cv2.medianBlur(self.img,5))
				#self.f0=np.int16(self.img)
				
			else:
				self.cur_markers = self.extractMarker(self.img)
				self.cur_markers, marker_motion = self.trackMarker(self.cur_markers, self.prev_markers, self.initial_markers)
				self.prev_markers = self.cur_markers
				img_show = self.img.copy()
				showScale = 2
				self.edge = self.displayMotion(img_show, self.initial_markers, marker_motion, showScale)

			
			filename = 'gelsight_'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
			cv2.imwrite(self.gel_path + '/' + filename, self.edge)
			self.gelsight_count += 1
		except CvBridgeError, e:
			print(e)

	def stopRecord(self):
		self.gel_sub.unregister()




	def __str__(self):
		return 'GelSight'
	def trackMarker(self, marker_present, marker_prev, marker_init):
		markerCount = len(marker_init)
		Nt  = len(marker_present)

		marker_motion = np.zeros((markerCount,2))
		no_seq2 = np.zeros(Nt)
		center_now = np.zeros([markerCount, 3])

		for i in range(Nt):
		    dif=np.abs(marker_present[i,0]-marker_prev[:,0])+np.abs(marker_present[i,1]-marker_prev[:,1])
		    no_seq2[i]=np.argmin(dif*(100+np.abs(marker_present[i,2]-marker_init[:,2])))

		for i in range(markerCount):
		    dif=np.abs(marker_present[:,0]-marker_prev[i,0])+np.abs(marker_present[:,1]-marker_prev[i,1])
		    t=dif*(100+np.abs(marker_present[:,2]-marker_init[i,2]))
		    a=np.amin(t)/100
		    b=np.argmin(t)

		    if marker_init[i,2]<a:   # for small area
		        center_now[i]=marker_prev[i]
		    elif i==no_seq2[b]:
		        marker_motion[i] = marker_present[b,0:2] - marker_init[i,0:2]
		        center_now[i] = marker_present[b]
		    else:
		        center_now[i] = marker_prev[i]

		return center_now, marker_motion


	def extractMarker(self, img):
		markerThresh=-40
		areaThresh1=2
		areaThresh2=900
		img_gaussian = np.int16(cv2.GaussianBlur(img, (101,101), 50))
		I = img.astype(np.double)-img_gaussian.astype(np.double)
		markerMask = ((np.max(I,2))<markerThresh).astype(np.uint8)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
		markerMask = cv2.morphologyEx(markerMask, cv2.MORPH_CLOSE, kernel)
		MarkerCenter=np.empty([0, 3])
		cnts, hierarchy = cv2.findContours(markerMask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for contour in cnts:
			AreaCount=cv2.contourArea(contour)
			if AreaCount>areaThresh1 and AreaCount<areaThresh2:
				t=cv2.moments(contour)
				MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)
		return MarkerCenter



	def displayMotion(self, img, marker_init, marker_motion, showScale):
		point2 = None
		timestamp = utils.get_current_time_stamp()
		img_gaussian = np.int16(cv2.GaussianBlur(img, (101,101), 50))
		img_diff = np.float32(img - self.f0)
		hsv = cv2.cvtColor(img_diff, cv2.COLOR_BGR2HSV)
		lower_color_bounds = np.array([50, 1, 35], np.uint8)
		upper_color_bounds = np.array([255, 255, 255], np.uint8)
		mask = cv2.inRange(hsv, lower_color_bounds, upper_color_bounds)
		kernel = np.ones((4,4),np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		kernel = np.ones((35,35),np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		#mask = mask[0:mask.shape[0], 0:mask.shape[1]-125]
		#img_diff_p = img_diff[0:img_diff.shape[0], 0+50:img_diff.shape[1]-125]
		contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		for c in contours:
			area = cv2.contourArea(c)
			perimeter = cv2.arcLength(c, False)
			if 500 < perimeter:
			#second approch - cv2.minAreaRect --> returns center (x,y), (width, height), angle of rotation )
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				img= cv2.drawContours(img ,[box],0,(0,0,255),2)
				#print(rect[1])
				alpha = rect[2]
				wh = rect[1]
				if wh[0] < wh[1]:
					alpha = np.int32(rect[2])
				else:
					alpha = np.int32(rect[2]) + 90
				point1, point2 = self.get_line_points(rect[0], alpha)
				point1 =  (np.int32(point1[0]), np.int32(point1[1]))
				point2 =  (np.int32(point2[0]), np.int32(point2[1]))
				img = cv2.line(img, point1, point2, (0,255,0), 3)
		#Marker tracking
		self.loc_markerArea()
		mask2=self.detectContactArea()
		markerCenter=np.around(marker_init[:,0:2]).astype(np.int16)
		for i in range(marker_init.shape[0]):
			if marker_motion[i,0]!=0:
				cv2.arrowedLine(img,(markerCenter[i,0], markerCenter[i,1]), \
					(int(marker_init[i,0]+marker_motion[i,0]*showScale), int(marker_init[i,1]+marker_motion[i,1]*showScale)),\
					(255,255,255),2)
	    #Marker trackin
		cv2.imshow('img2', img)
		cv2.waitKey(1)
		filecontour = 'contoured'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
		cv2.imwrite(self.gel_path + '/' + filecontour, img)
		filemask = 'mask'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
		cv2.imwrite(self.gel_path + '/' + filemask, mask)
		#Get the largest mover marker and store
		if point2 is not None:
			maxtop, maxbot, euc = self.largestMovers(marker_motion, markerCenter, point1[1])
			self.movementratio = maxtop/maxbot
		#cv2.imshow('video', mask)
		#cv2.waitKey(1)
		#self.edge = edges
		filenamediff = 'diff'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
		cv2.imwrite(self.gel_path + '/' + filenamediff, img_diff)
		return mask
	def getratio(self):
		return self.movementratio
	def setinitial(self):
		self.initialmove = self.maxmovement

	def largestMovers(self, marker_motion,centers, y):
		#Find the indexes of the 3 dots that moved the most
		euc = marker_motion[:,0]**2 + marker_motion[:,1]**2
		euc2 = euc.copy()
		invalidlocs = np.argwhere(centers[:,1]>y)
		validlocs = np.argwhere(centers[:,1]<y)
		euc[invalidlocs]= 0
		euc2[validlocs] = 0

		maxtopindex = np.argmax(euc, axis=0)
		maxtop = euc[maxtopindex]
		maxbotindex = np.argmax(euc2, axis=0)
		maxbot = euc2[maxbotindex]
		return maxtop, maxbot, euc

	def get_line_points(self, point, angle):
		length = 200
		theta = math.radians(angle)
		start_point = (point[0]+length*math.cos(theta), point[1]+length*math.sin(theta))
		end_point = (point[0]-length*math.cos(theta), point[1]-length*math.sin(theta))
		return start_point, end_point
	def loc_markerArea(self):
		'''match the area of the markers; work for the Bnz GelSight'''
		MarkerThresh=-30
		self.diff = np.array(self.img)-self.f0
		#print(np.amax(self.diff))
		#print(np.amin(self.diff))
		self.max_img = np.amax(self.diff,2)
		self.MarkerMask = self.max_img<MarkerThresh

	def detectContactArea(self):
		diffim=self.diff
		contactmap=self.max_img
		countnum=(contactmap>10).sum()
		contactmap[contactmap<10]=0
		image = np.zeros((480,640,3), np.uint8)
		maxC = np.max(contactmap)
		sec90C = np.percentile(contactmap, 90)
		sec95C = np.percentile(contactmap, 95)
		sec99C = np.percentile(contactmap, 99)
		contact_mask = contactmap>0.4*sec90C
		total_contact = np.sum(contact_mask)
		self.contact_ratio = total_contact/(image.shape[0]*image.shape[1])
		image[contact_mask,0] = 255
		image[contact_mask,1] = 255
		image[contact_mask,2] = 255
		return image


def start_data_collection(trial_dir):

	tactile = GelSight(trial_dir, video = False)

	return tactile
