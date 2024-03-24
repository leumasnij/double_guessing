#! /usr/bin/env python

'''
Basic functions for processing GelSight images. Including saving images, calculating and displaying marker motions, detect contact.
Before calling the classes, 'GelSight_driver' should be launched, which publish '/gelsight/image_raw' topic for the images from the camera

Wenzhen Yuan (yuanwenzhen@gmail.com)  Feb, 2017
'''

'''
This is also a version of the algorithm but help in making a video of the processed images.
'''

# import rospy
import cv2
# from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import Image
# import time
import numpy as np
import os
import copy
import pathlib
import json
import pandas as pd
from cv2 import aruco
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from random import randrange
import time
import math
from scipy import ndimage
import numba
import threading
import shutil
from norm_distribution import fit_norm, plot_norm
from OpticalFlow import *
from draw_perpendicular import *
from skimage.color import rgb2hsv
from scipy import ndimage

global translation
translation = False

@numba.jit(parallel = True,cache=True, fastmath=True)
def find_max(img, thresh):
    mask = np.amax(img, 2)<thresh
    return mask

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class GelSight_Img(object):

    #Checks for version of OpenCV
    def check_opencv_version(self, major, lib=None):
        if lib is None:
            import cv2 as lib
        return lib.__version__.startswith(major)

    #Check for OpenCV 2.x
    def is_cv2(self):
        return self.check_opencv_version("2.")

    #Check for OpenCV 3.x
    def is_cv3(self):
        return self.check_opencv_version("3.")
    #Check for OpenCV 4.x
    def is_cv4(self):
        return self.check_opencv_version("4.")

    #Used to create a black canvas in the size of the GelSight Image.
    def createVisualization(self):

        visualizationImage = np.zeros((self.img.shape[0], self.img.shape[1],3), np.uint8)
        #Making it aesthetic black.
        visualizationImage[:,:] = (40,40,40)

        self.img = visualizationImage



    def loc_markerArea(self):
        '''match the area of the markers; work for the Bnz GelSight'''
        MarkerThresh=-30
        # I=self.img.astype(np.double)-self.f0
        self.diff = np.array(self.img)-self.f0
        self.max_img = np.amax(self.diff,2)
        self.MarkerMask = self.max_img<MarkerThresh
        # self.MarkerMask = np.amax(I,2)<MarkerThresh
        # self.MarkerMask = find_max(I, MarkerThresh)


    def drawBox(self, cX, cY, disIm):

        delta = self.markerDelta

        x1 = cX-delta
        x2 = cX+delta

        y1 = cY-delta
        y2 = cY+delta

        cv2.line(disIm,(int(x1), int(y1)), \
                                    (int(x1), int(y2)),\
                                    (255, 255, 0),2)
        cv2.line(disIm,(int(x1), int(y1)), \
                            (int(x2), int(y1)),\
                            (255, 255, 0),2)
        cv2.line(disIm,(int(x2), int(y1)), \
                            (int(x2), int(y2)),\
                            (255, 255, 0),2)
        cv2.line(disIm,(int(x1), int(y2)), \
                            (int(x2), int(y2)),\
                            (255, 255, 0),2)

        return disIm

    def putBars(self, disIm, angle, orientation):

        # h = 480
        # w = 640


        h1, w1, _ = disIm.shape

        h = int(h1/12)
        w = int(w1/2)

        visualizationImage = np.zeros((h, w1, 3), np.uint8)
        #Making it aesthetic black.
        visualizationImage[:,:,:] = (40,40,0)

        # print(orientation)

        if(orientation != None):

            if (angle>10):

                if(orientation == 'CW'):
                    visualizationImage[:,int(w/2):w,:] = (0,0,255)
                    cv2.putText(visualizationImage, orientation, (int(0.75*w), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # print("1")
                elif(orientation == 'CCW'):
                    visualizationImage[:,0:int(w/2),:] = (0,0,255)
                    cv2.putText(visualizationImage, orientation, (int(w/4), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # print("2")


            else:

                if (orientation == 'CW'):

                    y = int((((w/2)-1)/10)*angle + (w/2))
                    visualizationImage[:,int(w/2):y,:] = (0,255,255)
                    cv2.putText(visualizationImage, orientation, (int(0.75*w), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 9, 229), 2)
                    # print("3")


                elif (orientation == 'CCW'):

                    y = int((((w/2)-1)/10)*(10-angle))
                    visualizationImage[:,y:int(w/2),:] = (0,255,255)
                    cv2.putText(visualizationImage, orientation, (int(w/4), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 9, 229), 2)
                    # print("4")

            cv2.line(visualizationImage,(int(w/2-1), int(0)), \
                                        (int(w/2-1), int(h-1)),\
                                        (20, 9, 229),1)

        elif(orientation == None):
            print()
            # print()

        if(not self.firstContact):
            cv2.putText(visualizationImage, "No contact yet!", (int(0.75*w1), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif(self.firstContact and not self.rotationOnset):
            cv2.putText(visualizationImage, "Contact made, no rotation yet!", (int(0.75*w1)-75, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif(self.rotationOnset):
            cv2.putText(visualizationImage, "Rotation!!! Angle:"+ str(angle), (int(0.75*w1)-75, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img = np.concatenate((disIm, visualizationImage), axis = 0)

        return img

    def drawMarkers(self):

        significantMotion = False

        markerCenter=np.around(self.flowcenter[:,0:2]).astype(np.int16)

        if (significantMotion):

            for i in range(self.MarkerCount):

                if self.markerU[i]!=0 and i in self.indexSignificantMotion:

                    cv2.arrowedLine(self.disIm,(markerCenter[i,0], markerCenter[i,1]), \
                        (int(self.flowcenter[i,0]+self.markerU[i]*self.showScale), int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)),\
                        (0, 255, 255),2)

        else:

            for i in range(self.MarkerCount):

                if self.markerU[i]!=0:

                    cv2.arrowedLine(self.disIm,(markerCenter[i,0], markerCenter[i,1]), \
                        (int(self.flowcenter[i,0]+self.markerU[i]*self.showScale), int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)),\
                        (0, 255, 255),2)

        # return disIm

    def drawContactVectors(self):

        markerCenter=np.around(self.flowcenter[:,0:2]).astype(np.int16)

        #Contact point marker vectors.
        # for i in self.contactMarkers:
        #
        #     x1, y1 = self.flowcenter[i,0], self.flowcenter[i,1]
        #     x2, y2 = self.contactLoc[i,0], self.contactLoc[i,1]
        #
        #     cv2.arrowedLine(self.disIm,(int(x1), int(y1)), \
        #         (int(x2), int(y2)),\
        #         (0, 255, 0),2)

        #Drawing markers in the cotact area.
        for i in range(self.MarkerCount):

            if i in self.contactMarkers:

                if self.markerU[i]!=0:

                    cv2.arrowedLine(self.disIm,(markerCenter[i,0], markerCenter[i,1]), \
                        (int(self.flowcenter[i,0]+self.markerU[i]*self.showScale), int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)),\
                        (0, 0, 255),2)
        # return disIm

    def drawCORvectors(self):


        for i in self.contactMarkers:

            # if i in self.indexSignificantMotion:


            # x1, y1 = self.flowcenter[i,0], self.flowcenter[i,1]
            # x2, y2 = self.contactLoc[i,0], self.contactLoc[i,1]
            # xcv1, ycv1 = self.markerStart[i]
            # xcv2, ycv2 = self.markerPresent[i]
            if self.markerU[i] == 0:
                continue
            xcv2 = self.flowcenter[i,0]+self.markerU[i]
            ycv2 = self.flowcenter[i,1]+self.markerV[i]

            xc = self.xVec[0]
            yc = self.xVec[1]

            # cv2.line(self.disIm,(int(xc), int(yc)), \
            #     (int(xcv1), int(ycv1)),\
            #     (64, 0, 255),1)

            cv2.line(self.disIm,(int(xc), int(yc)), \
                (int(xcv2), int(ycv2)),\
                (131, 228, 252),1)
            # (131, 228, 252)
        cv2.circle(self.disIm, (self.xVec[0], self.xVec[1]), 5, (0, 255, 255), -1)



        # xy = self.rotationIndices

        # for i in xy:

        #   xcv1, ycv1 = self.markerStart[i]
        #   xcv2, ycv2 = self.markerPresent[i]

        #   xc = self.xVec[0]
        #   yc = self.xVec[1]

        #   cv2.arrowedLine(disIm,(int(xc), int(yc)), \
        #       (int(xcv1), int(ycv1)),\
        #       (0, 0, 255),2)

        #   cv2.arrowedLine(disIm,(int(xc), int(yc)), \
        #       (int(xcv2), int(ycv2)),\
        #       (0, 255, 0),2)

        # return disIm

    def addToDictionary(self):

        self.dataDict[self.frameIndex] = {
        "flowcenter":       self.flowcenter,
        "markerPresent":    self.markerPresent,
        "markerMagnitude":  self.markerMagnitude,
        "markerAngle":      self.markerAngle,
        "indexPresent":     self.indexPresent,
        "indexPast":        self.indexPast,
        "markerStart":      self.markerStart,
        "indexContactPixels":       self.indexContactPixels,
        "indexSignificantMotion":   self.indexSignificantMotion,
        "xVec": self.xVec
        }

    def show_img_cb(self,event):
        try:
            cv2.imshow("ProcessedImage"+" "+ str(self.frameIndex), self.imgProcess)
            # cv2.resizeWindow('Processed Image', 50, 50)
            # cv2.imshow("Processed_Image",self.display_image)
            cv2.waitKey(3)
        except:
            pass


    def displayIm(self, angleOfRotation = 0, orientation = None):

        # self.createVisualization()
        self.disIm = copy.deepcopy(self.img)

        #Flowcenter
        markerCenter=np.around(self.flowcenter[:,0:2]).astype(np.int16)

        self.markerDelta = 10

        # for i in range(self.MarkerCount):

        #   cX = int(self.flowcenter[i,0]+self.markerU[i]*self.showScale)
        #   cY = int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)

        #   disIm = self.drawBox(cX, cY, disIm)

        # multi-thread
        threads = []

        #Before Contact Detection
        if (not self.firstContact and not self.rotationOnset):

            # print("1")

            self.drawMarkers()

        if(self.firstContact and not self.rotationOnset):

            # print("2")

            self.drawMarkers()
            self.drawContactVectors()


        if (self.rotationOnset and not self.badRotation):

            # print("3")
            # t1 = threading.Thread(target=self.drawMarkers)
            # t2 = threading.Thread(target=self.drawContactVectors)
            # t3 = threading.Thread(target=self.drawCORvectors)
            # # t2 = thread.start_new_thread( self.drawContactVectors )
            # # t3 = thread.start_new_thread( self.drawCORvectors )
            # threads.append(t1)
            # threads.append(t2)
            # threads.append(t3)
            # t1.start()
            # t2.start()
            # t3.start()
            # for t in threads:
            #     t.join()
            # start = time.time()
            self.drawCORvectors()
            # end = time.time()
            # print("drawCORvectors : " + str(end-start))
            # start = time.time()
            self.drawMarkers()
            # end = time.time()
            # print("drawMarkers : " + str(end-start))
            #
            # start = time.time()
            self.drawContactVectors()
            # end = time.time()
            # print("drawContactVectors : " + str(end-start))
            #
            motion_start = self.flowcenter[self.indexSignificantMotion,:]

            motionU = self.markerU[self.indexSignificantMotion]
            motionV = self.markerV[self.indexSignificantMotion]
            img = self.disIm
            # draw_perpendicular(motion_start,motionU,motionV,img)


        if (self.rotationOnset and self.badRotation):

            # print("4")

            self.drawMarkers()
            self.drawContactVectors()

            motion_start = self.flowcenter[self.indexSignificantMotion,:]

            motionU = self.markerU[self.indexSignificantMotion]
            motionV = self.markerV[self.indexSignificantMotion]
            img = self.disIm
            # draw_perpendicular(motion_start,motionU,motionV,img)


        self.imgProcess = self.disIm


        #Displays images with the processed and old image.
        imgNewOld = np.concatenate((self.disIm, self.img), axis = 1)
        #Displays the yellow bars of the angle rotated.
        imgOrient = self.putBars(imgNewOld, (round(angleOfRotation,2)), str(orientation))

        # cv2.circle(self.disIm, (self.xVec[0], self.xVec[1]), 7, (0, 255, 255), -1)

        # self.pub.publish(self.bridge.cv2_to_imgmsg(disIm, "bgr8"))

        if (self.startDisplay or self.rotationOnset):

            # cv2.putText(self.disIm, "Mean:   " + str(round(angleOfRotation,2)), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            # cv2.putText(self.disIm, "Orientation:   " + str(orientation), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29,223,217), 1)

            #Displays images with the processed and old image.
            imgNewOld = np.concatenate((self.disIm, self.img), axis = 1)
            #Displays the yellow bars of the angle rotated.
            imgOrient = self.putBars(imgNewOld, (round(angleOfRotation,2)), str(orientation))
            # cv2.imshow("ProcessedImage"+" "+ str(self.frameIndex), self.imgProcess)
            # Specify the amount of milli seconds for the image to be displayed.
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #   cv2.destroyAllWindows()
            self.startDisplay = False
        # cv2.destroyAllWindows()


        imgcheck = np.zeros((1080, 1920,3), np.uint8)
        #Making it aesthetic black.
        imgcheck[:,:,:] = (40,40,40)

        imgcheck[280:800, 320:1600, :] = imgOrient

        global stop
        if(self.frameIndex<(stop-1)):

            #Adds the current frame data to the dictionary values.
            if (self.frameIndex==self.contactIndex):
                self.dataDict["contactIndex"] = self.contactIndex
                self.dataDict["contactIndices"] = self.contactMarkers



            # self.addToDictionary()
            self.dataDict[self.frameIndex] = {
            "flowcenter":       self.flowcenter,
            "markerPresent":    self.markerPresent,
            "markerMagnitude":  self.markerMagnitude,
            "markerAngle":      self.markerAngle,
            "indexPresent":     self.indexPresent,
            "indexPast":        self.indexPast,
            "markerStart":      self.markerStart,
            "indexContactPixels":       self.indexContactPixels,
            "indexSignificantMotion":   self.indexSignificantMotion,
            "xVec": self.xVec,
            "rotationAngle": self.angleOfRotation
            }

            self.rotationAngles.append(self.angleOfRotation)
            self.out.write(imgcheck)
            # print("Writing")
            cv2.imwrite(self.resulPath+str(self.frameIndex)+'.jpg', imgcheck)
            cv2.imwrite(self.resultPathC+str(self.frameIndex)+'.jpg', self.disIm)


        if(self.frameIndex==(stop-1)):
            # print(self.dataDict)
            # saveFolderPath = self.dataPath + "dictionaryData/"
            saveFolderPath = self.raw_resultPath
            filename = self.dataPath.split('/')[-2]
            # print(filename)
            # if not os.path.isdir(saveFolderPath):
            #     os.mkdir(saveFolderPath)
            #     print ('folder made')
            self.dataDict[self.frameIndex] = {
            "flowcenter":       self.flowcenter,
            "markerPresent":    self.markerPresent,
            "markerMagnitude":  self.markerMagnitude,
            "markerAngle":      self.markerAngle,
            "indexPresent":     self.indexPresent,
            "indexPast":        self.indexPast,
            "markerStart":      self.markerStart,
            "indexContactPixels":       self.indexContactPixels,
            "indexSignificantMotion":   self.indexSignificantMotion,
            "xVec": self.xVec,
            "rotationAngle": self.angleOfRotation
            }
            # self.rotationAngles.append(round(angleOfRotation,2))
            self.dataDict["numFrames"] = self.frameIndex+1
            self.dataDict["rotationOnsetIndex"] = self.rotationOnsetIndex
            self.dataDict["rotationAngles"] = np.array(self.rotationAngles)
            np.save(saveFolderPath + str(filename)+'.npy', self.dataDict)
            # np.save('data.npy', self.dataDict)

            # with open('my_dict.json', 'w') as f:
            #   json.dump(self.dataDict, f)
            # jsonFile = json.dumps(self.dataDict, cls=NumpyEncoder)
            # f = open("dataDict.json", "w")
            # f.write(jsonFile)
            # f.close()
            print("Writing to dictionary done.")
            cv2.destroyAllWindows()
            # self.out.release()
            print("Finish")
            cv2.imwrite(self.resulPath + str(self.frameIndex)+'.jpg', imgcheck)
            cv2.imwrite(self.resultPathC+str(self.frameIndex)+'.jpg', self.disIm)




        self.frameIndex += 1

    def find_markers(self):
        self.loc_markerArea()
        areaThresh1=50
        areaThresh2=400
        #50, 400 are default values
        MarkerCenter=np.empty([0, 3])
        # img_grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.blur(img_grey, (3,3))
        # edges = cv2.Canny(img_blur,10,50)
        # thresh = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv2.THRESH_BINARY,11,2)

        # plt.imshow(edges);plt.show()

        # contours =cv2.findContours(self.MarkerMask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #In OpenCV 2 & 4, the contours can beget from the first return value.
        if self.is_cv2() or self.is_cv4():
            (cnts, _) = cv2.findContours(self.MarkerMask.astype(np.uint8), #Input Image
                                  cv2.RETR_EXTERNAL,           #Contour Retrieval Method
                                  cv2.CHAIN_APPROX_SIMPLE)     #Contour Approximation Method

        #In OpenCV 3, the contours can beget from the second return calue.
        elif self.is_cv3():
            (_, cnts, _) = cv2.findContours(self.MarkerMask.astype(np.uint8), #Input Image
                                  cv2.RETR_EXTERNAL,           #Contour Retrieval Method
                                  cv2.CHAIN_APPROX_SIMPLE)     #Contour Approximation Method

        # _, contours, _ =cv2.findContours(self.MarkerMask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts
        self.contourMask = np.zeros((self.img.shape[0],self.img.shape[1]))
        cv2.drawContours(self.contourMask, contours, -1, 255, -1) # Draw filled contour in mask

        print("Num contours: " + str(len(contours)))
        if len(contours)<25:  # if too little markers, then give up
            self.MarkerAvailable=False
            return MarkerCenter
        # cv2.drawContours(self.img, contours, -1, (0,255,0), 3)
        for contour in contours:
            AreaCount=cv2.contourArea(contour)
            if AreaCount>areaThresh1 and AreaCount<areaThresh2:
                t=cv2.moments(contour)
                if (t['m10']/t['m00'] > 20 and t['m10']/t['m00'] < 620 and t['m01']/t['m00'] > 20 and t['m01']/t['m00'] < 460):
                    MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)

        # 0:x 1:y
        return MarkerCenter

    def contourTrack(self, first_contour = False):
        dst = cv2.inpaint(self.img,self.contourMask.astype(np.uint8),3,cv2.INPAINT_TELEA)
        hsv_img = rgb2hsv(self.img)
        # thresh2 = threshold_otsu(hsv_img[:,:,2])
        binary2 = (hsv_img[:,:,2] > self.hsv_thresh*0.75)
        binary3 = (dst[:,:,2] > self.red_thresh*0.85)
        binary2 = (binary2 & binary3)

        # smooth the contour
        contour_img = 255*binary2.astype(np.uint8)
        ret,tmp = cv2.threshold(contour_img, 125, 255, cv2.THRESH_BINARY);
        blur = cv2.pyrUp(tmp)
        for i in range(5):
            blur = cv2.medianBlur(blur,7)
        blur = cv2.pyrDown(blur)
        ret,contour_img = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY);

        contours, hierarchy = cv2.findContours(contour_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print("Num contours: " + str(len(contours)))
        if len(contours) == 0:
            return None
        max_area = -1
        cnt = None
        idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max_area:
                cnt = contours[i]
                idx = i
                max_area = area
        print("max area is " + str(max_area))
        if max_area < 1000:
            return None
        # print("max idx is " + str(idx))
        t=cv2.moments(cnt)

        color = np.stack((contour_img,)*3, axis=-1)

        ellipse = cv2.fitEllipse(cnt)
        Ma, ma = ellipse[1]
        print("Ma is " + str(Ma))
        print("ma is " + str(ma))
        print("ratio is " + str(Ma/ma))
        if Ma/ma > 0.4:
            return None
        angle = ellipse[2]
        cv2.ellipse(color,ellipse,(0,255,0),2)
        # cv2.imshow("contour", color)
        # cv2.waitKey(1)
        if first_contour:
            self.first_ellipse_ang = angle
            ellipse_ang = 0.0
        else:
            # calculate the angle
            ellipse_ang = self.first_ellipse_ang-angle
            if ellipse_ang > 30:
                ellipse_ang = -180+ellipse_ang
            elif ellipse_ang < -30:
                ellipse_ang = 180+ellipse_ang
            print("ellipse angle is " + str(ellipse_ang))
            # all_ellipse_angles.append(ellipse_ang)
        return ellipse_ang

    def getCurrentMarkers(self):

        self.markerPresent = np.zeros((self.flowcenter.shape[0], 2))

        self.markerPresent[:, 0] = self.flowcenter[:, 0] + self.markerU[:]
        self.markerPresent[:, 1] = self.flowcenter[:, 1] + self.markerV[:]

        return self.markerPresent

    def getMarkerMagnitude(self):

        return np.round(np.sqrt((self.markerPresent[:, 0]-self.flowcenter[:, 0])**2 +(self.markerPresent[:, 1]-self.flowcenter[:, 1])**2), 2)

    def calculateAngle(self, xc1, yc1, xc2, yc2):

        v0 = [1, 0]
        v1 = [xc2-xc1, yc2-yc1]

        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

        return np.degrees(angle)

    def getMarkerAngle(self):

        # Store the angles each marker makes with the x axis aligning with the width of the image.
        markerAngles = []

        for i in range(self.flowcenter.shape[0]):

            # set_trace()
            xcv1, ycv1, _ = self.flowcenter[i]
            xcv2, ycv2 = self.markerPresent[i]

            xc1, yc1 = self.convertCV2ToCartesian(xcv1, ycv1)
            xc2, yc2 = self.convertCV2ToCartesian(xcv2, ycv2)

            angle = self.calculateAngle(xc1, yc1, xc2, yc2)

            markerAngles.append(round(angle,2))

        return np.array(markerAngles)

    def cal_marker_center_motion(self, MarkerCenter):
        Nt=len(MarkerCenter)
        no_seq2=np.zeros(Nt)
        center_now=np.zeros([self.MarkerCount, 3])
        for i in range(Nt):
            dif=np.abs(MarkerCenter[i,0]-self.marker_last[:,0])+np.abs(MarkerCenter[i,1]-self.marker_last[:,1])
            no_seq2[i]=np.argmin(dif*(100+np.abs(MarkerCenter[i,2]-self.flowcenter[:,2])))

        for i in range(self.MarkerCount):
            dif=np.abs(MarkerCenter[:,0]-self.marker_last[i,0])+np.abs(MarkerCenter[:,1]-self.marker_last[i,1])
            t=dif*(100+np.abs(MarkerCenter[:,2]-self.flowcenter[i,2]))
            a=np.amin(t)/100
            b=np.argmin(t)
            if self.flowcenter[i,2]<a:   # for small area
                self.markerU[i]=0
                self.markerV[i]=0
                center_now[i]=self.flowcenter[i]
                # center_now[i]=self.marker_last[i]
            elif i==no_seq2[b]:
                self.markerU[i]=MarkerCenter[b,0]-self.flowcenter[i,0]
                self.markerV[i]=MarkerCenter[b,1]-self.flowcenter[i,1]
                center_now[i]=MarkerCenter[b]
            else:
                self.markerU[i]=0
                self.markerV[i]=0
                center_now[i]=self.flowcenter[i]
                # self.markerU[i]=self.marker_last[i,0]-self.flowcenter[i,0]
                # self.markerV[i]=self.marker_last[i,1]-self.flowcenter[i,1]
                # center_now[i]=self.marker_last[i]

        self.markerPresent = self.getCurrentMarkers()
        self.markerMagnitude = self.getMarkerMagnitude()
        self.markerAngle = self.getMarkerAngle()

        return center_now, self.markerMagnitude, self.markerAngle

    def compareMarkerMotion(self):

        if (self.firstContact):
            indices = self.contactMarkers
            # print(self.contactMarkers)
        else:
            indices = np.arange(0,self.MarkerCount,1)
        self.markerMagnitude = np.sqrt(self.markerU[indices]**2+self.markerV[indices]**2)
        # self.markerMagnitudeMax = np.max(self.markerMagnitude)
        self.markerMagnitudeMax = np.percentile(self.markerMagnitude, 70, axis=0)
        # print(self.markerMagnitude)
        # print(self.markerMagnitudeMax)
        # new
        index = np.where(self.markerMagnitude>=0.3*self.markerMagnitudeMax)
        # print(np.shape(index))
        # print(indices[index[0]])

        return indices[index[0]]

    #This function to convert the cartesian coordinates to cv2 notation.
    def convertCartesianToCV2(self, x1, y1):

        h = self.img.shape[0]
        l = self.img.shape[1]

        x_cv2 = x1
        y_cv2 = h - y1

        return x_cv2, y_cv2

    def convertCV2ToCartesian(self, x_cv2, y_cv2):

        h = self.img.shape[0]
        l = self.img.shape[1]

        xcc = x_cv2
        ycc = h - y_cv2

        return xcc, ycc

    def normalCoordinates(self, x1, y1, x2, y2):

        l = self.length

        if ((x2- x1)==0):

            yc = (y1 + y2)/2

            yn1 = yc
            yn2 = yc

            xc = x1

            xn1 = x1 - l
            xn2 = x2 + l

            return int(xn1), int(yn1), int(xn2), int(yn2)

        else:

            m = ((y2-y1)/(x2-x1))

            xc = (x1 + x2)/2
            yc = (y1 + y2)/2

            xn1 = xc - ((l*m)/(np.sqrt(1+(m**2))))
            yn1 = yc + ((l)/(np.sqrt(1+(m**2))))

            xn2 = xc + ((l*m)/(np.sqrt(1+(m**2))))
            yn2 = yc - ((l)/(np.sqrt(1+(m**2))))

            return int(xn1), int(yn1), int(xn2), int(yn2)

    def RANSAC(self,indices):
        num_iter = 500
        [num_idx] = np.shape(indices)
        max_inliers = 0
        best_inliers = []
        for i in range(num_iter):
            #random choose two vectors to solve the angle
            a = randrange(num_idx)
            b = randrange(num_idx)
            AMatrix = []
            bMatrix = []
            idxa = indices[a]
            idxb = indices[b]
            x1a = self.flowcenter[idxa,0]
            y1a = self.flowcenter[idxa,1]

            # end point
            x2a = self.flowcenter[idxa,0] + self.markerU[idxa]
            y2a = self.flowcenter[idxa,1] + self.markerV[idxa]

            #mid point
            xma = (x1a+x2a)/2
            yma= (y1a+y2a)/2

            AMatrix.append([x1a-x2a,y1a-y2a])
            bMatrix.append([xma*(x1a-x2a)+yma*(y1a-y2a)])

            x1b = self.flowcenter[idxb,0]
            y1b = self.flowcenter[idxb,1]

            # end point
            x2b = self.flowcenter[idxb,0] + self.markerU[idxb]
            y2b = self.flowcenter[idxb,1] + self.markerV[idxb]

            #mid point
            xmb = (x1b+x2b)/2
            ymb = (y1b+y2b)/2

            AMatrix.append([x1b-x2b,y1b-y2b])
            bMatrix.append([xmb*(x1b-x2b)+ymb*(y1b-y2b)])

            xVec = self.leastSquares(np.array(AMatrix), np.array(bMatrix))
            xc = xVec[0]
            yc = xVec[1]

            direction1 = np.array([x1a-xc,y1a-yc])
            direction2 = np.array([x2a-xc,y2a-yc])
            cos_theta = np.dot(direction1.T,direction2)/(np.linalg.norm(direction1)*np.linalg.norm(direction2))
            theta = np.arccos(cos_theta)
            ang1 = np.degrees(theta)

            direction1 = np.array([x1b-xc,y1b-yc])
            direction2 = np.array([x2b-xc,y2b-yc])
            cos_theta = np.dot(direction1.T,direction2)/(np.linalg.norm(direction1)*np.linalg.norm(direction2))
            theta = np.arccos(cos_theta)
            ang2 = np.degrees(theta)
            # print("first angle is " + str(ang1) + "; second is " + str(ang2))
            if (abs(ang1 - ang2) > 0.2 or np.isnan(ang1) or np.isnan(ang2)):
                continue
            angle = (ang1 + ang2)/2.0

            #test all others, find inliers
            count_inliers = 0
            inliers = []
            for j in indices:
                x1, y1 = self.flowcenter[j,0], self.flowcenter[j,1]
                x2, y2 = self.markerPresent[j]
                direction1 = np.array([x1-xc,y1-yc])
                direction2 = np.array([x2-xc,y2-yc])
                cos_theta = np.dot(direction1.T,direction2)/(np.linalg.norm(direction1)*np.linalg.norm(direction2))
                theta = np.arccos(cos_theta)
                ang = np.degrees(theta)
                # print("avg angle is: " + str(angle) + " ; cur angle is: " + str(ang))
                if (abs(ang-angle) < 0.2 and (not np.isnan(ang))):
                    count_inliers += 1
                    inliers.append(j)
            # print("# inliers: " + str(count_inliers))
            if count_inliers > max_inliers:
                max_inliers = count_inliers
                best_inliers = inliers


        #find the best model
        print("# best inliers: " + str(max_inliers) + " out of total # " + str(num_idx))

        #refine the angle
        AMatrix = []
        bMatrix = []
        for i in best_inliers:
            x1 = self.flowcenter[i,0]
            y1 = self.flowcenter[i,1]

            # end point
            x2 = self.flowcenter[i,0] + self.markerU[i]
            y2 = self.flowcenter[i,1] + self.markerV[i]

            #mid point
            xm = (x1+x2)/2
            ym = (y1+y2)/2

            AMatrix.append([x1-x2,y1-y2])
            bMatrix.append([xm*(x1-x2)+ym*(y1-y2)])
        xVec = self.leastSquares(np.array(AMatrix), np.array(bMatrix))
        return xVec

    def checkTranslation(self):

        print("check translation")
        if (self.firstContact):
            # print("after contact")
            indices = self.contactMarkers
            # print(self.contactMarkers)
        else:
            # print("before contact")
            indices = self.indexSignificantMotion
            # print(self.indexSignificantMotion)

        x1 = self.flowcenter[self.indexSignificantMotion][:,0]
        y1 = self.flowcenter[self.indexSignificantMotion][:,1]

        x2 = x1 + self.markerU[self.indexSignificantMotion]
        y2 = y1 + self.markerV[self.indexSignificantMotion]
        # dx = (x1 - x2)
        # dy = (y1 - y2)
        AMatrix = np.array([x1-x2,y1-y2])
        # AMatrix = AMatrix.T
        [U,D,VT] = np.linalg.svd(AMatrix)
        print(D)
        print(D[1]/D[0])
        self.t_check = D[1]/D[0]
        if (D[1]/D[0] < 0.28):
            return True

        return False


    def calculateABMatrices(self):


        # AMatrix  = []       #Ax = b
        # bMatrix  = []

        if (self.firstContact):
            print("after contact")
            indices = self.contactMarkers
            # print(self.contactMarkers)
        else:
            print("before contact")
            indices = self.indexSignificantMotion
            # print(self.indexSignificantMotion)

        x1 = self.flowcenter[self.indexSignificantMotion][:,0]
        y1 = self.flowcenter[self.indexSignificantMotion][:,1]

        x2 = x1 + self.markerU[self.indexSignificantMotion]
        y2 = y1 + self.markerV[self.indexSignificantMotion]

        xm = (x1 + x2)/2
        ym = (y1 + y2)/2

        AMatrix = np.array([x1-x2,y1-y2])
        bMatrix = np.array([xm*(x1-x2)+ym*(y1-y2)])
        AMatrix = AMatrix.T
        bMatrix = bMatrix.T
        # print(np.shape(AMatrix))
        # print(np.shape(bMatrix))
        # print(AMatrix)
        # print(bMatrix)

        return AMatrix, bMatrix

    def leastSquares(self, AMatrix, bMatrix):

        if AMatrix.shape[0] != 0 and bMatrix.shape[0] != 0:

            xVec = np.linalg.lstsq(AMatrix, bMatrix, rcond=None)[0] #Centre of Rotation Calculation
            # print("xVec is", img.shape[0]-xVec[0], img.shape[1]-xVec[1])

            return xVec

        else: return np.array([0,0])

    def calculateTangentialVec(self):

        xc = self.xVec[0][0]
        yc = self.xVec[1][0]

        inliers = []

        if (self.firstContact):
            indices = self.contactMarkers
            # print(self.contactMarkers)
        else:
            indices = self.indexSignificantMotion
            # print(self.indexSignificantMotion)

        for i in indices:

            # start point
            x1 = self.flowcenter[i,0]
            y1 = self.flowcenter[i,1]

            # end point
            x2 = self.flowcenter[i,0] + self.markerU[i]
            y2 = self.flowcenter[i,1] + self.markerV[i]

            #mid point
            xm = (x1+x2)/2
            ym = (y1+y2)/2

            d1 = np.array([xc-xm,yc-ym])
            d1 = d1/ np.linalg.norm(d1)
            d2 = np.array([-d1[1],d1[0]])
            d2 = d2/ np.linalg.norm(d2)

            d = np.array([x2-x1,y2-y1])

            mag1 = abs(np.dot(d,d1))
            mag2 = abs(np.dot(d,d2))
            print("ratio is " + str(mag2/mag1))
            if (not np.isnan(mag2/mag1) and mag2/mag1 > 2):
                inliers.append(i)

        return np.array(inliers)

    def checkAngle(self,indices = None):

        xc = self.xVec[0][0]
        yc = self.xVec[1][0]

        inliers = []

        if (self.orientation == "CW"):
            indices = self.idx_cw
            # print(self.contactMarkers)
        else:
            indices = self.idx_acw
            # print(self.indexSignificantMotion)

        for i in indices:

            # start point
            x1 = self.flowcenter[i,0]
            y1 = self.flowcenter[i,1]

            # end point
            x2 = self.flowcenter[i,0] + self.markerU[i]
            y2 = self.flowcenter[i,1] + self.markerV[i]

            #mid point
            xm = (x1+x2)/2
            ym = (y1+y2)/2

            d1 = np.array([xc-xm,yc-ym])
            d1 = d1/ np.linalg.norm(d1)
            d2 = np.array([-d1[1],d1[0]])
            d2 = d2/ np.linalg.norm(d2)

            d = np.array([x2-x1,y2-y1])

            mag1 = abs(np.dot(d,d1))
            mag2 = abs(np.dot(d,d2))
            print("ratio is " + str(mag2/mag1))
            if (not np.isnan(mag2/mag1) and mag2/mag1 > 2):
                inliers.append(i)

            # d1 = np.array([xc-xm,yc-ym])
            # d1 = d1/ np.linalg.norm(d1)
            # d2 = np.array([x2-x1,y2-y1])
            # d2 = d2/ np.linalg.norm(d2)
            #
            # cos_theta = np.dot(d1.T,d2)
            # theta = np.arccos(cos_theta)
            # ang = np.degrees(theta)
            #
            # print("check angle is " + str(ang))
            # if (not np.isnan(ang) and (ang > 80 and ang < 100)):
            #     inliers.append(i)
        print(inliers)

        if (len(inliers) > 0.5*len(indices)):
            print("It is valid angle")
            return False
        else:
            print("Bad angle")
            return True
        # return np.array(inliers)


    def calculateCOR(self):
        # if self.firstContact and np.shape(self.contactMarkers)[0] <= 2:
        if self.firstContact and np.shape(self.indexSignificantMotion)[0] <= 2:
            self.badRotation = True
            return np.array([[320], [240]])
        #Form the A & B Matrices which represent the equation of normal lines.
        AMatrix, bMatrix = self.calculateABMatrices()
        # #Disabled for testing purpose
        # # visualizationImage = drawNormalLines(visualizationImage, normals)

        # #Solve it using the least squares method to obtain the location of the Centre of Rotation.
        xcVec = self.leastSquares(AMatrix, bMatrix)
        # print("center is " + str(xcVec))

        # #This conversion is vital because the input to display should be in the notation of cv2.
        # xcvVec = self.convertCartesianToCV2(xcVec[0], xcVec[1])

        if (not self.firstContact):
            return xcVec

        # self.inliers = self.calculateTangentialVec()
        # print(self.inliers)
        # print(xcVec)
        # print(self.xVec)
        self.xVec = xcVec
        angleR = self.calculateAngleRotation()
        if (angleR == 0.0):
            self.badRotation = True
            return xcVec
        if (abs(np.shape(self.idx_cw)[0] - np.shape(self.idx_acw)[0])/max(np.shape(self.idx_cw)[0], np.shape(self.idx_acw)[0]) < 0.5):
            # print("dis-ambiguity")
            num_iter = 10
            pre_vec = xcVec
            # for i in range(num_iter):
            # recalculate the COR based on cw/acw
            x1 = self.flowcenter[self.idx_cw][:,0]
            y1 = self.flowcenter[self.idx_cw][:,1]

            x2 = x1 + self.markerU[self.idx_cw]
            y2 = y1 + self.markerV[self.idx_cw]

            xm = (x1 + x2)/2
            ym = (y1 + y2)/2

            AMatrix = np.array([x1-x2,y1-y2])
            bMatrix = np.array([xm*(x1-x2)+ym*(y1-y2)])
            AMatrix = AMatrix.T
            bMatrix = bMatrix.T
            cw_vec = self.leastSquares(AMatrix, bMatrix)
            x1 = self.flowcenter[self.idx_acw][:,0]
            y1 = self.flowcenter[self.idx_acw][:,1]

            x2 = x1 + self.markerU[self.idx_acw]
            y2 = y1 + self.markerV[self.idx_acw]

            xm = (x1 + x2)/2
            ym = (y1 + y2)/2

            AMatrix = np.array([x1-x2,y1-y2])
            bMatrix = np.array([xm*(x1-x2)+ym*(y1-y2)])
            AMatrix = AMatrix.T
            bMatrix = bMatrix.T
            acw_vec = self.leastSquares(AMatrix, bMatrix)
            dis_cw = np.linalg.norm(cw_vec-pre_vec)
            dis_acw = np.linalg.norm(acw_vec-pre_vec)
            # print("cw err is " + str(cw_vec-pre_vec) + " norm is " + str(dis_cw))
            # print("acw err is " + str(acw_vec-pre_vec) + " norm is " + str(dis_acw))
            if (dis_cw > 100.0 and dis_cw > 100.0): # not a rotation
                self.badRotation = True
            if (dis_cw < dis_acw):
                xcVec = cw_vec
            else:
                xcVec = acw_vec
            # self.xVec = cw_vec
            # angle_cw = self.calculateAngleRotation()
            # ratio_cw = (abs(np.shape(self.idx_cw)[0] - np.shape(self.idx_acw)[0])/max(np.shape(self.idx_cw)[0], np.shape(self.idx_acw)[0]) < 0.5)
            #
            # self.xVec = acw_vec
            # angle_acw = self.calculateAngleRotation()
            # ratio_acw = (abs(np.shape(self.idx_cw)[0] - np.shape(self.idx_acw)[0])/max(np.shape(self.idx_cw)[0], np.shape(self.idx_acw)[0]) < 0.5)
            # if (ratio_cw < 0.5 and ratio_acw < 0.5):
            #     self.badRotation = True
            # elif (ratio_cw >= ratio_acw):
            #     xcVec = cw_vec
            # elif (ratio_acw >= ratio_cw):
            #     xcVec = acw_vec


        else: # recompute on the good inliers
            # print("improve the COR")
            if (np.shape(self.idx_cw)[0] - np.shape(self.idx_acw)[0] > 0):
                indices = self.idx_cw
            else:
                indices = self.idx_acw
            pre_vec = xcVec
            x1 = self.flowcenter[indices][:,0]
            y1 = self.flowcenter[indices][:,1]

            x2 = x1 + self.markerU[indices]
            y2 = y1 + self.markerV[indices]

            xm = (x1 + x2)/2
            ym = (y1 + y2)/2

            AMatrix = np.array([x1-x2,y1-y2])
            bMatrix = np.array([xm*(x1-x2)+ym*(y1-y2)])
            AMatrix = AMatrix.T
            bMatrix = bMatrix.T
            xcVec = self.leastSquares(AMatrix, bMatrix)
            dis = np.linalg.norm(xcVec-pre_vec)
            if dis > 100.0:
                self.badRotation = True
            # print("err is " + str(xcVec-pre_vec) + " norm is " + str(dis))
            # print(xcVec-pre_vec)

        # return xcvVec
        return xcVec


    def drawBBoxDynamic(self):

        #Final marker positions
        markerPosition = 0*self.flowcenter
        markerPosition[:, 0] = self.flowcenter[:, 0] + self.markerU[:]
        markerPosition[:, 1] = self.flowcenter[:, 1] + self.markerV[:]

        xDistance = np.abs(self.xVec[0] - markerPosition[:, 0])
        xDistanceInd = int(np.argmin(xDistance))

        yDistance = np.abs(self.xVec[1] - markerPosition[:, 1])
        yDistanceInd = int(np.argmin(yDistance))

        # set_trace()
        deltaxx = np.abs(markerPosition[xDistanceInd, 0]-self.xVec[0])
        deltayy = np.abs(markerPosition[yDistanceInd, 1]-self.xVec[1])

        x11 = deltaxx + self.delta
        # x22 = deltaxx - 100

        y11 = deltayy + self.delta
        # y22 = deltayy - 100

        self.x1 = self.xVec[0]-x11
        self.x2 = self.xVec[0]+x11

        self.y1 = self.xVec[1]-y11
        self.y2 = self.xVec[1]+y11

        # print(x1, y1, x2, y2)

        # # #Disabled for Analysis
        cv2.line(self.img,(int(self.x1), int(self.y1)), \
                            (int(self.x1), int(self.y2)),\
                            (255, 255, 0),2)
        cv2.line(self.img,(int(self.x1), int(self.y1)), \
                            (int(self.x2), int(self.y1)),\
                            (255, 255, 0),2)
        cv2.line(self.img,(int(self.x2), int(self.y1)), \
                            (int(self.x2), int(self.y2)),\
                            (255, 255, 0),2)
        cv2.line(self.img,(int(self.x1), int(self.y2)), \
                            (int(self.x2), int(self.y2)),\
                            (255, 255, 0),2)


        return x11, y11

    def checkBound(self, x1, x2, y1, y2, xVec, x11, y11):

        # set_trace()
        xVec = np.array((xVec))

        if x1 in range(int(xVec[0]-x11), int(xVec[0]+x11)) and x2 in range(int(xVec[0]-x11), int(xVec[0]+x11)) \
            and y1 in range(int(xVec[1]-y11), int(xVec[1]+y11)) and y2 in range(int(xVec[1]-y11), int(xVec[1]+y11)):
            return True
        else:
            return False

    def boundingBox(self):

        indices = self.compareMarkerMotion()

        self.delta = 200

        x11, y11 = self.drawBBoxDynamic()

        indexIterate = []

        for i in indices:

            x1 = self.flowcenter[i,0]
            y1 = self.flowcenter[i,1]

            x2 = self.flowcenter[i,0]+self.markerU[i]
            y2 = self.flowcenter[i,1]+self.markerV[i]

            if self.checkBound(int(x1), int(x2), int(y1), int(y2), (self.xVec), x11, y11):

                indexIterate.append(i)

        return np.array(indexIterate)

    def detectContactArea(self):

        diffim=self.diff
        # maxim=diffim.max(axis=2)
        contactmap=self.max_img
        countnum=(contactmap>10).sum()
        # print(countnum)

        contactmap[contactmap<10]=0
        # contactmap[contactmap<=0]=0

        image = np.zeros((480,640,3), np.uint8)
        # image = np.zeros((480,640))

        maxC = np.max(contactmap)
        sec90C = np.percentile(contactmap, 90)
        sec95C = np.percentile(contactmap, 95)
        sec99C = np.percentile(contactmap, 99)
        # image[contactmap>0.4*sec90C] = 1

        contact_mask = contactmap>0.4*sec90C

        total_contact = np.sum(contact_mask)
        self.contact_ratio = total_contact/(image.shape[0]*image.shape[1])
        print("contact ratio is " + str(self.contact_ratio))

        image[contact_mask,0] = 255
        image[contact_mask,1] = 255
        image[contact_mask,2] = 255
        # pause = input("pause here")


        #
        # for i in range(3):
        #     image[contactmap>0.3*maxC,0] = 255
        #     image[contactmap>0.3*maxC,1] = 255
        #     image[contactmap>0.3*maxC,2] = 255

        return image
    def detectStableContact(self):
        diffim=np.absolute(np.int16(self.img)-self.lastImg)
        diffmap=diffim.max(axis=2)
        countnum=np.where(diffmap>10).sum()
        print("unstable contact num: " + str(countnum))
        if countnum > 500:
            return False
        else:
            return True

    #Detects the onset of rotation in the pipeline.
    def rotationInit(self):

        idx_contact = self.contactMarkers

        fx = self.flowcenter[idx_contact][:, 0]
        fy = self.flowcenter[idx_contact][:, 1]

        px = self.markerPresent[idx_contact][:, 0]
        py = self.markerPresent[idx_contact][:, 1]

        cx = self.contactLoc[idx_contact][:,0]
        cy = self.contactLoc[idx_contact][:,1]

        d1x = cx - fx
        d1y = cy - fy

        d2x = px - fx
        d2y = py - fy

        d3x = px - cx
        d3y = py - cy

        contactMags = np.sqrt(d3x*d3x + d3y*d3y)

        direction1 = np.array([d1x,d1y])
        direction2 = np.array([d2x,d2y])
        direction1 = direction1.T # N*2
        direction2 = direction2.T # N*2
        direction1 = direction1/np.linalg.norm(direction1, axis = 1)[:, None]
        direction2 = direction2/np.linalg.norm(direction2, axis = 1)[:, None]
        # print("d1 is " + str(direction1))
        # print("d2 is " + str(direction2))
        cos_theta = direction1[:,0]*direction2[:,0] + direction1[:,1]*direction2[:,1]
        theta = np.arccos(cos_theta)
        ang = np.degrees(theta)
        contactAngles = ang[~np.isnan(ang)]

        self.anglesContact = np.array(contactAngles)
        # contactMags = np.array(contactMags)
        if (len(contactAngles) == 0):
            return False
        # print(contactMags)
        # print("median mag is " + str(np.median(contactMags)))
        # print(contactAngles)
        # print("total # " + str(len(idx_contact)))
        # print("change # " + str(len(contactAngles)))
        # print("median change is " + str(np.median(self.anglesContact)))
        # pause = input("Pause here, enter key:")
        nonZeroAngles = self.anglesContact[self.anglesContact!=0]
        if np.median(self.anglesContact) > 8.0 or np.median(contactMags) > 10.0:
            return True
        else:
            return False
        # return False

    def calculateWhitePixels(self, cX, cY):

        delta = self.markerDelta

        x1 = cX-delta
        x2 = cX+delta

        y1 = cY-delta
        y2 = cY+delta

        return ((self.contactPixels[y1:y2, x1:x2, :]==255).sum())


    def calculateAngleRotation(self):

        # print(self.frameIndex,  "\n")
        # print("cal angle")
        # print(self.contactMarkers)

        xcv1 = self.flowcenter[self.indexSignificantMotion][:,0]
        ycv1 = self.flowcenter[self.indexSignificantMotion][:,1]
        # print(xcv1)

        xcv2 = self.markerPresent[self.indexSignificantMotion][:,0]
        ycv2 = self.markerPresent[self.indexSignificantMotion][:,1]
        # print(xcv2)
        xc_Cv = self.xVec[0][0]
        yc_Cv = self.xVec[1][0]

        direction1 = np.array([xcv1-xc_Cv,ycv1-yc_Cv])
        direction2 = np.array([xcv2-xc_Cv,ycv2-yc_Cv])
        direction1 = direction1.T # N*2
        direction2 = direction2.T # N*2
        direction1 = direction1/np.linalg.norm(direction1, axis = 1)[:, None]
        direction2 = direction2/np.linalg.norm(direction2, axis = 1)[:, None]
        # print("d1 is " + str(direction1))
        # print("d2 is " + str(direction2))
        cos_theta = direction1[:,0]*direction2[:,0] + direction1[:,1]*direction2[:,1]
        theta = np.arccos(cos_theta)
        ang = np.degrees(theta)
        thetas = ang[~np.isnan(ang)]
        m1 = np.array([xcv2-xcv1,ycv2-ycv1,np.zeros(len(ang))])
        m2 = np.array([xcv1-xc_Cv,ycv1-yc_Cv,np.zeros(len(ang))])
        # print(m1)
        # print(m2)
        norm = np.cross(m2.T,m1.T)
        clock = norm[:,2][~np.isnan(ang)]
        idx = self.indexSignificantMotion[~np.isnan(ang)]

        # thetas = []
        # clock = []
        # idx = []
        #
        #
        # # for i in self.contactMarkers:
        # for i in self.indexSignificantMotion:
        #
        #         xcv1, ycv1 = self.flowcenter[i,0], self.flowcenter[i,1]
        #         x1, y1 = self.convertCV2ToCartesian(xcv1, ycv1)
        #         xcv2, ycv2 = self.markerPresent[i,0],self.markerPresent[i,1]
        #         x2, y2 = self.convertCV2ToCartesian(xcv2, ycv2)
        #
        #         # print(self.markerStart[i], self.markerPresent[i], "\n")
        #
        #         xc_Cv = self.xVec[0][0]
        #         yc_Cv = self.xVec[1][0]
        #
        #         direction1 = np.array([xcv1-xc_Cv,ycv1-yc_Cv])
        #         direction2 = np.array([xcv2-xc_Cv,ycv2-yc_Cv])
        #         cos_theta = np.dot(direction1.T,direction2)/(np.linalg.norm(direction1)*np.linalg.norm(direction2))
        #         theta = np.arccos(cos_theta)
        #         ang = np.degrees(theta)
        #         # print ("theta1 is " + str(ang))
        #         if (not np.isnan(ang)):
        #             thetas.append(ang)
        #             m1 = np.array([xcv2-xcv1,ycv2-ycv1,0])
        #             m2 = np.array([xcv1-xc_Cv,ycv1-yc_Cv,0])
        #             # print(m1)
        #             # print(m2)
        #             norm = np.cross(m2,m1)
        #             # print("norm is " + str(norm))
        #             clock.append(norm[2])
        #             idx.append(i)
        #
        #             # cv2.putText(self.img, str(np.round(ang, 2)), (int(xcv2), int(ycv2+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        #
        thetas = np.array(thetas)
        clock = np.array(clock)
        idx = np.array(idx)

        clockwise = 1*(clock > 0)
        index_cw = np.where(clockwise > 0)
        theta_cw = thetas[index_cw]
        self.idx_cw = idx[index_cw]
        # index_inlier = self.checkAngle(self.idx_cw)
        # print(index_inlier)
        # print("clockwise # " + str(np.shape(theta_cw)[0]))

        median_theta_cw = np.median(theta_cw)
        # var_theta_cw = np.var(theta_cw)
        theta_cw2 = theta_cw * (np.greater_equal(theta_cw, 0.3*median_theta_cw) & np.less_equal(theta_cw, 1.8*median_theta_cw))
        theta_cw3 = theta_cw2[np.where(theta_cw2 > 0)]
        sum_cw = np.shape(theta_cw3)[0]
        # sum_cw = np.sum(clockwise)
        # print("cw")
        # print("median is" + str(median_theta_cw))
        # print(theta_cw)
        # print(theta_cw3)
        # print("clockwise # " + str(sum_cw))
        anti_clockwise = 1*(clock < 0)
        index_acw = np.where(anti_clockwise > 0)
        theta_acw = thetas[index_acw]
        self.idx_acw = idx[index_acw]
        # index_inlier = self.checkAngle(self.idx_acw)
        # print(index_inlier)
        # print("anit-clockwise # " + str(np.shape(theta_acw)[0]))
        # sum_acw = np.sum(anti_clockwise)

        median_theta_acw = np.median(theta_acw)
        # var_theta_cw = np.var(theta_cw)
        theta_acw2 = theta_acw * (np.greater_equal(theta_acw, 0.3*median_theta_acw) & np.less_equal(theta_acw, 1.8*median_theta_acw))
        theta_acw3 = theta_acw2[np.where(theta_acw2 > 0)]
        sum_acw = np.shape(theta_acw3)[0]
        # print("acw")
        # print("median is" + str(median_theta_acw))
        # print(theta_acw)
        # print(theta_acw3)
        # print("anti clockwise # " + str(sum_acw))

        if (sum_acw >= sum_cw):
            # print(theta_final)
            self.orientation = "CCW"
            if (np.shape(theta_acw3)[0] <= 2):
                return 0.0
            angleR = np.percentile(theta_acw3, 70, axis=0)
            # print(theta_acw3.shape)
            # print(theta_cw3.shape)
            # all_theta = np.concatenate((theta_acw3, -1*theta_cw3))
            # print(all_theta.shape)
            # # pause = input("pause")
            # plot_norm(all_theta)
            # angleR = np.median(theta_acw3)
            # print("acw! final angle is " + str(angleR))
        else:
            self.orientation = "CW"
            # print(theta_final)
            if (np.shape(theta_cw3)[0] <= 2):
                return 0.0
            angleR = -1*np.percentile(theta_cw3, 70, axis=0)
            # print(theta_acw3.shape)
            # print(theta_cw3.shape)
            # all_theta = np.concatenate((theta_cw3, -1*theta_acw3))
            # print(all_theta.shape)
            # # pause = input("pause")
            # plot_norm(all_theta)
            # angleR = np.median(theta_cw3)
            # print("cw! final angle is " + str(angleR))


        self.indexPast = self.indexPresent
        return angleR

    def convertVideoFormat(self, img):

        h, w, _ = img.shape

        hMax = 1080
        wMax = 1920

        imgcheck = np.zeros((1080, 1920,3), np.uint8)
        #Making it aesthetic black.
        imgcheck[:,:,:] = (40,40,40)

        imgcheck[0:h, 0:w, :] = img

        return imgcheck

    def detectContactMarkers(self):
        self.contactPixels = self.detectContactArea()
        markerCenter=np.around(self.flowcenter[:,0:2]).astype(np.int16)

        self.markerDelta = 20

        index = []
        index_noncontact = []

        for i in range(self.MarkerCount):

            cX = int(self.flowcenter[i,0]+self.markerU[i]*self.showScale)
            cY = int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)



            # self.contactPixels = self.drawBox(cX, cY, self.contactPixels)
            # countnum.append(self.calculateWhitePixels(cX, cY))
            count = int(self.calculateWhitePixels(cX, cY))

            if count>1000:

                # cv2.putText(self.contactPixels, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                cv2.circle(self.contactPixels, (cX, cY), 3, (0, 255, 255), -1)
                index.append(i)
            else:
                index_noncontact.append(i)

        index = np.array(index)
        return index

    def contactStable(self):

        self.contactMarkers = self.detectContactMarkers()
        idx_contact = self.contactMarkers
        # print(idx_contact)

        dx = self.marker_last[idx_contact][:, 0] - self.markerPresent[idx_contact][:, 0]
        dy = self.marker_last[idx_contact][:, 1] - self.markerPresent[idx_contact][:, 1]

        contactMags = np.sqrt(dx*dx + dy*dy)
        # print("median contact mag is " + str(np.median(contactMags)))
        # here used to be 2.0
        if np.median(contactMags) < 0.375:
            return True
        else:
            return False

    def recordContactMarkers(self):

        #Checks if contact occurs and keeps checking until the 30th frame after contact.
        if (self.isContact and self.contactThresh<self.contactThreshValue):

            self.contactThresh += 1
            # print("contact # " + str(self.contactThresh))
        elif(self.isContact and not self.firstContact and self.contactThresh == self.contactThreshValue):
            self.contactThresh += 1
            #During this step, the pixels where contact occurs is stored. This information is further used to detect the onset of rotation.
            self.contactPixels = self.detectContactArea()

            self.firstContact = True
            # temp = cv2.medianBlur(self.contactPixels,5)
            markerCenter=np.around(self.flowcenter[:,0:2]).astype(np.int16)

            self.markerDelta = 20

            # idx = np.arange(0,self.MarkerCount,1)
            # cX = (self.flowcenter[idx][:,0]+self.markerU[idx]*self.showScale).astype(int)
            # cY = (self.flowcenter[idx][:,1]+self.markerV[idx]*self.showScale).astype(int)
            # # count = (self.calculateWhitePixels(cX, cY)).astype(int)
            # kernel = np.ones((self.markerDelta*2+1,self.markerDelta*2+1))
            # map = ndimage.convolve(self.contactPixels, kernel)
            # print(np.shape(map))
            # print(cX)
            # print(cY)
            # # map[cX,xY] > 1000
            # valid_idx = np.where(map[cY,cX] > 1000)
            # print(valid_idx)
            # index = idx[valid_idx]

            index = []
            index_noncontact = []

            for i in range(self.MarkerCount):

                cX = int(self.flowcenter[i,0]+self.markerU[i]*self.showScale)
                cY = int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)

                # self.contactPixels = self.drawBox(cX, cY, self.contactPixels)
                # countnum.append(self.calculateWhitePixels(cX, cY))
                count = int(self.calculateWhitePixels(cX, cY))

                if count>1000:

                    # cv2.putText(self.contactPixels, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                    cv2.circle(self.contactPixels, (cX, cY), 3, (0, 255, 255), -1)
                    index.append(i)
                else:
                    index_noncontact.append(i)

            index = np.array(index)
            #These are the markers where contact occurs. It is found by comparing the pixel values of area surrounding the markers.
            self.contactMarkers = index
            self.contactIndex = self.frameIndex

            # index_noncontact = np.array(index_noncontact)
            # self.noncontactMarkers = index_noncontact



            self.startDisplay = True
            for i in range(10):
                self.out.write(self.convertVideoFormat(self.contactPixels))
                show_img = 255-self.diff
                cv2.imwrite(self.resultPathC + str(self.frameIndex) + "Contact_diff"+'.jpg', show_img)
                cv2.imwrite(self.resulPath + str(self.frameIndex) + "Contact"+'.jpg', self.contactPixels)
                cv2.imwrite(self.resultPathC + str(self.frameIndex) + "Contact"+'.jpg', self.contactPixels)


            # cv2.imshow("Image", self.contactPixels)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("indexes are")
            # print(index)
            return index

        if(self.isContact and  self.contactThresh > 10 and not self.firstContact and self.contactStable() ):
            # self.contactThresh = self.contactThreshValue + 1
            #During this step, the pixels where contact occurs is stored. This information is further used to detect the onset of rotation.
            self.contactPixels = self.detectContactArea()

            self.firstContact = True
            # temp = cv2.medianBlur(self.contactPixels,5)
            markerCenter=np.around(self.flowcenter[:,0:2]).astype(np.int16)

            self.markerDelta = 20

            index = []
            index_noncontact = []

            for i in range(self.MarkerCount):

                cX = int(self.flowcenter[i,0]+self.markerU[i]*self.showScale)
                cY = int(self.flowcenter[i,1]+self.markerV[i]*self.showScale)

                # self.contactPixels = self.drawBox(cX, cY, self.contactPixels)
                # countnum.append(self.calculateWhitePixels(cX, cY))
                count = int(self.calculateWhitePixels(cX, cY))

                if count>1000:

                    # cv2.putText(self.contactPixels, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                    cv2.circle(self.contactPixels, (cX, cY), 3, (0, 255, 255), -1)
                    index.append(i)
                else:
                    index_noncontact.append(i)

            index = np.array(index)
            #These are the markers where contact occurs. It is found by comparing the pixel values of area surrounding the markers.
            self.contactMarkers = index
            self.contactIndex = self.frameIndex

            index_noncontact = np.array(index_noncontact)
            self.noncontactMarkers = index_noncontact



            self.startDisplay = True
            for i in range(10):
                self.out.write(self.convertVideoFormat(self.contactPixels))
                # show_img = (255-self.diff)*0.85
                # show_img = (128-self.diff)*1.5
                show_img = 255-self.diff*1.5
                cv2.imwrite(self.resultPathC + str(self.frameIndex) + "Contact_diff"+'.jpg', show_img)
                cv2.imwrite(self.resulPath + str(self.frameIndex) + "Contact"+'.jpg', self.contactPixels)
                cv2.imwrite(self.resultPathC + str(self.frameIndex) + "Contact"+'.jpg', self.contactPixels)


            # cv2.imshow("Image", self.contactPixels)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("indexes are")
            # print(index)
            return index

    def detectRotationOnset(self):

        #Now rotation onset is looked after.
        if(self.firstContact and not self.rotationOnset):

            if(self.firstContactInfo):
                self.contactMag = self.markerMagnitude
                self.contactAngle = self.markerAngle
                self.contactLoc = self.markerPresent
                self.indexPresent = self.compareMarkerMotion()

                self.firstContactInfo = False

            if(self.rotationInit()):
                print("Rotation Started")

                # self.contactMarkers = self.detectContactMarkers()

                self.indexSignificantMotion = self.compareMarkerMotion()
                self.xVec = self.calculateCOR()

                self.indexPresent = self.boundingBox()

                self.markerStart = self.markerPresent
                # self.calculateAngleRotation()
                image = np.zeros((self.img.shape[0],self.img.shape[1],3), np.uint8)
                image[:,:] = (69,24,88)

                cv2.putText(image, "Rotation Detected", (int(image.shape[0]/2-40), int(image.shape[1]/2)-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 221, 217), 2)
                cv2.imwrite(self.resulPath + str(self.frameIndex) + "Rotation"+'.jpg', image)
                cv2.imwrite(self.resultPathC + str(self.frameIndex) + "Rotation"+'.jpg', image)

                # cv2.imshow("Contact!", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                self.rotationOnset = True
                self.rotationOnsetIndex = self.frameIndex

    def detectDetach(self):

        motion_present = np.zeros((self.flowcenter.shape[0], 2))

        motion_present[:, 0] = self.markerPresent[:, 0]-self.flowcenter[:, 0]
        motion_present[:, 1] = self.markerPresent[:, 1]-self.flowcenter[:, 1]

        motion_last = np.zeros((self.flowcenter.shape[0], 2))

        motion_last[:, 0] = self.marker_last[:, 0]-self.flowcenter[:, 0]
        motion_last[:, 1] = self.marker_last[:, 1]-self.flowcenter[:, 1]

        motion_diff = motion_present-motion_last
        num_diff = np.sum(abs(motion_diff))

        median_motion_x = np.median(abs(motion_present[:,0]))
        median_motion_y = np.median(abs(motion_present[:,1]))
        # min_motion_x = np.min(abs(motion_present[:,0]))
        # min_motion_y = np.min(abs(motion_present[:,1]))

        # print(median_motion_x)
        # print(median_motion_y)

        # print(min_motion_x)
        # print(min_motion_y)

        # color_diff = np.sum(self.diff)
        # print(color_diff)
        print("Detach detection: " + str(num_diff))
        # print(self.flowcenter.shape[0])
        # if num_diff < self.flowcenter.shape[0]*0.25:
        if num_diff < self.flowcenter.shape[0]*0.25 and median_motion_x < 2 and median_motion_y < 2:
            return True
        else:
            return False

        # return num_diff

    def update_markerMotion(self, img=None):

        #Updating of MarkerMotion happens here.
        if img is not None:
            self.img=img

        #Finds the markers in the current frame.
        # start = time.time()
        MarkerCenter=self.find_markers()
        # end = time.time()
        # print("find marker time " + str(end - start) + "s")


        #Calculates the markermotion and now the marker_last becomes the present marker position.
        marker_last, markerMagnitudeLast, markerAngleLast = self.cal_marker_center_motion(MarkerCenter)


        #Checks if contact is made in the present frame.
        # start = time.time()
        self.isContact, k = self.detect_contact(self.img)
        # end = time.time()
        # print("contact detection" + str(end - start) + "s")
        print("is contact?"  + str(self.isContact))

        #Once the contact detection occurs, contact pixels information is recorded after the 30th frame from the initial contact.
        # self.contactThreshValue = 30
        self.contactThreshValue = 30

        # Records the markers where the contact detection occurs
        # start = time.time()
        self.recordContactMarkers()
        # end = time.time()
        # print("record detection" + str(end - start) + "s")

        if (self.firstContact and self.firstContour and self.contact_ratio < 0.6):
            angle = self.contourTrack(self.firstContour)
            self.firstContour = False
            if angle is None:
                self.smallContour = False
            else:
                pause = input("small contour pause")
                self.smallContour = True
            self.ellipse_ang = angle
        elif (self.firstContact and self.smallContour):
            angle = self.contourTrack(self.firstContour)
            if angle is None:
                self.ellipse_ang = 0.0
                # self.smallContour = False
            else:
                self.ellipse_ang = angle



        # Detection Rotation Onset
        # start = time.time()
        self.detectRotationOnset()
        # end = time.time()
        # print("detect rotation onset" + str(end - start) + "s")
        # if self.frameIndex == 33:
        #     self.rotationOnset = True


        if (self.rotationOnset and not self.detach):
        # if (self.rotationOnset):
            self.detach = self.detectDetach()

        if (self.rotationOnset):
            self.translate = self.checkTranslation()

        #Setting Up parameters for index significant motion.
        self.marker_last = marker_last
        self.markerMagnitudeLast =  markerMagnitudeLast
        self.markerAngleLast     =  markerAngleLast

        #Calculation of COR happens here.
        # start = time.time()
        self.indexSignificantMotion = self.compareMarkerMotion()
        # end = time.time()
        # print("compare motion" + str(end - start) + "s")

        # start = time.time()
        self.xVec = self.calculateCOR()
        # end = time.time()
        # print("calculate COR" + str(end - start) + "s")

        # self.indexSignificantMotion = self.boundingBox()
        # self.xVec = self.calculateCOR()
        # print("xvec is " + str(self.xVec))

        #Calculation of Rotation Angle.
        angleOfRotation  = 0.0
        self.angleOfRotation = 0.0
        orientation = "None"
        if (self.rotationOnset and not self.badRotation):

            # start = time.time()
            self.angleOfRotation = self.calculateAngleRotation()
            # end = time.time()
            # print("calculate angle" + str(end - start) + "s")

            # self.badRotation = self.badRotation | self.checkAngle()
            # print("bad rotation is " + str(self.badRotation))
            if (self.badRotation or self.detach or self.translate):
            # if (self.badRotation):
                self.angleOfRotation = 0
                self.badRotation = False
            else:
                angleOfRotation = self.angleOfRotation
                orientation = self.orientation
        if (self.firstContact and self.smallContour):
            if (self.ellipse_ang > 0):
                self.orientation = "CCW"
                self.angleOfRotation = self.ellipse_ang
            elif (self.ellipse_ang < 0):
                self.orientation = "CW"
                self.angleOfRotation = self.ellipse_ang
        if self.IsDisplay:
            # publish image
            # start = time.time()
            if self.rotationOnset and not self.badRotation:
                self.displayIm(angleOfRotation, orientation)
            else:
                self.displayIm()
            # end = time.time()
            # print("display img" + str(end - start) + "s")
        ###Zilin print out rotation angle
        self.badRotation = False
        self.lastImg = np.int16(cv2.GaussianBlur(self.img, (101,101), 50))
        # print("gel sight angle is " + str(self.angleOfRotation))

    def iniMarkerPos(self):
        # set the current marker position as the initial positions of the markers
        self.flowcenter=self.marker_last

    def start_display_markerIm(self):
        self.IsDisplay=True

    def stop_display_markerIm(self):
        self.IsDisplay=False

    def detect_contact(self, img=None, ColorThresh=1):
        # if not self.calMarker_on:
        #   self.update_markerMotion(img)

        isContact=False

        # contact detection based on color
        # diffim=np.int16(self.img)-self.f0
        diffim=self.diff
        max_map = self.max_img
        # max_map = diffim.max(axis=2)
        self.contactmap=max_map-diffim.min(axis=2)
        countnum=np.logical_and(self.contactmap>10, max_map>0).sum()
        ColorThresh = 1.5
        if countnum>self.touchthresh*ColorThresh:  # there is touch
            isContact=True
            # print("case1", self.touchthresh, ColorThresh, countnum)
            # print "Contact--Color Detected"

        # contact detection based on marker motion
        motion=np.abs(self.markerU)+np.abs(self.markerV)
        MotionNum=(motion>self.touchMarkerMovThresh*np.sqrt(ColorThresh)).sum()
        if MotionNum>self.touchMarkerNumThresh:
            isContact=True
            # print "Contact--Marker Detected"
            # print("case2")

        return isContact, countnum

    def ini_contactDetect(self):

        diffim=np.int16(self.img)-self.f0
        maxim=diffim.max(axis=2)
        contactmap=maxim-diffim.min(axis=2)
        countnum=np.logical_and(contactmap>10, maxim>0).sum()
        print (countnum)

        contactmap[contactmap<10]=0
        contactmap[maxim<=0]=0
        cv2.imwrite('iniContact.png', contactmap)

        self.touchthresh=round((countnum+1500)*1.0)
        # self.touchMarkerMovThresh=0.6
        self.touchMarkerMovThresh=1
        self.touchMarkerNumThresh=20

        hsv_img = rgb2hsv(self.img)
        self.hsv_thresh = np.max(hsv_img[:,:,2])
        self.red_thresh = np.max(self.img[:,:,2])


    def reinit(self, frame, frame0=None):
        self.img=frame   #current frame
        if frame0 is not None:
            self.f0=frame0   # frame0 is the low
        else:
            self.f0=np.int16(cv2.GaussianBlur(self.img, (101,101), 50))
        self.ini_contactDetect()


        # for markers
        self.flowcenter=self.find_markers()  # center of all the markers; x,y
        self.marker_last=self.flowcenter
        self.MarkerCount=len(self.flowcenter)
        self.markerSlipInited=False
        self.markerU=np.zeros(self.MarkerCount) # X motion of all the markers. 0 if the marker is not found
        self.markerV=np.zeros(self.MarkerCount) # Y motion of all the markers. 0 if the marker is not found
        self.markerMagnitudeLast=np.zeros(self.MarkerCount)
        self.markerAngleLast=np.zeros(self.MarkerCount)
        self.markerPresent = np.zeros((self.flowcenter.shape[0], 2))
        self.markerMagnitude = np.zeros(self.MarkerCount)
        self.markerAngle = np.zeros(self.MarkerCount)

        self.indexPresent = np.arange(0, self.MarkerCount)
        self.indexPast    = np.arange(0, self.MarkerCount)
        self.markerStart  = self.markerPresent

        self.startDisplay = False

        self.lastImg = self.f0


    def __init__(self, frame, frame0=None, dataPath = None):
        self.reinit(frame, frame0)

        # self.bridge = CvBridge()
        self.contactmap=np.zeros([480, 640])

        # initialte marker locations
        self.MarkerAvailable=True   # whether the markers are found
        self.calMarker_on=False

        self.markerMagnitude = None
        self.length = 150
        self.firstContact = False
        self.rotationOnset = False
        self.contactThresh = 0
        # self.contactStable = False
        self.contactIndex = 0
        self.indexContactPixels = None
        self.firstContactInfo = True
        self.badRotation = False
        self.detach = False
        self.translate = False
        self.t_check = 0.0
        self.contourMask = None
        self.firstContour = True
        self.smallContour = False
        self.ellipse_ang  = 0.0
        self.contact_ratio = None
        # self.tracker = FlowTracker(frame)

        self.frameIndex = 1
        self.orientation = None
        self.orientationOnce = True

        #Stores the indices of the markers considered for the calculation of the rotation angle; these are the intersection of the significant marker motions in the previous and the current frame.
        self.rotationIndices = None

        #Initiating the angle of contact to zero so that we can store it in the form of a dictionary.
        self.angleOfRotation = 0
        self.rotationAngles = []

        self.rotationOnsetIndex = 0

        self.size = (1280, 1000)
        global objectName
        self.out = cv2.VideoWriter(dataPath + "result" + '.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1920,1080))
        # #Process_Image object has a variety of helper functions for processing GelSight Image
        # self.function = Process_Image(self.img, self.flowcenter, self.markerU, self.markerV)
        self.dataPath = dataPath
        self.resulPath = dataPath + "GelSightProcessed/"
        self.resultPathC = dataPath + "GelSightProcessedC/"
        self.truthPath = dataPath + "trueLabelMergeBase/"
        self.raw_resultPath = result_path

        #Creating a nested dictionary to store the final results so that it becomes easy to debug and run algorithms super fast.
        self.dataDict = {}

        if os.path.isdir(self.resulPath):
                shutil.rmtree(self.resulPath, ignore_errors=True)
                print ('folder clean')

        if os.path.isdir(self.resultPathC):
                shutil.rmtree(self.resultPathC, ignore_errors=True)
                print ('folder clean')

        if not os.path.isdir(self.resulPath):
                os.mkdir(self.resulPath)
                print ('folder made')

        if not os.path.isdir(self.resultPathC):
                os.mkdir(self.resultPathC)
                print ('folder made')

        # rospy.Timer(rospy.Duration(0.03), self.show_img_cb)


        # paremeters for display
        self.IsDisplay=False   #whether to publish the marker motion image
        self.showScale=1
        # self.pub = rospy.Publisher('/gelsight/MarkerMotion', Image, queue_size=2)

global start
start = 0

global stop
stop = 0

global objectName
objectName = None

class GelSight_Bridge(object):
    def save_iniImg(self):
        img = cv2.imread(self.dataPath+"/0.jpg")
        self.ini_img=img
        # cv2.imshow("init img",img)
        # cv2.waitKey(0)
        self.frame0=np.int16(cv2.GaussianBlur(self.ini_img, (101,101), 50))

    def __init__(self, dataPath=None):
        # self.bridge = CvBridge()
        self.img = np.zeros((100,100,3))
        self.writepath='/home/robot/catkin_ws/data/'+time.strftime("%y%m%d")+'/'
        self.writecount=0
        self.topic = '/usb_cam/image_raw'
# CardboardBox Hammer MustardBottle Priggles RubberBox Spatula Spoon
        # self.dataObject = "Priggles/Depth1/L5"
        self.dataObject = "CBox_H1_P1_005/"

        global objectName
        objectName = self.dataObject
        # objectName = 'CBox/Depth1/L1'

        # self.dataPath = "../../../data/images/RotationTestData/Robot_Multi_Mode/"+self.dataObject
        # self.dataPath = "../../data/images/09062020/"+self.dataObject+"GelSightImages/sorted/"
        # self.dataPath = dataPath + "GelSightImages/sorted/"
        self.dataPath = dataPath + "gelSightMerge/"
        self.trueDataPath = dataPath + "trueLabelMergeBase/"
        self.folderPath = dataPath
        self.raw_resultPath = result_path
        print("self.resultPath", self.raw_resultPath)
        print("self.dataPath", self.dataPath)

        self.save_iniImg()

        # self.sub_IncomeIm=rospy.Subscriber(self.topic, Image,  self.callback_incomeIm)
        # self.sub_IncomeIm.unregister()
        img = cv2.imread(self.dataPath+"/0.jpg", cv2.COLOR_BGR2GRAY)

        # self.callback_incomeIm(img)

        self.sub_IncomeIm_working=False

        self.saveim_on=False
        self.touchtest_on=False
        self.calMarker_on=False
        self.isContact=False
        self.callbackCount=0
        self.startContactDetectFlag=False

        #Testing
        self.rotationDetect_on = False

        self.args=None
        self.contactFunc=None
        self.contact_Thresh=1
        self.touchtestInied=False
        self.savename='Im'
        self.img_record = np.zeros((480,640,3,500),dtype = np.uint8)
        self.t = 0
        self.trial = 0

        # detect marker motion
        self.GelSightIm=GelSight_Img(self.ini_img, self.frame0, dataPath)


    def show_MarkerImg(self):
        self.GelSightIm.start_display_markerIm()
    def stop_show_MarkerImg(self):
        self.GelSightIm.stop_display_markerIm()

    def start_rotationDetect(self):

        # self.frameIndex = 1
        if not self.rotationDetect_on:
            self.rotationDetect_on = True
            # self.rotationDetect_on+=1
            # self.sub_IncomeIm=rospy.Subscriber(self.topic, Image,  self.callback_incomeIm, queue_size=1)
            global start
            global stop

            count = 0

            for path in pathlib.Path(self.dataPath).iterdir():

                if path.is_file():

                    count += 1


            # print("count", count)
            # print("true label path", self.trueDataPath)
            # small_objects = ['Hammer', 'Mustard','Box']
            # small = False
            # for object in small_objects:
            #   if object in self.trueDataPath:
            #       small = True
            #       break
            ar = int((self.folderPath[:-1]).split('_')[-1])
            # print("ar id is " + str(ar))
            # pause = input("Pause here, please enter any key:")
            #For static analysis, it starts from a certain no. of images and ends analysis at a certain no. of image.
            start = 1
            stop = count-1
            # stop = min(190, count-1)
            gelsightAngles = []
            truelabelAngles = []
            translation = []
            for i in range(start, stop):
                img  = cv2.imread(self.dataPath + str(i) + ".jpg")
                # print(self.dataPath + str(i) + ".jpg")
                self.callback_incomeIm(img)
                gelsightAngles.append(self.GelSightIm.angleOfRotation)
                print("gelsight angle " + str(self.GelSightIm.angleOfRotation))

                translation.append(self.GelSightIm.t_check)

                true_img = cv2.imread( self.trueDataPath + str(i) + ".jpg")
                # print(self.trueDataPath + str(i) + ".jpg")
                if (i == start):
                    first_frame = True
                else:
                    first_frame = False
                true_angle = process_GTimage(true_img,ar,first_frame)
                # true_angle = 0
                truelabelAngles.append(true_angle)
                print("Frame # " + str(i) + ", true angle is " + str(true_angle))
                # self.frameIndex += 1
            # print(gelsightAngles)
            # print(truelabelAngles)
            print("true label path", self.trueDataPath)
            draw_Diag(gelsightAngles,truelabelAngles)
            # draw_translate(gelsightAngles,truelabelAngles, translation)
            # pause = input("Pause here, please enter any key:")


    def callback_incomeIm(self,img):
        self.img= img

        # cv2.imshow("Init", self.img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #Testing
        if self.rotationDetect_on:
            # self.GelSightIm.tracker.track(img)
            self.GelSightIm.update_markerMotion(self.img)

############# Interface with GT #############
def convertCartesianToCV2(img, x1, y1):

    h = img.shape[0]
    l = img.shape[1]

    x_cv2 = x1
    y_cv2 = h - y1

    return x_cv2, y_cv2

def convertCV2ToCartesian(img, x_cv2, y_cv2):

    h = img.shape[0]
    l = img.shape[1]

    xcc = x_cv2
    ycc = h - y_cv2

    return xcc, ycc

def draw_Diag(gelsightAngles,truelabelAngles):
    T = np.array(truelabelAngles)
    T = T[T!=None]
    # print(trueLabel)
    T = -1*T
    G = np.array(gelsightAngles)

    a = int(T.shape[0])
    b = int(G.shape[0])
    low = min(a, b)

    if low == T.shape[0]:
        x = np.arange(0, T.shape[0])
        y1 = T[0:T.shape[0]]
        y2 = G[0:T.shape[0]]
    else:
        x = np.arange(0, G.shape[0])
        y1 = T[0:G.shape[0]]
        y2 = G[0:G.shape[0]]

    plt.plot(x,y1,x,y2)
    plt.show()

def draw_translate(gelsightAngles,truelabelAngles,translation):
    T = np.array(truelabelAngles)
    T = T[T!=None]
    # print(trueLabel)
    T = -1*T
    G = np.array(gelsightAngles)

    C = np.array(translation)

    a = int(T.shape[0])
    b = int(G.shape[0])
    c = int(C.shape[0])
    low = min(a, b)

    if low == T.shape[0]:
        x = np.arange(0, T.shape[0])
        y1 = T[0:T.shape[0]]
        y2 = G[0:T.shape[0]]
        y3 = C[0:T.shape[0]]
    else:
        x = np.arange(0, G.shape[0])
        y1 = T[0:G.shape[0]]
        y2 = G[0:G.shape[0]]
        y3 = C[0:G.shape[0]]

    # plt.plot(x,y1,x,y2,x,y3)
    # plt.show()


    fig, ax1 = plt.subplots()

    # color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('angle')
    ax1.plot(x, y1, color='tab:green')
    ax1.plot(x, y2, color='tab:orange')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    ax2.set_ylabel('translate')  # we already handled the x-label with ax1
    ax2.plot(x, y3, color='tab:blue')
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

global firstImage
firstImage = False

global coordinateInit
coordinateInit = None

global initDirection
initDirection = None

def process_GTimage(frame, ar,first_frame):

    if (ar==2):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        size = 0.02
    elif (ar==4):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        size = 0.048
    elif (ar==6):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        size = 0.044
    else:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        size = 0.048
    # frame = img
    img = frame

    #Giving the Camera Matrix Values
    mtx = np.array([[703.9175373484547, 0.0, 329.62810408111864], [ 0.0, 700.8982411971419, 224.93294160737707], [0.0, 0.0, 1.0]])
    dist = np.array([[-0.064410596], [-0.07054740605], [-0.007974908], [-0.0060625018079], [0.0]])

    #self.Converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detector parameters can be set here
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    global firstImage
    global coordinateInit
    global initDirection

    if (first_frame):
        firstImage = False

    #Lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print(ids)
    # print(ids)
    #Font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    #Making sure the no. of Id's > 0
    if np.all(ids != None):

        ids = np.array(ids)

        if (ids.shape[0]>=2):

            font = cv2.FONT_HERSHEY_SIMPLEX


            #Estimate the pose and get the rotational and translational vectors
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, size, mtx, dist)
            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.048, mtx, dist)


            yc1 = int((corners[0][0][0][1] + corners[0][0][2][1]) / 2)
            xc1 = int((corners[0][0][0][0] + corners[0][0][2][0]) / 2)

            yc2 = int((corners[1][0][0][1] + corners[1][0][2][1]) / 2)
            xc2 = int((corners[1][0][0][0] + corners[1][0][2][0]) / 2)

            if xc1>xc2:

                xc1, xc2 = xc2, xc1
                yc1, yc2 = yc2, yc1

            x1, y1, z1 = tvec[0][0]
            x2, y2, z2 = tvec[1][0]

            # print(z1-z2)

            if xc1>xc2:

                x2, y2 = x1, y1

            xcs1, ycs1 = convertCV2ToCartesian(img, xc1, yc1)
            xcs2, ycs2 = convertCV2ToCartesian(img, xc2, yc2)

            if (firstImage):

                m = coordinateInit

                ycs2k = m*(100) + ycs1

                xc2, yc2k = convertCartesianToCV2(img, xcs2, ycs2k)
                xcs1, ycs1 = convertCV2ToCartesian(img, xc1, yc1)
                xcs2, ycs2 = convertCV2ToCartesian(img, xc2, yc2)

                v0 = [xcs2-xcs1, ycs2k-ycs1]
                v1 = [xcs2-xcs1, ycs2-ycs1]

                angle2 = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
                theta2 = np.degrees(angle2)


                cur_direction = np.array([xc2-xc1,yc2-yc1])
                cos_theta = np.dot(initDirection.T,cur_direction)/(np.linalg.norm(initDirection)*np.linalg.norm(cur_direction))
                theta = np.arccos(cos_theta)
                ang = np.degrees(theta)
                if (np.isnan(ang)):
                    return np.round(0.0,2)

                move = cur_direction-initDirection
                m1 = np.array([move[0],move[1],0])
                m2 = np.array([initDirection[0],initDirection[1],0])
                norm = np.cross(m2,m1)
                if norm[2] > 0:
                    ang = -1*ang
                    print("clockwise")
                else:
                    print("anti-clockwise")

                # if(theta2>0):
                #   print("CCW")
                # else:
                #   print("CW")

                # return np.round(theta2, 2)
                return np.round(ang,2)

            if(not firstImage):
                coordinateInit = ((ycs2-ycs1)/float((xcs2-xcs1)))
                initDirection = np.array([xc2-xc1,yc2-yc1])
                firstImage = True
                return 0


############# Test Functions #################
def detect_rotation():

    # rospy.init_node('recordAnalysis', anonymous=True)

    ''' We are using the color change and marker motion to detect contact. Note that for different sensors, the threshold for the color detection may differ. Please contact Wenzhen for details about adjusting the parameters'''
    ContactThresh=1.5  # the higher the threshold is, the harder for the sensor to decide contact
    def ContactFunc():
        print ('!!!!!In Contact')

    # path = '../../09062020/'
    path = '../../RobotData'
    # path = '../../AdditionalStaticCases'
    # path = '../../VideoData/Sync/'

    files = os.listdir(path)

    sorted_files =  sorted(files)
    # 1008
    # Boxes_P4_D2_S1_30_L1_5_6
    # CBox_H1_P2_008_4
    # CBox_H2_P5_005_4
    # Cup_P2_S1_30_6

    # video
    # sorted_files = ["Boxes_P2_D1_S1_30_L1_5_6","Boxes_P4_D2_S1_30_L1_5_6","Boxes_P5_D1_S1_30_L1_5_6"]
    # sorted_files = ["Boxes_P4_D2_S1_30_L1_5_6"]
    # sorted_files = ["Boxes_P5_D1_S1_30_L1_5_6"]
    # sorted_files = ["CBox_H1_P1_008_4","CBox_H1_P3_008_4","CBox_H2_P6_005_4"]
    # sorted_files = ["CBox_H1_P3_008_4"]
    # sorted_files = ["CBox_H2_P6_005_4"]
    # sorted_files = ["Cylinder_P1_S2_50_L1_5_6","Cylinder_P3_S2_50_L1_5_6","Cylinder_P7_S2_50_L1_5_6"]
    # sorted_files = ["Cylinder_P3_S2_50_L1_5_6"]
    # sorted_files = ["Cylinder_P7_S2_50_L1_5_6"]
    # sorted_files = ["Hammer_P1_S2_50_L1_5_6"]
    # sorted_files = ["MetalPlate_P3_S1_30_6","MetalPlate_P5_S1_30_6"]
    # sorted_files = ["MetalPlate_P5_S1_30_6"]
    # sorted_files = ["Mustard_P1_S2_50_6","Mustard_P2_S1_30_6"]
    # sorted_files = ["Mustard_P2_S1_30_6"]
    # sorted_files = ["SoftScrub_P4_S1_30_6"]
    # sorted_files = ["UnevenBox_P4_S1_30_2","UnevenEvenBox_P2_S1_30_L1_5_6","UnevenEvenBox_P4_S1_30_L2_10_6"]
    # sorted_files = ["UnevenEvenBox_P2_S1_30_L1_5_6"]
    # sorted_files = ["UnevenEvenBox_P4_S1_30_L2_10_6"]

    # sorted_files = ["CBox_H2_P6_005_4"]
    # sorted_files = ["Cylinder_P1_S2_50_L1_5_6"]
    # sorted_files = ["Cylinder_P3_S2_50_L1_5_6"]
    # sorted_files = ["Cylinder_P7_S2_50_L1_5_6"]
    # sorted_files = ["Hammer_P1_S2_50_L1_5_6"]
    # sorted_files = ["MetalPlate_P3_S1_30_6"]
    # sorted_files = ["MetalPlate_P5_S1_30_6"]
    # sorted_files = ["Mustard_P1_S2_50_6"]

    # drawing
    # sorted_files = ["Boxes_P2_D1_S1_30_L1_5_6"]
    # sorted_files = ["CBox_H2_P6_005_4"] #01 # COR
    # sorted_files = ["CleaningLiquid_2"] # not good
    # sorted_files = ["Cylinder_P3_S2_50_L1_5_6"] #02 # COR
    # sorted_files = ["Cylinder_P7_S2_50_L1_5_6"] # not good
    # sorted_files = ["EvenBox_P2_S2_50_L1_5_6"] # not good
    # sorted_files = ["Hammer_P3_S2_50_L1_5_6"] #03
    # sorted_files = ["MetalPlate_P4_S1_30_6"] #04 # super good
    # sorted_files = ["Mustard_P2_S1_30_6"]# 05 # COR
    # sorted_files = ["UnevenBox_P4_S1_30_2"] # not good
    # sorted_files = ["WoodenHammer_P2_S1_40_6"] #06 # good
    # sorted_files = ["Wrench_2"]# 07
    # sorted_files = ["Boxes_P5_D1_S1_30_L1_5_6"]
    # end drawing

    # drawing
    # sorted_files = ["CBox_H1_P3_005_4"] #001
    # sorted_files = ["CBox_H1_P3_008_4"] 002
    # sorted_files = ["CBox_H2_P6_005_4"] 003
    # sorted_files = ["Cylinder_P7_S2_50_L1_5_6"] 004
    # sorted_files = ["CBox_H1_P1_008_4"]
    # end drawing

    # draw contact
    # sorted_files = ["WoodenHammer_P1_S1_40_6"]


    # sorted_files = ["MetalPlate_P1_S2_50_6"]
    # sorted_files = ["MetalPlate_P1_S1_30_6"]
    # sorted_files = ["MetalPlate_P2_S2_50_6"]
    # sorted_files = ["MetalPlate_P5_S2_50_6"]
    # sorted_files = ["UnevenEvenBox_P5_S1_30_L1_5_6"]
    # sorted_files = ["CBox_H2_P5_005_4"]
    # sorted_files = ["CBox_H2_P6_005_4"]
    # sorted_files = ["Boxes_P5_D2_S1_30_L1_5_6"]
    # sorted_files = ["CleaningLiquid_2_PS_S50_F40_G1_6"]
    # last_file = "CleaningLiquid_2_PS_S50_F50_G1_6"
    # start_file = False

    # sorted_files = ["TeaCup_P1_D1_30_L1_5_6"]
    sorted_files = ["Spatula_2"]
    # CBox_H2_P5_005_4 # lifting shaking
    # CBox_H1_P4_005_4 # lifting shaking
    # EvenBox_P3_S1_30_L1_5_6 # lifting shaking
    # Mustard_H1_P1_2 # contact too small
    # SoftScrub_P4_S2_50_6 # contact too small
    # Spatula_2 # contact too small
    # Wrench_P5_S1_30_6 # too hard
    # WoodenHammer_P3_S2_40_6 # contact too small

    fileNames = []

    count = 0
    test_total = 3
    for f in sorted_files:
        if f == ".DS_Store" or f == "Results":
            continue
        # if count == test_total:
        #   break
        # if f == last_file:
        #     start_file = True
        # if start_file == False:
        #     continue
        # path = "../../09062020/"+f+'/'
        path = "../../RobotData/"+f+'/'
        # path = "../../AdditionalStaticCases/"+f+'/'
        # path = '../../VideoData/Sync/'+f+'/'

        GelSight=GelSight_Bridge(path)
        print ("ini done")
        GelSight.show_MarkerImg()
        GelSight.start_rotationDetect()
        count += 1

global result_path
result_path = None

def save_results():
    global result_path
    cur_path = input("Enter your expecting result folder name: ")
    result_path = "../../RobotData/Results/Raw_Data/" + cur_path + "/"
    # result_path = "../../AdditionalStaticCases/Results/Raw_Data/" + cur_path + "/"
    # result_path = "../../VideoData/Sync/Results/Raw_Data/" + cur_path + "/"
    if not os.path.isdir(result_path):
            os.makedirs(result_path)
            print ("folder made : " + cur_path)

if __name__ == '__main__':
    save_results()
    detect_rotation()
    # x = np.array([1,2,3,4,5])
    # y1 = np.array([3,4,6,7,7])
    # y2 = np.array([1,1,2,3,3])
    #
    # plt.plot(x,y1,x,y2)
    # plt.show()
