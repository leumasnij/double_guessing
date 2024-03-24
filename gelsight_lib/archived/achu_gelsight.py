import cv2
import time
import numpy as np
import os
'''

Version V2
    added support for Gelsight Inc Sensor
'''

#Finds the location of the markers on the Gelpad Image.
#It uses findContours algorithm of OpenCV to find markers as areas bounded by the countours.
#It returns MarkerCenters, a numpy array whose each row represents x & y location of the markers
#and the area of each marker or contour area.

#findContour method returns different values in different OpenCV versions. Hence ensuring cross version compatibility.

#Checks for version of OpenCV
def check_opencv_version(major, lib=None):
    if lib is None:
        import cv2 as lib
    return lib.__version__.startswith(major)

#Check for OpenCV 2.x
def is_cv2():
    return check_opencv_version("2.")

#Check for OpenCV 3.x
def is_cv3():
    return check_opencv_version("3.")
#Check for OpenCV 4.x
def is_cv4():
    return check_opencv_version("4.")

# img: Present Image, img_init: Initial Image
def find_markers(img, numMarkers=None):
    img_gaussian = (cv2.GaussianBlur(img, (5,5), 0))
    img_gray  = cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2GRAY)
    ret1, img_thr =cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
    #cv2.imshow("Gaussian BLR",img_gaussian)
    #cv2.imshow("Binary",img_thr)

    MarkerCenter=np.empty([0, 3])

    contours, hierarchy = cv2.findContours(img_thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    #cnt = sorted(cnts, key=cv2.contourArea, reverse =True)

    for i in range(len(contours)):
        h = hierarchy[0,i]
        #to get rid of boundary
        if(h[2]==-1 and h[3]==0):
            #cv2.drawContours(img, contours, i, (0,255,0), 3)
            AreaCount=cv2.contourArea(contours[i])
            #calculate moments and contour centers
            t=cv2.moments(contours[i])
            MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)

    return MarkerCenter[:numMarkers]

#Calculates motion of marker at each markercentre.
#In each iteration, it finds corresponding markers by finding the markers that are the nearest coupled with finding the markers that match in contour areas between the frames being compared.
def update_markerMotion(marker_present, marker_prev, marker_init):

    #No. of markers identified in the initial frame. This remains constant through all the frames.
    markerCount=len(marker_init)

    markerU=np.zeros(markerCount)    # X motion of all the markers. 0 if the marker is not found
    markerV=np.zeros(markerCount)    # Y motion of all the markers. 0 if the marker is not found

    #No. of markers in the present frame.
    Nt         = len(marker_present)

    #Temporary variable used for analysis
    no_seq2 = np.zeros(Nt)

    #center_now is the variable that will be returned by the function that contains equal no. of centers as the initial frame.
    center_now = np.zeros([markerCount, 3])

    for i in range(Nt):

        #Calculating the motion of each marker in the present frame w.r.to all the markers in the previous frame.
        dif=np.abs(marker_present[i,0]-marker_prev[:,0])+np.abs(marker_present[i,1]-marker_prev[:,1])
        #Multiplying the above variable with the difference of the contour area of the each marker w.r.to the contour areas of all the markers in the previous frame.
        no_seq2[i]=np.argmin(dif*(100+np.abs(marker_present[i,2]-marker_init[:,2])))

    for i in range(markerCount):

        dif=np.abs(marker_present[:,0]-marker_prev[i,0])+np.abs(marker_present[:,1]-marker_prev[i,1])
        t=dif*(100+np.abs(marker_present[:,2]-marker_init[i,2]))

        #a is a threshold further used in the analysis to filter out markers that might not have significant marker motion.
        a=np.amin(t)/100
        b=np.argmin(t)

        #If the contour area of a marker in the present frame is less than obtained 'a', set the x and y motion of that marker 0
        if marker_init[i,2]<a:   # for small area
            markerU[i]=0
            markerV[i]=0
            center_now[i]=marker_prev[i]

        #When the index i matches the index of the variable b, x and y motion are calculated w.r.to the initial marker location.
        elif i==no_seq2[b]:
            markerU[i]=marker_present[b,0]-marker_init[i,0]
            markerV[i]=marker_present[b,1]-marker_init[i,1]
            center_now[i]=marker_present[b]
        else:
            markerU[i]=0
            markerV[i]=0
            center_now[i]=marker_prev[i]

    return center_now, markerU, markerV

# Displays the marker motion as yellow lines between the present marker position w.r.to the marker position in the initial frame.
def draw_markers(img, marker_init, markerU, markerV, showScale):

    #Just rounding markerCenters location
    markerCenter=np.around(marker_init[:,0:2]).astype(np.int16)
    blk  = np.zeros((img.shape[0],img.shape[1]))
    for i in range(marker_init.shape[0]):

        if markerU[i]!=0:

            cX = int(marker_init[i,0])
            cY = int(marker_init[i,1])

            #This line draws lines from the marker position in the initial frame to the marker position in the current frame.
            cv2.arrowedLine(img,(markerCenter[i,0], markerCenter[i,1]), \
                (int(marker_init[i,0]+markerU[i]*showScale), int(marker_init[i,1]+markerV[i]*showScale)),\
                (0, 255, 255),2)
            cv2.arrowedLine(blk,(markerCenter[i,0], markerCenter[i,1]), \
                (int(marker_init[i,0]+markerU[i]*showScale), int(marker_init[i,1]+markerV[i]*showScale)),\
                (255),2)

    return  img, blk
    #cv2.imshow("marker_init Image", img)
    #cv2.destroyAllWindows()
    #cv2.waitKey(0)
