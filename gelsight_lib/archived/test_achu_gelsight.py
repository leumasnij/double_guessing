from get_image import imageReader
import cv2
import time
import numpy as np

from gelsight_lib.archived.achu_gelsight import draw_markers, find_markers, update_markerMotion
#Scale of marker movement in visualization.
showScale = 3


cam0=imageReader('http://gsr15009.local:8080/?action=stream')
#c1=imageReader('http://192.168.255.10:8089/stream')

backSub = cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold = 400.0,detectShadows = False )
#backSub = cv2.createBackgroundSubtractorMOG2()

init_frame = cam0.getFrame()
#init_frame =  correctImage(init_frame)
marker_init = find_markers(init_frame)
marker_prev = marker_init.copy()

while True:

    img = cam0.getFrame()
    fgMask = backSub.apply(img, learningRate = -1)
    diff = cv2.absdiff(img, init_frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret1, img_thr =cv2.threshold(mask,0,255,cv2.THRESH_OTSU)


    #img = correctImage(im0)
    #im1 = c1.getFrame()
    marker_present = find_markers(img)
    marker_prev,markerU,markerV = update_markerMotion(marker_present,marker_prev,marker_init)
    imc=img.copy()
    marker_im, marker_im_gray = draw_markers(imc,marker_init,markerU,markerV,showScale)
    #cv2.imshow("IMAGE0",im0)
    cv2.imshow("IMAGE1",img)
    cv2.imshow("Markers",marker_im)
    cv2.imshow('FG Mask', fgMask)
    cv2.imshow('Diff', mask)
    cv2.imshow("Thr", img_thr)

    cv2.waitKey(1)
