from gelsight_lib import gelsight
from gelsight_lib import markerMotion as mm
import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
ret, frameI = cap.read()
markerI = mm.find_markers(frameI, frameI)
while True:
    ret, frame = cap.read()
    # markers = gelsight.extract_markers(frame)
    markers = mm.find_markers(frame, frameI)
    markers, U, V = mm.update_markerMotion(markers, markerI, markerI)
    frame = mm.displaycentres(frame, markerI, U, V, 1)



    display_frame = frame.copy()
    display_frame = cv2.resize(display_frame, (656, 493))
    cv2.imshow('frame', display_frame)
    # print(display_frame.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break