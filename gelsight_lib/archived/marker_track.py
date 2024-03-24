import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import skimage.morphology
import pickle
import argparse
from gelsight import extract_markers


class MarkerTracking:
  def __init__(self, images):
    self.images = images

  def track(self, init_image):
    # Extract the first markers
    init_markers = extract_markers(init_image)[:, :2].astype(np.float32)
    self.n_markers = init_markers.shape[0]

    init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2GRAY)
    curr_markers = init_markers.copy()
    corners = init_markers.copy()
    self.corners = corners.copy()
    self.firstPos = np.array([corners[:, 0], corners[:, 1]]).transpose()
    self.lastPos = np.array([corners[:,0], corners[:,1]]).transpose()

    self.marker_x = -np.ones([0, self.n_markers])
    self.marker_y = -np.ones([0, self.n_markers])

    for img in self.images:
      image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      curr_markers = self.match(init_image, image, init_markers, curr_markers)
      self.lastPos = self.curPos.copy()


  def match(self, init_image, image, init_markers, curr_markers):
    pl, st, err = cv2.calcOpticalFlowPyrLK(init_image, image, init_markers, curr_markers, maxLevel=5, winSize=(100, 100),
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.001),
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    marker_x = -np.ones([self.n_markers])
    marker_y =  -np.ones([self.n_markers])
    curPos = np.array([pl[:, 0], pl[:, 1]]).transpose()
    for i in range(0, self.n_markers):
      marker_x[i] = curPos[i,0]-self.firstPos[i,0]
      marker_y[i] = curPos[i,1]-self.firstPos[i,1]
    self.curPos = curPos.copy()
    self.marker_x = np.row_stack((self.marker_x, marker_x))
    self.marker_y = np.row_stack((self.marker_y, marker_y))
    return pl

def track_markers_opticalflow(init_image, images):
  mk = MarkerTracking(images)
  mk.track(init_image)
  init_markers = mk.corners
  temporal_marker_dxs = mk.marker_x
  temporal_marker_dys = mk.marker_y
  return init_markers, temporal_marker_dxs, temporal_marker_dys
