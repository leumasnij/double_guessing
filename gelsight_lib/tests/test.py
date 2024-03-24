import os
import cv2
import matplotlib.pyplot as plt
from gelsight_lib.gelsight import (
    extract_markers, track_markers, annotate_markers, annotate_marker_motions,
    warp_gelwedge)

def test_warp_gelwedge():
  """ Test the warp_gelwedge function. """
  image = cv2.imread('gelsight_lib/tests/test_gelwedge_initial.png')
  image = warp_gelwedge(image)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.show()

def test_extract_markers1():
  """ Test the extract_markers function. """
  image = cv2.imread('gelsight_lib/tests/test_initial.png')
  markers = extract_markers(image)
  annotated_image = image.copy()
  annotate_markers(annotated_image, markers)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  plt.imshow(annotated_image)
  plt.show()

def test_extract_markers2():
  """ Test the extract_markers function. """
  image = cv2.imread('gelsight_lib/tests/test_gelwedge_initial.png')
  image = warp_gelwedge(image)
  markers = extract_markers(image, marker_threshold=-120, threshold_mode='sum')
  annotated_image = image.copy()
  annotate_markers(annotated_image, markers)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  plt.imshow(annotated_image)
  plt.show()

def test_track_markers():
  """ Test the track_markers function. """
  init_image = cv2.imread('gelsight_lib/tests/test_initial.png')
  prev_image = cv2.imread('gelsight_lib/tests/test_current.png')
  init_markers = extract_markers(init_image)
  prev_markers = extract_markers(prev_image)
  prev_markers, marker_dxs, marker_dys = track_markers(
      prev_markers, init_markers, init_markers)
  annotated_image = init_image.copy()
  annotate_marker_motions(annotated_image, init_markers, marker_dxs, marker_dys)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  plt.imshow(annotated_image)
  plt.show()

if __name__ == "__main__":
  #test_warp_gelwedge()
  #test_extract_markers1()
  test_extract_markers2()
  #test_track_markers()
