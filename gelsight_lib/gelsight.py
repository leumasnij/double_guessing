import cv2
import numpy as np
import matplotlib.pyplot as plt

AREA_LOWER_THRESHOLD = 2
AREA_UPPER_THRESHOLD = 900

def warp_gelwedge(image):
  """
  Warp the GelWedge image.
  :param image: PIL.image; the GelWedge image to be warped.
  :return image: PIL.image; the warped image.
  """
  # Manually pick the matching points
  src = np.array([[29., 196.],
                  [29., 461.],
                  [457., 495.],
                  [453., 153.]], dtype=np.float32)
  dst = np.array([[30., 30.],
                  [450., 30.],
                  [450., 610.],
                  [30., 610.]], dtype=np.float32)
  M = cv2.getPerspectiveTransform(src, dst)
  warped = cv2.warpPerspective(image, M, (480, 640), flags=cv2.INTER_LINEAR)
  warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
  return warped

def extract_markers(image, marker_threshold = -40, threshold_mode="max"):
  """
  Extract the markers position.

  :param image: np.ndarray; the image.
  :param marker_threshold: float; the color threshold to get markers.
  :param threshold_mode: string; using "max" or "sum" for thresholding.
  :return markers: np.ndarray (n, (x, y, area)); marker states.
  """
  image_gaussian = np.int16(cv2.GaussianBlur(image, (101, 101), 50))
  I = image.astype(np.double) - image_gaussian.astype(np.double)
  if threshold_mode == "max":
    marker_mask = ((np.max(I, 2)) < marker_threshold).astype(np.uint8)
  elif threshold_mode == "sum":
    marker_mask = ((np.sum(I, 2)) < marker_threshold).astype(np.uint8)
  else:
    raise ValueError("Extract Markers Error: Threshold Mode Not Exist.")
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, kernel)
  contours, _ = cv2.findContours(
      marker_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  markers = []
  for contour in contours:
    area = cv2.contourArea(contour)
    if area > AREA_LOWER_THRESHOLD and area < AREA_UPPER_THRESHOLD:
      moment = cv2.moments(contour)
      markers.append([moment['m10'] / moment['m00'],
                      moment['m01'] / moment['m00'], 
                      area])
  markers = np.array(markers)
  return markers

def track_markers(raw_curr_markers, prev_markers, init_markers):
  """
  Track the markers. Return the marker motion and the current markers in
  the initial marker orders.

  :param raw_curr_markers: np.ndarray (n, (x, y, area));
      current untracked markers.
  :param prev_markers: np.ndarray (n, (x, y, area)); previous tracked markers.
  :param init_markers: np.ndarray (n, (x, y, area)); initial markers.
  :return
    curr_markers: np.ndarray (n, (x, y, area)); current tracked markers.
    marker_dxs: np.ndarray (n, ); current marker movement in x.
    marker_dys: np.ndarray (n, ); current marker movement in y.
  """
  n_markers = len(init_markers)
  n_raw_curr_markers = len(raw_curr_markers)
  # Current marker motions and states in init orders
  marker_dxs = np.zeros(n_markers)
  marker_dys = np.zeros(n_markers)
  curr_markers = np.zeros_like(init_markers)
  # Raw current marker orders
  marker_orders = np.zeros(n_raw_curr_markers)
  # Match raw current markers with the initial markers
  for i in range(n_raw_curr_markers):
    dxs = np.abs(raw_curr_markers[i, 0] - prev_markers[:, 0])
    dys = np.abs(raw_curr_markers[i, 1] - prev_markers[:, 1])
    dareas = np.abs(raw_curr_markers[i, 2] - init_markers[:, 2])
    marker_orders[i] = np.argmin((dxs + dys) * (100 + dareas))
  for i in range(n_markers):
    dxs = np.abs(raw_curr_markers[:, 0] - prev_markers[i, 0])
    dys = np.abs(raw_curr_markers[:, 1] - prev_markers[i, 1])
    dareas = np.abs(raw_curr_markers[:, 2] - init_markers[i, 2])
    metrics = (dxs + dys) * (100 + dareas)
    area_threshold = np.amin(metrics) / 100.0
    inv_marker_order = np.argmin(metrics)
    if init_markers[i, 2] < area_threshold:
      # If marker is small and move little
      marker_dxs[i] = 0
      marker_dys[i] = 0
      curr_markers[i] = prev_markers[i]
    elif i == marker_orders[inv_marker_order]:
      # Tracked motion calculated
      marker_dxs[i] = raw_curr_markers[inv_marker_order, 0] - init_markers[i, 0]
      marker_dys[i] = raw_curr_markers[inv_marker_order, 1] - init_markers[i, 1]
      curr_markers[i] = raw_curr_markers[inv_marker_order]
    else:
      # Unmatched markers
      marker_dxs[i] = 0
      marker_dys[i] = 0
      curr_markers[i] = prev_markers[i]
  return curr_markers, marker_dxs, marker_dys

def track_markers_opticalflow(init_gray_image, curr_gray_image, init_markers, prev_markers):
  """
  Track the markers using optical flow. Return the marker motions and current markers.

  :param init_gray_image: np.ndarray; the initial gray image.
  :param curr_gray_image: np.ndarray; the current gray image.
  :param init_markers: np.ndarray (float32) (n, (x, y)); initial markers.
  :param prev_markers: np.ndarray (float32) (n, (x, y)); previous tracked markers.
  :return
    curr_markers: np.ndarray (n, (x, y)); current tracked markers.
    marker_dxs: np.ndarray (n, ); current marker movement in x.
    marker_dys: np.ndarray (n, ); current marker movement in y.
  """
  curr_markers, _, _ = cv2.calcOpticalFlowPyrLK(
      init_gray_image, curr_gray_image, init_markers, prev_markers,
      maxLevel=5, winSize=(100, 100),
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.001),
      flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
  marker_dxs = curr_markers[:, 0] - init_markers[:, 0]
  marker_dys = curr_markers[:, 1] - init_markers[:, 1]
  return curr_markers, marker_dxs, marker_dys

def annotate_markers(image, markers):
  """
  Annotate the input image with markers.

  :param image: np.ndarray; the image to be annotated.
  :param markers: np.ndarray (n, (x, y, area)); marker states.
  """
  for marker in markers:
    center = (int(marker[0]), int(marker[1]))
    radius = np.sqrt(marker[2] / np.pi)
    cv2.circle(image, center, int(radius), (0, 255, 255), 3)

def annotate_marker_motions(
    image, init_markers, marker_dxs, marker_dys, arrow_scale = 2):
  """
  Annotate the input image with marker motions.

  :param image: np.ndarray; the image to be annotated.
  :param init_markers: np.ndarray (n, (x, y, area)); inital markers.
  :param marker_dxs: np.ndarray (n, ); current marker movement in x.
  :param marker_dys: np.ndarray (n, ); current marker movement in y.
  :param arrow_scale: int; the scale of the arrow when annotating.
  """
  for i in range(init_markers.shape[0]):
    if marker_dxs[i] != 0 or marker_dys[i] != 0:
      begin_center = (int(init_markers[i, 0]), int(init_markers[i, 1]))
      end_center = (int(init_markers[i, 0] + marker_dxs[i] * arrow_scale),
                    int(init_markers[i, 1] + marker_dys[i] * arrow_scale))
      cv2.arrowedLine(image, begin_center, end_center, (0, 255, 255), 3)
