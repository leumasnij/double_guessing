import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gelsight_lib.gelsight import (
    extract_markers, track_markers, annotate_marker_motions)

class MarkerTracker(object):
  """ Track the marker and publish marker motion annotated image. """
  def __init__(self):
    self.bridge = CvBridge()
    self.prev_markers = None
    self.init_markers = None
    rospy.Subscriber('/gelsight/usb_cam/image_raw', Image, self.cb_image)
    self.publisher = rospy.Publisher(
        '/gelsight/usb_cam/marker_motion', Image, queue_size=10)

  def cb_image(self, msg):
    """
    Image callback function. Track the markers and publish annotated image.

    :param msg: Image; the GelSight image message.
    """
    image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    raw_curr_markers = extract_markers(image)
    if self.init_markers is None:
      self.init_markers = raw_curr_markers
      self.prev_markers = raw_curr_markers
    else:
      curr_markers, marker_dxs, marker_dys = track_markers(
          raw_curr_markers, self.prev_markers, self.init_markers)
      self.prev_markers = curr_markers
      annotated_image = image.copy()
      annotate_marker_motions(
          annotated_image, self.init_markers, marker_dxs, marker_dys)
      annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
      self.publisher.publish(annotated_msg)

if __name__ == "__main__":
  rospy.init_node("marker_tracker_app")
  marker_tracker = MarkerTracker()
  rospy.spin()

