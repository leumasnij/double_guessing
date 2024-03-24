import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Zilin version
def detect_contact_area(img, f0):
  diffim = np.abs(np.int16(img) - np.int16(f0))
  max_img = np.sum(diffim,2)
  contactmap=max_img
  countnum=(contactmap>20).sum()
  # print(countnum)
  contactmap[contactmap<20]=0
  # contactmap[contactmap<=0]=0
  image = np.zeros((480,640))
  maxC = np.max(contactmap)
  sec90C = np.percentile(contactmap, 90)
  sec95C = np.percentile(contactmap, 95)
  sec99C = np.percentile(contactmap, 99)
  contact_mask = contactmap>0.8*sec90C
  img[contact_mask] = np.array([0, 0, 0])
  return img

# Uksang version
#def detect_contact_area(image, f0):
#  image0 = np.int16(cv2.GaussianBlur(f0, (101,101), 50))
#  diffim = np.array(image) - image0
#  contactmap = np.amax(diffim, 2)
#  countnum = (contactmap > 10).sum()
#  contactmap[contactmap < 10] = 0
#  image = np.zeros((480, 640, 3), np.uint8)
#  maxC = np.max(contactmap)
#  sec90C = np.percentile(contactmap, 90)
#  sec95C = np.percentile(contactmap, 95)
#  sec99C = np.percentile(contactmap, 99)
#  contact_mask = contactmap > 0.4 * sec90C
#  image[contact_mask, 0] = 255
#  image[contact_mask, 1] = 255
#  image[contact_mask, 2] = 255
#  return image

class DetectContactArea(object):
  """ Detect the contact area and masked out from input image. """
  def __init__(self):
    self.bridge = CvBridge()
    rospy.Subscriber('/gelsight/usb_cam/image_raw', Image, self.cb_image)
    self.publisher = rospy.Publisher(
        '/gelsight/usb_cam/contact_area', Image, queue_size=10)
    self.init_image = None

  def cb_image(self, msg):
    """
    Image callback function. Detect Contact area and mask image.

    :param msg: Image; the GelSight image message.
    """
    image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    if self.init_image is None:
      self.init_image = image
    else:
      masked_image = detect_contact_area(image, self.init_image)
      masked_msg = self.bridge.cv2_to_imgmsg(masked_image, 'bgr8')
      self.publisher.publish(masked_msg)

if __name__ == "__main__":
  rospy.init_node("detect_contact_area_app")
  gelsight = DetectContactArea()
  rospy.spin()

