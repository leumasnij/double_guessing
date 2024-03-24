from re import I
import cv2 as cv
from sklearn.cluster import KMeans
import os
import argparse
import numpy as np
import rospy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from ur5_lib import gripper
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from wsg_50_common.msg import Status, Cmd
from scipy.interpolate import interp1d
from scipy import interpolate
from image_decomp import image_decomp_main
import copy
from pancake_drawing import pourring

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
  # Add arguments
    argparse.add_argument("--depth", default = 4.5, help = "The depth of the pancake") 
    argparse.add_argument("--wfratio", default = 1.3, help = "The water flour ratio of the pancake")
    args = argparse.parse_args()
    depth = args.depth
    wfr = args.wfratio
    wfr = float(wfr)
    depth = float(depth)
    rospy.init_node("pancake_pouring")
    runner = pourring.Pourring()
    # runner.tele_op(depth)
    experiment_dir = '/home/rocky/samueljin/pancake_bot/experiments/manual/' + str(wfr) + '1'
    while os.path.exists(experiment_dir):
        experiment_dir = experiment_dir + '1'
    os.makedirs(experiment_dir)
    experiment_dir = experiment_dir + '/'
    runner.line_width_measurement(experiment_dir, 0)
