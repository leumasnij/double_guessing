from re import I
import cv2 as cv
from sklearn.cluster import KMeans
import os
import argparse
import numpy as np
import rospy
import matplotlib
matplotlib.use('Agg')
from pancake_drawing import pourring
import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # Input layer with 2 inputs
        self.fc2 = nn.Linear(32, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 1)  # Output layer with 1 output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
  # Add arguments
    argparse.add_argument('-d', "--depth", default = 4.5, help = "The depth of the pancake")
    argparse.add_argument('-r', "--wfratio", default = 1.3, help = "The water flour ratio of the pancake")
    argparse.add_argument('-w', "--width", default = 5, help = "The width of the batter stroke")
    args = argparse.parse_args()
    depth = args.depth
    wfratio = args.wfratio
    width = args.width
    depth = float(depth)
    wfratio = float(wfratio)
    width = float(width)

    if wfratio < 1.0 or wfratio > 2.0:
        print("Invalid water flour ratio, please enter a number between 1.0 and 2.0.")
        exit()
    if width > 5:
        print("Invalid wdith, please enter a number less than 5.")
        exit()
    model = SimpleNet()
    model.load_state_dict(torch.load('stroke_width.pth'))
    input = torch.tensor([[wfratio, width]])
    velocity = model(input).item()
    print(velocity)
    rospy.init_node("pancake2")
    runner = pourring.Pourring()
    #runner.drop()
    # runner.single_line_drawing(depth, velocity)
    # runner.line_width_measurement( )
    runner.run_single_line([velocity], depth, wfratio)