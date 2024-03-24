from re import I
from sklearn.cluster import KMeans
import argparse
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
    argparse.add_argument('-a', "--diameter", default = 10, help = "The diameter of the pancake")
    argparse.add_argument('-m', "--method", default = 'our', help = "our method[our] or simple method[base]")
    args = argparse.parse_args()
    depth = args.depth
    wfratio = args.wfratio
    diameter = args.diameter
    method = args.method
    depth = float(depth)
    wfratio = float(wfratio)
    diameter = float(diameter)
    if method != 'our' and method != 'base':
        print("Invalid method, please enter 'our' or 'base'.")
        exit()
    
    if wfratio < 1.0 or wfratio > 2.0:
        print("Invalid water flour ratio, please enter a number between 1.0 and 2.0.")
        exit()
    if diameter > 25:
        print("Invalid diameter, please enter a number less than 25.")
        exit()
    model = SimpleNet()
    model.load_state_dict(torch.load('round_prediction.pth'))
    input = torch.tensor([[wfratio, diameter]])
    time = model(input).item()
    print(time)
    rospy.init_node("pancake2")
    runner = pourring.Pourring()
    runner.pickup()
    if method == 'our':
        runner.fixed_pour(depth=depth, time=time)
    else:
        runner.baseline_pour(depth=depth, time=time)
    runner.drop()
    runner.calc_round_area()

    