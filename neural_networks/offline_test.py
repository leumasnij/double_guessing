import os
import cv2
import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_helpers import Net, HapticDataset, GelResNet, GelSightDataset, GelDifDataset, GelRefDataset, GelHapResNet, RegNet, GelRefResNet
import argparse
argparser = argparse.ArgumentParser()
# argparser.add_argument('--dataset', type=str, default='GelDifDataset')
argparser.add_argument('-m', '--model', type=str, default='gel')
argparser.add_argument('--directory', '-d', type=str, default='/media/okemo/extraHDD31/samueljin/CoM_dataset')
args = argparser.parse_args()
model = args.model
address = args.directory
if model != 'gel' and model != 'haptic' and model != 'diff' and model != 'gelhap':
    raise ValueError('Invalid model type')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_lists = []
for file in os.listdir(address):
    file_lists.append(os.path.join(address, file))

random.shuffle(file_lists)
test_size = 20
test_lists = file_lists[:test_size]
gel_offset = [0,0]
haptic_offset = [0,0]
ref_offset = [0,0]
gelhap_offset = [0,0]
diff_offset = [0,0]
for test in test_lists:
    img = cv2.imread(os.path.join(test, 'marker.png'))
    img = cv2.resize(img, (480, 640))/255.0
    ref = cv2.imread(os.path.join(test, 'marker_ref.png'))
    ref = cv2.resize(ref, (480, 640))/255.0
    imgdiff = img - ref
    imgdiff = torch.tensor(imgdiff).float()
    imgdiff = imgdiff.permute(2, 0, 1)
    imgref = np.concatenate((img, ref), axis=2)
    imgref = torch.tensor(imgref).float()
    imgref = imgref.permute(2, 0, 1)
    img = torch.tensor(img).float()
    img = img.permute(2, 0, 1)
    dic = np.load(os.path.join(test, 'data.npy'), allow_pickle=True, encoding='latin1').item()
    hap = dic['force']
    hap = torch.tensor(hap).float()
    loc = np.loadtxt(os.path.join(test, 'loc.txt'))[:2]*100
    print('True: ', loc)


    model = GelResNet().to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gel_best_model.pth'))
    model.eval()
    with torch.no_grad():
        loc_pred = model(img.unsqueeze(0).to(device))
        print('Gel: ', loc_pred.cpu().numpy())
        gel_offset += np.abs(loc - loc_pred.cpu().numpy())
    
    model = GelResNet().to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/dif_best_model.pth'))
    model.eval()
    with torch.no_grad():
        loc_pred = model(imgdiff.unsqueeze(0).to(device))
        print('Diff: ', loc_pred.cpu().numpy())
        diff_offset += np.abs(loc - loc_pred.cpu().numpy())
    model = RegNet(input_size=6).to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/hap_best_model.pth'))
    model.eval()
    with torch.no_grad():
        loc_pred = model(hap.unsqueeze(0).to(device))
        print('Haptic: ', loc_pred.cpu().numpy())
        haptic_offset += np.abs(loc - loc_pred.cpu().numpy())
    model = GelRefResNet().to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/ref_best_model.pth'))
    model.eval()
    with torch.no_grad():
        loc_pred = model(imgref.unsqueeze(0).to(device))
        print('Ref: ', loc_pred.cpu().numpy())
        ref_offset += np.abs(loc - loc_pred.cpu().numpy())
    model = GelHapResNet().to(device)
    model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gelhap_best_model.pth'))
    model.eval()
    with torch.no_grad():
        loc_pred = model(imgref.unsqueeze(0).to(device), hap.unsqueeze(0).to(device))
        print('GelHap: ', loc_pred.cpu().numpy())
        gelhap_offset += np.abs(loc - loc_pred.cpu().numpy())


print('Average offset:')
print('Gel: ', gel_offset/test_size)
print('Haptic: ', haptic_offset/test_size)
print('Ref: ', ref_offset/test_size)
print('GelHap: ', gelhap_offset/test_size)
print('Diff: ', diff_offset/test_size)
    



# if model == 'gel':
#     model = GelResNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gel_best_model.pth'))
# elif model == 'haptic':
#     model = RegNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/hap_best_model.pth'))
# elif model == 'diff':
#     model = GelResNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/diff_best_model.pth'))
# elif model == 'gelhap':
#     model = GelHapResNet().to(device)
#     model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gelhap_best_model.pth'))

