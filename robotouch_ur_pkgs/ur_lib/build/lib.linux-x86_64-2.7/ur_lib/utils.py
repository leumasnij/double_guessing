#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
Hung-Jui Huang (hungjuih@andrew.cmu.edu) Sept, 2021
'''

import os
import time
import csv
from datetime import datetime
import numpy as np

def get_current_timestamp():
  """ Get current timestamp in integer. """
  now = datetime.now()
  x = np.datetime64(now)
  timestamp = x.view('i8')
  return int(timestamp) 

def create_data_dir(parent_dir):
  """
  Create the directory that saved the data.

  :param parent_dir: string; the parent directory.
  :return data_dir: string; the data directory.
  """
  timestamp = get_current_timestamp()
  prefix = raw_input('Enter Directory Prefix Name: ')
  data_subdir = prefix + '_' + str(timestamp)
  data_dir = os.path.join(parent_dir, data_subdir)
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    print('Current object directory: %s'%data_dir)
  return data_dir

def robot_movement_direction_user_input():
  """
  User input robot movement.

  :return direction: int;
    'w': up, 's': down, 'a': left, 'd': right, 'z': front,
    'x': back, 'g': grasp, 'r': release, 'q': quit
  """
  while True:
    try:
      print('Select Movement Direction: [UP: \'w\', DOWN: \'s\', LEFT: \'a\', RIGHT: \'d\', FRONT: \'z\', BACK: \'x\', GRASP: \'g\', RELEASE: \'r\', EXIT: \'q\']')
      direction = raw_input()
    except ValueError:
      print('Invalid selection! Please try again')
      continue
    if direction not in ['w', 's', 'a', 'd', 'z', 'x', 'g', 'r', 'q']:
      print('Invalid selection! Select from the given options')
      continue
    else:
      break
  return direction  
