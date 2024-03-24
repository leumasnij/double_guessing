#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
'''

import os
import time
import csv
from datetime import datetime
import numpy as np

def select_operation():
	while True:
		try:
			print('Select Operation [1/2]:')
			print('1. Robot Movement')
			print('2. Data Collection')
			operation = raw_input()
		except ValueError:
			print('Invalid selection. Please try again')
			continue
		if operation not in ['1', '2']:
			print('Invalid selection! Select from the given options')
			continue
		else:
			break
	return operation

def get_current_time_stamp():
	now = datetime.now()
	x = np.datetime64(now)
	timestamp = x.view('i8')
	return int(timestamp)	


def create_data_dir(parent_dir):
	timestamp = get_current_time_stamp()
	object_name = raw_input('Enter Object Name: ')

	trial_path = object_name + '_' + str(timestamp)
	trial_dir = os.path.join(parent_dir, trial_path)
	os.mkdir(trial_dir)
	print('Current Object Directory:')
	print(trial_dir)

	return trial_dir, trial_path

# User Input Handlers
def robot_movement_direction_user_input():
	while True:
		try:
			print('Select Moverment Direction: [UP: \'w\', DOWN: \'s\', LEFT: \'a\', RIGHT: \'d\', FRONT: \'z\', BACK: \'x\', GRASP: \'g\', RELEASE: \'r\']')
			direction = raw_input()
		except ValueError:
			print('Invalid selection! Please try again')
			continue
		if direction not in ['w', 's', 'a', 'd', 'z', 'x', 'g', 'r']:
			print('Invalid selection! Select from the given options')
			continue
		else:
			break
	return direction  

def reached_destination():
	while True:
		try:
			print('Reached Destination? [y/n]')
			out = raw_input()
		except ValueError:
			print('Invalid selection. Please try again')
			continue
		if out not in ['y', 'n']:
			print('Invalid selection! Select from the given options')
			continue
		else:
			break
	if out == 'y':
		return True
	return False



# Data Annotation Helper Functions
def _label(swipe, writer):
	# choices = []
	while True:
		choice = int(raw_input('Label: Enter your choice [0/1/2/3]- '))
		if choice in [0, 1, 2, 3]:
			# choices.append(choice)
			if choice == 0:
				writer.writerow([swipe, 'blank'])
			elif choice == 1:
				writer.writerow([swipe, 'miss'])
			elif choice == 2:
				writer.writerow([swipe, 'insert'])
			elif choice == 3:
				writer.writerow([swipe, 'not present'])
			break
		else:
			print('Invalid Selection')
			continue
	return choice

def label_data(trial_dir, v):
	label_path = os.path.join(trial_dir, 'label.csv')

	print('*'*30)
	print('Data Labelling: ')
	print('0. Blank')
	print('1. Miss')
	print('2. Insert')
	print('3. Not Present')

	f = open(label_path, 'a+')
	writer = csv.writer(f)
	
	swipe = 'swipe_' + str(v)
	choice = _label(swipe, writer)
	# writer.writerow(['force', force])
	f.close()
	print('*'*30)
	return choice

	# f = open(log_dir, 'a+')
	# writer = csv.writer(f)
	# writer.writerow([trial_dir, choices])

