from re import I

import os
import argparse

import rospy
import matplotlib
matplotlib.use('Agg')

from pancake_drawing import pourring

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
  # Add arguments
    argparse.add_argument('-d',"--depth", default = 4.5, help = "The depth of the pancake")
    args = argparse.parse_args()
    depth = args.depth
    depth = float(depth)
    Fill_letter = ['E', 'F', 'H', 'I', 'K', 'T', 'X', 'Y']
    while True:
        file_name = raw_input("Graphical mode[G] or Text mode[T]: ")
        if file_name == 'G' or file_name == 'g':
            adr = '/home/rocky/samueljin/pancake_bot/Images/'
            break
        elif file_name == 'T' or file_name == 't':
            adr = '/home/rocky/samueljin/pancake_bot/Alphabet_Dataset/'
            break
        else:
            print("Invalid mode, please enter 'G' or 'T'.")
    if file_name == 'G' or file_name == 'g':
        while True:
            file_name2 = raw_input("Please enter image number: ")
            if file_name2.isdigit():
                if file_name2 == '1':
                    mode = 'outline'
                else:
                    mode = 'fill'
                adr += 'file' + file_name2 + '.png'
                break
            else:
                print("Invalid input, please enter a number.")
    else:
        while True:
            file_name2 = raw_input("Please enter the alphabet: ")
            if file_name2.isalpha():
                if file_name2.islower():
                    file_name2 = file_name2.upper()
                adr += file_name2 + '.png'
                if file_name2 in Fill_letter:
                  mode = 'fill'
                else:
                  mode = 'outline'
                break
            else:
                print("Invalid input, please enter a letter.")
 
    rospy.init_node("pancake_pouring")
    runner = pourring.Pourring()
    print(adr, mode)
    runner.pickup()
    runner.pour_shape(depth, adr, mode)
    runner.drop()
    runner.camera_capture()