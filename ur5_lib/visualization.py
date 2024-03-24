import numpy as np
import os
import math
import matplotlib.pyplot as plt
import subprocess

folder = '/media/okemo/extraHDD1/Shubham/PegInHole/final/dia_6_dist_1_1621350401175063/processed/Swipe_9'
rgb_path = folder + '/rgb'
gelsight_path = folder + '/gelsight'
marker_path = folder + '/marker_magnitudes.npy'
sidecam_path = folder + '/side_cam'

def truncate(number, decimals):
  if not isinstance(decimals, int):
    raise TypeError("decimal places must be an integer")
  elif decimals < 0:
    raise TypeError("decimal places has to be 0 or more")
  elif decimals == 0:
    return math.trunc(number)
  factor = 10.0 ** decimals
  return math.trunc(number * factor) / factor

def getRgbData(idx):
  _, _, files = next(os.walk(rgb_path))
  files.sort(key=lambda f: int(filter(str.isdigit, f)))
  return files[idx]

def getGelsightData(idx):
  _, _, files = next(os.walk(gelsight_path))
  files.sort(key=lambda f: int(filter(str.isdigit, f)))
  return files[idx]

def getSidecamData(idx):
  _, _, files = next(os.walk(sidecam_path))
  files.sort(key=lambda f: int(filter(str.isdigit, f)))
  return files[idx]

def visualize(rgb_sync, gelsight_sync, sidecam_sync, marker_sync, marker_multiplier):
  plt.figure()
  plt.ion()

  for i in range(len(rgb_sync)):
    plt.subplot(221)
    rgb_image = plt.imread(rgb_sync[i])
    plt.axis('off')
    plt.imshow(rgb_image)
    plt.xlabel('RGB Data')

    plt.subplot(222)
    gelsight_image = plt.imread(gelsight_sync[i])
    plt.axis('off')
    plt.imshow(gelsight_image)
    plt.xlabel('Gelsight Data')

    plt.subplot(223)
    sidecam_image = plt.imread(sidecam_sync[i])
    plt.axis('off')
    plt.imshow(sidecam_image)
    plt.xlabel('Sidecam Data')  

    plt.subplot(224)
    plt.plot(marker_sync)
    plt.axvline(x=(i*marker_multiplier), color='#df00fe')
    plt.xlabel('Iterations')
    plt.ylabel('Marker Magnitude')

    plt.suptitle('Data Visualization')
    plt.show()
    plt.savefig(folder + "/file%02d.png" % i)
    plt.pause(0.01)
    plt.clf() 


def main():
  _, _, rgb_files = next(os.walk(rgb_path))
  rgb_count = len(rgb_files)

  _, _, gelsight_files = next(os.walk(gelsight_path))
  gelsight_count = len(gelsight_files)

  _, _, sidecam_files = next(os.walk(sidecam_path))
  sidecam_count = len(sidecam_files)


  marker_files = np.load(marker_path)
  marker_count = marker_files.shape[0]

  print("RGB Count: ", rgb_count, "Gelsight Count: ", gelsight_count, "Sidecam Count: ", sidecam_count)  

  idx_counter = min(rgb_count, gelsight_count, sidecam_count, marker_count)
  print("Counter Range: ", idx_counter)

  rgb_multiplier = truncate((float(rgb_count) / idx_counter), 2)
  gelsight_multiplier = truncate((float(gelsight_count) / idx_counter), 2)
  marker_multiplier = truncate((float(marker_count) / idx_counter), 2)
  sidecam_multiplier = truncate((float(sidecam_count) / idx_counter), 2)

  print('Multipliers: ', rgb_multiplier, gelsight_multiplier, sidecam_multiplier, marker_multiplier)

  rgb_sync, gelsight_sync, sidecam_sync, marker_sync = [], [], [], []

  for idx in range(idx_counter):
    rgb_idx = getRgbData(int(math.floor(idx*rgb_multiplier)))
    rgb_sync.append(rgb_path + '/' + rgb_idx)

    gelsight_idx = getGelsightData(int(math.floor(idx*gelsight_multiplier)))
    gelsight_sync.append(gelsight_path + '/' + gelsight_idx)

    sidecam_idx = getSidecamData(int(math.floor(idx*sidecam_multiplier)))
    sidecam_sync.append(sidecam_path + '/' + sidecam_idx)

    marker_sync.append(marker_files[int(idx*marker_multiplier)])
  print(marker_sync)

  visualize(rgb_sync, gelsight_sync, sidecam_sync, list(marker_files), marker_multiplier)

  os.chdir(folder)
  subprocess.call(['ffmpeg', '-framerate', '1', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p', 'video_name.mp4'])

if __name__ == '__main__':
  main()
