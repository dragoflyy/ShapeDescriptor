import numpy as np
import cv2
import os

dest_folder="data"
squares_folder="squares"
other_folder="others"

directory_path = dest_folder+"\\"+squares_folder
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
print(files)