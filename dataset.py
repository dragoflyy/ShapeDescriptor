#My dataset
import numpy as np
import cv2
import os



class Dataset :
    static_dataset_number = 1

    def __init__(self, image, i_type, center=None, size=None, rotation=None) :
        self._number = Dataset.static_dataset_number
        Dataset.static_dataset_number += 1
        self._image = image               # np image
        self._type = i_type               # shape classification [str]
        self._center = center             # np point
        self._size = size                 # shape size [various depending to shape]
        self.rotation = rotation          # shape rotation [0:180], as rectangle 0° = 180°
        self.dest_folder="data"

    def __str__(self):
        return str(self._type) + " | " + str(self._center) + " | " + str(self._size)

    def __add__(self, other):
        if isinstance(other, Dataset) and other._type == "Noise" :
            return Dataset(self._image + other._image, self._type, self._center, self._size)
        raise TypeError("Only noise can be added to Image type")

    def Write(self) :
        destination = self.dest_folder+"\\"+self._type
        name = str(self._number) + "." + str(self._center)+str(self._size)+".png"
        if not os.path.exists(destination):
           os.makedirs(destination, exist_ok=True)
        cv2.imwrite(destination+"\\"+name, self._image)
