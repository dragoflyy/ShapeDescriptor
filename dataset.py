#My dataset
import numpy as np
import cv2
import os

# and dataset should have a map of all image by type
class Dataset : 
    def __init__(self) :
        self.data = []

    def __str__(self) : 
        return "dataset ["+str(len(self.data))+" items]"

    def Write(self) :
        for data in self.data :
            data.Write()
    
    def Reshape (self) :
        for data in self.data : 
            data.Reshape()

    def append(self,other) :
        if isinstance(other, Data) : 
            self.data.append(other)
        elif isinstance(other, Dataset)
            for item in other.data : 
                self.data.append(item)

class Data :
    # static parameters
    static_dataset_number = 1
    static_normalized_size = (300, 300)

    def __init__(self, image, i_type, center=None, size=None, rotation=None) :
        self._number = Dataset.static_dataset_number
        Data.static_dataset_number += 1
        self._image = image               # np image
        self._type = i_type               # shape classification [str]
        self._center = center             # np point
        self._size = size                 # shape size [various depending to shape]
        self._rotation = rotation          # shape rotation
        self.dest_folder="data"

    def __str__(self):
        return str(self._type) + " | " + str(self._center) + " | " + str(self._size)

    def __add__(self, other):
        if isinstance(other, Data) and other._type == "Noise" :
            return Data(self._image + other._image, self._type, self._center, self._size)
        raise TypeError("Only noise can be added to Image type")

    def Write(self) :
        destination = self.dest_folder+"\\"+self._type
        name = str(self._number) + "_" + str(self._center) + "_" + str(self._size) + "_" + str(self._rotation) + ".png"
        if not os.path.exists(destination):
           os.makedirs(destination, exist_ok=True)
        cv2.imwrite(destination+"\\"+name, self._image)

    def Reshape(self) :
        f_agrandissement = Data.static_normalized_size/np.array(self._image.shape[0:2])

        self._image = cv2.resize(self._image, Data.static_normalized_size)
        if not isinstance(self._center, type(None)) :
            self._center = self._center * f_agrandissement
        if not isinstance(self._size, type(None)) :
            if isinstance(self._size, int) :
                self._size = self._size * f_agrandissement[0]
            else :
                self._size = self._size * f_agrandissement

def ParseData(data) :
    if "[" in data :
        # it's a list
        data = data.replace("[", "")
        data = data.replace("]", "")
        result = []
        for item in data.split(" ") :
            if not item.isdigit() :
                continue
            item = item.replace(" ", "")
            result.append(int(item))
        return np.array(result)
    else :
        try :
            return int(data)
        except :
            return None

def LoadDataSet() :
    print("load dataset")
    dest_folder="data"
    dataset = Dataset()

    for im_type in os.listdir(dest_folder) :
        for image in os.listdir(dest_folder + "\\" + im_type) :
            infos = image.split(".")[0]
            splitted = infos.split("_")
            nb = splitted[0]
            center = ParseData(splitted[1])
            size = ParseData(splitted[2])
            rotation = ParseData(splitted[3])
            
            image = cv2.imread(dest_folder+"\\"+im_type+"\\"+image)

            data = Data(image, im_type, center, size, rotation)
            dataset.append(data)
    return dataset


if __name__ == "__main__":
    dataset = LoadDataSet()
    print("size of dataset : " + str(len(dataset)))
    print("exemple one : " + str(dataset[0]))
    print("exemple one : " + str(dataset[510]))
    print("exemple one : " + str(dataset[1010]))
    print("exemple one : " + str(dataset[2000]))
