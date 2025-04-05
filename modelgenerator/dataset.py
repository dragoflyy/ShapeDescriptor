import numpy as np
import cv2
import os

# and dataset should have a map of all image by type
class Dataset : 
    def __init__(self) :
        self.data = []

    def __str__(self) : 
        return "dataset ["+str(len(self.data))+" items]"
        
    def __getitem__(self, key):
        if isinstance(key, int) :
            return self.data[key]
        if isinstance(key, str) :
            return self.ConstructSortedImageList(key)

    def __add__(self, other):
        result = Dataset()
        if isinstance(other, int) :
            for i in range(len(self.data)) :
                result.append(self.data[i] + other)
        elif isinstance(other, Dataset) :
            if len(other.data) == len(self.data) :
                for i in range(len(self.data)) :
                    result.append(self.data[i] + other.data[i])
        else :
            return None
        return result

    def __sub__(self, other) :
        result = Dataset()
        if (isinstance(other, int)) :
            for i in range(len(self.data)) :
                result.append(self.data[i] - other)
        else :
            return None
        return result

    def __mul__(self, other) :
        result = Dataset()
        if (isinstance(other, (int, float))) :
            for i in range(len(self.data)) :
                result.append(self.data[i]*other)
        else :
            return None
        return result

    def __len__(self):
        return len(self.data)

    def ConstructSortedImageList(self, key) :
        result_dataset = Dataset()
        for item in self.data :
            if (item._type == key) :
                result_dataset.append(item)
        return result_dataset

    # GETTERS
    def Images(self) :
        result = []
        for item in self.data :
            result.append(np.array(item._image, dtype=np.int16))
        return np.array(result)

    def Types(self) :
        result = []
        for item in self.data :
            result.append(item._type)
        return np.array(result)

    def Center(self) :
        result = []
        for item in self.data :
            result.append(item._center)
        return np.array(result)

    def Size(self) :
        result = []
        for item in self.data :
            result.append(item._size)
        return np.array(result)

    def Rotation(self) :
        result = []
        for item in self.data :
            result.append(item._rotation)
        return np.array(result)
        
    def Write(self) :
        for data in self.data :
            data.Write()
    
    def Reshape (self, size = None) :
        for data in self.data :
            data.Reshape(size)

    def append(self,other) :
        if isinstance(other, Data) :
            self.data.append(other)
        elif isinstance(other, Dataset) :
            for item in other.data : 
                self.data.append(item)

class Data :
    # static parameters
    static_dataset_number = 1
    static_normalized_size = (300, 300)

    def __init__(self, image, i_type, center=None, size=None, rotation=None) :
        self._number = Data.static_dataset_number
        Data.static_dataset_number += 1
        self._image = image               # np image
        self._type = i_type               # shape classification [str]
        self._center = center             # np point
        self._size = size                 # shape size [various depending to shape]
        self._rotation = rotation         # shape rotation
        self.dest_folder="data"

    def __str__(self):
        return str(self._type) + " | " + str(self._center) + " | " + str(self._size)

    def __add__(self, other):
        if isinstance(other, Data) and other._type == "Noise" :
            if other._image.shape != self._image.shape :
                other.Reshape(self._image.shape)
            return Data(np.uint8(np.clip(self._image + other._image, a_min = 0, a_max = 255)), self._type + "_Noisy", self._center, self._size)
        if isinstance(other, int) :
            return Data(self._image + other, self._type, self._center, self._size)
        raise TypeError("Only noise or integers can be added to Data type")

    def __sub__(self, other) :
        if isinstance(other, int) :
            return Data(self._image - other, self._type, self._center, self._size)
        raise TypeError("Only int can be substracted to Data type")

    def __mul__(self, other) :
        if isinstance(other, (int, float)) :
            return Data(np.int16(other*self._image), self._type, self._center, self._size)

    def Write(self) :
        destination = self.dest_folder+"\\"+self._type
        name = str(self._number) + "_" + str(self._center) + "_" + str(self._size) + "_" + str(self._rotation) + ".png"
        if not os.path.exists(destination):
           os.makedirs(destination, exist_ok=True)
        cv2.imwrite(destination+"\\"+name, self._image)

    def Reshape(self, size = None) :
        if size == None :
            size = Data.static_normalized_size
        f_agrandissement = size/np.array(self._image.shape[0:2])
        self._image = cv2.resize(self._image, [size[1], size[0]])
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
            item = item.replace(" ", "")
            try: 
                result.append(float(item))
            except : 
                continue
        return np.array(result)
    else :
        try :
            return float(data)
        except :
            return None

def LoadDataSet() :
    print("load dataset")
    dest_folder="data"
    dataset = Dataset()

    for im_type in os.listdir(dest_folder) :
        for image in os.listdir(dest_folder + "\\" + im_type) :
            infos = image.split(".png")[0]
            splitted = infos.split("_")
            nb = splitted[0]
            center = ParseData(splitted[1])
            size = ParseData(splitted[2])
            rotation = ParseData(splitted[3])
            
            image = cv2.imread(dest_folder+"\\"+im_type+"\\"+image, cv2.IMREAD_GRAYSCALE)

            data = Data(image, im_type, center, size, rotation)
            dataset.append(data)
    return dataset


if __name__ == "__main__":
    dataset = LoadDataSet()
    print("size of dataset : " + str(dataset))
    print("exemple one : " + str(dataset[0]))
    print("exemple one : " + str(dataset[510]))
    print("exemple one : " + str(dataset[1010]))
    print("exemple one : " + str(dataset[2000]))
    print("nb of rectangles : " + str(len(dataset["Rectangle"])))
    print("nb of noisy rectangles : " + str(len(dataset["Rectangle_Noisy"])))
    print("nb of images : " + str(len(dataset.Images())))
