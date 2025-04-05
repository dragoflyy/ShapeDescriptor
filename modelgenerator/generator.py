# Dataset Generator
import random
import numpy as np
import cv2
import shutil
import os
from modelgenerator.dataset import Data, Dataset

dest_folder="data"

# Generate Size of the image
def GenerateSize() :
    rand_size = 0.4 + 0.8*random.random()
    rand_size_diff = random.random()*0.2
    h = int(5 + (rand_size+rand_size_diff)*100)
    w = int(5 + (rand_size-rand_size_diff)*100)
    return np.array([h, w])

def GenerateSquare() :
    shape_size = GenerateSize()
    center = np.int64(shape_size/2)
    angle = int(180*random.random())

    rect = ((int(center[0]), int(center[1])), (int(shape_size[0]), int(shape_size[1])), angle)
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    # correct rectangle position to fit it in the image again
    x_array, y_array = box[:,1], box[:,0]
    x_negativ_array, y_negativ_array = box[:,1]*(box[:,1]<0), box[:,0]*(box[:,0]<0)
    oversize = 10 + int(random.random()*100)
    offset = np.int64([oversize*(0.2 + 0.8*random.random()), oversize*(0.2 + 0.8*random.random())])
    final_offset = np.array([abs(min(y_negativ_array)), abs(min(x_negativ_array))]) + offset
    box = box + final_offset
    
    im_size = [oversize + max(y_array) + abs(min(y_array)), oversize +  max(x_array) + abs(min(x_array))]
    image = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)
    
    cv2.drawContours(image, [box], 0, 255, 1 + int(2*random.random()))

    return Data(image, "Rectangle", center + final_offset, shape_size, angle)

def GeneratePosition(im_size, shape_size) :
    pos_range = im_size - 2*shape_size
    return np.int64(np.array([random.random()*pos_range[0], random.random()*pos_range[1]]) + shape_size)
    
def GenerateCircle():
    im_size = GenerateSize()
    im_size = np.array([im_size[0], im_size[0]])
    shape_size = int((0.05 + 0.35*random.random())*min(im_size))
    shape_pos = GeneratePosition(im_size, shape_size)

    image = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)
    cv2.circle(image, shape_pos, shape_size, 255, 1 + int(2*random.random()))
    return Data(image, "Circle", shape_pos, shape_size)

def GenerateNoisyImage(size = None) :
    if (size != None ):
        im_size = size
    else :
        im_size = GenerateSize()
    return Data(np.int16(255*np.random.random(im_size)), "Noise")

def Noisit(image, ratio=4) :
    noise = GenerateNoisyImage(image.shape)._image
    return image + ( noise - 255/2)/ratio

def GenerateLinesImage() :
    im_size = GenerateSize()
    image = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)

    lines_nb = 1 + int(random.random()*20)
    for i in range(lines_nb) :
        sp = np.array([random.random(), random.random()]) * im_size
        ep = np.array([random.random(), random.random()]) * im_size
        cv2.line(image,np.int64(sp),np.int64(ep),255,1 + int(2*random.random()))
    return Data(image, "Lines", None, None)

def SpawnBlackSquare(image) :
    shape_size = np.int64(np.array(image.shape[0:2])/( 3 + int(random.random()*4)))
    shape_pos = GeneratePosition(image.shape[0:2], shape_size)
    cv2.rectangle(image, shape_pos-shape_size, shape_pos+shape_size, (0,0,0), -1)
    return image

def WriteImage(name, image) :
    cv2.imwrite(dest_folder+"\\"+name+".png", image)

def CleanGenerated() :
    try :
        shutil.rmtree(dest_folder)
    except :
        print("can't delete " + dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

def CreateDatasetFolders(squares=0, incomplete_s=0, noisy_s=0, circle=0, incomplete_c=0, noisy_c=0, noise=0, lines=0) :
    # -- Cleaning of older data
    CleanGenerated()

    # -- Generation of datas 

    for i in range(squares) :
        incomplete, noised = False, False
        if (random.random() < incomplete_s) :
            incomplete = True
        if (random.random() < noisy_s) :
            noised = True
        GenerateShape("Rectangle", 1, occult_shape=incomplete, noisy=noised)[0].Write()

    for i in range(circle) :
        incomplete, noised = False, False
        if (random.random() < incomplete_c) :
            incomplete = True
        if (random.random() < noisy_c) :
            noised = True
        GenerateShape("Circle", 1, occult_shape=incomplete, noisy=noised)[0].Write()

    noises = GenerateShape("Noise", noise)
    for item in noises :
        item.Write()
    lines = GenerateShape("Lines", lines)
    for item in lines :
        item.Write()



def GenerateShape(shape, count, occult_shape=False, noisy=False) :
    match shape:
        case "Rectangle":
            generator=GenerateSquare
        case "Circle":
            generator=GenerateCircle
        case "Lines":
            generator=GenerateLinesImage
        case "Noise":
            generator=GenerateNoisyImage
        case _:
            print("Wrong Shape Name")
            return None
    dataset = Dataset()
    for i in range(count) :
        data = generator()
        if occult_shape :
            data._image = SpawnBlackSquare(data._image)
        if noisy :
            noise = GenerateNoisyImage()
            data += noise - 128
        dataset.append(data)
    return dataset

