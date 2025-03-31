import random
import numpy as np
import cv2
import shutil
import os

dest_folder="data"
squares_folder="squares"
other_folder="others"

# Generate Size of the image
def GenerateSize() :
    rand_size = 0.4 + 0.8*random.random()
    rand_size_diff = random.random()*0.2
    h = int(5 + (rand_size+rand_size_diff)*100)
    w = int(5 + (rand_size-rand_size_diff)*100)
    return np.array([h, w])

def GenerateSquare() :
    shape_size = GenerateSize()
    center = shape_size/2
    angle = int(360*random.random())

    rect = ((int(center[0]), int(center[1])), (int(shape_size[0]), int(shape_size[1])), angle)
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    x_array, y_array = box[:,1], box[:,0]
    x_negativ_array, y_negativ_array = box[:,1]*(box[:,1]<0), box[:,0]*(box[:,0]<0)
    oversize = 10 + int(random.random()*100)
    offset = [int(oversize*(0.2 + 0.8*random.random())), int(oversize*(0.2 + 0.8*random.random()))]
    box = box + np.array([abs(min(y_negativ_array)), abs(min(x_negativ_array))]) + offset
    
    im_size = [oversize + max(y_array) + abs(min(y_array)), oversize +  max(x_array) + abs(min(x_array))]
    image = np.zeros((im_size[1], im_size[0], 3), dtype=np.uint8)
    
    cv2.drawContours(image, [box], 0, (255, 255, 255), 1 + int(2*random.random()))

    return image

def GeneratePosition(im_size, shape_size) :
    pos_range = im_size - shape_size
    return np.int64(np.array([random.random()*pos_range[0], random.random()*pos_range[1]]))
    
def GenerateCircle():
    im_size = GenerateSize()
    shape_size = int((0.05 + 0.35*random.random())*im_size[0])
    shape_pos = GeneratePosition(im_size, 2*shape_size)

    image = np.zeros((im_size[1], im_size[0], 3), dtype=np.uint8)
    cv2.circle(image, shape_pos, shape_size, (255, 255, 255), 1 + int(2*random.random()))
    return image

def GenerateNoisyImage() :
    im_size = GenerateSize()
    return 255*np.random.random(im_size)

def GenerateLinesImage() :
    im_size = GenerateSize()
    image = np.zeros((im_size[1], im_size[0], 3), dtype=np.uint8)

    lines_nb = 1 + int(random.random()*20)
    for i in range(lines_nb) :
        sp = np.array([random.random(), random.random()]) * im_size
        ep = np.array([random.random(), random.random()]) * im_size
        cv2.line(image,np.int64(sp),np.int64(ep),(255,255,255),1 + int(2*random.random()))
    return image

def SpawnBlackSquare(image) :
    shape_size = np.int64(np.array(image.shape[0:2])/( 3 + int(random.random()*4)))
    shape_pos = GeneratePosition(image.shape[0:2], shape_size)
    cv2.rectangle(image, shape_pos-shape_size, shape_pos+shape_size, (0,0,0), -1)

def WriteImage(name, image) :
    cv2.imwrite(dest_folder+"\\"+name+".png", image)

def CleanGenerated() :
    try :
        shutil.rmtree(dest_folder)
    except :
        print("can't delete " + dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    os.makedirs(dest_folder+"\\"+squares_folder, exist_ok=True)
    os.makedirs(dest_folder+"\\"+other_folder, exist_ok=True)

# -- Cleaning of older data
CleanGenerated()

# -- Generation of datas 
square_number = 600
incomplete_squares_percent = 0.7
noise_number = 50
circle_number = 450
incomplete_circle_percent = 0.5
lines_images = 200

for i in range(square_number) :
    im = GenerateSquare()
    if (random.random() < incomplete_squares_percent) :
        SpawnBlackSquare(im)
    WriteImage(squares_folder + "\\"+str(i+1), im)

for i in range(noise_number) :
    im = GenerateNoisyImage()
    WriteImage(other_folder + "\\noisy"+str(i+1), im)

for i in range(circle_number) :
    im = GenerateCircle()
    if (random.random() < incomplete_circle_percent) :
        SpawnBlackSquare(im)
    WriteImage(other_folder + "\\circle"+str(i+1), im)

for i in range(lines_images) :
    im = GenerateLinesImage()
    WriteImage(other_folder + "\\lines"+str(i+1), im)

