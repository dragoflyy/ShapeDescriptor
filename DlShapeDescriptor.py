import keras.models
from modelgenerator import DlBBGenerator
from modelgenerator import dataset
from modelgenerator import generator
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def Erode(image) :
    kernel = np.ones((3, 3), np.uint8) 
    return cv2.erode(image, kernel, iterations=1) 

def main() :
    denoising_model_path = 'Models\\ShapeDescriptor.keras'
    input_size = [300,300]

    if not os.path.exists(denoising_model_path) :
        # if the model didn't exists, we generate a new one
        DlBBGenerator.generate(input_size)

    print("Loading neural model")
    # let's load the model for denoising
    model = keras.models.load_model(denoising_model_path)
    
    # Check its architecture
    model.summary()

    count = 7
    NewTest = generator.GenerateShape("Rectangle", count, occult_shape=True, noisy=False)
    NewTest.Reshape(input_size)

    X_secondTest = (NewTest).Images()


    YResult = model.predict(X_secondTest)

    # Number of rows and columns in the grid
    rows = 2
    cols = count

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the images and display them in the grid
    for i in range(count) :
        axes[i].imshow(X_secondTest[i], cmap='gray')
        axes[i].axis('off')  # Hide the axes

        Y_result = np.copy(X_secondTest[i])
        sample = YResult[i]*np.array(input_size+input_size+[180])
        print("center : " + str(sample[0:2]))
        print("size : " + str(sample[2:4]))
        print("angle : " + str(int(sample[4:5])))

        rect = (sample[0:2], sample[2:4], int(sample[4:5]))
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        Y_result_rgb = np.stack((Y_result,)*3, axis=-1)
        cv2.drawContours(Y_result_rgb, [box], 0, (0, 255, 0), 1)

        axes[count+i].imshow(Y_result_rgb)
        axes[count+i].axis('off')  # Hide the axes

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()