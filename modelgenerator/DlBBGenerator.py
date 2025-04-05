from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import modelgenerator.dataset as dataset
from modelgenerator.generator import *

def generate(img_size) :
    Images_Data = dataset.LoadDataSet()
    Images_Data.Reshape(img_size)

    # split into input and output variables
    population = len(Images_Data)
    centers = Images_Data.Center().reshape(2,population)
    sizes = Images_Data.Size().reshape(2, population)
    rotations = Images_Data.Rotation().reshape(1, population)

    print(centers.shape)
    print(sizes.shape)
    print(rotations.shape)

    X = Images_Data.Images()
    Y = np.concatenate((centers, sizes, rotations), axis=0).transpose()/np.array(img_size+img_size+[180])

    print(Y.shape)

    # split the data into training and testing
    train_prct = 0.7
    split = int(train_prct*population)
    (X_train, X_test, Y_train, Y_test) = (X[0:split], X[split:population], Y[0:split], Y[split:population])

    im_size = X[0].shape
    vector_size = im_size[0]*im_size[1]
    print("image sizes : " + str(im_size))
    print("vector size then : " + str(vector_size))
    print("image count : " + str(len(Images_Data)))

    print("creating the model")
    # create the model
    model = Sequential()
    model.add(layers.Input(shape=(im_size[0], im_size[1], 1)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding="same"))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))

    print("compiling the model")
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()

    print("fitting the model ...")
    # fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, shuffle=True)

    print("evaluating the model")
    # evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("\n\nAccuracy: %.2f%% ------------------\n\n" %(scores[1]*100))

    model.save("Models\\ShapeDescriptor.keras")

