# organize imports
from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from datasets import dataset

# seed for reproducing same results
seed = 9
np.random.seed(seed)

dataset = datasets.load_digits()
maximum = dataset.images.max()

exit
# split into input and output variables
X = dataset.images
Y = dataset.target

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)
X_train, X_test = X_train / 255.0, X_test / 255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
print("creating the model")
# create the model
model = Sequential()
model.add(layers.Flatten(input_shape=[8, 8]))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print("compiling the model")
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("fitting the model ...")
# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100)

print("evaluating the model")
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n\nAccuracy: %.2f%% ------------------\n\n" %(scores[1]*100))


reverif = 30
X_secondTest = X_test[0:reverif]
Y_secondTest = Y_test[0:reverif]


YResult = model.predict(X_secondTest)

print("prediction result : ")
for i in range(len(YResult)) :
    print("sorted : " + str(np.argmax(YResult[i])))
    print("expected : " + str(np.argmax(Y_secondTest[i])))
    print("----")
