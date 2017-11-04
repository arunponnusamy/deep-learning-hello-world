# import required modules
import keras
import sys
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# import MNIST dataset
from keras.datasets import mnist

# load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# preprocessing
x_test = x_test.reshape(x_test.shape[0], 784)
x_test = x_test.astype('float32')

x_test /= 255
y_test = keras.utils.to_categorical(y_test, 10)

# create model
model = Sequential()
model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# load the weights already learned
model.load_weights('mnist_weights_epoch300.h5')

# evaluate the model on test set 
score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)

print("\nTest set loss: ", score[0])
print("Test set accuracy: ", score[1])

# load test input if provided
if(len(sys.argv) == 2):
    img = cv2.imread(sys.argv[1])
    if(not img.data):
        print("Could not load image")
        exit

    # preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    img = img.reshape(1, 784)
    img = img.astype('float32')
    img /= 255    

    # predict the handwritten digit in the input image 
    score = model.predict(img, batch_size=1, verbose=0)
    
    # display scores    
    print("\nPrediction score for test input: " + sys.argv[1])
    sort = sorted(range(len(score[0])), key=lambda k:score[0][k],reverse=True)
    for index in sort:
        print(str(index) + ": " + str(score[0][index]))  
    percent = format(score[0][sort[0]] * 100, '.2f')
    print("\nI am " + str(percent) + "% confident that it is " + str(sort[0]))
