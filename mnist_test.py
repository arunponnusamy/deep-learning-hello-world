# import required modules
import keras
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

# save learned weights
score = model.evaluate(x_test, y_test, batch_size=100)

print("\nTest loss: ", score[0])
print("Test accuracy: ", score[1])
