import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import BatchNormalization

class ModelTraining():
    # define cnn model
    def define_model(self):
	
        model=Sequential()
        model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
        model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

        # Pooling Layer
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Batch Normalization to normalize hidden layers and speed up training
        model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Batch Normalization to normalize hidden layers and speed up training
        model.add(BatchNormalization())   

        model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

        # Pooling Layer
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Flattening Layer    
        model.add(Flatten())

        # Batch Normalization to normalize hidden layers and speed up training
        model.add(BatchNormalization())
        model.add(Dense(512,activation="relu"))

        # Dense output layer with size = 10, representing each of numbers [0-9]
        model.add(Dense(10,activation="softmax"))

        return model

    def train(self, model, data):
        
        # compile model with Stochastic-Gradient-Descent as training alg
        train_alg = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer = train_alg, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(data[0], data[1])

        return model

    # scaling
    def apply_scaling(self, X):
	
        # convert from integers to floats
	    X_float = X.astype('float32')
	
        # normalize to range 0-1
	    X_norm = (X_float - X_float.min()) / (X_float.max() - X_float.min())
	
        # return normalized images
	    return X_norm
