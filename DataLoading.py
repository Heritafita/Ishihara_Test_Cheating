import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

class DataLoading():
    def load_mnist_enriched(self):
        
        # Load Dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
    
        # Reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
    
        """ Inversed color images as additional training data """
        # the original MNIST dataset contains images of handwritten numbers of white numbers on darker background
        # this adds training images inverting colors of the iamges
        # --> dark numbers on white background
        trainX_inverse = 255 - trainX
        testX_inverse = 255 - testX
    
        # adding inversed images as training data
        trainX_total = np.concatenate((trainX, trainX_inverse), axis = 0)
        trainY_total = np.concatenate((trainY, trainY), axis = 0)
        testX_total = np.concatenate((testX, testX_inverse), axis = 0)
        testY_total = np.concatenate((testY, testY), axis = 0)

        # one hot encode target values
        trainY_total = to_categorical(trainY_total)
        testY_total = to_categorical(testY_total)

        return trainX_total, trainY_total, testX_total, testY_total
