from ModelTraining import ModelTraining
from DataLoading import DataLoading

dl = DataLoading()
ml = ModelTraining()

# loading enriched MNIST dataset
trainX, trainY, testX, testY = dl.load_mnist_enriched()

# scaling the pixels
trainX = ml.apply_scaling(trainX)
testX = ml.apply_scaling(testX)

# defining the model
model = ml.define_model()
model = ml.train(model, data = (trainX, trainY))

# saving trained model
model.save("CNN")
