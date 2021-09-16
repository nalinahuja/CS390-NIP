# Modified By Nalin Ahuja, ahuja15@purdue.edu

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras

# End Imports----------------------------------------------------------------------------------------------------------------------------------------------------------

# Seed Value
SEED_VALUE = 1618

# Information On Dataset
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Selected Algorithm ("guesser", "custom_net", "tf_net")
ALGORITHM = "guesser"

# End Embedded Constants-----------------------------------------------------------------------------------------------------------------------------------------------

# Setting Random Seeds To Maintain Deterministic Behavior
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
# tf.set_random_seed(SEED_VALUE) Uncomment for TF1.

# Disable Tensorflow Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.logging.set_verbosity(tf.logging.ERROR) Uncomment for TF1.

# End Module Initialization--------------------------------------------------------------------------------------------------------------------------------------------

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        # Initialize Class Members From Arguments
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate

        # Initialize Weights Randomly
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation Function
    def __sigmoid(self, x):
        # Compute Sigmoid Value At X
        return (1 / (1 + np.exp(-x)))

    # Activation Function Derivative
    def __sigmoidDerivative(self, x):
        # Compute Sigmoid Value At X
        s = self.__sigmoid(x)

        # Return Derivative Of Sigmoid
        return (s * (1 - s))

    # Un-Randomized Batch Generator For Mini-Batches
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        pass

    # Forward Pass Function.
    def __forward(self, input):
        # Calculate Sigmoid Values For Layers
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))

        # Return Result
        return (layer1, layer2)

    # Prediction Function
    def predict(self, xVals):
        # Perform Forward Pass
        _, layer2 = self.__forward(xVals)

        # Return Result
        return (layer2)

# End Neural Network Layer Class---------------------------------------------------------------------------------------------------------------------------------------

# Classifier Function That Guesses The Class Label
def guesserClassifier(xTest):
    # Initialize Answer Vector
    ans = []

    # Iterate Over xTest Sample
    for entry in (xTest):
        # Initialize Base Prediction Vector
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Randomly Set Class Label
        pred[random.randint(0, 9)] = 1

        # Append Predicted Class To Answer Vector
        ans.append(pred)

    # Return
    return (np.array(ans))

# End Subroutine Functions---------------------------------------------------------------------------------------------------------------------------------------------

def getRawData():
    # Fetch MNIST Dataset From Tensorflow Imporrt
    mnist = tf.keras.datasets.mnist

    # Load Data From MNIST Dataset
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    # Display Information About Dataset
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))

    # Return Data
    return ((xTrain, yTrain), (xTest, yTest))

def preprocessData(raw):
    # Unpack Data From Raw Input
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).

    # Normalize Input Data To Range
    xTrainP = np.divide(xTrain, 255)
    xTestP = np.divide(xTest, 255)

    print(xTrainP)

    import sys
    sys.exit()

    # Process Integer Arrays Into Binary Class Matrices
    yTrainP = keras.utils.to_categorical(yTrain, NUM_CLASSES)
    yTestP = keras.utils.to_categorical(yTest, NUM_CLASSES)

    # Display Information About Dataset
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))

    # Return Preprocessed Data
    return ((xTrain, yTrainP), (xTest, yTestP))

def trainModel(data):
    # Unpack Training Data
    xTrain, yTrain = data

    # Run Training Algorithm
    if (ALGORITHM == "guesser"):
        return (None)
    elif (ALGORITHM == "custom_net"):
        print("Building and training custom neural network")
        print("Not yet implemented.")                                   #TODO: Write code to build and train your custon neural net.
        return (None)
    elif (ALGORITHM == "tf_net"):
        print("Building and training TF_NN.")
        print("Not yet implemented.")                                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")

def runModel(data, model):
    if (ALGORITHM == "guesser"):
        return guesserClassifier(data)
    elif (ALGORITHM == "custom_net"):
        print("Testing Custom_NN.")
        print("Not yet implemented.")                                   #TODO: Write code to run your custon neural net.
        return (None)
    elif (ALGORITHM == "tf_net"):
        print("Testing TF_NN.")
        print("Not yet implemented.")                                   #TODO: Write code to run your keras neural net.
        return (None)
    else:
        raise ValueError("Algorithm not recognized.")

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    # Unpack Data
    xTest, yTest = data

    # Initialize Accuracy
    acc = 0

    # Iterate Over Predicted Values
    for i in range(preds.shape[0]):
        # Verify Predicted Values Match Expected Values
        if (np.array_equal(preds[i], yTest[i])):
            # Increment Accuracy Metric
            acc += 1

    # Calculate Accuracy
    acc /= preds.shape[0]

    # Display Classifier Metrics
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))

# End Pipeline Functions-----------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):
    # Get Raw Data
    raw = getRawData()

    # Preprocess Raw Data
    data = preprocessData(raw)

    # Train Model On Raw Data
    model = trainModel(data[0])

    # Run Model On Raw Data
    preds = runModel(data[1][0], model)

    # Evaluate Model Results
    evalResults(data[1], preds)

# End Main Function----------------------------------------------------------------------------------------------------------------------------------------------------
