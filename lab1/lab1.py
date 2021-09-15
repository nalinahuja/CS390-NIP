# Modified By Nalin Ahuja, ahuja15@purdue.edu

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.utils import to_categorical

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
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

# End Subroutine Functions---------------------------------------------------------------------------------------------------------------------------------------------

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()

# End Pipeline Functions-----------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

# End Main Function----------------------------------------------------------------------------------------------------------------------------------------------------
