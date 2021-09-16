# Modified By Nalin Ahuja, ahuja15@purdue.edu

import os
import sys
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras

# End Imports----------------------------------------------------------------------------------------------------------------------------------------------------------

# Seed Value
SEED_VALUE = 1618

# Model Training Constants
NUM_EPOCHS = 5
INPUT_SIZE = 784
HIDDEN_SIZE = 15
OUTPUT_SIZE = 10

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
    def __init__(self, input_size, output_size, neurons_per_layer, learning_rate = 0.1):
        # Initialize Class Members From Arguments
        self.input_size = inputSize
        self.output_size = outputSize
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate

        # Initialize Weights Randomly
        self.W1 = np.random.randn(self.input_size, self.neurons_per_layer)
        self.W2 = np.random.randn(self.neurons_per_layer, self.output_size)

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

    # Unrandomized Batch Generator For Mini-Batches
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        pass

    # Forward Pass Function
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

def guesser_classifier(xTest):
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

def get_raw_data():
    # Fetch MNIST Dataset From Tensorflow Imporrt
    mnist = tf.keras.datasets.mnist

    # Load Data From MNIST Dataset
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    # Display Information About Dataset
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))

    # Return Data
    return ((xTrain, yTrain), (xTest, yTest))

def preprocess_data(raw):
    # Unpack Data From Raw Input
    ((xTrain, yTrain), (xTest, yTest)) = raw

    # Normalize Input Data To Range
    xTrainP = np.divide(xTrain, 255)
    xTestP = np.divide(xTest, 255)

    # Process Integer Arrays Into Binary Class Matrices
    yTrainP = keras.utils.to_categorical(yTrain, OUTPUT_SIZE)
    yTestP = keras.utils.to_categorical(yTest, OUTPUT_SIZE)

    # Display Information About Dataset
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))

    # Return Preprocessed Data
    return ((xTrain, yTrainP), (xTest, yTestP))

def train_model(data):
    # Unpack Training Data
    xTrain, yTrain = data

    # Run Training Algorithm
    if (ALGORITHM == "guesser"):
        # Display Status
        print("Using guesser algorithm...")

        # Return No Model
        return (None)
    elif (ALGORITHM == "custom_net"):
        # Display Status
        print("Using custom neural network...")

        # Initialize New Neural Network Instance
        model = NeuralNetwork_2Layer(INPUT_SIZE, OUTPUT_SIZE, NUM_NEURONS)

        # Train Model With Training Data
        model.train(xTrain, yTrain, epochs = NUM_EPOCHS)

        # Return Model
        return (model)
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Using Tensorflow neural network...")

        # Initialize New Sequential Instance
        model = keras.Sequential()

        # Initialize Loss Function
        loss_func = keras.losses.categorical_crossentropy

        # Initialize Model Optimizer
        opt_func = tf.train.AdamOptimizer()

        # Add Neuron Hidden Layer To Model
        model.add(keras.layers.Dense(HIDDEN_SIZE, input_shape = INPUT_SIZE, activation = tf.nn.relu))

        # Add Neuron Output Layer To Model
        model.add(keras.layers.Dense(OUTPUT_SIZE, input_shape = HIDDEN_SIZE, activation = tf.nn.softmax))

        # Compile Model
        model.compile(optimizer = opt_func, loss = loss_func)

        # Train Model
        model.train(xTrain, yTrain, epochs = NUM_EPOCHS)

        # Return Model
        return (model)
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("Algorithm not recognized...")

def run_model(data, model):
    if (ALGORITHM == "guesser"):
        # Display Status
        print("Running guesser classifier...")

        # Return Prediction
        return (guesser_classifier(data))
    elif (ALGORITHM == "custom_net"):
        # Display Status
        print("Running custom neural network...")

        # Run Custom Model
        preds = model.predict(data)

        # Return Prediction
        return (preds)
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Running Tensorflow neural network...")

        # Run Keras Model
        preds = model.predict(data)

        # TODO: one hot encoding

        # Return Prediction
        return (preds)
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("Algorithm not recognized...")

def eval_results(data, preds):
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

    # TODO: Add F1 score and confusion matrix

    # Initialize Prediction Metrics
    tp = tn = fp = fn = 0

    # # Iterate Over Predicted Values
    # for i in range(preds.shape[0]):
    #     # Verify Predicted Values Match Expected Value

    # Display Classifier Metrics
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))

# End Pipeline Functions-----------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):
    # Get Raw Data
    raw = get_raw_data()

    # Preprocess Raw Data
    data = preprocess_data(raw)

    # Train Model On Raw Data
    model = train_model(data[0])

    # Run Model On Raw Data
    preds = run_model(data[1][0], model)

    # Evaluate Model Results
    eval_results(data[1], preds)

# End Main Function----------------------------------------------------------------------------------------------------------------------------------------------------