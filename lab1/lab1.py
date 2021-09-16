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

# Model Structure Constants
INPUT_SIZE = 784
HIDDEN_SIZE = 15
OUTPUT_SIZE = 10

# Model Training Constants
NUM_EPOCHS = 5
LEARNING_RATE = 0.1

# Selected Algorithm ("guesser", "custom_net", "tf_net")
ALGORITHM = "tf_net"

# End Embedded Constants-----------------------------------------------------------------------------------------------------------------------------------------------

# Setting Random Seeds To Maintain Deterministic Behavior
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Tensorflow Settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.set_random_seed(SEED_VALUE) # Uncomment for TF1
# tf.logging.set_verbosity(tf.logging.ERROR) # Uncomment for TF1

# End Module Initialization--------------------------------------------------------------------------------------------------------------------------------------------

class NeuralNetwork_2Layer():
    def __init__(self, input_size, output_size, neurons_per_layer, activation = "sigmoid", learning_rate = 0.1):
        # Initialize Neural Network Attributes
        self.input_size = inputSize
        self.output_size = outputSize
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate

        # Set Activation Function
        if (activation == "relu"):
            self.__activation = self.__relu
            self.__activation_prime = self.__relu_derivative
        elif (activation == "sigmoid"):
            self.__activation = self.__sigmoid
            self.__activation_prime = self.__sigmoid_derivative
        else:
            # Throw Error Due To Invalid Activation Function
            raise ValueError("activation function not recognized")

        # Initialize Weights Randomly
        self.W1 = np.random.randn(self.input_size, self.neurons_per_layer)
        self.W2 = np.random.randn(self.neurons_per_layer, self.output_size)

    # ReLU Activation Function
    def __relu(self, x):
        # Compute ReLU Value At X
        return (max(0, x))

    # ReLU Activation Function Derivative
    def __relu_derivative(self, x):
        # Return Derivative Of ReLU
        return ((1) if (x > 0) else (0))

    # Sigmoid Activation Function
    def __sigmoid(self, x):
        # Compute Sigmoid Value At X
        return (1 / (1 + np.exp(-x)))

    # Sigmoid Activation Function Derivative
    def __sigmoid_derivative(self, x):
        # Compute Sigmoid Value At X
        s = self.__sigmoid(x)

        # Return Derivative Of Sigmoid
        return (s * (1 - s))

    # Unrandomized Batch Generator For Mini-Batches
    def __batchGenerator(self, l, n):
        # Iterate Over Training Samples
        for i in range(0, len(l), n):
            # Return Slice Of Training Samples
            yield l[i : i + n]

    # Training with backpropagation
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        pass

    # Forward Pass Function
    def __forward(self, input):
        # Calculate Sigmoid Values For Layers
        layer1 = self.__activation(np.dot(input, self.W1))
        layer2 = self.__activation(np.dot(layer1, self.W2))

        # Return Result
        return (layer1, layer2)

    # Prediction Function
    def predict(self, xVals):
        # Perform Forward Pass
        _, layer2 = self.__forward(xVals)

        # Return Result
        return (layer2)

# End Neural Network Layer Class---------------------------------------------------------------------------------------------------------------------------------------

def get_raw_data():
    # Fetch MNIST Dataset From Tensorflow Imporrt
    mnist = tf.keras.datasets.mnist

    # Load Data From MNIST Dataset
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    # Display Information About Dataset
    print("Shape of xTrain dataset: %s" % str(xTrain.shape))
    print("Shape of yTrain dataset: %s" % str(yTrain.shape))
    print("Shape of xTest dataset: %s" % str(xTest.shape))
    print("Shape of yTest dataset: %s" % str(yTest.shape))

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
    print("New shape of xTrain dataset: %s" % str(xTrainP.shape))
    print("New shape of yTrain dataset: %s" % str(yTrainP.shape))
    print("New shape of xTest dataset: %s" % str(xTestP.shape))
    print("New shape of yTest dataset: %s" % str(yTestP.shape))

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
        print("Training custom neural network...")

        # Initialize New Neural Network Instance
        model = NeuralNetwork_2Layer(INPUT_SIZE, OUTPUT_SIZE, NUM_NEURON, activation = "sigmoid")

        # Train Model With Training Data
        model.train(xTrain, yTrain, epochs = NUM_EPOCHS)

        # Return Model
        return (model)
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Training Tensorflow neural network...")

        # Initialize New Sequential Instance
        model = keras.Sequential()

        # Initialize Loss Function
        loss_func = keras.losses.categorical_crossentropy

        # Initialize Model Optimizer
        opt_func = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

        # Add Neuron Hidden Layer To Model
        model.add(keras.layers.Dense(HIDDEN_SIZE, input_shape = [INPUT_SIZE], activation = tf.nn.relu))

        # Add Neuron Output Layer To Model
        model.add(keras.layers.Dense(OUTPUT_SIZE, input_shape = [HIDDEN_SIZE], activation = tf.nn.softmax))

        # Compile Model
        model.compile(optimizer = opt_func, loss = loss_func)

        # Train Model
        model.fit(xTrain, yTrain, epochs = NUM_EPOCHS)

        # Return Model
        return (model)
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("algorithm not recognized")

def run_model(data, model):
    if (ALGORITHM == "guesser"):
        # Display Status
        print("Running guesser algorithm...")

        # Initialize Answer Vector
        ans = []

        # Iterate Over xTest Sample
        for entry in (data):
            # Initialize Base Prediction Vector
            pred = [0] * 10

            # Randomly Set Class Label
            pred[random.randint(0, 9)] = 1

            # Append Predicted Class To Answer Vector
            ans.append(pred)

        # Return Prediction
        return (np.array(ans))
    elif (ALGORITHM == "custom_net"):
        # Display Status
        print("Running custom neural network...")

        # Run Custom Model
        preds = model.predict(data)

        # Return Prediction
        return (np.array(preds))
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Running Tensorflow neural network...")

        # Run Keras Model
        preds = model.predict(data)

        # TODO: one hot encoding

        # Return Prediction
        return (np.array(preds))
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("algorithm not recognized")

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
    print("Classifier accuracy: %f%%" % (acc * 100))

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
