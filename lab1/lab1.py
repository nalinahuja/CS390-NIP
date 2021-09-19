# Modified By Nalin Ahuja, ahuja15@purdue.edu

import os
import sys
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras

# End Imports----------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Seed Value
SEED_VALUE = 1618

# General Neural Network Structure Constants
INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10

# Custom Neural Network Training Constants
NN2L_NUM_EPOCHS = 15
NN2L_USE_BATCHES = True
NN2L_LEARNING_RATE = 0.001

# Keras Neural Network Training Constants
KERAS_NUM_EPOCHS = 15
KERAS_LEARNING_RATE = 0.001

# Selected Algorithm ("guesser", "custom_net", "tf_net")
ALGORITHM = "guesser"

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
    def __init__(self, input_size, output_size, neurons_per_layer, learning_rate = 0.001):
        # Initialize Neural Network Attributes
        self.input_size = input_size
        self.output_size = output_size
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate

        # Initialize Weights Randomly
        self.W1 = np.random.randn(self.input_size, self.neurons_per_layer)
        self.W2 = np.random.randn(self.neurons_per_layer, self.output_size)

    # Sigmoid Activation Function
    def __sigmoid(self, x):
        # Return Sigmoid Value At X
        return (1 / (1 + np.exp(-x)))

    # Sigmoid Activation Function Derivative
    def __sigmoid_prime(self, x):
        # Compute Sigmoid Value At X
        s = self.__sigmoid(x)

        # Return Derivative Of Sigmoid
        return (np.multiply(s, (1 - s)))

    # Unrandomized Batch Generator For Mini Batches
    def __batch_generator(self, x_vals, y_vals, n):
        # Iterate Over Training Samples
        for i in range(0, x_vals.shape[0], n):
            # Return Slice Of Training Samples
            yield (x_vals[i : i + n], y_vals[i : i + n])

    # Forward Pass Function
    def __forward(self, input):
        # Calculate Sigmoid Values For Layers
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))

        # Return Result
        return (layer1, layer2)

    # Loss Function
    def __loss(self, target, out):
        # Calculate Squared Differences
        loss = np.square(np.subtract(target, out))

        # Scalar Divide Squared Differences
        loss = np.divide(loss, 2)

        # Sum Over All Squared Differences
        loss = np.sum(loss)

        # Return Loss
        return (loss)

    # Loss Function Derivative
    def __loss_prime(self, target, out):
        # Return Loss
        return (np.subtract(out, target))

    # Training With Backpropagation
    def train(self, x_vals, y_vals, epochs = 10, mini_batches = True, batch_size = 100):
        # Run Training Epochs
        for i in range(1, epochs + 1):
            # Display Status
            print("\rEpoch %d/%d " % (i, epochs), end = "")

            # Determine Training Strategy
            if (mini_batches):
                # Train Neural Network Using Mini Batches
                for x_batch, y_batch in (self.__batch_generator(x_vals, y_vals, batch_size)):
                    # Get Feed Forward Output
                    l1_out, l2_out = self.__forward(x_batch)

                    # Calculate Loss Derivative For Batch
                    loss_prime = self.__loss_prime(y_batch, l2_out)

                    # Calculate Activation Derivative For Batch
                    logistic_prime = self.__sigmoid_prime(l2_out)

                    # Calculate Output Layer Output Differences
                    l2_diffs = np.multiply(loss_prime, logistic_prime)

                    # Calculate Error Derivative For All Hidden Layer Neurons
                    error_prime = np.dot(l2_diffs, np.transpose(self.W2))

                    # Calculate Activation Derivative For Batch
                    logistic_prime = self.__sigmoid_prime(l1_out)

                    # Calculate Hidden Layer Output Differences
                    l1_diffs = np.multiply(error_prime, logistic_prime)

                    # Compute Output Layer Weight Adjustments
                    l2_adj = np.matmul(np.transpose(l1_out), l2_diffs)

                    # Adjust Output Layer Weights
                    self.W2 -= np.multiply(l2_adj, self.learning_rate)

                    # Compute Hidden Layer Weight Adjustments
                    l1_adj = np.matmul(np.transpose(x_batch), l1_diffs)

                    # Adjust Hidden Layer Weights
                    self.W1 -= np.multiply(l1_adj, self.learning_rate)
            else:
                # Get Feed Forward Output
                l1_out, l2_out = self.__forward(x_vals)

                # Calculate Loss Derivative For Samples
                loss_prime = self.__loss_prime(y_vals, l2_out)

                # Calculate Activation Derivative For Samples
                logistic_prime = self.__sigmoid_prime(l2_out)

                # Calculate Output Layer Output Differences
                l2_diffs = loss_prime * logistic_prime

                # Calculate Error Derivative For All Hidden Layer Neurons
                error_prime = np.dot(l2_diffs, np.transpose(self.W2))

                # Calculate Activation Derivative For Samples
                logistic_prime = self.__sigmoid_prime(l1_out)

                # Calculate Hidden Layer Output Differences
                l1_diffs = error_prime * logistic_prime

                # Compute Output Layer Weight Adjustments
                l2_adj = np.matmul(np.transpose(l1_out), l2_diffs)

                # Adjust Output Layer Weights
                self.W2 -= np.multiply(l2_adj, self.learning_rate)

                # Compute Hidden Layer Weight Adjustments
                l1_adj = np.matmul(np.transpose(x_vals), l1_diffs)

                # Adjust Hidden Layer Weights
                self.W1 -= np.multiply(l1_adj, self.learning_rate)

    # Prediction Function
    def predict(self, xVals):
        # Perform Forward Pass
        _, layer2 = self.__forward(xVals)

        # Return Result
        return (layer2)

# End Neural Network Layer Class---------------------------------------------------------------------------------------------------------------------------------------

def encode_preds(preds):
    # Initialize Encoded Predictions Representation
    enc = np.zeros(preds.shape)

    # Determine Indicies With Maximum Probability
    mpi = np.argmax(preds, axis = 1)

    # Iterate Over Predictions
    for i in range(preds.shape[0]):
        # Set Position Of Maximum Probability
        enc[i][mpi[i]] = 1

    # Return Encoded Predictions Representation
    return (enc)

# End Utility Functions------------------------------------------------------------------------------------------------------------------------------------------------

def get_data():
    # Load Data From MNIST Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Display Information About Dataset
    print("Shape of x_train dataset: %s" % (str(x_train.shape)))
    print("Shape of y_train dataset: %s" % (str(y_train.shape)))
    print("Shape of x_test dataset: %s" % (str(x_test.shape)))
    print("Shape of y_test dataset: %s" % (str(y_test.shape)))
    print("\n" * 1, end = "")

    # Return Data
    return ((x_train, y_train), (x_test, y_test))

def process_data(raw):
    # Unpack Data From Raw Input
    ((x_train, y_train), (x_test, y_test)) = raw

    # Normalize Input Data
    x_train = np.divide(x_train, 255, dtype = np.float16)
    x_test = np.divide(x_test, 255, dtype = np.float16)

    # Reshape Input Data
    x_train = x_train.reshape(-1, 28 ** 2)
    x_test = x_test.reshape(-1, 28 ** 2)

    # Process Integer Arrays Into Binary Class Matrices
    y_train = keras.utils.to_categorical(y_train, OUTPUT_SIZE)
    y_test = keras.utils.to_categorical(y_test, OUTPUT_SIZE)

    # Display Information About Dataset
    print("New shape of x_train dataset: %s" % (str(x_train.shape)))
    print("New shape of y_train dataset: %s" % (str(y_train.shape)))
    print("New shape of x_test dataset: %s" % (str(x_test.shape)))
    print("New shape of y_test dataset: %s" % (str(y_test.shape)))
    print("\n" * 1, end = "")

    # Return Preprocessed Data
    return ((x_train, y_train), (x_test, y_test))

def train_model(data):
    # Unpack Training Data
    x_train, y_train = data

    # Run Training Algorithm
    if (ALGORITHM == "guesser"):
        # Return No Model
        return (None)
    elif (ALGORITHM == "custom_net"):
        # Display Status
        print("Training custom neural network...")

        # Initialize New Neural Network Instance
        model = NeuralNetwork_2Layer(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, learning_rate = NN2L_LEARNING_RATE)

        # Train Model With Training Data
        model.train(x_train, y_train, epochs = NN2L_NUM_EPOCHS, mini_batches = NN2L_USE_BATCHES)

        # Print Separator
        print("\n" * 2, end = "")

        # Return Model
        return (model)
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Training Tensorflow neural network...")

        # Initialize New Sequential Model Instance
        model = keras.Sequential()

        # Add Flattening Layer To Model
        model.add(keras.layers.Flatten())

        # Add Neuron Hidden Layer To Model
        model.add(keras.layers.Dense(HIDDEN_SIZE, input_shape = [INPUT_SIZE], activation = tf.nn.relu))

        # Add Neuron Output Layer To Model
        model.add(keras.layers.Dense(OUTPUT_SIZE, input_shape = [HIDDEN_SIZE], activation = tf.nn.softmax))

        # Initialize Loss Function
        loss_func = keras.losses.categorical_crossentropy

        # Initialize Model Optimizer
        opt_func = tf.optimizers.Adam(learning_rate = KERAS_LEARNING_RATE)

        # Compile Model
        model.compile(loss = loss_func, optimizer = opt_func, metrics = ["accuracy"])

        # Train Model
        model.fit(x_train, y_train, epochs = KERAS_NUM_EPOCHS)

        # Print Separator
        print("\n" * 1, end = "")

        # Return Model
        return (model)
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("algorithm not recognized")

def run_model(data, model):
    if (ALGORITHM == "guesser"):
        # Display Status
        print("Running guesser algorithm...\n")

        # Initialize Predictions Vector
        preds = []

        # Iterate Over Data Sample
        for i in range(data.shape[0]):
            # Initialize Base Prediction Vector
            pred = np.zeros(OUTPUT_SIZE)

            # Randomly Set Class Label
            pred[random.randint(0, 9)] = 1

            # Append Predicted Class To Predictions Vector
            preds.append(pred)

        # Return Predictions
        return (np.array(preds))
    elif (ALGORITHM == "custom_net"):
        # Display Status
        print("Running custom neural network...\n")

        # Run Custom Model On Data
        preds = model.predict(data)

        # One Hot Encode Predictions
        preds = encode_preds(preds)

        # Return Predictions
        return (np.array(preds))
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Running Tensorflow neural network...\n")

        # Run Keras Model On Data
        preds = model.predict(data)

        # One Hot Encode Predictions
        preds = encode_preds(preds)

        # Return Predictions
        return (np.array(preds))
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("algorithm not recognized")

    # Print Separator
    print()

def eval_results(data, y_preds):
    # Unpack Output Test Data
    _, y_test = data

    # Format Test Data
    y_test = np.argmax(y_test, axis = 1)

    # Format Prediction Data
    y_preds = np.argmax(y_preds, axis = 1)

    # Initialize Accuracy
    accuracy = 0

    # Iterate Over Predicted Values
    for i in range(y_preds.shape[0]):
        # Verify Predicted Values Match Expected Values
        if (y_test[i] == y_preds[i]):
            # Increment Accuracy Metric
            accuracy += 1

    # Calculate Accuracy
    accuracy /= y_preds.shape[0]

    # Initialize Confusion Matrix Representation
    cm = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype = np.int32)

    # Iterate Over Predicted Values
    for i in range(y_preds.shape[0]):
        # Update Confusion Matrix
        cm[y_test[i]][y_preds[i]] += 1

    # Calculate Confusion Matrix Sum
    cm_sum = np.sum(cm)

    # Initialize F1 Scores Vector Representation
    f1 = np.zeros(OUTPUT_SIZE)

    # Iterate Over Output Dimension
    for i in range(OUTPUT_SIZE):
        # Determine True Positives
        tp = cm[i][i]

        # Determine False Positives
        fp = sum(cm[i][j] for j in range(OUTPUT_SIZE) if (i != j))

        # Determine False Negatives
        fn = sum(cm[j][i] for j in range(OUTPUT_SIZE) if (i != j))

        # Calculate F1 Score For Class
        f1[i] = float(tp / (tp + (0.5 * (fp + fn))))

    # Display Classifier Metrics
    print("Classifier algorithm: %s" % (ALGORITHM))
    print("Classifier accuracy: %f%%" % (accuracy * 100))

    # Print Confusion Matrix Header
    print("\nConfusion Matrix:")

    # Print Column Label Padding
    print(" " * 3, end = "")

    # Print Matrix Column Labels
    for i in range(OUTPUT_SIZE):
        print("%3d" % (i), end = " ")

    # Print Separator
    print("\n" * 1, end = "")

    # Print Labeled Classifier Confusion Matrix
    for i in range(OUTPUT_SIZE):
        print(i, cm[i])

    # Print F1 Score Header
    print("\nF1 Scores:")

    # Print Column Label Padding
    print(" " * 1, end = "")

    # Print F1 Score Column Labels
    for i in range(OUTPUT_SIZE):
        print("%6d" % (i), end = " ")

    # Print Separator
    print("\n" * 1, end = "")

    # Print F1 Scores
    print(np.around(f1, decimals = 4))

# End Pipeline Functions-----------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):
    # Ignore Numpy Warnings
    np.seterr(all = "ignore")

    # Get Raw Data
    raw = get_data()

    # Process Raw Data
    data = process_data(raw)

    # Train Model On Raw Data
    model = train_model(data[0])

    # Run Model On Raw Data
    preds = run_model(data[1][0], model)

    # Evaluate Model Results
    eval_results(data[1], preds)

# End Main Function----------------------------------------------------------------------------------------------------------------------------------------------------
