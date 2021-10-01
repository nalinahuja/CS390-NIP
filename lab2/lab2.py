# Modified By Nalin Ahuja, ahuja15@purdue.edu

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras

# End Imports----------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Seed Value
SEED_VALUE = 1618

# TensorFlow Classifier Training Constants
TF_NUM_EPOCHS = 15
TF_LEARNING_RATE = 0.001

# Selected Algorithm ("guesser", "tf_net", "tf_conv")
ALGORITHM = "guesser"

# Selected Dataset ("mnist_d", "mnist_f", "cifar_10", "cifar_100_f", "cifar_100_c")
DATASET = "mnist_d"

# Conditonally Initialize TensorFlow Network Structure
if (DATASET == "mnist_d"):
    # Set Number Of Classes
    NUM_CLASSES = 10

    # Set Input Dimensions
    IH = 28
    IW = 28
    IZ = 1
    IS = IH * IW * IZ
elif (DATASET == "mnist_f"):
    # Set Number Of Classes
    NUM_CLASSES = 10

    # Set Input Dimensions
    IH = 28
    IW = 28
    IZ = 1
    IS = IH * IW * IZ
elif (DATASET == "cifar_10"):
    # Set Number Of Classes
    NUM_CLASSES = 10

    # Set Input Dimensions
    IH = 32
    IW = 32
    IZ = 3
    IS = IH * IW * IZ
elif (DATASET == "cifar_100_f"):
    # Set Number Of Classes
    NUM_CLASSES = 100

    # Set Input Dimensions
    IH = 32
    IW = 32
    IZ = 3
    IS = IH * IW * IZ
elif (DATASET == "cifar_100_c"):
    # Set Number Of Classes
    NUM_CLASSES = 20

    # Set Input Dimensions
    IH = 32
    IW = 32
    IZ = 3
    IS = IH * IW * IZ

# End Embedded Constants------------------------------------------------------------------------------------------------------------------------------------------------

# Setting Random Seeds To Maintain Deterministic Behavior
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Tensorflow Settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.set_random_seed(SEED_VALUE) # Uncomment for TF1
# tf.logging.set_verbosity(tf.logging.ERROR) # Uncomment for TF1

# End Module Initialization---------------------------------------------------------------------------------------------------------------------------------------------

def build_tf_neural_net(x, y, eps = 6):
    # TODO: Implement a standard ANN here.
    return None

def build_tf_conv_net(x, y, eps = 10, dropout = True, dropRate = 0.2):
     # TODO: Implement a CNN here. dropout option is required.
    return None

# End Classifier Functions----------------------------------------------------------------------------------------------------------------------------------------------

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
    if (DATASET == "mnist_d"):
        # Load Data From MNIST Dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif (DATASET == "mnist_f"):
        # Load Data From MNIST Dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif (DATASET == "cifar_10"):
        # Load Data From CIFAR10 Dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif (DATASET == "cifar_100_f"):
        # Load Fine Data From CIFAR100 Dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode = "fine")
    elif (DATASET == "cifar_100_c"):
        # Load Coarse Data From CIFAR100 Dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode = "coarse")
    else:
        # Throw Error Due To Invalid Dataset
        raise ValueError("dataset not recognized")

    # Display Information About Dataset
    print("Dataset: %s" % DATASET)
    print("Shape of x_train dataset: %s." % str(x_train.shape))
    print("Shape of y_train dataset: %s." % str(y_train.shape))
    print("Shape of x_test dataset: %s." % str(x_test.shape))
    print("Shape of y_test dataset: %s." % str(y_test.shape))
    print("\n" * 1, end = "")

    # Return Data
    return ((x_train, y_train), (x_test, y_test))

def process_data(raw):
    # Unpack Data From Raw Input
    ((x_train, y_train), (x_test, y_test)) = raw

    # Conditionally Reshape Input Data
    if (ALGORITHM == "tf_conv"):
        # Reshape Input Data To Fit Convolutional Networks
        x_train = x_train.reshape((x_train.shape[0], IH, IW, IZ))
        x_test = x_test.reshape((x_test.shape[0], IH, IW, IZ))
    else:
        # Reshape Input Data To Fit Non-Convolutional Networks
        x_train = x_train.reshape((x_train.shape[0], IS))
        x_test = x_test.reshape((x_test.shape[0], IS))

    # Process Integer Arrays Into Binary Class Matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Display Information About Dataset
    print("New shape of x_train dataset: %s." % str(x_train.shape))
    print("New shape of x_test dataset: %s." % str(x_test.shape))
    print("New shape of y_train dataset: %s." % str(y_train.shape))
    print("New shape of y_test dataset: %s." % str(y_test.shape))
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
    elif (ALGORITHM == "tf_net"):
        # Display Status
        print("Training TensorFlow neural network...")

        # Return Model
        return (build_tf_neural_net(x_train, y_train))
    elif (ALGORITHM == "tf_conv"):
        # Display Status
        print("Training Tensorflow convolutional network...")

        # Return Model
        return (build_tf_conv_net(x_train, y_train))
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
        for i in range(len(data)):
            # Initialize Base Prediction Vector
            pred = np.zeros(NUM_CLASSES)

            # Randomly Set Class Label
            pred[random.randint(0, 9)] = 1

            # Append Predicted Class To Predictions Vector
            preds.append(pred)

        # Return Predictions
        return (np.array(preds))
    elif ALGORITHM == "tf_net":
        # Display Status
        print("Running TensorFlow neural network...\n")

        # Run TensorFlow Neural Model On Data
        preds = model.predict(data)

        # One Hot Encode Predictions
        preds = encode_preds(preds)

        # Return Predictions
        return (np.array(preds))
    elif ALGORITHM == "tf_conv":
        # Display Status
        print("Running Tensorflow convolutional network...\n")

        # Run TensorFlow Convolutional Model On Data
        preds = model.predict(data)

        # One Hot Encode Predictions
        preds = encode_preds(preds)

        # Return Predictions
        return (np.array(preds))
    else:
        # Throw Error Due To Invalid Algorithm
        raise ValueError("algorithm not recognized")

def eval_results(data, y_preds):
    # Unpack Output Test Data
    _, y_test = data

    # Initialize Accuracy Metric
    accuracy = 0

    # Iterate Over Predicted Values
    for i in range(y_preds.shape[0]):
        # Verify Predicted Values Match Expected Values
        if (np.array_equal(y_test[i], y_preds[i])):
            # Increment Accuracy Metric
            accuracy += 1

    # Calculate Accuracy
    accuracy /= y_preds.shape[0]

    # Display Classifier Metrics
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("\n" * 1, end = "")

# End Pipeline Functions------------------------------------------------------------------------------------------------------------------------------------------------

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

# End Main Function-----------------------------------------------------------------------------------------------------------------------------------------------------
