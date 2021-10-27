# Modified By Nalin Ahuja, ahuja15@purdue.edu

import os
import sys

# Set TensorFlow Logging Level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# End Module Imports----------------------------------------------------------------------------------------------------------------------------------------------------

# Random Seed Value
SEED_VALUE = 1618

# Media Directory
MEDIA_DIR = "./media"

# Content Image File Path
CONTENT_IMG_PATH = os.path.join(MEDIA_DIR, "content/john.jpg")

# Content Image Dimensions
CONTENT_IMG_W = 500
CONTENT_IMG_H = 500

# Style Image File Path
STYLE_IMG_PATH = os.path.join(MEDIA_DIR, "style/stars.jpg")

# Style Image Dimensions
STYLE_IMG_W = 500
STYLE_IMG_H = 500

# Style Transfer Weights
CONTENT_WEIGHT = 0.300    # Alpha Weight
STYLE_WEIGHT = 100.0      # Beta Weight
TOTAL_WEIGHT = 1.000

# Number Of Style Transfers
STYLE_TRANSFER_ROUNDS = 3

# Gradient Decent Values
GRADIENT_DECENT_FUNC = 25
GRADIENT_DECENT_ITER = 1000

# End Embedded Constants------------------------------------------------------------------------------------------------------------------------------------------------

import random
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb

from tensorflow import keras
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set Deterministic Random Seeds
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# TensorFlow Settings
# tf.set_random_seed(SEED_VALUE) # Uncomment for TF1
# tf.logging.set_verbosity(tf.logging.ERROR) # Uncomment for TF1
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# End Module Imports----------------------------------------------------------------------------------------------------------------------------------------------------

def deprocess_image(img):
    # Copy Image Representation
    img = img.copy()

    # Reshape Deprocessed Image
    img = img.reshape((STYLE_IMG_H, STYLE_IMG_W, 3))

    # Reverse VGG19 Transformation
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.680

    # Convert Image Colorspace To RGB
    img = img[:, :, ::-1]

    # Limit Array Values To Range
    return (np.clip(img, 0, 255).astype("uint8"))

def gram_matrix(x):
    # Check Image Data Format
    if (kb.image_data_format() == "channels_first"):
        # Flatten The Input Matrix
        features = kb.flatten(x)
    else:
        # Flatten The Input With Permuted Dimensions
        features = kb.batch_flatten(kb.permute_dimensions(x, (2, 0, 1)))

    # Compute Gram Matrix
    gram = kb.dot(features, kb.transpose(features))

    # Return Gram Matrix
    return (gram)

# End Helper Functions--------------------------------------------------------------------------------------------------------------------------------------------------

def content_loss(content, gen):
    # Return Content Loss
    return (kb.sum(kb.square(gen - content)))

def style_loss(style, gen):
    # Return Style Loss
    return (kb.sum(kb.square(gram_matrix(style) - gram_matrix(gen))) / (4.0 * np.square(style.shape[2]) * np.square(STYLE_IMG_H * STYLE_IMG_W)))

# End Loss Functions----------------------------------------------------------------------------------------------------------------------------------------------------

def get_data():
    # Print Status
    print("  Loading Images")
    print("    Content Image Path: \"%s\"" % CONTENT_IMG_PATH)
    print("    Style Image Path:   \"%s\"" % STYLE_IMG_PATH)

    # Load Content Image
    c_img = load_img(CONTENT_IMG_PATH)

    # Load Transfer Image
    t_img = c_img.copy()

    # Load Style Image
    s_img = load_img(STYLE_IMG_PATH)

    # Return Image Data
    return ((c_img, CONTENT_IMG_H, CONTENT_IMG_W), (s_img, STYLE_IMG_H, STYLE_IMG_W), (t_img, CONTENT_IMG_H, CONTENT_IMG_W))

def preprocess_data(raw):
    # Unpack Image Data
    img, ih, iw = raw

    # Convert Image To Numpy Array
    img = img_to_array(img)

    # Convert Image Data Type
    img = img.astype("float64")

    # Resize Image To Specified Dimensions
    img = cv.resize(img, dsize = (iw, ih))

    # Add Dimension To Numpy Array
    img = np.expand_dims(img, axis = 0)

    # Preprocess Image Through VGG19
    img = vgg19.preprocess_input(img)

    # Return Image Data
    return (img)

def style_transfer(c_data, s_data, t_data):
    # Print Status
    print("\n  Building Style Transfer Model")

    # Create Content Tensor
    content_tensor = kb.variable(c_data)

    # Create Style Tensor
    style_tensor = kb.variable(s_data)

    # Create Placeholder Generator Tensor
    gen_tensor = kb.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))

    # Create Input Tensor From Previous Tensors
    input_tensor = kb.concatenate([content_tensor, style_tensor, gen_tensor], axis = 0)

    # Load VGG19 Model
    model = vgg19.VGG19(include_top = False, weights = "imagenet", input_tensor = input_tensor)

    # Extract Layer Data From Model
    output_dict = dict([(layer.name, layer.output) for layer in (model.layers)])

    # Print Status
    print("    VGG19 Model Loaded")

    # Initialize Loss Metric
    loss = 0

    # Initialize Layer Names
    content_layer_name, style_layer_names = "block5_conv2", ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

    # Print Status
    print("    Calculating Content Loss")

    # Extract Corresponding Content Layers
    content_layer = output_dict[content_layer_name]
    content_output = content_layer[0, :, :, :]
    content_gen_output = content_layer[2, :, :, :]

    # Compute Content Loss
    loss += CONTENT_WEIGHT * content_loss(content_output, content_gen_output)

    # Print Status
    print("    Calculating Style Loss")

    # Iterate Over Style Layers
    for style_layer_name in (style_layer_names):
        # Extract Corresponding Style Layers
        style_layer = output_dict[style_layer_name]
        style_output = style_layer[1, :, :, :]
        style_gen_output = style_layer[2, :, :, :]

        # Compute Style Loss
        loss += STYLE_WEIGHT * style_loss(style_output, style_gen_output)

    # Setup Gradients
    grads = kb.gradients(loss, gen_tensor)

    # Setup Outputs List
    outputs = [loss, grads]

    # Initialize Loss Function
    kf = kb.function([gen_tensor], outputs)

    # Initialize Loss Function
    def loss_func(x):
        # Reshape Input To Generator Tensor Dimensions
        x = x.reshape((1, STYLE_IMG_H, STYLE_IMG_W, 3))

        # Get Loss
        loss, _ = kf([x])

        # Return Loss
        return (np.array(loss).flatten().astype("float64"))

    # Initialize Gradient Function
    def grad_func(x):
        # Reshape Input To Generator Tensor Dimensions
        x = x.reshape((1, STYLE_IMG_H, STYLE_IMG_W, 3))

        # Get Gradient
        _, grad = kf([x])

        # Return Gradient
        return (np.array(grad).flatten().astype("float64"))

    # Print Status
    print("\n  Beginning Style Transfer")

    # Perform Style Transfer
    for i in range(1, STYLE_TRANSFER_ROUNDS + 1):
        # Print Step Increment
        print("    Step %d" % i)

        # Perform Gradient Decent Using fmin_l_bfgs_b
        t_data, t_loss, _ = fmin_l_bfgs_b(loss_func, t_data.flatten(), grad_func, maxfun = GRADIENT_DECENT_FUNC, maxiter = GRADIENT_DECENT_ITER)

        # Print Total Loss
        print("      Loss: %f" % t_loss)

        # Deprocess Image
        t_img = deprocess_image(t_data)

        # Convert Array Into Image
        t_img = cv.cvtColor(t_img, cv.COLOR_BGR2RGB)

        # Create Saved Image File Path
        save_file = os.path.join(MEDIA_DIR, f"st_{i}.jpg")

        # Save Image To Disk
        cv.imwrite(save_file, t_img)

        # Print Status
        print("      Image Path: \"%s\"\n" % save_file)

# End Pipeline Functions------------------------------------------------------------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    # Print Status
    print("Starting Style Transfer Program\n")

    # Get Data
    raw = get_data()

    # Preprocess Content Image
    c_data = preprocess_data(raw[0])

    # Preprocess Style Image
    s_data = preprocess_data(raw[1])

    # Preprocess Transfer Image
    t_data = preprocess_data(raw[2])

    # Apple Style Transfer
    style_transfer(c_data, s_data, t_data)

# End Main Function-----------------------------------------------------------------------------------------------------------------------------------------------------
