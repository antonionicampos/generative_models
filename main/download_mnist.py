import numpy as np
import os
import tensorflow as tf

file_folder = os.path.join(os.getcwd(), "datasets")
file_path = os.path.join(os.getcwd(), "datasets", "mnist.npz")

# Download and save MNIST data
if not os.path.isdir(file_folder):
    os.mkdir(file_folder)

train_data, test_data = tf.keras.datasets.mnist.load_data(path=file_path)
