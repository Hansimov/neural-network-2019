# Convolutional Neural Networks with TensorFlow
#   https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = input_data.read_data_sets('fashion',one_hot=True)

print("\n+++ Data Extracted Successfully +++\n")

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))

print("")


# Create dictionary of target classes
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

# plt.figure(figsize=[5,5])

# # Display the first image in training data
# plt.subplot(121)
# curr_img = np.reshape(data.train.images[0], (28,28))
# curr_lbl = np.argmax(data.train.labels[0,:])
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# # Display the first image in testing data
# plt.subplot(122)
# curr_img = np.reshape(data.test.images[0], (28,28))
# curr_lbl = np.argmax(data.test.labels[0,:])
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# plt.show()

# print(data.train.images[0])
# print(np.max(data.train.images[0]), np.min(data.train.images[0]))

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)
print(train_X.shape, test_X.shape)

train_y = data.train.labels
test_y = data.test.labels
print(train_y.shape, test_y.shape)


# Some parameters
training_iters  = 200
learning_rate   = 0.001
batch_size      = 128

# MNIST data input (img shape: 28*28)
n_input = 28

# MNIST total classes (0-9 digits)
n_classes = 10

# both placeholders are of type float
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])

# input x, weights W, bias b, strides
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x

# input x, kernel size k
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32),      initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64),     initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128),    initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128),   initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32),    initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64),    initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128),   initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128),   initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10),    initializer=tf.contrib.layers.xavier_initializer()),
}