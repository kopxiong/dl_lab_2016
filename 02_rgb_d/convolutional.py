# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional rgbd_10 model example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# TODO
# These are some useful constants that you can use in your code.
# Feel free to ignore them or change them.
# TODO 
IMAGE_SIZE = 32
NUM_LABELS = 10
SEED = 66478    # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 15 
EVAL_BATCH_SIZE = 1024
EVAL_FREQUENCY = 100    # Number of steps between evaluations.
LAMBDA = 5e-4    # Regularization parameter

# This is where the data gets stored
TRAIN_DIR = 'data'
# HINT:
# if you are working on the computers in the pool and do not want
# to download all the data you can use the pre-loaded data like this:
# TRAIN_DIR = '/home/mllect/data/rgbd'


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])

def fake_data(num_images, channels):
    """Generate a fake dataset that matches the dimensions of rgbd_10 dataset."""
    data = numpy.ndarray(
        shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, channels),
        dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    
    return data, labels

def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.self_test:
        print('Running self-test.')
        NUM_CHANNELS = 1
        train_data, train_labels = fake_data(256, NUM_CHANNELS)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
        num_epochs = 1
    else:
        if (FLAGS.use_rgbd):
            NUM_CHANNELS = 4
            print('****** RGBD_10 dataset ******') 
            print('* Input: RGB-D              *')
            print('* Channels: 4               *') 
            print('*****************************')
        else:
            NUM_CHANNELS = 3
            print('****** RGB_10 dataset ******') 
            print('* Input: RGB                *')
            print('* Channels: 3               *') 
            print('*****************************')
      
        # Load input data
        data_sets = input_data.read_data_sets(TRAIN_DIR, FLAGS.use_rgbd)
        num_epochs = NUM_EPOCHS

        train_data = data_sets.train.images
        train_labels= data_sets.train.labels
        test_data = data_sets.test.images
        test_labels = data_sets.test.labels 
        validation_data = data_sets.validation.images
        validation_labels = data_sets.validation.labels

    train_size = train_labels.shape[0]

    # TODO:
    # After this you should define your network and train it.
    # Below you find some starting hints. For more info have
    # a look at online tutorials for tensorflow:
    # https://www.tensorflow.org/versions/r0.11/tutorials/index.html
    # Your goal for the exercise will be to train the best network you can
    # please describe why you did chose the network architecture etc. in
    # the one page report, and include some graph / table showing the performance
    # of different network architectures you tried.
    #
    # Your end result should be for RGB-D classification, however, you can
    # also load the dataset with NUM_CHANNELS=3 to only get an RGB version.
    # A good additional experiment to run would be to compare how much
    # you can gain by adding the depth channel (how much better the classifier can get)
    # TODO:

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        data_type(),
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        data_type(),
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # TODO
    # define weights and biases initialization functions
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, 
                                    stddev=0.1, 
                                    dtype=data_type(),
                                    seed=SEED)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=data_type())
        #initial = tf.constant(0.1, dtype=data_type(), shape=shape)
        return tf.Variable(initial)

    # define convolutional and pooling functions (2D convolution with 'SAME' padding)
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Global variables: parameters for the Lenet5-like network
    # conv1: 5*5 filter with depth 32
    W_conv1 = weight_variable([5, 5, NUM_CHANNELS, 32])
    b_conv1 = bias_variable([32])

    # conv2: 5*5 filter with depth 64
    W_conv2 = weight_variable([5, 5, 32, 64])   
    b_conv2 = bias_variable([64])
    
    # conv3: 5*5 filter with depth 128
    W_conv3 = weight_variable([3, 3, 64, 128])   
    b_conv3 = bias_variable([128])
    
    # conv4: 5*5 filter with depth 256
    W_conv4 = weight_variable([3, 3, 128, 256])   
    b_conv4 = bias_variable([256])

    # fc1: 8*8*64 num_units with depth 512
    W_fc1 = weight_variable([2*2*256, 512])
    b_fc1 = bias_variable([512])

    # fc2: 512 num_units with depth: NUM_LABELS
    W_fc2 = weight_variable([512, NUM_LABELS])
    b_fc2 = bias_variable([NUM_LABELS])
        
    # define your model here
    def model(data, train=False):

        h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)
        
        h_pool2_flat = tf.reshape(h_pool4, [-1, 2*2*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # add a 50% dropout during training only
        if train:
            h_fc1 = tf.nn.dropout(h_fc1, 0.5, seed=SEED)
        
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
        
        return h_fc2

    # TODO
    # compute the loss of the model here
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                          (logits, train_labels_node, name='xentropy'), 
                          name='xentropy_mean')
    
    # add L2 regularizer for the fully connected parameters
    l2_reg = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2)
    loss += LAMBDA * l2_reg

    # TODO
    # then create an optimizer to train the model
    # HINT: you can use the various optimizers implemented in TensorFlow.
    #       For example, google for: tf.train.AdamOptimizer()
    
    #tf.scalar_summary(loss.op.name, loss)
    
    global_step = tf.Variable(0, dtype=data_type())
    learning_rate = tf.train.exponential_decay(0.01, 
                                               global_step * BATCH_SIZE, 
                                               train_size,
                                               0.95, 
                                               staircase=True)
    # Use momentum for the optimization
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    
    # Use AdamOptimizer for the optimization
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(loss, global_step=global_step)
    
    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # TODO
    # Make sure you also define a function for evaluating on the validation
    # set so that you can track performance over time
    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction, 
                                                     feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(eval_prediction, 
                                             feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin-size:, :]
        return predictions

    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        # TODO
        # Make sure you initialize all variables before starting the tensorflow training
        tf.initialize_all_variables().run()
        print('Initialized!')

        # Loop through training steps here
        # HINT: always use small batches for training (as in SGD in your last exercise)
        # WARNING: The dataset does contain quite a few images if you want to test something quickly
        #          It might be useful to only train on a random subset!
        # For example use something like :
        # for step in max_steps:
        # Hint: make sure to evaluate your model every once in a while
        # For example like so:
        #print('Minibatch loss: {}'.format(loss))
        #print('Validation error: {}'.format(validation_error_you_computed)
        
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset: (offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset: (offset + BATCH_SIZE)]
            
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            
            # Run the optimizer to update parameters
            sess.run(optimizer, feed_dict=feed_dict)
            
            if step % EVAL_FREQUENCY == 0:
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction], 
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
        
                # Flush the internal I/O buffer
                sys.stdout.flush()

        # Finally, after the training! calculate the test result!
        # WARNING: You should never use the test result to optimize
        # your hyperparameters/network architecture, only look at the validation error to avoid
        # overfitting. Only calculate the test error in the very end for your best model!
        # if test_this_model_after_training:
        #     print('Test error: {}'.format(test_error))
        #     print('Confusion matrix:') 
        #     # NOTE: the following will require scikit-learn
        #     print(confusion_matrix(test_labels, numpy.argmax(eval_in_batches(test_data, sess), 1)))
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)
        
        if FLAGS.self_test:
            print('test_error', test_error)
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error, )
        
        # NOTE: the following will require scikit-learn
        print('Confusion matrix:') 
        print(confusion_matrix(test_labels, numpy.argmax(eval_in_batches(test_data, sess), 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_rgbd',
        default=False,
        help='Use rgb-d input data (4 channels).',
        action='store_true')
    parser.add_argument(
        '--self_test',
        default=False,
        action='store_true',
        help='True if running a self test.')
    FLAGS = parser.parse_args()

    tf.app.run()
