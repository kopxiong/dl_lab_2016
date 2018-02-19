import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from random import randrange

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

# set some constants or parameters for the neural network
NUM_LABELS   = 5
LAMBDA       = 5e-4 	# regularizer
SEED         = 66478    # Set to None for random seed.
NUM_EPOCHS   = 15

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])

""" train a neural network to predict the actions of the A* planner """

############################################################################
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
############################################################################

# 0. initialization
opt   = Options()
sim   = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# both train_data and valid_data contain tupes of images and labels
train_data = trans.get_train()  # train_data[0] shape: (16000, 2500); train_data[1] shape: (16000, 5)
valid_data = trans.get_valid()  # valid_data[0] shape: (500, 2500); valid_data[1] shape: (500, 5)

# alternatively you can get one random mini batch like this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.

train_data_node = tf.placeholder(tf.float32,
    shape=(opt.minibatch_size, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))
train_labels_node = tf.placeholder(tf.int32, shape=(opt.minibatch_size, ))

eval_data = tf.placeholder(tf.float32,
    shape=(opt.valid_size, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))

train_size = len(train_data[0])


# define convolutional and pooling functions (2D convolution with 'SAME' padding)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Global variables: parameters for the Lenet5-like network
# conv1: 5*5 filter with depth 32
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, opt.hist_len, 32],
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_conv1')
b_conv1 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='b_conv1')

# conv2: 3*3 filter with depth 32
W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32],
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_conv2')
b_conv2 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='b_conv2')

# fc1: 7*7*32 num_units with depth 64
W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*32, 64],
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_fc1')
b_fc1 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_fc1')

# fc2: 64 num_units with depth: NUM_LABELS
W_fc2 = tf.Variable(tf.truncated_normal(shape=[64, NUM_LABELS],
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros(shape=[NUM_LABELS], dtype=tf.float32), name='b_fc2')


# define your model here
def model(data, train=False):

    h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # add a 50% dropout during training only
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, 0.5, seed=SEED)
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    return h_fc2

# TODO
# compute the loss of the model here
logits = model(train_data_node, True)
loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
		      (labels=train_labels_node, logits=logits, name='xentropy'),
		      name='xentropy_mean')

# add L2 regularizer for the fully connected parameters
l2_reg = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2)
loss  += LAMBDA * l2_reg

# TODO
# then create an optimizer to train the model
# HINT: you can use the various optimizers implemented in TensorFlow.
#       For example, google for: tf.train.AdamOptimizer()

global_step   = tf.Variable(0, dtype=tf.float32)
learning_rate = tf.train.exponential_decay(0.01,
					    global_step * opt.minibatch_size,
					    train_size,
					    0.95,
					    staircase=True)

# Use AdamOptimizer for the optimization
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
				   epsilon=0.1).minimize(loss, global_step=global_step)

# Predictions for the current training minibatch.
train_prediction = tf.nn.softmax(logits)

# Predictions for the test and validation, which we'll compute less often.
eval_prediction = tf.nn.softmax(model(eval_data))

# TODO
# Make sure you also define a function for evaluating on the validation
# set so that you can track performance over time
def eval_in_batches(data, sess):
    size = data.shape[0]
    if size < opt.valid_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)

    for begin in range(0, size, opt.valid_size):
        end = begin + opt.valid_size
    if end <= size:
	    predictions[begin:end, :] = sess.run(eval_prediction,
						  feed_dict={eval_data: data[begin:end, ...]})
    else:
        batch_predictions = sess.run(eval_prediction,
					  feed_dict={eval_data: data[-opt.valid_size:, ...]})
        predictions[begin:, :] = batch_predictions[begin-size:, :]
    return predictions

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
save_file = "./model/trained_model.ckpt"
start_time = time.time()

sess = tf.Session()

# TODO
# Make sure you initialize all variables before starting the tensorflow training
sess.run(tf.global_variables_initializer())
print("Initialized!")

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

for step in range(int(NUM_EPOCHS * train_size) // opt.minibatch_size):
	offset = (step * opt.minibatch_size) % (train_size - opt.minibatch_size)
	batch_data = train_data[0][offset: (offset +
                    opt.minibatch_size)].reshape((opt.minibatch_size,
					opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len))
	batch_labels = train_data[1][offset: (offset + opt.minibatch_size)].astype(np.int32)
	batch_labels = np.argmax(batch_labels, axis=1)

	feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

	# Run the optimizer to update parameters
	sess.run(optimizer, feed_dict=feed_dict)

	if step % opt.eval_freq == 0:
		l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
					feed_dict=feed_dict)
		elapsed_time = time.time() - start_time
		start_time = time.time()
		print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * opt.minibatch_size / \
            train_size, 1000 * elapsed_time / opt.eval_freq))
		print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
		print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
		print('Validation error: %.1f%%' % error_rate(eval_in_batches(valid_data[0].reshape((opt.n_minibatches, \
            opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len)), sess), np.argmax(valid_data[1], axis=1)))

saver.save(sess, save_file)
print("Trained model saved!")
