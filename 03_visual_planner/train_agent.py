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
LAMBDA       = 5e-4	# regularizer
SEED = 66478    # Set to None for random seed.
NUM_EPOCHS = 15 

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
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
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

# conv2: 5*5 filter with depth 64
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], 
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_conv2')  
b_conv2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_conv2')

# fc1: 7*7*64 num_units with depth 128
W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 128], 
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_fc1')
b_fc1 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32), name='b_fc1')

# fc2: 128 num_units with depth: NUM_LABELS
W_fc2 = tf.Variable(tf.truncated_normal(shape=[128, NUM_LABELS], 
										  stddev=0.1, dtype=tf.float32, seed=SEED), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros(shape=[NUM_LABELS], dtype=tf.float32), name='b_fc2')
 

# define your model here
def model(data, train=False):

    h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
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

global_step = tf.Variable(0, dtype=tf.float32)
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

#print('train_prediction: ', train_prediction)  
#'train_prediction: ', <tf.Tensor 'Softmax:0' shape=(32, 5) dtype=float32>

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
    
    for begin in xrange(0, size, opt.valid_size):
		end = begin + opt.valid_size
	if end <= size:
	    predictions[begin:end, :] = sess.run(eval_prediction, 
						  feed_dict={eval_data: data[begin:end, ...]})
	else:
	    batch_predictions = sess.run(eval_prediction, 
					  feed_dict={eval_data: data[-opt.valid_size:, ...]})
	    predictions[begin:, :] = batch_predictions[begin-size:, :]
    return predictions

# Create a local session to run the training.
start_time = time.time()

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()

# Remember 'logits' we want to run by adding it to a collection.
#tf.add_to_collection('logits', logits)


# test_agent
# Add ops to save and restore all the variables.
#new_saver = tf.train.Saver()

#agent = None

# 1. control loop
if opt.disp_on:
	win_all = None
	win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network



with tf.Session() as sess:
    # TODO
    # Make sure you initialize all variables before starting the tensorflow training
	tf.global_variables_initializer().run()
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
    
	for step in xrange(int(NUM_EPOCHS * train_size) // opt.minibatch_size):
		offset = (step * opt.minibatch_size) % (train_size - opt.minibatch_size)
		#print('type of offset: ', type(offset))
		#print('offset: ', offset)
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
			
			print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * opt.minibatch_size / train_size, 
											1000 * elapsed_time / opt.eval_freq))
			print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
			print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
			print('Validation error: %.1f%%' % error_rate(eval_in_batches(valid_data[0].reshape((opt.n_minibatches, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len)), sess), np.argmax(valid_data[1], axis=1)))
		
			# Flush the internal I/O buffer
			#sys.stdout.flush()			
			
# 2. save your trained model
	
	#meta_graph_def = tf.train.export_meta_graph(filename='./logs/trained_model.meta', 
	#										 collection_list=["input_tensor", "output_tensor"])
    #saver.save(sess, "./model/trained_model")
    
	print('Training session completed!')
	
	
#######################################
# 3. Here is the test phase!
#######################################
	
	# start a new game
	state = sim.newGame(opt.tgt_y, opt.tgt_x)
	for step in range(opt.eval_steps):
		print('step: ', step)

		# check if episode ended
		if state.terminal or epi_step >= opt.early_stop:
			epi_step = 0
			nepisodes += 1
			if state.terminal:
				nepisodes_solved += 1
			# start a new game
			state = sim.newGame(opt.tgt_y, opt.tgt_x)
			
		elif epi_step == 0:
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			# TODO: here you would let your agent take its action
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			# this just gets a random action
			
			# Restore variables from disk
			#new_saver = tf.train.import_meta_graph('./model/trained_model.meta')
			#new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
			
			#test_data_node = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.cub_siz*opt.pob_siz, 
			#											opt.cub_siz*opt.pob_siz, opt.hist_len))
			#logits = tf.get_collection('logits')
			
			# Print all the trainable variables
			#all_vars = tf.trainable_variables()
			#for v in all_vars:
			#	print('v_name: ', v.name)
			
			first_state = rgb2gray(state.pob).reshape((opt.cub_siz*opt.pob_siz, 
												opt.cub_siz*opt.pob_siz)).astype(np.float32)
			hist_states = np.dstack([first_state] * opt.hist_len)
			
			#print('step {} hist_states {}'.format(epi_step, hist_states))
			
			# Copy the hist_states for 32 times to fit into the placeholder
			feed_data = np.repeat(hist_states[np.newaxis, ...], 32, axis=0)
			
			logits_0 = sess.run(logits, feed_dict={train_data_node: feed_data})
			
			#print('logits: ', logits.eval())
			
			# random action
			#action = randrange(opt.act_num)
			action = np.argmax(logits_0[0])
			state = sim.step(action)
			
			print('step {} action {}'.format(epi_step, action))
			#print('action_0: ', action)
		
			epi_step += 1
				
		else:
			next_state = rgb2gray(state.pob).reshape((opt.cub_siz*opt.pob_siz, 
												opt.cub_siz*opt.pob_siz)).astype(np.float32)
			
			# Replace the earliest state with the latest state
			hist_states = np.delete(hist_states, 0, 2)
			#print('delete hist_states: ', hist_states)
			#print('delete hist_states: ')
			
			hist_states = np.insert(hist_states, -1, next_state, axis=2)			
			#print('insert hist_states: ', hist_states)
			#print('insert hist_states: ')
			
			feed_data = np.repeat(hist_states[np.newaxis, ...], 32, axis=0)
			
			logits_next = sess.run(logits, feed_dict={train_data_node: feed_data})
			
			#print('logits: ', logits.eval())

			# random action
			#action = randrange(opt.act_num)
			if epi_step < 10:
				action = randrange(opt.act_num)
			else:
				action = np.argmax(logits_next[0])
			state = sim.step(action)
			
			print('step {} action {}'.format(epi_step, action))
			
			epi_step += 1
						
		if state.terminal or epi_step >= opt.early_stop:
			epi_step = 0
			nepisodes += 1
			if state.terminal:
				nepisodes_solved += 1
			# start a new game
			state = sim.newGame(opt.tgt_y, opt.tgt_x)

		if step % opt.prog_freq == 0:
			print('step: ', step)

		if opt.disp_on:
			if win_all is None:
				plt.figure()
				plt.subplot(121)
				win_all = plt.imshow(state.screen)
				plt.subplot(122)
				win_pob = plt.imshow(state.pob)
			else:
				win_all.set_data(state.screen)
				win_pob.set_data(state.pob)
			plt.pause(opt.disp_interval)
			plt.draw()

# 4. calculate statistics
	print('Accuracy: ', float(nepisodes_solved) / float(nepisodes))
