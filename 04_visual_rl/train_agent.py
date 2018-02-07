import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    target_q = reward + (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 
														  1, keep_dims=True) 
    
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)    
    
    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    
    # here normalize the loss
    loss = tf.reduce_sum(tf.square(selected_q - target_q)) / opt.minibatch_size   
    return loss

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# set some constants or parameters for the neural network
SEED = 66478               # set to None for random seed.
EPSILON = 0.2              # epsilon greedy

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 200000
epi_step = 0
nepisodes = 0    
factor = 2.          # factor for initialization

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None

# setup placeholders for states(x), actions(u), rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))


# initialize network parameters using get_variable
W_conv1 = tf.get_variable("W_conv1", [5, 5, opt.hist_len, 32], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_conv1 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='b_conv1')

W_conv2 = tf.get_variable("W_conv2", [5, 5, 32, 64], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_conv2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_conv2')

W_fc1 = tf.get_variable("W_fc1", [8*8*64, 64], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_fc1 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_fc1')

W_fc2 = tf.Variable(tf.truncated_normal(shape=[64, opt.act_num], 
										stddev=1e-3, dtype=tf.float32, seed=SEED), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros(shape=[opt.act_num], dtype=tf.float32), name='b_fc2')

"""
# conv1: 5*5 filter with depth 32
n = ((5 * 5 * opt.hist_len + 32) / 2.)
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, opt.hist_len, 32], 
										stddev=np.sqrt(factor / n), dtype=tf.float32, seed=SEED), name='W_conv1')
b_conv1 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='b_conv1')

# conv2: 5*5 filter with depth 64
n = ((5 * 5 * 32 + 64) / 2.)
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], 
										stddev=np.sqrt(factor / n), dtype=tf.float32, seed=SEED), name='W_conv2')  
b_conv2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_conv2')

# fc1: 8*8*64 num_units with depth 64
n = ((8 * 8 * 64 + 64) / 2.)
W_fc1 = tf.Variable(tf.truncated_normal(shape=[8*8*64, 64], 
										stddev=np.sqrt(factor / n), dtype=tf.float32, seed=SEED), name='W_fc1')
b_fc1 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_fc1')

# fc2: 64 num_units with depth: opt.act_num
W_fc2 = tf.Variable(tf.truncated_normal(shape=[64, opt.act_num], 
										stddev=1e-3, dtype=tf.float32, seed=SEED), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros(shape=[opt.act_num], dtype=tf.float32), name='b_fc2')
"""

# define your model here
def model_forward(stateInput, train=False):

	stateInput = tf.reshape(stateInput, [opt.minibatch_size, opt.hist_len, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz])
	
	# change order of channels and image width/height and normalize
	stateInput = tf.transpose(stateInput, perm=[0, 2, 3, 1]) / 255.
	
	h_conv1 = tf.nn.relu(conv2d(stateInput, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	# add a 50% dropout during training only
	if train:
		h_fc1 = tf.nn.dropout(h_fc1, 0.5, seed=SEED)
	
	Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	return Q_value

# define convolutional and pooling functions (2D convolution with 'SAME' padding)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# get the output from your network
Q = model_forward(x, False)
Qn = model_forward(xn, False)

# compute the loss of the model here
loss = Q_loss(Q, u, Qn, ustar, r, term)

# use AdamOptimizer for the optimization
train_op = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

# take an action according to the maximal Q value of current state
#Q_action = tf.reduce_max(tf.argmax(Q, axis=1))
Q_action = tf.argmax(Q, 1)

# launch a session
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()    #This function will be deprecated after 2017-03-02.

# save the trained model
saver = tf.train.Saver()

tf.add_to_collection('params', W_conv1)
tf.add_to_collection('params', b_conv1)
tf.add_to_collection('params', W_conv2)
tf.add_to_collection('params', b_conv2)
tf.add_to_collection('params', W_fc1)
tf.add_to_collection('params', b_fc1)
tf.add_to_collection('params', W_fc2)
tf.add_to_collection('params', b_fc2)

#tf.get_default_graph().finalize()

sess = tf.Session()
sess.run(init)


state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

for step in xrange(steps):
	if state.terminal or epi_step >= opt.early_stop:
		epi_step = 0
		nepisodes += 1
		
		# reset the game
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# TODO: here you would let your agent take its action
	#       remember this just gets a random action
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	#Input_state = np.repeat(state_with_history, opt.minibatch_size).reshape([opt.minibatch_size, -1])
	Input_state = np.tile(state_with_history.reshape(1, -1), [opt.minibatch_size, 1])
	
	#plt.imshow(state_with_history[0].reshape([30, 30]))
	#plt.show()
	#print(Input_state.shape, state_with_history.shape, state.pob.shape)
	#exit()
	
	#action = sess.run(Q_action, feed_dict = {x: Input_state})
	action = sess.run(Q_action, feed_dict = {x: Input_state})[0]
	
	if step <= 3000:
		action = randrange(opt.act_num)
	elif np.random.random() <= EPSILON:
		action = randrange(opt.act_num)
	else:
		action = action
		
	action_onehot = trans.one_hot_action(action)
	next_state = sim.step(action)
	
	epi_step += 1
    
	# append to history
	append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
	# add to the transition table
	trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), 
				next_state.reward, next_state.terminal)
	# mark next state as current state
	state_with_history = np.copy(next_state_with_history)
	state = next_state
	
	if step <= 3000:
		print('Total steps: {}, epi_step: {}'.format(step, epi_step))

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# TODO: here you would train your agent
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if step > 3000:
		state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

		# TODO train me here
		# this should proceed as follows:
		# 1) pre-define variables and networks as outlined above
		
		# 2) here: calculate best action for next_state_batch
		# TODO:
		# action_batch_next = CALCULATE_ME
		
		#action_batch_next = sess.run(Qn, feed_dict = {xn: state_batch})
		action_batch_next = sess.run(Q_action, feed_dict = {x: state_batch}).reshape((-1,1))
		action_batch_next = trans.one_hot_action(action_batch_next)
		_, Q_loss = sess.run([train_op, loss], feed_dict = {x: state_batch, u: action_batch, ustar: action_batch_next, 
										xn: next_state_batch, r: reward_batch, term : terminal_batch})

		# 3) with that action make an update to the q values
		#    as an example this is how you could print the loss 
		if state.terminal or epi_step >= opt.early_stop:
			# print('Q - Qn: ', sess.run(Q-Qn, feed_dict = {x: state_batch, x: state_batch}))
			print('Episodes: {} with steps: {}, Q_loss: {}, total steps: {}'.format(nepisodes, epi_step, Q_loss, step))
		
		# TODO every once in a while you should test your agent here so that you can track its performance

		if opt.disp_on:
			if win_all is None:
				plt.subplot(121)
				win_all = plt.imshow(state.screen)
				plt.subplot(122)
				win_pob = plt.imshow(state.pob)
			else:
				win_all.set_data(state.screen)
				win_pob.set_data(state.pob)
			plt.pause(opt.disp_interval)
			plt.draw()

# save the trained model			
saver.save(sess, "./model/trained_model")
print('Training session completed!')



#######################################
# 1. perform a final test of your model
#######################################
# TODO

# setup some parameters for testing phase
epi_step = 0
nepisodes = 0
nepisodes_solved = 0       # calculate the trained agent's accuracy
test_steps = 30000         # number of steps for testing

# Restore variables from disk
new_saver = tf.train.import_meta_graph('./model/trained_model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))

# Print all the trainable variables
all_vars = tf.get_collection('params')

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

for step in range(test_steps):

	# check if episode ended
	if state.terminal or epi_step >= opt.early_stop:
            
                print('Episodes: {}, test steps: {}'.format(nepisodes, step))
		epi_step = 0
		nepisodes += 1
		if state.terminal:
			nepisodes_solved += 1
		# start a new game
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        state_shaped = rgb2gray(state.pob).reshape(opt.state_siz)
        state_with_history = np.tile(state_shaped, [1, opt.hist_len])
        test_state = np.tile(state_with_history.reshape(1, -1), [opt.minibatch_size, 1])
        """
        
        test_state = np.tile(state_with_history.reshape(1, -1), [opt.minibatch_size, 1])
        
        action = sess.run(Q_action, feed_dict = {x: test_state})[0]
        
        # epsilon greedy exploration
        if np.random.random() <= EPSILON:
		action = randrange(opt.act_num)
	else:
		action = action
        
        action_onehot = trans.one_hot_action(action)
	next_state = sim.step(action)
	
	epi_step += 1
    
	# append to history
	append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
	# add to the transition table
	trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), 
				next_state.reward, next_state.terminal)
	# mark next state as current state
	state_with_history = np.copy(next_state_with_history)
	state = next_state
        
        #print(test_state.shape, state_with_history.shape)
        #exit()

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

# 2. calculate statistics
print('nepisodes_solved: {}, nepisodes: {}'.format(nepisodes_solved, nepisodes))
print('Trained agent accuracy: ', float(nepisodes_solved) / float(nepisodes))
		
sess.close()
