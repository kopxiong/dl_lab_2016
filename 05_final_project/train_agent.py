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
    #       use it as the target for Q_s (fixed target Q-Network)
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
steps = 100000
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


# 1. Initialize the Q-network parameters
W_conv1 = tf.get_variable("W_conv1", [8, 8, opt.hist_len, 32], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_conv1 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='b_conv1')

W_conv2 = tf.get_variable("W_conv2", [5, 5, 32, 64], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_conv2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_conv2')

W_conv3 = tf.get_variable("W_conv3", [5, 5, 64, 64], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_conv3 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='b_conv3')


W_fc1 = tf.get_variable("W_fc1", [8*8*64, 128], 
						  initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
																   mode='FAN_AVG', uniform=True, seed=SEED, 
																   dtype=tf.float32))
b_fc1 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32), name='b_fc1')

W_fc2 = tf.Variable(tf.truncated_normal(shape=[128, opt.act_num], 
										stddev=1e-3, dtype=tf.float32, seed=SEED), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros(shape=[opt.act_num], dtype=tf.float32), name='b_fc2')

# define Q-network model here
def Q_model_forward(stateInput):

	stateInput = tf.reshape(stateInput, [opt.minibatch_size, opt.hist_len, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz])
	
	# change order of channels and image width/height and normalize
	stateInput = tf.transpose(stateInput, perm=[0, 2, 3, 1]) / 255.
	
	h_conv1 = tf.nn.relu(conv2d(stateInput, W_conv1, strides=[1, 2, 2, 1]) + b_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1]) + b_conv2)
	
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)	

	h_conv3_flat = tf.reshape(h_conv3, [-1, 8*8*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
	
	# Vanilla DQN 
	Q_out = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	"""
	# Dueling DQN
	# Take the output from the final fully connected layer and split it into separate advantage and value streams.
	stream_adv, stream_value = tf.split(1, 2, h_fc1)
	flatten_adv = tf.contrib.layers.flatten(stream_adv)
	flatten_value = tf.contrib.layers.flatten(stream_value)
	W_adv = tf.Variable(tf.random_normal([128/2, opt.act_num]))
	W_value = tf.Variable(tf.random_normal([128/2, 1]))
	Advantage = tf.matmul(flatten_adv, W_adv)
	Value = tf.matmul(flatten_value, W_value)
	
	# Then combine them together to get our final Q-values.
	# tf.sub(x, y): returns x-y element-wise, supports broadcasting. 
	Q_out = Value + tf.sub(Advantage, tf.reduce_mean(Advantage, reduction_indices=1, keep_dims=True))
	"""
	
	"""
	predict = tf.argmax(Q_out, 1)
	
	# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
	targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
	actions = tf.placeholder(shape=[None], dtype=tf.int32)
	actions_onehot = tf.one_hot(actions, opt.act_num, dtype=tf.float32)
	
	Q = tf.reduce_sum(tf.mul(Q_out, actions_onehot), reduction_indices=1)
	
	td_error = tf.square(targetQ - self.Q)
	loss = tf.reduce_mean(td_error)
	trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
	updateModel = self.trainer.minimize(loss)
	"""
	
	return Q_out


# 2. Initialize the target Q-network parameters as the same with Q-network parameters
target_W_conv1 = W_conv1
target_b_conv1 = b_conv1

target_W_conv2 = W_conv2
target_b_conv2 = b_conv2

target_W_conv3 = W_conv3
target_b_conv3 = b_conv3

target_W_fc1 = W_fc1
target_b_fc1 = b_fc1

target_W_fc2 = W_fc2
target_b_fc2 = b_fc2

# define target Q-network model here
def target_model_forward(stateInput):

	stateInput = tf.reshape(stateInput, [opt.minibatch_size, opt.hist_len, opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz])
	
	# change order of channels and image width/height and normalize
	stateInput = tf.transpose(stateInput, perm=[0, 2, 3, 1]) / 255.
	
	target_h_conv1 = tf.nn.relu(conv2d(stateInput, target_W_conv1, strides=[1, 2, 2, 1]) + target_b_conv1)

	target_h_conv2 = tf.nn.relu(conv2d(target_h_conv1, target_W_conv2, strides=[1, 2, 2, 1]) + target_b_conv2)
	
	target_h_conv3 = tf.nn.relu(conv2d(target_h_conv2, target_W_conv3) + target_b_conv3)	

	target_h_conv3_flat = tf.reshape(target_h_conv3, [-1, 8*8*64])
	target_h_fc1 = tf.nn.relu(tf.matmul(target_h_conv3_flat, target_W_fc1) + target_b_fc1)
	 
	target_Q_out = tf.matmul(target_h_fc1, target_W_fc2) + target_b_fc2
	
	return target_Q_out


# define convolutional function (2D convolution with 'SAME' padding)
def conv2d(x, W, strides=[1, 1, 1, 1]):
	return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

"""
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
"""

# get the output from your network
Q = Q_model_forward(x)
Qn = Q_model_forward(xn)

# compute the loss of the model here
loss = Q_loss(Q, u, Qn, ustar, r, term)

# use AdamOptimizer for the optimization
train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

# take an action according to the maximal Q value of current state
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
tf.add_to_collection('params', W_conv3)
tf.add_to_collection('params', b_conv3)
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

	Input_state = np.tile(state_with_history.reshape(1, -1), [opt.minibatch_size, 1])
	
	action = sess.run(Q_action, feed_dict = {x: Input_state})[0]
	
	if step <= 5000 or np.random.random() <= EPSILON:
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
	
	if step <= 5000:
		print('Total steps: {}, epi_step: {}'.format(step, epi_step))

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# TODO: here you would train your agent
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if step > 5000:
                
                # every 1000 steps updates the target Q-network
                if (step % 1000) == 0:
                        target_W_conv1 = W_conv1
                        target_b_conv1 = b_conv1
                        target_W_conv2 = W_conv2
                        target_b_conv2 = b_conv2
                        target_W_conv3 = W_conv3
                        target_b_conv3 = b_conv3
                        target_W_fc1 = W_fc1
                        target_b_fc1 = b_fc1
                        target_W_fc2 = W_fc2
                        target_b_fc2 = b_fc2
                
                
		state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

		# TODO train me here
		# this should proceed as follows:
		# 1) pre-define variables and networks as outlined above
		
		# 2) here: calculate best action for next_state_batch
		# TODO:
		# action_batch_next = CALCULATE_ME
		
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

# Restore variables from disk (In test phase, the model folder should contain the corresponding trained models)
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
