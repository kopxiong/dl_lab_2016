import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import randrange

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from train_agent import train_data_node

# set some constants or parameters for the neural network
NUM_LABELS   = 5
SEED = 66478    # Set to None for random seed.

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# TODO: load your agent
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

test_data_node = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.cub_siz*opt.pob_siz, 
													 opt.cub_siz*opt.pob_siz, opt.hist_len))

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.eval_steps):

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # this just gets a random action
        
		sess = tf.Session()
		
		# Restore variables from disk
		new_saver = tf.train.import_meta_graph('./model/trained_model.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
		
		logits = tf.get_collection('logits')
		
		# Print all the trainable variables
		#all_vars = tf.trainable_variables()
		#for v in all_vars:
		#	print('v_name: ', v.name)
		
		if epi_step == 0:
			first_state = rgb2gray(state.pob).reshape((opt.cub_siz*opt.pob_siz, 
												opt.cub_siz*opt.pob_siz)).astype(np.float32)
			hist_states = np.dstack([first_state] * opt.hist_len)
			
			# Copy the hist_states for 32 times to fit into the placeholder
			feed_data = np.repeat(hist_states[np.newaxis, ...], 32, axis=0)
			
			logits_0 = sess.run(logits, feed_dict={test_data_node: feed_data})
			
			# random action
			#action = randrange(opt.act_num)
			action = np.argmax(logits_0[0])
			state = sim.step(action)
			
			print('step {} action {}'.format(epi_step, action))
		
			epi_step += 1
			
		else:
			next_state = rgb2gray(state.pob).reshape((opt.cub_siz*opt.pob_siz, 
												opt.cub_siz*opt.pob_siz)).astype(np.float32)
			
			# Replace the earliest state with the latest state
			hist_states = np.delete(hist_states, 0, 2)
			hist_states = np.insert(hist_states, -1, next_state, axis=2)
			
			feed_data = np.repeat(hist_states[np.newaxis, ...], 32, axis=0)
			
			logits_next = sess.run(logits, feed_dict={test_data_node: feed_data})

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

# 2. calculate statistics
print('Accuracy: ', float(nepisodes_solved) / float(nepisodes))

# 3. TODO perhaps do some additional analysis
