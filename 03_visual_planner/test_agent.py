import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import randrange

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
#from train_agent import model, eval_in_batches

# set some constants or parameters for the neural network
NUM_LABELS = 5
SEED       = 66478    # Set to None for random seed.

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# TODO: load your agent
# Add ops to save and restore all the variables.
g = tf.reset_default_graph()

test_data = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.cub_siz*opt.pob_siz,
													 opt.cub_siz*opt.pob_siz, opt.hist_len))

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

save_file = "./model/trained_model.ckpt"

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step         = 0    # #steps in current episode
nepisodes        = 0    # total #episodes executed
nepisodes_solved = 0
action           = 0    # action to take given by the network

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

        with tf.Session(graph = g) as sess:
            sess.run(tf.global_variables_initializer())
            # Restore graph
            #saver = tf.train.import_meta_graph("./model/trained_model.ckpt.meta")
            saver = tf.train.Saver()
            saver.restore(sess, save_file)
            #print("Load trained model successfully!")
            logits = model(test_data)

            if epi_step == 0:
                first_state = rgb2gray(state.pob).reshape((opt.cub_siz*opt.pob_siz, \
                    opt.cub_siz*opt.pob_siz)).astype(np.float32)
                hist_states = np.dstack([first_state] * opt.hist_len)

    			# Copy the hist_states for 32 times to fit into the placeholder
                feed_data = np.repeat(hist_states[np.newaxis, ...], 32, axis=0)
                logits_0 = sess.run(logits, feed_dict={test_data: feed_data})

                # random action
                action = randrange(opt.act_num)
                state = sim.step(action)
                print('step {} action {}'.format(epi_step, action))
                epi_step += 1
            else:
                next_state = rgb2gray(state.pob).reshape((opt.cub_siz*opt.pob_siz, \
                    opt.cub_siz*opt.pob_siz)).astype(np.float32)

                # Replace the earliest state with the latest state
                hist_states = np.delete(hist_states, 0, 2)
                hist_states = np.insert(hist_states, -1, next_state, axis=2)

                feed_data = np.repeat(hist_states[np.newaxis, ...], 32, axis=0)
                logits_next = sess.run(logits, feed_dict={test_data: feed_data})

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
