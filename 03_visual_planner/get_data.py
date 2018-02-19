import numpy as np; np.random.seed(0)
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
import matplotlib.pyplot as plt

# 0. initialization
opt    = Options()
sim    = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
states = np.zeros([opt.data_steps, opt.state_siz], float)
labels = np.zeros([opt.data_steps], int)

# Note I am forcing the display to be off here to make data collection fast
# you can turn it on again for debugging purposes

#opt.disp_on = False
opt.disp_on = True

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 1   # total #episodes executed

state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.data_steps):
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        state = sim.step() 	# will perform A* actions

    print('shape:{}'.format(state.pob.shape))			    # (25, 25, 3)
    print('shape:{}'.format(rgb2gray(state.pob).shape))		# (25, 25)
    print('shape:{}'.format(opt.state_siz))			        # (625)

    # save data & label
    states[step, :] = rgb2gray(state.pob).reshape(opt.state_siz)
    labels[step]    = state.action

    epi_step += 1

    if step % opt.prog_freq == 0:
        print(step)

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

# 2. save to disk
print('saving data ...')
np.savetxt(opt.states_fil, states, delimiter=',')
np.savetxt(opt.labels_fil, labels, delimiter=',')
print("states saved to " + opt.states_fil)
print("labels saved to " + opt.labels_fil)
