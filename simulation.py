# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:40:16 2017

@author: matthew
"""
from environment import Environment
import player
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def simulate(n_trials=10000):
    #Initialize player and environment
    p = player.Basic_Player()
    env = Environment()
    env.add_primary_agent(p)
    
    #Initialize state and reward list
    states_p = np.zeros([n_trials])
    states_d = np.zeros([n_trials])
    rewards = np.zeros([n_trials])
    
    #Beging trials
    for i in range(n_trials):
        print("Playing game {}".format(i+1))
        state,reward = env.play_game()
        states_p[i] = state['p_sum']
        states_d[i] = state['d_start']
        rewards[i] = reward
        
    return states_p,states_d,rewards
    
    
if __name__ == "__main__":
    sp,sd,r = simulate()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_trisurf(sp,sd,r)
        
        
