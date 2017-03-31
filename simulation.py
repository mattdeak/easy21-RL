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

def simulate(n_trials=10000,agent='Basic',debug=False):
    #Initialize player and environment

    if agent == 'Basic':
        print("Selected basic player")
        p = player.Basic_Player(debug=debug)
    elif agent == 'Naive':
        print("Selected naive player")
        p = player.Naive_Player(debug=debug)
    
    env = Environment()
    env.add_primary_agent(p)
    
    #Initialize state and reward list
    states_p = np.zeros([n_trials])
    states_d = np.zeros([n_trials])
    rewards = np.zeros([n_trials])
    
    #Beging trials
    print("Beginning Trials")
    for i in range(n_trials):
        state,reward = env.play_game()
        states_p[i] = state['p_sum']
        states_d[i] = state['d_start']
        rewards[i] = reward
        
    return states_p,states_d,rewards
    
    
if __name__ == "__main__":
    sp,sd,r = simulate(n_trials=2000)
    n_p,nd,r2 = simulate(n_trials=2000,agent='Naive')
    fig = plt.figure()
    ax = fig.add_subplot(121,projection='3d')
    ax.plot_trisurf(sp,sd,r)
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot_trisurf(n_p,nd,r2)
    fig.show()
        
        
