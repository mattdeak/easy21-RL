# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:40:16 2017

@author: matthew
"""
from environment import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate(n_trials=1000):
    env = Environment()
    states_p = []
    states_d = []
    rewards = []
    for i in range(n_trials):
        print("Playing game {}".format(i+1))
        state,reward = env.play_game()
        states_p.append(state['p_sum'])
        states_d.append(state['d_start'])
        rewards.append(reward)
        
    return states_p,states_d,rewards
    
    
if __name__ == "__main__":
    sp,sd,r = simulate()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_wireframe(sp,sd,r)
        
        
