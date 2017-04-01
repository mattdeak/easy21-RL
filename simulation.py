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
    VALID_AGENTS = ['Basic','Naive']
    if agent == 'Basic':
        print("Selected basic player")
        p = player.Basic_Player(debug=debug)
    elif agent == 'Naive':
        print("Selected naive player")
        p = player.Naive_Player(debug=debug)
    else:
        raise ValueError("Invalid Agent Type. Select from: {}".format(VALID_AGENTS))
    
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
    
    
def Q_Simulate(n_trials=1000):
    p = player.QLearner()
    env = Environment()
    env.add_primary_agent(p)
    wins = []
    
    print("Simulating...")
    for i in range(n_trials):
        _,_,result = env.play_game()
        wins.append(result)
        
    win_cumulative_sum = np.cumsum(wins)
        
    
        
    table = p.Q_Table
    
    p_sums = []
    d_starts = []
    q_vals = []
    
    print("Starting...")    
    
    for key,action_dict in table.items():
        p_sum,d_start = key
        for actions,values in action_dict.items():
            p_sums.append(p_sum)
            d_starts.append(d_starts)
            q_vals.append(q_vals)
            
    print("Done")
            
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(range(1,n_trials+1),win_cumulative_sum)
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot_trisurf(p_sums,d_starts,q_vals)
    fig.show()
    
    
def sim_1():
    sp,sd,r = simulate(n_trials=2000)
    n_p,nd,r2 = simulate(n_trials=2000,agent='Naive')
    fig = plt.figure()
    ax = fig.add_subplot(121,projection='3d')
    ax.plot_trisurf(sp,sd,r)
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot_trisurf(n_p,nd,r2)
    fig.show()
    
if __name__ == "__main__":
    Q_Simulate()
    

        
        
