# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:40:16 2017

@author: matthew
"""
__author__ = "Matthew Deakos"

from environment import Easy21_Environment
from q_players import QLearner
import traceback as tb
import sys
import matplotlib.pyplot as plt
import numpy as np
    
def simulate(player,environment,n_trials=1000,verbose=False):
    
    environment.player = player
    rewards = []
    
    for i in range(1,n_trials+1):
        if i % (n_trials/5) == 0:
            if verbose:
                print ("Loading game {}".format(i))
        try:
            result = environment.play_game()
            rewards.append(result)
        except Exception:
            tb.print_exc(file=sys.stdout)
                 
    return rewards
    
    
def process_simulation_metrics(rewards_list):
    rewards = np.array(rewards_list)
    wins = np.where(rewards == 1)[0].size
    draws = np.where(rewards == 0)[0].size
    losses = np.where(rewards == -1)[0].size
    
    win_rate = float(wins)/len(rewards)
    draw_rate = float(draws)/len(rewards)
    loss_rate = float(losses)/len(rewards)
    
    metrics = {'wins':wins,'draws':draws,'losses':losses,
               'win_rate':win_rate,'draw_rate':draw_rate,'loss_rate':loss_rate,
               'rewards':rewards}
    
    return metrics
    
if __name__ == "__main__":
    env = Easy21_Environment()
    p = QLearner()
    results = simulate(p,env,n_trials=1000)
    metrics = process_simulation_metrics(results)
    print(metrics['wins'])

    

        
        
