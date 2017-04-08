
"""
Module containing useful functions for simulating and processing games.
"""
__author__ = "Matthew Deakos"

import traceback as tb
import numpy as np
import sys
    
    
def simulate(player,environment,n_trials=1000,verbose=False):
    """Simulates games in an environment.
    
    Arguments:
        player: a player capable of interacting with an environment via an act() method
        environment: an environment capable of being acted upon
        n_trials: the number of games to simulate
        verbose: if true, displays progress of simulation
        
    Returns:
        rewards: a list of terminal rewards given by the games
    """
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
    """Processes simulation output into a metrics dictionary
    
    Arguments:
        rewards_list: A list of rewards as given by the simulate() function
    
    Returns:
        metrics: A dictionary of useful metrics with respect to the simulation
    """
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


    

        
        
