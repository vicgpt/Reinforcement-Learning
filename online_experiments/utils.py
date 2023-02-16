# pylint: disable=import-error, invalid-name
'''
support functions for online_experiments notebook -
* plotting the actual rewards vs distribution obtained using MAB
* regret plots
'''

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from multi_arm_bandits import ArmsList

def plot_rewards(
    arms_list: ArmsList,
    rewards_list: List
    ) -> None:
    'plotting empirical distributions of mean rewards for each arm with actual'
    n_arms = arms_list.n_arms
    arm_label = ['Arm' + str(i+1) for i in np.arange(n_arms)]

    actual_mean_rewards = [arm.mu for arm in arms_list.arms]
    mean_rewards = [np.mean(rewards_list[i]) for i in np.arange(n_arms)]
    stddev_rewards = [np.std(rewards_list[i])/np.sqrt(len(rewards_list[i]))
                     for i in np.arange(n_arms)]

    plt.errorbar(arm_label, mean_rewards, stddev_rewards,
                linestyle='None', label='estimated', marker = 'o')
    plt.errorbar(arm_label, actual_mean_rewards,
                linestyle='None', label='actual', marker='^')
    plt.legend()
    plt.show()
