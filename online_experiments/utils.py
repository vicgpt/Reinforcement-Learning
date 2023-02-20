# pylint: disable=import-error, invalid-name
'''
support functions for online_experiments notebook -
* plotting the actual rewards vs distribution obtained using MAB
* regret plots
'''

from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multi_arm_bandits import ArmsList, MAB

def plot_rewards(
    arms_list: ArmsList,
    rewards_list: List
    ) -> None:
    'plotting empirical distributions of mean rewards for each arm with actual'
    n_arms = arms_list.n_arms
    arm_label = ['Arm' + str(i+1) for i in np.arange(n_arms)]

    actual_mean_rewards = [arm.mu for arm in arms_list.arms]
    mean_rewards = [np.mean(rewards_list[i]) for i in np.arange(n_arms)]
    # assuming each trial to be a binomial distributions
    # hence the empirical distribution will be normal by clt ~ N(p, p*(1-p)/n)
    ci_rewards = [2*np.sqrt(mean_rewards[i]*(1 - mean_rewards[i])/len(rewards_list[i]))
                     for i in np.arange(n_arms)]

    plt.errorbar(arm_label, mean_rewards, ci_rewards,
                linestyle='None', label='estimated', marker = 'o')
    plt.errorbar(arm_label, actual_mean_rewards,
                linestyle='None', label='actual', marker='^')
    plt.xlabel('Arms')
    plt.ylabel('CTR')
    plt.legend()
    plt.show()


def get_regret(
    arms_list: ArmsList,
    observed_rewards: List,
    n_trials: int
):
    'calculating regret up to t'
    expected_reward = [arms_list.get_best_arm().mu]*n_trials
    regret = [(
        expected_reward[i] - np.sum(observed_rewards[:i+1])/(i+1)) for i in range(n_trials)]

    return regret


def plot_regret(
    arms_list: ArmsList,
    mab_list: Dict[str, MAB]
    ) -> None:
    'plot regret(t): |actual_reward(t) - observed_reward(t)|'
    for mab_name, mab in mab_list.items():
        n_trials = mab.n_trials
        observed_rewards = mab.observed_rewards
        regret = get_regret(arms_list, observed_rewards, n_trials)

        plt.plot(np.arange(n_trials), regret, label=mab_name)
    plt.title('Regret for different MAB approaches')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('regret')
    plt.show()


def plot_clicks_received(
    n_trials: int,
    mab_list: Dict[str, MAB],
    arms_list: ArmsList
    ) -> None:
    'plotting clicks from selecting the best arm vs different mab approaches'

    for mab_name, mab in mab_list.items():
        reward = pd.DataFrame(mab.observed_rewards)
        plt.plot(reward.cumsum(), label=mab_name)

    mean_ctr = arms_list.get_best_arm().mu
    plt.plot(mean_ctr*np.arange(n_trials), label='best arm')
    plt.xlabel('Number of trials')
    plt.ylabel('Number of clicks')
    plt.legend()
    plt.show()


def plot_optimal_arm_selection(
    n_trials: int,
    mab_list: Dict[str, MAB],
    arms_list: ArmsList
    ) -> None:
    'percentage of time best arm is selected by the algo'

    optimal_arm_selection_df = pd.DataFrame(columns=['best_arm'])
    optimal_arm_selection_df['best_arm'] = [arms_list.get_best_arm_idx()]*n_trials

    for mab_name, mab in mab_list.items():
        optimal_arm_selection_df[f'selected_arms_{mab_name}'] = mab.selected_arms
        optimal_arm_selection_df[f'optimal_selection_{mab_name}'] = (
            optimal_arm_selection_df['best_arm'] ==
            optimal_arm_selection_df[f'selected_arms_{mab_name}']
        ).cumsum()/np.arange(1, n_trials+1)

        plt.plot(optimal_arm_selection_df[f'optimal_selection_{mab_name}'], label=mab_name)

    plt.xlabel('Number of trials')
    plt.ylabel('Optimal selection %')
    plt.legend()
    plt.show()
