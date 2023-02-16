'''
Implementation of multi arm bandits approaches for online testing
'''
# pylint: disable=import-error, invalid-name
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from scipy.stats import beta, norm
import pandas as pd
from environment import ArmsList

class MAB(ABC):
    '''
    Base class for multi arm bandits algo used for online experiments
    '''
    def __init__(
        self,
        n_arms: int,
        n_trials: int
    ) -> None:

        self.n_arms = n_arms
        self.n_trials = n_trials
        self.selected_arms = np.ones(n_trials)*-1
        self.rewards_list = {}

        for j in range(n_arms):
            self.rewards_list[j] = [0]

    def get_best_arm(self) -> int:
        'to get the best arm for MAB algos'
        best_arm = np.argmax([
            sum(self.rewards_list[t])/len(self.rewards_list[t])
            for t in range(self.n_arms)])
        return best_arm

    def get_selected_arms(self) -> np.array:
        'to get the list of selected arm after each trails'
        return self.selected_arms

    @abstractmethod
    def run_test(
        self,
        arms_list: ArmsList
    ) -> Dict[int, List]:
        'base function to run the trials'
        raise NotImplementedError()

    def plot_arms_distribution(self) -> None:
        'plot number of trials an arm is selected for'
        pd.Series(self.selected_arms).hist()


class ArmEmpiricalDistribution:
    '''Empirical distribution for Thompson sampling'''
    def __init__(self, a=1, b=1) -> None:
        self.a = a
        self.b = b
        self.distribution = beta(a, b)

    def sample_reward(self, n_trial: int=1) -> float:
        'sample reward based on distribution'
        return self.distribution.rvs(size=n_trial)

    def update_distribution(self, reward: List):
        'update distribution after each trial'
        for r in reward:
            if r == 1:
                self.a += 1
            else:
                self.b += 1
        self.distribution = beta(self.a, self.b)

    def get_mean(self) -> float:
        'get mean of the empirical distribution'
        return self.a/(self.a + self.b)


class EpsGreedy(MAB):
    '''
    Implementation of epsilon greedy algo for arm selection
    '''
    def __init__(
        self,
        n_arms: int,
        n_trials: int=10000,
        eps_init: float=0.05,
        buffer: int=100
    ) -> None:

        super().__init__(n_arms, n_trials)
        self.eps_list = np.concatenate((
            np.array([0]*buffer),
            np.linspace(start=eps_init, stop=1, num=n_trials-buffer)))

    def run_test(
        self,
        arms_list: ArmsList
    ) -> Dict[int, List]:
        np.random.seed(100)
        for t in range(self.n_trials):
            eps = self.eps_list[t]
            if np.random.choice([1, 0], p=[eps, 1-eps]):
                best_arm = self.get_best_arm()
            else:
                best_arm = np.random.choice(self.n_arms)

            self.selected_arms[t] = best_arm
            self.rewards_list[best_arm] += arms_list.arms_reward(selected_arm=best_arm)

        return self.rewards_list


class UCB(MAB):
    '''
    Implementation of upper confidence bound algo for arm selection
    '''
    def __init__(
        self,
        n_arms: int,
        n_trials: int=10000,
        ci: float=0.95,
        buffer: int=100
    ) -> None:
        super().__init__(n_arms, n_trials)
        self.buffer = buffer
        self.c = norm.interval(alpha=ci)[1]
        self.upper_bound_list = []
        self.t = 0

    def get_upper_bound(
        self,
        selected_arm
    ) -> float:
        'to get upper uncertainty bound based on observed rewards for an arm'
        n = self.t
        n_a = sum([1 for arm in self.selected_arms if arm==selected_arm])
        return np.sum(self.rewards_list[selected_arm])/n_a + self.c * np.sqrt(np.log(n)/n_a)

    def get_best_arm(
        self
    ) -> int:
        return np.argmax([self.get_upper_bound(arm) for arm in range(self.n_arms)])

    def run_test(
        self,
        arms_list: ArmsList
    ) -> Dict[int, List]:
        np.random.seed(100)
        for t in range(self.n_trials):
            if t < self.buffer:
                best_arm = np.random.choice(self.n_arms)
            else:
                best_arm = self.get_best_arm()
            self.selected_arms[t] = best_arm
            self.rewards_list[best_arm] += arms_list.arms_reward(selected_arm=best_arm)

            self.t += 1

        return self.rewards_list


class ThompsonSampling(MAB):
    '''
    Implementation of upper confidence bound algo for arm selection
    '''
    def __init__(self, n_arms: int, n_trials: int=10000) -> None:
        super().__init__(n_arms, n_trials)
        self.arms_distribution = {}
        for t in range(self.n_arms):
            self.arms_distribution[t] = ArmEmpiricalDistribution()

    def get_best_arm(self) -> int:
        sample_reward_list = [
            arm_distr.sample_reward()
            for arm_distr in self.arms_distribution.values()]
        return np.argmax(sample_reward_list)

    def run_test(self, arms_list: ArmsList) -> Dict[int, List]:
        np.random.seed(100)
        for t in range(self.n_trials):
            best_arm = self.get_best_arm()
            self.selected_arms[t] = best_arm

            reward = arms_list.arms_reward(selected_arm=best_arm)
            self.rewards_list[best_arm] += reward
            self.arms_distribution[best_arm].update_distribution(reward)

        return self.rewards_list
