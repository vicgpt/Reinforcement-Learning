import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd


class Arm:
    '''Arm object with defined click through probability'''
    def __init__(self, p: float) -> None:
        # bernoulli distribution with mean, and variance given the click through rate
        self.mu = p
        self.var = p * (1 - p)
        
    def reward(self, pulls: int=1) -> float:
        return list(st.bernoulli.rvs(self.mu, size=pulls))


class ArmsList(List[Arm]):
    
    def __init__(self, n_arms: int, p_arms: List[float]) -> None:
        self.n_arms = n_arms
        self.arms = []
        for i in range(n_arms):
            self.arms.append(Arm(p_arms[i]))
            
    def arms_reward(self, selected_arm: int, pulls: int=1) -> float:
        return self.arms[selected_arm].reward(pulls)
    
    def get_best_arm(self) -> Arm:
        best_arm = self.arms[0]
        for arm in self.arms:
            if arm.mu > best_arm.mu:
                best_arm = arm
        return best_arm
    
    # TODO: add a function to track the selection of different arms over time


class ArmEmpiricalDistribution:
    from scipy.stats import beta
    def __init__(self, a=1, b=1) -> None:
        self.a = a
        self.b = b
        self.distribution = st.beta(a, b)
        
    def sample_reward(self, n_trial: int=1) -> float:
        return self.distribution.rvs(size=n_trial)
    
    def update_distribution(self, reward: List):
        for r in reward:
            if r == 1:
                self.a += 1
            else:
                self.b += 1
        self.distribution = st.beta(self.a, self.b)
        
    def get_mean(self) -> float:
        return self.a/(self.a + self.b)


class MAB(ABC):
    def __init__(self, n_arms: int, n_trials: int) -> None:
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.selected_arms = np.ones(n_trials)*-1
        self.rewards_list = {}
        
        for j in range(n_arms):
            self.rewards_list[j] = [0]
            
    def get_best_arm(self) -> int:
        best_arm = np.argmax([sum(self.rewards_list[i])/len(self.rewards_list[i]) for i in range(self.n_arms)])
        return best_arm
        
    def get_selected_arms(self) -> np.array:
        return self.selected_arms
            
    @abstractmethod
    def run_test(self)  -> Dict:
        raise NotImplemented
    
    def plot_arms_distribution(self) -> None:
        pd.Series(self.selected_arms).hist()


class EpsGreedy(MAB):
    def __init__(self, n_arms: int, n_trials: int=10000, eps_init: float=0.05, buffer: int=100) -> None:
        super().__init__(n_arms, n_trials)
        self.eps_list = np.concatenate((
            np.array([0]*buffer),
            np.linspace(start=eps_init, stop=1, num=n_trials-buffer))) 
            
    def run_test(self, arms_list: ArmsList) -> Dict[int, List]:
        np.random.seed(100)
        for i in range(self.n_trials):
            eps = self.eps_list[i]
            if np.random.choice([1, 0], p=[eps, 1-eps]):
                best_arm = self.get_best_arm()
            else:
                best_arm = np.random.choice(self.n_arms)

            self.selected_arms[i] = best_arm
            self.rewards_list[best_arm] += arms_list.arms_reward(selected_arm=best_arm)

        return self.rewards_list


class UCB(MAB):
    def __init__(self, n_arms: int, n_trials: int=10000, ci: float=0.95, buffer: int=100) -> None:
        super().__init__(n_arms, n_trials)
        self.buffer = buffer
        self.c = st.norm.interval(alpha=ci)[1]
        self.upper_bound_list = []
        
    def get_upper_bound(self, selected_arm, n) -> float:
        n_a = sum([1 for arm in self.selected_arms if arm==selected_arm])
        return np.sum(self.rewards_list[selected_arm])/n_a + self.c * np.sqrt(np.log(n)/n_a)

    def get_best_arm(self, n) -> int:
        return np.argmax([self.get_upper_bound(arm, n) for arm in range(self.n_arms)])
    
    def run_test(self, arms_list) -> Dict[int, List]:
        np.random.seed(100)
        for i in range(self.n_trials):
            if i < self.buffer:
                best_arm = np.random.choice(self.n_arms)
            else:
                best_arm = self.get_best_arm(i)
            
            self.selected_arms[i] = best_arm
            self.rewards_list[best_arm] += arms_list.arms_reward(selected_arm=best_arm)

        return self.rewards_list 
    # TODO: check if the option with ucb[a_i] < ucb[a_j] for every j != i, needs to be removed


class ThompsonSampling(MAB):
    def __init__(self, n_arms: int, n_trials: int=10000) -> None:
        super().__init__(n_arms, n_trials)
        self.arms_distribution = {}
        for i in range(self.n_arms):
            self.arms_distribution[i] = ArmEmpiricalDistribution()

    def get_best_arm(self) -> int:
        sample_reward_list = [arm_distr.sample_reward() for arm_distr in self.arms_distribution.values()]
        return np.argmax(sample_reward_list)
    
    def run_test(self, arms_list: List) -> Dict[int, List]:
        np.random.seed(100)
        for i in range(self.n_trials):
            best_arm = self.get_best_arm()
            self.selected_arms[i] = best_arm
            
            reward = arms_list.arms_reward(selected_arm=best_arm)
            self.rewards_list[best_arm] += reward
            self.arms_distribution[best_arm].update_distribution(reward)

        return self.rewards_list 