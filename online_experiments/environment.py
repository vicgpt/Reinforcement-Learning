# pylint: disable=import-error, invalid-name, fixme
'''
setting the environment based on number of arms
for each trial, selected arm will provide the reward for learning
'''

from typing import List
from scipy.stats import bernoulli

class Arm:
    '''Arm object with defined click through probability'''
    def __init__(self, p: float) -> None:
        'Bernoulli distribution with mean, and variance given the click through rate'
        self.mu = p
        self.var = p * (1 - p)

    def reward(self, pulls: int=1) -> List:
        'sample reward from an arm`s distribution'
        return list(bernoulli.rvs(self.mu, size=pulls))


class ArmsList(List[Arm]):
    '''List of Arm object for simulated environment'''
    def __init__(self, n_arms: int, p_arms: List[float]) -> None:
        self.n_arms = n_arms
        self.arms = []
        for t in range(n_arms):
            self.arms.append(Arm(p_arms[t]))

    def arms_reward(self, selected_arm: int, pulls: int=1) -> float:
        'sample reward for the selected arm'
        return self.arms[selected_arm].reward(pulls)

    def get_best_arm(self) -> Arm:
        'return the best arm in the environment'
        best_arm = self.arms[0]
        for arm in self.arms:
            if arm.mu > best_arm.mu:
                best_arm = arm
        return best_arm

    def get_best_arm_idx(self) -> int:
        'return the index of best arm in the environment'
        best_arm_idx = 0
        for idx in range(self.n_arms):
            if self.arms[idx].mu > self.arms[best_arm_idx].mu:
                best_arm_idx = idx
        return best_arm_idx
