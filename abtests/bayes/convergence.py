from typing import Dict, Text, Any, List
import matplotlib.pyplot as plt
import numpy as np
from bayesian_bandits import Bandit

n_trials = 2000
bandit_probs = [0.2, 0.5, 0.75]


class BanditsExperiment2:
    def __init__(self, bandit_probs: List, n_trials: int, **kwargs):

        self.n_trials = n_trials
        self.bandit_probs = bandit_probs
        self.bandits = [Bandit(p) for p in bandit_probs]
        self.sample_points = kwargs.get("sample_points", np.linspace(5, n_trials, 5))
        self.rewards = np.zeros(n_trials)

    def run(self):
        for i in range(self.n_trials):
            chosen_k = np.argmax([b.sample() for b in self.bandits])
            reward = self.bandits[chosen_k].pull()
            self.bandits[chosen_k].update(reward)
            self.rewards[i] = reward

    def plot(self):
        self.cumulative_reward = np.cumsum(self.rewards) / (
            np.arange(self.n_trials) + 1
        )
        plt.plot(self.cumulative_reward)
        for p in self.bandit_probs:
            plt.plot(np.ones(self.n_trials) * p)
        plt.ylim((0, np.max(self.cumulative_reward) + 0.1))
        plt.xscale("log")
        plt.title("Convergence of Cumulative Rewards Averages")
        plt.xlabel("trial")
        plt.ylabel("cumulative average of rewards")

        plt.show()


if __name__ == "__main__":
    n_trials = 2000
    bandit_probs = [0.2, 0.5, 0.75]
    experiment = BanditsExperiment2(
        bandit_probs, n_trials, sample_points=[5, 10, 50, 100, 500, 1000, 1999]
    )
    experiment.run()
    experiment.plot()
