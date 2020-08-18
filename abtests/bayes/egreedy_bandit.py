from typing import Dict, Text, Any, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


class Bandit:
    def __init__(self, p:float):
        self.p = p  # true probability this is usually not know in reality!
        self.rewards = []
        self.cumulative_rewards = []

    def pull(self):
        """
        simulares an answer from pulling the bandit.
        """
        # also, this reward law is usually not know in reality !
        return int(np.random.random() < self.p)

    def update(self, reward:int):
        # we do not update any belief, i.e, we learn nothing but we save
        # this reward to decide which bandit is better so far
        self.rewards.append(reward)
        self.cumulative_rewards = np.cumsum(self.rewards) / (
            np.arange(len(self.rewards)) + 1
            )

        pass



class BanditsExperiment:
    def __init__(self, eps: float, bandit_probs: List, n_trials: int, **kwargs):

        self.n_trials = n_trials
        self.eps = eps
        self.bandit_probs = bandit_probs
        self.bandits = [Bandit(p) for p in bandit_probs]
        self.sample_points = kwargs.get("sample_points", np.linspace(5, n_trials, 5))
        # this reward is calculated to our whole experiment, not for each bandit indivudually
        # we want to see the whole picture of rewards
        self.rewards = np.zeros(n_trials)

    def run(self, initial_pulls=100):
        for trial in range(self.n_trials):
            all_samples = []
            best_bandit = None
            for b in self.bandits:
                # we find which bandit will be pulled
                if np.random.random() < self.eps or trial < initial_pulls:
                    # explore
                    chosen_k = np.random.randint(0, len(self.bandits))
                    chosen = self.bandits[chosen_k]
                    explore = True

                else:
                    # exploit best bandit so far
                    chosen_k = np.argmax([b.cumulative_rewards[-1] for b in self.bandits])
                    chosen = self.bandits[chosen_k]
                    best_bandit_k = chosen_k
                    explore = False

                all_samples.append(chosen)

                # pull the bandit to update results
                reward = chosen.pull()
                chosen.update(reward)

                # save total rewards on experiment so far
                self.rewards[trial] = reward

    def plot_rewards(self):
        # whole experiment rewards
        self.cumulative_reward = np.cumsum(self.rewards) / (
            np.arange(self.n_trials) + 1
        )
        plt.plot(self.cumulative_reward)
        for p in self.bandit_probs:
            plt.plot(np.ones(self.n_trials) * p)

        plt.xscale("log")
        plt.title("Convergence of Cumulative Rewards Averages")
        plt.xlabel("trial")
        plt.ylabel("cumulative average of rewards")
        plt.tight_layout

        plt.show()

    def plot_individual_rewards(self):
        for b in self.bandits:
            plt.plot(b.cumulative_rewards)
            plt.tight_layout

        plt.show()


if __name__ == "__main__":
    n_trials = 1000
    bandit_probs = [0.2, 0.5, 0.75]
    eps = 0.05
    experiment = BanditsExperiment(
        eps, bandit_probs, n_trials, sample_points=[100, 101, 102, 103, 105, 109])
    experiment.run(initial_pulls=100)
    experiment.plot_rewards()
    experiment.plot_individual_rewards()
