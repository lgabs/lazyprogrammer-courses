from typing import Dict, Text, Any, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


class Bandit:
    def __init__(self, p: float):
        self.p = p  # this is usually not know in reality!
        self.a = 1
        self.b = 1

    def pull(self):
        """
        simulares an answer from pulling the bandit.
        """
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x: int):
        self.a += x
        self.b += 1 - x


def plot(bandits: Bandit, trial: int):
    x = np.linspace(0, 1, 200)
    for bandit in bandits:
        y = beta.pdf(x, bandit.a, bandit.b)
        plt.plot(x, y, label=f"real p: %.4f" % bandit.p)

    plt.title(f"Bandits distributions after {trial} trials")
    plt.legend()
    plt.show()


class BanditsExperiment:
    def __init__(self, bandit_probs: List, n_trials: int, **kwargs):

        self.n_trials = n_trials
        self.bandits = [Bandit(p) for p in bandit_probs]
        self.sample_points = kwargs.get("sample_points", np.linspace(5, n_trials, 5))

    def run(self):
        for trial in range(self.n_trials):
            best_bandit = None
            maxsample = -1
            all_samples = []
            for b in self.bandits:
                # we find which bandit will be pulled
                sample = b.sample()
                all_samples.append(round(sample, 4))
                if sample > maxsample:
                    maxsample = sample
                    best_bandit = b

            if trial in self.sample_points:
                print("current sample: ", trial)
                plot(self.bandits, trial)

            # pull the bandit to update results
            x = best_bandit.pull()
            best_bandit.update(x)


if __name__ == "__main__":
    n_trials = 2000
    bandit_probs = [0.2, 0.5, 0.75]
    experiment = BanditsExperiment(
        bandit_probs, n_trials, sample_points=[5, 10, 50, 100, 500, 1000, 1999]
    )
    experiment.run()
