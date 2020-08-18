from typing import Dict, Text, Any, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm

T = 501
true_ctr = 0.5
a, b = 1, 1
plot_indices = [10, 50, 100, 250, 500]
data = np.empty(T)

for i in range(T):
    reward = 1 if np.random.random() < true_ctr else 0
    data[i] = reward
    a += reward
    b += 1 - reward

    if i in plot_indices:
        p = data[:i].mean()  # data so far
        n = i + 1
        std = np.sqrt(p * (1 - p) / n)

        x = np.linspace(0, 1, 200)
        g = norm.pdf(x, loc=p, scale=std)
        plt.plot(x, g, label="Gaussian Approximation")

        posterior = beta.pdf(x, a=a, b=b)
        plt.plot(x, posterior, label="Beta Posterior")

        plt.legend()
        plt.title(f"Distributions after {i}")
        plt.show()
