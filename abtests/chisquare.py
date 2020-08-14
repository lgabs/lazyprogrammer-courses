import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

class DataClicksGenerator():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def next(self):
        click1 = 1 if (np.random.random() < self.p1) else 0
        click2 = 1 if (np.random.random() < self.p2) else 0

        return click1, click2

def get_pvalue(T):
    #det = np.linalg.det(T)
    det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    c2 = (float(det)**2 * T.sum() )/ T[0].sum() / T[1].sum() / T[:, 0].sum() / T[:, 1].sum()
    p = 1 - chi2.cdf(x=c2, df=1)

    return p

def run_experiment(p1, p2, N, alpha):
    """
    p1, p2: probabilities of each variant
    N: number of trials
    """
    data = DataClicksGenerator(p1, p2)
    T = np.zeros((2,2)).astype(np.float32)
    p_values = np.empty(N)
    for i in range(N):
        c1, c2 = data.next()
        T[0, c1] += 1
        T[1, c2] += 1
        if i < 10:
            p_values[i] = None
        else:
            p_values[i] = get_pvalue(T)

    return p_values

experiments = 3
N = 20000
results = []
alpha = 0.05

for k in range(experiments):
    results.append(run_experiment(0.1, 0.1, N, alpha))


# plot results
f, ax = plt.subplots()
for p_values in results:
    ax.plot(p_values)

ax.plot(np.ones(N)*alpha)
plt.show()
