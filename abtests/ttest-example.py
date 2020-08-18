import examples
from scipy import stats
import numpy as np

a = examples.generate_data(10, 2)
b = examples.generate_data(10, 0)


def calculate_ttest_statistic(x1, x2, N):
    """
    Results from t test for two data samples with equal sizes N.
    """

    var_x1 = x1.var(ddof=1)
    var_x2 = x2.var(ddof=1)
    s = np.sqrt((var_x1 + var_x2) / 2)

    # test statistic
    t = (x1.mean() - x2.mean()) / (s * np.sqrt(2.0 / N))
    # degrees of freedom
    df = 2 * N - 2
    p = 1 - stats.t.cdf(t, df=df)

    return {"t": t, "p-value": 2 * p}


results = calculate_ttest_statistic(a, b, 10)
t1, p1 = results["t"], results["p-value"]
print(f"t1: {t1}, p-value: {p1}")
# print(results)

t2, p2 = stats.ttest_ind(a, b)
print(f"t2: {t2}, p-value: {p2}")
