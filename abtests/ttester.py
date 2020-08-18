from scipy import stats
import numpy as np


def calculate_ttest_statistic(x1, x2, N, one_side=False):
    """
    Results from t test for two data samples with equal sizes N.
    """

    var_x1 = x1.var(ddof=1)
    var_x2 = x2.var(ddof=1)  # unbiased estimator, divide by N-1 instead of N
    s = np.sqrt((var_x1 + var_x2) / 2)

    # test statistic
    t = (x1.mean() - x2.mean()) / (s * np.sqrt(2.0 / N))  # balanced standard deviation
    # degrees of freedom
    df = 2 * N - 2
    # one-sided test
    p = 1 - stats.t.cdf(np.abs(t), df=df)
    pvalue = p if one_side else 2 * p

    return {"t": t, "p-value": pvalue}
