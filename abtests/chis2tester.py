import numpy as np
from scipy.stats import chi2

def get_pvalue_chi2(T):
    det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    c2 = (float(det)**2 * T.sum() )/ T[0].sum() / T[1].sum() / T[:, 0].sum() / T[:, 1].sum()
    p = 1 - chi2.cdf(x=c2, df=1)
    print("testing for H0 Xb - Xa:")


    return c2, p
