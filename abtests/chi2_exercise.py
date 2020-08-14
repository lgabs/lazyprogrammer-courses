import pandas as pd
import numpy as np
import chis2tester

df = pd.read_csv("advertisement_clicks.csv")
a = df.query("advertisement_id == 'A'")["action"].values
b = df.query("advertisement_id == 'B'")["action"].values

print("mean A: ", a.mean())
print("mean B: ", b.mean())


# lets construct the table T
T = np.zeros((2,2)).astype(np.float32)
alternatives = ["A", "B"]
action_values = [0, 1]
for i, alternative in enumerate(alternatives):
    for j, value in enumerate(action_values):
        T[i, j] = df.query(f"advertisement_id == '{alternative}' and action == {value} ")["action"].count()

print(T)

t, p  = chis2tester.get_pvalue_chi2(T)
print(f"t: {t}  p: {p}")
