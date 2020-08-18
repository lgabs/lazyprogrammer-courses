import pandas as pd
import ttester

df = pd.read_csv("advertisement_clicks.csv")
a = df.query("advertisement_id == 'A'")["action"].values
b = df.query("advertisement_id == 'B'")["action"].values

print("mean A: ", a.mean())
print("mean B: ", b.mean())

r = ttester.calculate_ttest_statistic(a, b, len(a))
print(r)
