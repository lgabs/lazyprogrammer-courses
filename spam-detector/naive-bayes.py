from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

# read data
data = pd.read_csv("spambase.data").as_matrix()

# shuffle inplace
np.random.shuffle(data)

X = data[:, :48]
y = data[:, -1]

N = 100
X_train, X_test = X[:-N,], X[-N:,]
y_train, y_test = y[:-N,], y[-N:,]

clfs = {
    "NB": MultinomialNB(),
    "AdaBoost": AdaBoostClassifier(),
}

for name, clf in clfs.items():
    clf.fit(X_train, y_train)
    print(f"model score for {name}: {clf.score(X_test, y_test)}")
