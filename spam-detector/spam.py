# Author: http://lazyprogrammer.me
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from wordcloud import WordCloud

df = pd.read_csv("datasets_483_982_spam.csv", encoding="ISO-8859-1")
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.rename(mapper={"v1": "label", "v2": "data"}, axis=1, inplace=True)

# binary labels
df["is_spam"] = df["label"].map({"ham": 0, "spam": 1})

# get X and y
y = df["is_spam"].values
count_vectorizer = CountVectorizer(decode_error="ignore")
X = count_vectorizer.fit_transform(df["data"])

# separate train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# time to model
model = MultinomialNB()
model.fit(X_train, y_train)
print("train score: ", model.score(X_train, y_train))
print("test score: ", model.score(X_test, y_test))

# visualize the data
def visualize(label):
  words = ''
  for msg in df[df['label'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

# visualize('spam')
# visualize('ham')

# make predictions col
df["pred"] = model.predict(X)

# sneaky spams (real spam but predicted as not) ---> false negative
sneaky_spams = df.query("is_spam == 1 and pred == 0")

# false positives
fp = df.query("is_spam == 0 and pred == 1")
