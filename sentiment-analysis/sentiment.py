import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

lemmatizer = WordNetLemmatizer()

with open("stopwords.txt") as f:
    stopwords = set(w.rstrip for w in f)

positive_reviews = BeautifulSoup(open("electronics/positive.review").read(), features="html5lib").findAll("review_text")
negative_reviews = BeautifulSoup(open("electronics/negative.review").read(), features="html5lib").findAll("review_text")

# lets balance the dataset because one group is much higher than the other
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


def my_tokenizer(sentence:str):
    s = sentence.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]

    return tokens

# now we need to create a global index for each word
# we need to find the size of vocabulary
word_index_map = {}
current_index = 0
positive_tokens = []
negative_tokens = []


for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


def tokens2vector(tokens:list, label):
    """
    simple way here to gather both vector and label for later usage
    """

    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x/x.sum()
    x[-1] = label

    return x

N = len(positive_tokens) + len(negative_tokens)
data = np.zeros((N, len(word_index_map) + 1))

i = 0
for tokens in positive_tokens:
    xy = tokens2vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokens:
    xy = tokens2vector(tokens, 0)
    data[i,:] = xy
    i += 1

np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

# batch_test = 100
# X_train, y_train = X[:-batch_test,], Y[:-batch_test,]
# X_test, y_test = X[-batch_test:,], Y[-batch_test:,]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)

print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))


ths = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if abs(weight) > ths:
        print(f"word: {word}, {weight}")
