import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import csv

X = []
y = []
# Load the data from the csv file
with open('letters_training.csv', 'rU') as csvfile:
    training = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in training:
        # row.remove(1025)
        try:
            tempRow = []
            tempRow.append([float(i) for i in row])
            X.append(tempRow[0][0:1023])
            y.append(tempRow[0][1024])
        except:
            continue

X = np.asarray(X)
y = np.asarray(y)

from sklearn import linear_model
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=0)

logreg = linear_model.LogisticRegression(C=1)
# train the classifier
lr = logreg.fit(X_train, y_train)
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
print '#############Problem 1, part 2#######################'
print 'Accuracy on the 85/15 set=' + str(test_score) + '\n\n'

from sklearn.cross_validation import ShuffleSplit
n_samples, n_features = X_train.shape
X_train, X_validate, y_train, y_validate = train_test_split(
    X_train, y_train, test_size=0.20, random_state=0)

n_Cs = 10
n_iter = 3

train_scores = np.zeros((n_Cs, n_iter))
test_scores = np.zeros((n_Cs, n_iter))
Cs = np.logspace(-5, 5, n_Cs)

for i, C in enumerate(Cs):
    # for j, (train, test) in enumerate(cv):
    # print i
    lr = linear_model.LogisticRegression(C=C).fit(X_train, y_train)
    train_scores[i] = lr.score(X_train, y_train)
    test_scores[i] = lr.score(X_validate, y_validate)
    print 'score for C=' + str(C) + 'is: ' + str(lr.score(X_validate, y_validate))

test_score = lr.score(X_test, y_test)
print '#############Problem 1, part 2 and 3#######################'
print 'Accuracy on the 70/15/15 set=' + str(test_score)

f = open('9_54.txt', 'r')
img = []
for line in f:
    for l in range(len(line)):
        if line[l] != '\r' and line[l] != '\n':
            img.append(float(line[l]))
img.pop(0)
print '#############Problem 1, part 4#######################'
print '\nPrediction for 9_54.txt = ' + str(logreg.predict(img)[0])

n_Cs = 10
n_iter = 3
# cv = ShuffleSplit(n_samples, n_iter=n_iter, test_size=0.15, random_state=0)

train_scores = np.zeros((n_Cs, n_iter))
test_scores = np.zeros((n_Cs, n_iter))
Cs = np.logspace(-3, 5, n_Cs)

for i, C in enumerate(Cs):
    # for j, (train, test) in enumerate(cv):
    print i
    lr = linear_model.LogisticRegression(penalty='l1', C=C).fit(X_train, y_train)
    train_scores[i] = lr.score(X_train, y_train)
    test_scores[i] = lr.score(X_validate, y_validate)


for i in range(n_iter):
    plt.semilogx(Cs, train_scores[:, i], alpha=0.4, lw=2, c='b')
    plt.semilogx(Cs, test_scores[:, i], alpha=0.4, lw=2, c='g')
plt.title("Effect of L1 regularization")
plt.ylabel("score for LR(penalty='l1', C=C)")
plt.xlabel("C")
plt.show()

def display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                      for k, v in params.items())
    line = "{0}:\t{1:.3f} (+/-{2:.3f})".format(
        params, np.mean(scores), sem(scores))
    if append_star:
        line += " *"
    return line

def display_grid_scores(grid_scores, top=None):
    """Helper function to format a report on a grid of scores"""

    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    if top is not None:
        grid_scores = grid_scores[:top]
    _, best_mean, best_scores = grid_scores[0]
    threshold = best_mean - 2 * sem(best_scores)

    for params, mean_score, scores in grid_scores:
        append_star = mean_score + 2 * sem(scores) > threshold
        print(display_scores(params, scores, append_star=append_star))

train_sizes = np.logspace(1, 2.9, 10, base=6).astype(np.int)

n_iter = 5
train_scores = np.zeros((train_sizes.shape[0], n_iter), dtype=np.float)
test_scores = np.zeros((train_sizes.shape[0], n_iter), dtype=np.float)

C = [0.0001, 0.01, 100.0, 10000.0]
for k in C:
    logreg = linear_model.LogisticRegression(C=k)

    for i, train_size in enumerate(train_sizes):
        cv = StratifiedShuffleSplit(y, n_iter=n_iter, train_size=train_size)
        for j, (train, test) in enumerate(cv):
            lr = logreg.fit(X[train], y[train])
            train_scores[i, j] = lr.score(X[train], y[train])
            test_scores[i, j] = lr.score(X[test], y[test])

    # We can now plot the mean scores with error bars that reflect the standard errors of the means:

    mean_train = np.mean(train_scores, axis=1)
    confidence = sem(train_scores, axis=1) * 2

    plt.fill_between(train_sizes, mean_train - confidence, mean_train + confidence,
                    color = 'b', alpha = .2)
    plt.plot(train_sizes, mean_train, 'o-k', c='b', label='Train score')

    mean_test = np.mean(test_scores, axis=1)
    confidence = sem(test_scores, axis=1) * 2

    plt.fill_between(train_sizes, mean_test - confidence, mean_test + confidence,
                    color = 'g', alpha = .2)
    plt.plot(train_sizes, mean_test, 'o-k', c='g', label='Test score')

    plt.xlabel('Training set size')
    plt.ylabel('Score')
    # plt.xlim(0, X_train.shape[0])
    plt.xlim(0, max(train_sizes))
    plt.ylim((None, 1.01))  # The best possible score is 1.0
    plt.legend(loc='best')
    plt.title('Main train and test scores +/- 2 standard errors, C=' + str(k))

    plt.show()

