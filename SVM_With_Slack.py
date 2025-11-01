import numpy as np
from cvxopt import matrix, solvers
from collections import defaultdict, Counter

train_data = np.loadtxt("wdbc_train.data", delimiter=",")
X = train_data[:, 1:]
y = train_data[:, 0]
y = y.reshape(-1, 1)

test_data = np.loadtxt("wdbc_test.data", delimiter=",")
X_test = test_data[:, 1:]
y_test = test_data[:, 0].reshape(-1, 1)

n, d = X.shape
print(f"Loaded {n} samples with {d} features.")

k = 10
fold_size = n // k
folds = []
for i in range(k):
    start = i * fold_size
    end = (i + 1) * fold_size if i != k - 1 else n
    folds.append((X[start:end], y[start:end]))
print("Data partitioned into 10 contiguous folds.")

def train_linear_svm(X_train, y_train, C):
    n, d = X_train.shape
    K = np.dot(y_train * X_train, (y_train * X_train).T)

    P = matrix(K)
    q = matrix(-np.ones((n, 1)))
    G_std = np.vstack((-np.eye(n), np.eye(n)))
    h_std = np.vstack((np.zeros((n, 1)), np.ones((n, 1)) * C))
    A = matrix(y_train.reshape(1, -1))
    b = matrix(np.zeros(1))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, matrix(G_std), matrix(h_std), A, b)
    lambdas = np.array(sol['x'])

    support = (lambdas > 1e-5).flatten()
    w = np.sum(lambdas[support] * y_train[support] * X_train[support], axis=0)

    b_vals = y_train[support] - np.dot(X_train[support], w)
    b = np.mean(b_vals)

    return w, b

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

C_values = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
C_scores = defaultdict(list)

print("\nStarting 10-fold cross-validation...\n")

for C in C_values:
    for i in range(k):
        X_val, y_val = folds[i]
        X_train = np.vstack([folds[j][0] for j in range(k) if j != i])
        y_train = np.vstack([folds[j][1] for j in range(k) if j != i])

        w, b = train_linear_svm(X_train, y_train, C)

        preds = predict(X_val, w, b)
        acc = np.mean(preds == y_val.flatten()) * 100
        C_scores[C].append(acc)

    mean_acc = np.mean(C_scores[C])
    print(f"C = {C:>7},  mean validation accuracy = {mean_acc:.2f}%")

best_C = max(C_scores, key=lambda C: np.mean(C_scores[C]))
best_acc = np.mean(C_scores[best_C])

print("\n===================================")
print(f"Best hyperparameter: C = {best_C}")
print(f"Average validation accuracy: {best_acc:.2f}%")
print("===================================")

print(f"\nRetraining final model with C = {best_C} ...")
w, b = train_linear_svm(X, y, best_C)

print("Complete.")

preds = predict(X_test, w, b)
test_acc = np.mean(preds == y_test.flatten()) * 100
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

