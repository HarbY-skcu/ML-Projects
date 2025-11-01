import numpy as np
import pandas as pd
from collections import namedtuple, Counter
import math

train = np.loadtxt("heart_train.data", delimiter=",", dtype=int)
test = np.loadtxt("heart_test.data", delimiter=",", dtype=int)

y_train = np.where(train[:, 0] == 1, 1, -1)
X_train = train[:, 1:]
y_test = np.where(test[:, 0] == 1, 1, -1)
X_test = test[:, 1:]

n_train, d = X_train.shape
n_test = X_test.shape[0]

Tree = namedtuple("Tree", ["feature", "left_label", "right_label"])

def weighted_majority_label(labels, weights):
    score = np.sum(weights * labels)
    if score >= 0:
        return 1
    else:
        return -1

def find_best_tree(X, y, w):
    n, d = X.shape
    best_err = np.inf
    best_tree = None

    for j in range(d):
        f = X[:, j]
        left_mask = (f == 0)
        right_mask = (f == 1)

        left_label = weighted_majority_label(y[left_mask], w[left_mask]) if np.any(left_mask) else 1
        right_label = weighted_majority_label(y[right_mask], w[right_mask]) if np.any(right_mask) else 1

        preds = np.where(f == 0, left_label, right_label)
        err = np.sum(w * (preds != y))

        if err < best_err or (abs(err - best_err) < 1e-10 and (best_tree is None or j < best_tree.feature)):
            best_err = err
            best_tree = Tree(feature=j, left_label=left_label, right_label=right_label)

    return best_tree, best_err

def tree_predict(tree, X):
    f = X[:, tree.feature]
    return np.where(f == 0, tree.left_label, tree.right_label)

T = 10
w = np.ones(n_train) / n_train
trees, epsilons, mus = [], [], []
train_accs, test_accs = [], []

def ensemble_predict(X, trees, mus):
    if not trees:
        return np.ones(X.shape[0], dtype=int)
    F = np.zeros(X.shape[0])
    for tree, a in zip(trees, mus):
        F += a * tree_predict(tree, X)
    return np.sign(F)

for t in range(1, T + 1):
    tree, err = find_best_tree(X_train, y_train, w)
    eps = max(min(err, 1 - 1e-12), 1e-12)
    mu = 0.5 * math.log((1 - eps) / eps)

    trees.append(tree)
    epsilons.append(eps)
    mus.append(mu)

    h = tree_predict(tree, X_train)
    w *= np.exp(-mu * y_train * h)
    w /= np.sum(w)

    train_pred = ensemble_predict(X_train, trees, mus)
    test_pred = ensemble_predict(X_test, trees, mus)

    train_acc = np.mean(train_pred == y_train) * 100
    test_acc = np.mean(test_pred == y_test) * 100
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# --------------------------
# Display Results as Table
# --------------------------
rows = []
for i, (tree, eps, a, tr_acc, te_acc) in enumerate(zip(trees, epsilons, mus, train_accs, test_accs), start=1):
    desc = f"where... F{tree.feature+1}=0 → {tree.left_label:+d}, else → {tree.right_label:+d}"
    rows.append({
        "Round": i,
        "Feature": f"F{tree.feature+1}",
        "Tree": desc,
        "Train Acc (%)": round(tr_acc, 2),
        "Test Acc (%)": round(te_acc, 2)
    })

df = pd.DataFrame(rows, columns=["Round", "Feature", "Tree", "Train Acc (%)", "Test Acc (%)"])
print("\nAdaBoost Decision Trees (10 rounds):\n")
print(df.to_string(index=False))
