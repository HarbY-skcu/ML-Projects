import numpy as np
from collections import Counter

train = np.loadtxt("heart_train.data", delimiter=",", dtype=int)
test  = np.loadtxt("heart_test.data",  delimiter=",", dtype=int)

y_train = np.where(train[:, 0] == 1, 1, -1)
X_train = train[:, 1:]
y_test  = np.where(test[:, 0] == 1, 1, -1)
X_test  = test[:, 1:]

n_train, d = X_train.shape

def majority_label(y):
    c = Counter(y)
    return 1 if c[1] > c[-1] else -1

def train_tree(X, y):
    best_feat = None
    best_acc = -1
    best_left_label, best_right_label = None, None

    for j in range(X.shape[1]):
        f = X[:, j]
        left_mask = (f == 0)
        right_mask = (f == 1)

        left_label = majority_label(y[left_mask])
        right_label = majority_label(y[right_mask])

        preds = np.where(f == 0, left_label, right_label)
        acc = np.mean(preds == y)

        if acc > best_acc:
            best_acc = acc
            best_feat = j
            best_left_label, best_right_label = left_label, right_label

    return {
        "feature": best_feat,
        "left_label": best_left_label,
        "right_label": best_right_label,
        "train_acc": best_acc
    }

def predict_tree(model, X):
    f = X[:, model["feature"]]
    return np.where(f == 0, model["left_label"], model["right_label"])

B = 20
n = len(y_train)
models = []

for b in range(B):
    idx = np.random.choice(n, size=n, replace=True)
    Xb, yb = X_train[idx], y_train[idx]
    model = train_tree(Xb, yb)
    models.append(model)

def predict_bagged(models, X):
    all_preds = np.array([predict_tree(m, X) for m in models])
    votes = np.sign(np.sum(all_preds, axis=0))
    votes[votes == 0] = 1
    return votes

train_accs, test_accs = [], []

print("\n=== Bagging Results ===")
print(f"Number of bootstrap samples: {B}\n")
print(f"{'tree':>5} {'Feature':>10} {'Train Acc (%)':>15} {'Test Acc (%)':>15}")
print("-" * 50)

for i, m in enumerate(models, start=1):
    train_preds = predict_tree(m, X_train)
    test_preds = predict_tree(m, X_test)

    train_acc = np.mean(train_preds == y_train) * 100
    test_acc = np.mean(test_preds == y_test) * 100

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"{i:5d} {f'F{m['feature']+1}':>10} {train_acc:15.2f} {test_acc:15.2f}")

avg_train_tree = np.mean(train_accs)
avg_test_tree  = np.mean(test_accs)

print("-" * 50)
print(f"{'Average':>15} {avg_train_tree:15.2f} {avg_test_tree:15.2f}")

train_preds = predict_bagged(models, X_train)
test_preds  = predict_bagged(models, X_test)

train_acc = np.mean(train_preds == y_train) * 100
test_acc  = np.mean(test_preds == y_test) * 100

print("\n=== Average Classifier ===")
print(f"Training accuracy: {train_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%")
