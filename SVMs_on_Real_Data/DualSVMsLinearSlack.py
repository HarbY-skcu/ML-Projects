import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

def gaussian_kernel_ij(xi, xj, sigma):
    diff = xi - xj
    return np.exp(-(diff @ diff) / (2.0 * (sigma ** 2)))

def compute_kernel(X, sigma):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = gaussian_kernel_ij(X[i], X[j], sigma)
    return K

def train_dual_svm(X, y, C, sigma):
    n = len(X)
    K = compute_kernel(X, sigma)

    Y = np.outer(y, y)

    P = matrix(Y * K, tc='d')
    q = matrix(-np.ones(n), tc='d')

    G = matrix(np.vstack([-np.eye(n), np.eye(n)]), tc='d')
    h = matrix(np.hstack([np.zeros(n), np.full(n, C)]), tc='d')

    sol = solvers.qp(P, q, G, h)
    lambdas = np.ravel(sol['x'])
    sv_mask = lambdas > 1e-6
    return lambdas[sv_mask], X[sv_mask], y[sv_mask]

def decision_at(x, lam, X_sv, y_sv, sigma):
    return sum(a * yi * gaussian_kernel_ij(xi, x, sigma)
               for a, yi, xi in zip(lam, y_sv, X_sv))

def predict(X, lam, X_sv, y_sv, sigma):
    return np.array([1 if decision_at(x, lam, X_sv, y_sv, sigma) >= 0 else -1 for x in X])

def accuracy(X, y, lam, X_sv, y_sv, sigma):
    y_pred = predict(X, lam, X_sv, y_sv, sigma)
    return np.mean(y_pred == y)

X_list, y_list = [], []
with open("magic.data") as f:
    for line in f:
        parts = line.strip().replace(",", " ").split()
        if not parts or len(parts) < 11:
            continue
        *x_vals, lab = parts
        X_list.append(list(map(float, x_vals)))
        y_list.append(1 if float(lab) == 1.0 else -1)

X_all = np.array(X_list, dtype=float)
y_all = np.array(y_list, dtype=int)

X_train, y_train = X_all[:1000], y_all[:1000]
X_val, y_val = X_all[1000:2000], y_all[1000:2000]
X_test, y_test = X_all[2000:], y_all[2000:]

mu = X_train.mean(axis=0)
sigma_x = X_train.std(axis=0) + 1e-12
def standardize(A):
  return (A - mu) / sigma_x
X_train, X_val, X_test = standardize(X_train), standardize(X_val), standardize(X_test)

C_values = [.1, 1, 10, 100, 1000, 10000, 100000]
sigma_values = [0.01, 0.1, 1, 10, 100, 1000]

best_acc_val, best_params = 0, (None, None)

print(f"{'C':>10} {'σ':>8} {'TrainAcc':>10} {'ValAcc':>10} {'TestAcc':>10}")
for C in C_values:
    for sigma in sigma_values:
        lam, X_sv, y_sv = train_dual_svm(X_train, y_train, C, sigma)
        train_acc = accuracy(X_train, y_train, lam, X_sv, y_sv, sigma)
        val_acc = accuracy(X_val, y_val, lam, X_sv, y_sv, sigma)
        test_acc = accuracy(X_test, y_test, lam, X_sv, y_sv, sigma)
        print(f"{C:>10.1e} {sigma:>8.2f} {train_acc:>10.4f} {val_acc:>10.4f} {test_acc:>10.4f}")
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            best_params = (C, sigma)

C_best, sigma_best = best_params
print("\nBest (C, σ) =", best_params, f"→ Validation Accuracy = {best_acc_val:.4f}")

X_comb = np.vstack([X_train, X_val])
y_comb = np.hstack([y_train, y_val])

lam, X_sv, y_sv = train_dual_svm(X_comb, y_comb, C_best, sigma_best)
train_comb_acc = accuracy(X_comb, y_comb, lam, X_sv, y_sv, sigma_best)
test_acc = accuracy(X_test, y_test, lam, X_sv, y_sv, sigma_best)

print(f"Using best (C, σ) = {best_params}")
print(f"Combined Train+Val Accuracy: {train_comb_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")