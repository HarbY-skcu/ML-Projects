import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

def standardize_train_val_test(X_train, X_val, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-12
    return norm(X_train, mu, sigma), norm(X_val, mu, sigma), norm(X_test, mu, sigma)

def norm(A, mu, sigma):
    return (A - mu) / sigma

def solve_primal_l2_slack(X_tr, y_tr, C):
    n, d = X_tr.shape
    m = d + 1 + n
    eps_w = 1e-8
    eps_xi = 1e-8

    P = np.zeros((m, m))
    P[:d, :d] = np.eye(d) * (1.0 + eps_w)
    q = np.zeros(m)
    q[d+1:] = C

    G1 = np.zeros((n, m))
    G1[:, :d] = -(y_tr[:, None] * X_tr)
    G1[:, d]  = -y_tr
    G1[:, d+1 + np.arange(n)] = -1.0
    h1 = -np.ones(n)

    G2 = np.zeros((n, m))
    G2[:, d+1 + np.arange(n)] = -1.0
    h2 = np.zeros(n)

    G = np.vstack([G1, G2])
    h = np.concatenate([h1, h2])

    P[d + 1:, d + 1:] = P[d + 1:, d + 1:] + 2.0 * eps_xi * np.eye(n)

    P_c, q_c = matrix(P, tc='d'), matrix(q, tc='d')
    G_c, h_c = matrix(G, tc='d'), matrix(h, tc='d')

    sol = solvers.qp(P_c, q_c, G_c, h_c)
    z = np.array(sol['x']).reshape(-1)
    w, b = z[:d], float(z[d])
    return w, b

def predict(X, w, b):
    return np.sign(X @ w + b)

def accuracy(X, y, w, b):
    y_pred = predict(X, w, b)
    return np.mean(y_pred == y)

X_list, y_list = [], []
with open("magic.data") as f:
    for line in f:
        parts = line.strip().replace(",", " ").split()
        if not parts or len(parts) < 11:
            continue
        *xvals, lab = parts
        X_list.append(list(map(float, xvals)))
        y_list.append(1.0 if float(lab) == 1.0 else -1.0)

X_all = np.array(X_list, dtype=float)
y_all = np.array(y_list, dtype=float)

X_train, y_train = X_all[:1000], y_all[:1000]
X_val,   y_val   = X_all[1000:2000], y_all[1000:2000]
X_test,  y_test  = X_all[2000:], y_all[2000:]

X_train, X_val, X_test = standardize_train_val_test(X_train, X_val, X_test)

C_values = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]

print(f"{'C':>10} {'TrainAcc':>10} {'ValAcc':>10} {'TestAcc':>10}")
best_val, best_C = -1.0, None
for C in C_values:
    w, b = solve_primal_l2_slack(X_train, y_train, C)
    tr = accuracy(X_train, y_train, w, b)
    va = accuracy(X_val,   y_val,   w, b)
    te = accuracy(X_test,  y_test,  w, b)
    print(f"{C:>10.1e} {tr:>10.4f} {va:>10.4f} {te:>10.4f}")
    if va > best_val:
        best_val, best_C = va, C

print(f"\nBest C = {best_C}  Validation Accuracy = {best_val:.4f}")

X_comb = np.vstack([X_train, X_val])
y_comb = np.hstack([y_train, y_val])

w_best, b_best = solve_primal_l2_slack(X_comb, y_comb, best_C)
acc_comb = accuracy(X_comb, y_comb, w_best, b_best)
acc_test = accuracy(X_test,  y_test,  w_best, b_best)

print(f"Combined Set Accuracy: {acc_comb:.4f}")
print(f"Test Accuracy: {acc_test:.4f}")