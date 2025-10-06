import numpy as np
from cvxopt import matrix, solvers

X_list = []
y_list = []
with open("mystery.data") as f:
    for line in f:
        parts = line.strip().replace(",", " ").split()
        if not parts:
            continue
        x1, x2, x3, x4, y = parts
        X_list.append([float(x1), float(x2), float(x3), float(x4)])
        y_list.append(int(y))

X = np.asarray(X_list, dtype=float)
y = np.asarray(y_list, dtype=float).reshape(-1)

n, d = X.shape

feats_list = [X, X**2]
cross_cols = []
for i in range(d):
    for j in range(i + 1, d):
        cross_cols.append((np.sqrt(2.0) * X[:, i] * X[:, j])[:, None])
if cross_cols:
    X_phi = np.hstack(feats_list + cross_cols)
else:
    X_phi = np.hstack(feats_list)

n, phi = X_phi.shape

P = np.zeros((phi + 1, phi + 1))
P[:phi, :phi] = np.eye(phi)
q = np.zeros(phi + 1)

G = np.zeros((n, phi + 1))
G[:, :phi] = -(y[:, None] * X_phi)
G[:, phi]   = -y
h = -np.ones(n)

P_c = matrix(P, tc='d')
q_c = matrix(q, tc='d')
G_c = matrix(G, tc='d')
h_c = matrix(h, tc='d')

solvers.options['show_progress'] = False
sol = solvers.qp(P_c, q_c, G_c, h_c)

z = np.array(sol['x']).reshape(-1)
w = z[:phi]
b = float(z[phi])

scores = X_phi @ w + b
y_hat = np.where(scores >= 0.0, 1.0, -1.0)
acc = (y_hat == y).mean()

w_norm = np.linalg.norm(w)
geom_margin = np.inf if w_norm == 0 else 1.0 / w_norm

print("w shape:", w.shape)
print("b:", b)
print(f"training accuracy: {acc:.3f}")
print("Feature Space:", geom_margin)

