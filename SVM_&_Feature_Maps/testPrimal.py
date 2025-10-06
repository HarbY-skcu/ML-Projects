# !!!!! THIS TESTING PROGRAM WAS GENERATED WITH AI TO CHECK FOR ACCURACY !!!!!
# !!!!! DO NOT GRADE !!!!!
import numpy as np
from cvxopt import matrix, solvers

# =======================
# Load TRAIN (80%)  ---- expects 4 features + label per line
# =======================
Xtr_list, ytr_list = [], []
with open("mystery_80_nonoverlap.data") as f:
    for line in f:
        parts = line.strip().replace(",", " ").split()
        if not parts:
            continue
        x1, x2, x3, x4, y = parts
        Xtr_list.append([float(x1), float(x2), float(x3), float(x4)])
        ytr_list.append(int(y))

Xtr = np.asarray(Xtr_list, dtype=float)          # (n_tr, d)
ytr = np.asarray(ytr_list, dtype=float).reshape(-1)  # (n_tr,)

n_tr, d = Xtr.shape

# =======================
# Load TEST (20%)
# =======================
Xte_list, yte_list = [], []
with open("mystery_20_nonoverlap.data") as f:
    for line in f:
        parts = line.strip().replace(",", " ").split()
        if not parts:
            continue
        x1, x2, x3, x4, y = parts
        Xte_list.append([float(x1), float(x2), float(x3), float(x4)])
        yte_list.append(int(y))

Xte = np.asarray(Xte_list, dtype=float)          # (n_te, d)
yte = np.asarray(yte_list, dtype=float).reshape(-1)  # (n_te,)

# =======================
# Build feature map φ(x) (NO functions)  --- using the same "poly2" map
# φ(x) = [x, x^2, sqrt(2)*x_i*x_j for i<j]
# =======================

feats_list_tr = [Xtr, Xtr**2]
feats_list_te = [Xte, Xte**2]

cross_cols_tr = []
cross_cols_te = []
for i in range(d):
    for j in range(i + 1, d):
        cross_cols_tr.append((np.sqrt(2.0) * Xtr[:, i] * Xtr[:, j])[:, None])
        cross_cols_te.append((np.sqrt(2.0) * Xte[:, i] * Xte[:, j])[:, None])

if cross_cols_tr:
    Xtr_phi = np.hstack(feats_list_tr + cross_cols_tr)
    Xte_phi = np.hstack(feats_list_te + cross_cols_te)
else:
    Xtr_phi = np.hstack(feats_list_tr)
    Xte_phi = np.hstack(feats_list_te)

n_tr, phi = Xtr_phi.shape
n_te = Xte_phi.shape[0]

# =======================
# Hard-margin primal SVM on TRAIN (variables z = [w(phi); b])
# min (1/2)||w||^2  s.t.  y_i (w^T φ(x_i) + b) >= 1
# G z <= h with rows: [ -y_i φ(x_i)^T , -y_i ] <= -1
# =======================
P = np.zeros((phi + 1, phi + 1))
P[:phi, :phi] = np.eye(phi)
q = np.zeros(phi + 1)

G = np.zeros((n_tr, phi + 1))
G[:, :phi] = -(ytr[:, None] * Xtr_phi)
G[:, phi]   = -ytr
h = -np.ones(n_tr)

P_c = matrix(P, tc='d')
q_c = matrix(q, tc='d')
G_c = matrix(G, tc='d')
h_c = matrix(h, tc='d')

solvers.options['show_progress'] = False
sol = solvers.qp(P_c, q_c, G_c, h_c)

z = np.array(sol['x']).reshape(-1)
w = z[:phi]
b = float(z[phi])

# =======================
# Evaluate on TRAIN and TEST
# =======================
scores_tr = Xtr_phi @ w + b
yhat_tr = np.where(scores_tr >= 0.0, 1.0, -1.0)
acc_tr = (yhat_tr == ytr).mean()

scores_te = Xte_phi @ w + b
yhat_te = np.where(scores_te >= 0.0, 1.0, -1.0)
acc_te = (yhat_te == yte).mean()

# Optional: simple confusion counts (labels in {-1,+1})
tp_tr = np.sum((ytr == 1) & (yhat_tr == 1))
tn_tr = np.sum((ytr == -1) & (yhat_tr == -1))

tp_te = np.sum((yte == 1) & (yhat_te == 1))
tn_te = np.sum((yte == -1) & (yhat_te == -1))

# geometric margin in feature space = 1 / ||w||
w_norm = np.linalg.norm(w)
geom_margin = np.inf if w_norm == 0 else 1.0 / w_norm

# =======================
# Report
# =======================
print("w shape:", w.shape)
print("b:", b)
print("Feature Space:", geom_margin)
print("")
print("=== TRAIN (80%) ===")
print(f"n = {n_tr}, accuracy = {acc_tr:.3f}")
print(f"TP={tp_tr}  TN={tn_tr}")
print("")
print("=== TEST  (20%) ===")
print(f"n = {n_te}, accuracy = {acc_te:.3f}")
print(f"TP={tp_te}  TN={tn_te}")
