import numpy as np

train = np.loadtxt("heart_train.data", delimiter=",", dtype=int)
y = np.where(train[:, 0] == 1, 1, -1).astype(int)
X = train[:, 1:].astype(int)
n, d = X.shape

H_list = []
names = []
for j in range(d):
    fj = X[:, j]
    h_plus  = np.where(fj == 0, -1, +1)
    h_minus = -h_plus
    H_list.append(h_plus)
    H_list.append(h_minus)
    names.append(f"F{j+1}:0→-1,1→+1")
    names.append(f"F{j+1}:0→+1,1→-1")

H = np.stack(H_list, axis=1)
m = H.shape[1]

def exp_loss(y, F):
    return float(np.sum(np.exp(-y * F)))

def update_mu_k(H_k, y, F, mu_k):
    F_minus_k = F - mu_k * H_k
    w = np.exp(-y * F_minus_k)

    correct = np.sum(w[(y == H_k)])
    incorrect = np.sum(w[(y != H_k)])

    eps = incorrect / (correct + incorrect + 1e-300)
    eps = min(max(eps, 1e-12), 1 - 1e-12)

    new_mu = 0.5 * np.log((1 - eps) / eps)
    F_new = F_minus_k + new_mu * H_k
    return new_mu, F_new, eps

mu = np.zeros(m)
F = np.zeros(n)

max_passes = 50
tol = 1e-8

history = []

for p in range(max_passes):
    mu_change = 0.0
    for k in range(m):
        H_k = H[:, k]
        new_mu_k, F, eps_k = update_mu_k(H_k, y, F, mu[k])
        mu_change = max(mu_change, abs(new_mu_k - mu[k]))
        mu[k] = new_mu_k

    L = exp_loss(y, F)
    history.append((p + 1, L))
    if mu_change < tol:
        break

order = np.argsort(-np.abs(mu))
print("Top coordinates (sorted by mu):")
print(f"{'Rank':>4}  {'Coord':>5}  {'Hypothesis':<20}  {'mu':>12}")
for r, k in enumerate(order[:min(20, m)], start=1):
    print(f"{r:>4}  {k:>5}  {names[k]:<20}  {mu[k]:>12.6f}")

k_opt = np.argmax(np.abs(mu))
mu_opt = mu[k_opt]
loss_opt = exp_loss(y, F)
hypothesis_opt = names[k_opt]

# -----------------------------
# Final Output
# -----------------------------
print("\n=== Coordinate Descent Results ===")
print(f"Optimal Tree: {hypothesis_opt}")
print(f"Optimal mu: {mu_opt:.6f}")
print(f"Exponential loss on training set: {loss_opt:.6f}")