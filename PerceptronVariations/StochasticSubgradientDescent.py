import numpy as np

def phi(x):
  x1, x2 = x
  return np.array([x1, x2, x1**2 + x2**2])

dataset = []
with open("perceptron.data") as f:
  for line in f:
    parts = line.strip().replace(",", " ").split()
    if not parts or parts[0].startswith("#"):
      continue
    x1, x2, y = map(float, parts)
    dataset.append(((x1, x2), int(y)))

w = np.zeros(3)
b = 0.0
step = 1

def lossFunction(w, b, dataset):
  M = len(dataset)
  total = 0.0
  for(x1, x2), y in dataset:
    margin = y * (w @ phi((x1, x2)) + b) # w^t * phi(x) + b
    total += max(0.0, -margin)
  return total / M

M = len(dataset)
max_iters = 100_000
checkpoints = {1, 10, 100, 1_000, 10_000, 100_000}

print("Reports (checkpoint | w | b | loss result):")
for t in range(1, max_iters + 1):
  i = (t - 1) % M
  x, y = dataset[i]
  phi_x = phi(x)
  margin = y * (float(w @ phi_x) + b)
  if margin <= 0.0:
    w = w + step * y * phi_x
    b = b + step * y

  if t in checkpoints:
    loss = lossFunction(w, b, dataset)

    print(f"{t:7d} | {w} | {b:.4f} | {loss:.4f}")
