from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

M = None

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "imports-85.data"

df = pd.read_csv(DATA_PATH, header=None, na_values='?')
data = df.iloc[:, [16, 23, 24, 25]]
data.columns = ["wt", "city-mpg", "hwy-mpg", "price"]
print(data)

def grad_a(a, b, x, y):
  residuals = x.dot(a) + b - y
  return 2 / M * np.sum(x * (a * x + b - y))

def grad_b(a, b, x, y):
  residuals = x.dot(a) + b - y
  return 2 / M * np.sum(a * x + b - y)

x = data[["wt", "city-mpg"]]
y = data["hwy-mpg"]
M = len(x)

x_add = x.copy()
x_add["bias"] = 1.0

theta = pd.Series([0.0] * x_add.shape[1], index=x_add.columns)

def grad_theta(theta, x_add, y):
  residuals = x_add.dot(theta) - y
  return 2 / M * x_add.mul(residuals, axis=0).sum()

int_step = 2e-4
for i in range(1, 2001):
  step_function = int_step / np.sqrt(i)
  theta = theta - step_function * grad_theta(theta, x_add, y)

pred_x = pd.Series(
    np.linspace(x["city-mpg"].min(), x["city-mpg"].max(), 10),
    name="city-mpg"
)
pred_df = pd.DataFrame({ "wt": x["wt"].mean(), "city-mpg": pred_x, "bias": 1.0 })
pred_y = pred_df.dot(theta)

plt.scatter(x["city-mpg"], y, label="True")
plt.plot(pred_x, pred_y, color="red", label="Prediction")
plt.ylabel("Highway MPG")
plt.xlabel("City MPG")
plt.legend()
plt.title("Highway MPG from City MPG (with weight as extra feature)")
plt.show()