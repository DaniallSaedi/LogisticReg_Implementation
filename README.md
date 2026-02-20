# Logistic Regression From Scratch (NumPy)

A lightweight and educational implementation of **Logistic Regression using Gradient Descent**, written entirely with NumPy.  
This project is ideal for learning how binary classification works under the hood — without relying on scikit‑learn.

---

## 🚀 Features

- Sigmoid activation  
- Binary cross‑entropy loss  
- Gradient Descent optimization  
- Optional feature scaling  
- Optional L2 regularization (Ridge)  
- Early stopping based on gradient norm  
- Probability prediction (`predict_proba`)  
- Class prediction (`predict`)  
- Tracks loss history  

---

## 📂 Project Structure

LogisticReg_Implementation/
│
├── logistic_reg.py
│
└── examples/
└── test_logistic_reg.py

---

## 🧠 How It Works

### Sigmoid Function



\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]



### Binary Cross‑Entropy Loss



\[
J(\theta) = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]
\]



### Gradient Descent Update



\[
\theta := \theta - \alpha \cdot \frac{1}{m} X^T(\hat{y} - y)
\]



With L2 regularization:



\[
\theta_j := \theta_j - \alpha \left( \frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m}\theta_j \right)
\]



---

## 🧪 Example Usage

```python
from LogisticReg_Implementation.logistic_reg import LogisticReg
import numpy as np

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

model = LogisticReg()
model.fit(X, y, alpha=0.01, max_iter=2000, scale=True)

print("Predictions:", model.predict(X))
print("Probabilities:", model.predict_proba(X))
print("First 5 losses:", model.loss_history[:5])

---

## Parameters
X        : feature matrix
y        : binary target
alpha    : learning rate
lam      : l2 regularization parameter
scale    : Standardize features
max_iter : Max iterations
tol      : Stop early if gradient norm < tol

---

## Installation
If you want to import it like a package:
    pip install -e .

---

## ▶️ How to Run the Example
python -m LogisticReg_Implementation.examples.test_logistic_reg

---

If you want, I can also generate:

- a combined README for both Linear & Logistic Regression  
- a comparison table  
- a math‑heavy version with derivations  
- a more aesthetic README with badges and visuals  

Just tell me what style you want.
