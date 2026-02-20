# Logistic Regression From Scratch (NumPy)

A lightweight and educational implementation of **Logistic Regression using Gradient Descent**, written entirely with NumPy.  
This project is ideal for learning how binary classification works under the hood — without relying on scikit‑learn.

---

## Features

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

## Example Usage

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
```
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

## How to Run the Example
python -m LogisticReg_Implementation.examples.test_logistic_reg
