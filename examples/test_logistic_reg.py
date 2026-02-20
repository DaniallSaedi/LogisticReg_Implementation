import numpy as np
import pandas as pd
from ..logistic_reg import LogisticReg

# Just for splitting data and use metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def make_dataset(m = 300):
    
    X = np.random.randn(m, 2)
    true_theta = np.array([[0.5], [2.0], [-1.0]])

    ones = np.ones((m, 1))
    X_aug = np.hstack([ones, X])

    logits = X_aug @ true_theta
    probs = 1 / (1 + np.exp(-logits))

    y = (probs >= 0.5).astype(int)

    return X, y, true_theta



X, y, true_theta = make_dataset(m = 1000)


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.25, random_state=42)

model = LogisticReg()
model.fit(X_train,
          y_train,
          alpha = 0.01)


print("Model Theta: ")
print(model.theta)

print("Real theta values : ")
print(true_theta)

print('f1 score : ', end='')
print(f1_score(model.predict(X_test), y_test))
