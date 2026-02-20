import numpy as np
import pandas as pd
from numpy import exp

class LogisticReg:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.scale = None
        self.loss_history = []
            
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self,
            X,
            y,
            alpha = 0.001,
            lam = 0.0,
            scale = False,
            max_iter = 1000,
            tol = 1e-8):

        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float)
        y = y.reshape(-1, 1)

        m, n = X.shape

        # Scaling (if needed)
        self.scale = scale
        
        if scale:
            self.mu = X.mean(axis =0)
            self.sigma = X.std(axis =0)
            X = (X - self.mu) / self.sigma
        else:
            self.mu = np.zeros(n)
            self.sigma = np.ones(n)
            
        # Initializing Thetas
        self.theta = np.zeros((n + 1, 1))

        # Bias
        ones = np.ones((m, 1))
        X = np.hstack([ones, X])

        eps = 1e-8
        
        for _ in range(max_iter):
            y_hat = self._sigmoid(X @ self.theta)

            grad = X.T @ (y_hat - y)
            grad[1:] += lam* self.theta[1:]
            self.theta -= alpha / m * grad 
            
            loss = -1 / m * np.sum( y * np.log(y_hat + eps) + ( 1 - y) * np.log( 1- y_hat + eps))
            self.loss_history.append(loss)

            if np.linalg.norm(grad) < tol:
                break

    def predict(self, 
                X, 
                threshold = 0.5):
        if self.theta is None:
            raise RuntimeError("Model must be fitted first")
        
        X = np.asarray(X, dtype = float)
        if self.scale:
            X = (X - self.mu) / self.sigma

        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

        y_hat = self._sigmoid(X @ self.theta)
        
        return (y_hat >= threshold).astype(int)

    def predict_proba(self, 
                X, 
                threshold = 0.5):
        if self.theta is None:
            raise RuntimeError("Model must be fitted first")
        
        X = np.asarray(X, dtype = float)
        if self.scale:
            X = (X - self.mu) / self.sigma

        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

        y_hat = self._sigmoid(X @ self.theta)
        
        return y_hat