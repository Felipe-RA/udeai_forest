from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

class MultipleRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, **hyperparameters):
        self.model = LinearRegression(**hyperparameters)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
