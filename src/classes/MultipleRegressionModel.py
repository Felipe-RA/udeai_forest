from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

class MultipleRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, copy_X=True, positive=True):
        """
        Hyperparameters:
        - fit_intercept: bool, default=True
            Whether to calculate the intercept for this model.
        - copy_X: bool, default=True
            If True, X will be copied; else, it may be overwritten.
        - positive: bool, default=True
            When set to True, forces the coefficients to be positive.
        """
        
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive
        
        self.model = LinearRegression(
            fit_intercept=self.fit_intercept, 
            copy_X=self.copy_X,
            n_jobs=-1,  # Always use the maximum number of jobs
            positive=self.positive
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
