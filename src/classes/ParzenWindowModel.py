from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor

class ParzenWindowRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski', metric_params=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, 
            weights=self.weights, 
            algorithm=self.algorithm, 
            leaf_size=self.leaf_size, 
            p=self.p, 
            metric=self.metric,
            metric_params=self.metric_params
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
