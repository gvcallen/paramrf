import numpy as np
from sklearn.decomposition import TruncatedSVD

class ThresholdedSVD:
    def __init__(self, threshold=0.00005, min_components=1, max_components=100):
        self.svd = None
        self.threshold = threshold
        self.min_components = min_components
        self.max_components = max_components
        
    def fit(self, X):
        for n in range(self.min_components, self.max_components):
            self.svd = TruncatedSVD(n_components=n).fit(X)
            ratios = self.svd.explained_variance_ratio_
            if ratios[-1] < self.threshold:
                break
            
        if n >= self.max_components:
            raise Exception('Error: maximum number of components reached in thresholded SVD')
        
        return self
        
    def transform(self, X):
        return self.svd.transform(X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @property
    def explained_variance_ratio_(self):
        return self.svd.explained_variance_ratio_
    
class ArcsinhScaler():
    def __init__(self, A=0.0, k=1.0):
        self.A = A
        self.k = k
        
    def fit(self, X):
        self.A = np.max(X)
        return self
        
    def transform(self, X):
        X_offset = X - self.A
        return np.arcsinh(X_offset / self.k)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)