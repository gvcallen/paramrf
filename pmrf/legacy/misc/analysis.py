import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD

class ThresholdedSVD:
    def __init__(self, threshold=0.00005, min_components=2, max_components=100):
        self.threshold = threshold
        self.min_components = min_components
        self.max_components = max_components
        
        self.singular_values_ = None
        self.n_components_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        for n in range(self.min_components, self.max_components):
            self.n_components_ = n
            _, s, Vh = svd(X, full_matrices=False)
            self.singular_values_ = s[:n]
            self.components_ = Vh[:n]
            
            X_transformed = self.transform(X)

            self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
            full_var = np.var(X, axis=0).sum()
            self.explained_variance_ratio_ = exp_var / full_var
            if self.explained_variance_ratio_[-1] < self.threshold:
                break

        if n >= self.max_components:
            raise Exception('Error: maximum number of components reached in thresholded SVD')
        
        return self
        
    def transform(self, X):
        return np.dot(X, self.components_.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
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