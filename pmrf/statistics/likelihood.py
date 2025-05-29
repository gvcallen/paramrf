from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, rice

"""
This file contains functionality for evaluating log-likelihoods for a probabilistic dataset.
For the Gaussian likelihood, for example, the log-likelihood for a number of independently and identically distributed
Gaussian random variables, with the same standard deviation but centred around an input array x, is calculated.
"""

class Likelihood(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, x_meas, x_model):
        pass
    
    @abstractmethod
    def kind(self):
        pass
    
    @abstractmethod
    def update_params(self, theta):
        pass
    
    @abstractmethod
    def params(self) -> dict:
        pass

class GaussianLikelihood(Likelihood):
    """
    The likelihood of n independent Guassian random variables with the same standard deviation sigma.
    x values passed in must be real.
    """
    def __init__(self, sigma = 0.1):
        self.sigma = sigma

    def __call__(self, x_meas, x_model):
        # Rather use scipy
        return np.sum(norm.logpdf(np.real(x_meas), loc=np.real(x_model), scale=self.sigma))
        # sigma2 = self.sigma * self.sigma     
        # logL = -np.log(np.sqrt(2 * np.pi * sigma2)) - (np.abs(x)**2 / (2 * sigma2))
        # logL = np.sum(logL)
        # return logL
    
    def kind(self):
        return 'gaussian'
    
    def update_params(self, theta):
        self.sigma = theta[0]

    @property
    def num_params(self):
        return 1
    
    def params(self) -> dict:
        return {
            'sigma': self.sigma,
        }
        
class CircularComplexGaussianLikelihood(Likelihood):
    """
    The likelihood of n independent complex gaussian random variables with standard deviations sigma_real and sigma_imag representing independent (circularly symmetrc) real and imaginary parts.
    Note that, unless scaled_sigma==True, if sigma_real == sigma_imag == sigma, then this is equivalent to the sum of two Gaussian likelihoods with sigma' == sigma / sqrt(2).
    However, if scaled_sigma=True is passed, then the sigma represents that of the UNDERLYING component gaussians.
    x values passed in must be complex.
    """
    def __init__(self, sigma_real = 0.1, sigma_imag = 0.1, scaled_sigma = False):
        self.sigma_real = sigma_real
        self.sigma_imag = sigma_imag
        if scaled_sigma:
            self.sigma_scale = 1.0
        else:
            self.sigma_scale = np.sqrt(2)

    def __call__(self, x_meas, x_model):
        return np.sum(norm.logpdf(np.real(x_meas), loc=np.real(x_model), scale=self.sigma_real/self.sigma_scale)) + np.sum(norm.logpdf(np.imag(x_meas), loc=np.imag(x_model), scale=self.sigma_imag/self.sigma_scale))
    
    def kind(self):
        return 'gaussian'
    
    def update_params(self, theta):
        self.sigma = theta[0]

    @property
    def num_params(self):
        return 1
    
    def params(self) -> dict:
        return {
            'sigma': self.sigma,
        }
               
class RicianLikelihood(Likelihood):
    """
    The likelihood of n independent Rician random variables with standard deviation sigma.
    Note that a random variable R ~ Rice with standard deviation sigma has the same pdf as |X+iY| if X and Y ~ gaussian with standard deviations sigma.
    x values passed in must be magnitudes.
    """
    def __init__(self, sigma = 0.1):
        self.sigma = sigma

    def __call__(self, x_meas, x_model):
        b = x_model / self.sigma
        return np.sum(rice.logpdf(x_meas, b=b, scale=self.sigma))
    
    def kind(self):
        return 'rician'
    
    def update_params(self, theta):
        self.sigma = theta[0]

    @property
    def num_params(self):
        return 1
    
    def params(self) -> dict:
        return {
            'sigma': self.sigma,
        }