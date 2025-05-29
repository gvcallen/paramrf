from asteval import Interpreter
from abc import abstractmethod, ABC

import numpy
from scipy.special import erfinv

"""
A list of priors, which usually will be grouped on a per-parameter basis, allowing for bayesian sampling of that parameter.
"""

class PDF(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def __str__(self, x):
        pass
    
    @property
    @abstractmethod
    def mean(self):
        pass

    @staticmethod
    @abstractmethod
    def kind() -> str:
        pass


class UniformPDF(PDF):
    def __init__(self, a, b, norm_input=True):
        if not norm_input:
            raise Exception('PDF input must currently be between 0.0 and 1.0')

        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a + (self.b-self.a) * x
    
    @property
    def mean(self):
        return (self.a + self.b) / 2.0
    
    @property
    def value(self):
        return (self.a, self.b)
    
    @value.setter
    def value(self, value):
        self.a, self.b = value
    
    @property
    def min(self):
        return self.a
    
    @min.setter
    def min(self, value):
        self.a = value
    
    @property
    def max(self):
        return self.b
    
    @max.setter
    def max(self, value):
        self.b = value
    
    @staticmethod
    def kind():
        return 'uniform'
    
    def __str__(self):
        return f'uniform({self.a}, {self.b})'


class GaussianPDF(PDF):
    def __init__(self, mu, sigma, bounds_sigma=5.0, norm_input=True):
        if not norm_input:
            raise Exception('Probability input must currently be between 0.0 and 1.0')
        
        self.mu = mu
        self.sigma = sigma
        self.bounds_sigma = bounds_sigma

    def __call__(self, x):
        return self.mu + self.sigma * numpy.sqrt(2) * erfinv(2*x-1)
    
    @property
    def mean(self):
        return self.mu
    
    @property
    def min(self):
        return self.mu - self.bounds_sigma*self.sigma
    
    @property
    def max(self):
        return self.mu + self.bounds_sigma*self.sigma
    
    @staticmethod
    def kind():
        return 'gaussian'

    def __str__(self):
        return f'gaussian({self.mu}, {self.sigma})'


class LogUniformPDF(UniformPDF):
    def __call__(self, x):
        return self.a * (self.b/self.a) ** x
    
    @staticmethod
    def kind():
        return 'log-uniform'


def forced_indentifiability_transform(x):
    N = len(x)
    t = numpy.zeros(N)
    t[N-1] = x[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = x[n]**(1./(n+1)) * t[n+1]
    return t


class SortedUniformProbability(UniformPDF):
    def __call__(self, x):
        t = forced_indentifiability_transform(x)
        return super(SortedUniformProbability, self).__call__(t)
    
    @staticmethod
    def kind():
        return 'sorted-uniform'


class LogSortedUniformPDF(LogUniformPDF):
    def __call__(self, x):
        t = forced_indentifiability_transform(x)
        return super(LogSortedUniformPDF, self).__call__(t)

    @staticmethod
    def kind():
        return 'log-sorted-uniform'
    

def from_string(s: str):
    class_map = {
        UniformPDF.kind(): UniformPDF,
        GaussianPDF.kind(): GaussianPDF,
        LogUniformPDF.kind(): LogUniformPDF,
        LogSortedUniformPDF.kind(): LogSortedUniformPDF,
    }

    try:
        # Parse the string "<class>(arg1, arg2, ...)"
        # Split the class name and arguments
        class_name, args_str = s.split("(", 1)
        args_str = args_str.rstrip(")")
        
        # Parse arguments
        # args = ast.literal_eval(f"({args_str})")  # Ensure it's a tuple
        aeval = Interpreter()
        args = aeval(f"({args_str})")
        
        # Find the class in the map
        if class_name in class_map:
            cls = class_map[class_name]
            return cls(*args)  # Dynamically instantiate the class
        else:
            raise ValueError(f"Class '{class_name}' is not allowed.")
    except Exception as e:
        raise ValueError(f"Failed to create instance: {e}")