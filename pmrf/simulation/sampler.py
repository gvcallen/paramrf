from copy import deepcopy
from typing import Iterator

import numpy as np
from scipy.stats import qmc

from pmrf.modeling import CircuitSystem

class CircuitSampler:
    def __init__(self, system: CircuitSystem, engine = 'lhs', **kwargs):
        self._system = system
        self._engine = engine 
        self._n = None

    def __iter__(self) -> Iterator[CircuitSystem]:
        if self._n is None:
            raise Exception('Error: to use this class as an iterator, call e.g. enumerate (CircuitSampler.range(n))')

        X = self._generate_hypercube_samples(self._n)
        for i in range(self._n):
            self._system.update_params(X[i,:], hypercube=True)
            yield self._system

        self._n = None

    def range(self, n) -> CircuitSystem:
        """Allows the CircuitSampler to be used as an iterable. To use, call e.g:
            for i, system in enumerate(sampler.range(10)).

        Args:
            n (int): The number of samples to generate

        Returns:
            CircuitSystem: self
        """
        self._n = n
        return self

    def generate(self, n=10, printout=False, callback=None, inplace=False):
        """Generates n random systems using the sampler's engine.

        Note that, if you want to generate samples one-by-one for efficiency, you can either use a callback with inplace=True,
        or use this class in iterator mode by passing N to the constructor and using e.g. python's "enumerate".

        Args:
            n (int, optional): The number of samples to generate. Defaults to 10.
            printout (bool, optional): Whether or not to print progress. Defaults to False.
            callback (callable, optional): A function to call-back each generatione, with parameters (idx, system). Defaults to None.
            inplace (bool, optional): If set to true, new systems are not created. Defaults to False, in which case a list of systems is created and returned.

        Returns:
            _type_: CircuitSystem | None
        """
        system = self._system
        
        if printout:
            print('Generating samples...')
        X = self._generate_hypercube_samples(n)
        
        if not inplace:
            circuits = []
        for i in range(n):
            if printout:
                print(f'Simulating network {i}')
            
            if not inplace:
                system = deepcopy(self._system)

            self._system.update_params(X[i,:], hypercube=True)

            if not inplace:
                circuits.append(system)

            if callback:
                callback(system, i)

        if not inplace:
            return circuits
    
    def _generate_hypercube_samples(self, n):
        d = self._system.num_free_params

        if self._engine == 'lhs':
            X = qmc.LatinHypercube(d).random(n)
        elif self._engine == 'uniform':
            X = np.random.uniform(0.0, 1.0, size=(n, d))
        else:
            raise Exception('Unknown sampler type')        
        
        return X