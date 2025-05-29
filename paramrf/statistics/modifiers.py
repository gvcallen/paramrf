from __future__ import annotations
from copy import deepcopy
import ast

import numpy as np

from paramrf.misc.math import dB20, norm

"""
This file contains the "Modifier" classes, which essentially encapsulate various functionality to be performed
on arrays. This makes testing new types of cost functions on the highest script level very easy,
as it is simply the combination of multiple "operations" specified using a list of strings.
"""

class Modifier:
    """
    A modifier class is used to encapsulate modification on a numpy array of data.
    """
    def __init__(self, mode, **kwargs):
        """Initialize a modifier class.

        Args:
            operation (str): The operation to be performed. Should be one of the following (note that most options accept suffix '-ax<int>' to specify an axis to perform the operation along):

                    - None (no-op)
                    - 'dB', 'abs': maths operations
                    - 'sum', 'min', 'max': maths operations (default axis == None)
                    - '<norm>': norm with <norm> in ['Linf', 'L1', 'L2'] (default axis == 0)
                    - '<norm>-M': norm with <norm> in ['Linf', 'L1', 'L2'] (axis == None)
                    - 'multiply-every-<int>': multiply every group of <int> rows/columns (default axis == 1)
                    - 'weight-by-<list>': weighting by array (default axis == 1):
                    - 'sum-every-<int>': sum every group of <int> rows/columns (default axis == 1)
                    - 'convolve-interleaved' convolve arrays formed by interleaving (default axis == 1)
                    - 'multiply-by-<list> multiply by array (default axis == 1)
                    - 'callback' (some )
          
        """
        self.mode = mode
        self.kwargs = kwargs

        axis_default = x_default = n_default = callback_default = None
         
        if self.mode == None:
            pass
        elif self.mode.startswith('L'):
            if self.mode.endswith('M'):
                axis_default = None
                self.mode = self.mode[0:-2]
            elif '-' in self.mode:
                axis_default = int(self.mode.split('-')[1])
            else:
                axis_default = 0
        elif self.mode.startswith('convolve-interleaved'):
            axis_default = 1
        elif self.mode.startswith('sum-every'):
            axis_default = 1
            nbase = len('sum-every')
            if len(mode) > nbase:
                n_default = int(ast.literal_eval(mode[nbase+1:]))
        elif self.mode.startswith('multiply-every'):
            axis_default = 1
            nbase = len('multiply-every')
            if len(mode) > nbase:
                n_default = int(ast.literal_eval(mode[nbase+1:]))
        elif self.mode.startswith('multiply-by'):
            axis_default = 1
            nbase = len('multiply-by')
            if len(mode) > nbase:
                x_default = np.array(ast.literal_eval(mode[nbase+1:]), dtype=np.float64)
        elif self.mode.startswith('weight-by'):
            axis_default = 1
            nbase = len('weight-by')
            if len(mode) > nbase:
                x_default = np.array(ast.literal_eval(mode[nbase+1:]), dtype=np.float64)

        if '-ax' in self.mode:
            axis_default = int(self.mode[self.mode.rindex('-ax')+3:])

        self.kwargs.setdefault('axis', axis_default)
        self.kwargs.setdefault('x', x_default)
        self.kwargs.setdefault('n', n_default)
        self.kwargs.setdefault('callback', callback_default) 

    def __call__(self, y) -> np.float64 | np.ndarray:    
        mode = self.mode
        kwargs = self.kwargs
        axis = kwargs['axis']
        x = kwargs['x']
        n = kwargs['n']
        callback = kwargs['callback']
        
        # Switch on mode
        if mode == None:
            y = y
        elif mode == 'dB':
            y = dB20(y)
        elif mode == 'abs':
            y = np.abs(y)
        elif mode == 'sum':
            y = np.sum(y, axis=axis)
        elif mode == 'sqr':
            y = y**2
        elif mode == 'min':
            y = np.min(y, axis=axis)
        elif mode == 'max':
            y = np.max(y, axis=axis)
        elif mode == 'mean':
            y = np.mean(y, axis=axis)
        elif mode[0] == 'L':
            y = norm(y, mode=mode, axis=axis)
        elif mode.startswith('convolve-interleaved'):
            y = self._convolve_interleaved(y, axis)
        elif mode.startswith('sum-every'):
            y = self._sum_every(y, n, axis=axis)
        elif mode.startswith('multiply-every'):
            y = self._multiply_every(y, n, axis=axis)
        elif mode.startswith('multiply-by'):
            y = self._multiply_by(y, x, axis)
        elif mode.startswith('weight-by'):
            nbase = len('weight-by')
            n = len(x)
            y = self._multiply_by(y, x, axis)
            y = self._sum_every(y, n, axis)
        elif mode == 'callback':
            y = callback(y)
        else:
            raise Exception('Unknown combiner type')

        return y        
    

    def _multiply_by(self, y, x, axis) -> np.float64 | np.ndarray:
        if x.shape == y.shape:
            y *= x
        else:
            n = len(x)

            if len(y.shape) == 1:
                if axis == 0:
                    y = y.reshape(len(y), 1)
                else:
                    y = y.reshape(1, len(y))
            
            if axis and y.shape[axis] % n != 0:
                raise ValueError(f"The length of the specified axis ({y.shape[axis]}) is not divisible by {n}.")

            if axis == 0:
                x = np.tile(x, (int(y.shape[0] / n), y.shape[1]))
            elif axis == 1:
                x = np.tile(x, (y.shape[0], int(y.shape[1] / n)))

            y *= x

        return y

    def _sum_every(self, y, n, axis) -> np.float64 | np.ndarray:
        if len(y.shape) == 1 and len(y) % n == 0:
            if axis == 0:
                y = y.reshape(len(y), 1)
            else:
                y = y.reshape(1, len(y))

        # Ensure the axis length is divisible by i
        if y.shape[axis] % n != 0:
            raise ValueError(f"The length of the specified axis ({y.shape[axis]}) is not divisible by {n}.")

        # Reshape and sum
        if axis == 0:
            # Group rows
            y = y.reshape(-1, n, y.shape[1]).sum(axis=1)
        elif axis == 1:
            # Group columns
            y = y.reshape(y.shape[0], -1, n).sum(axis=2)

        return y
    
    def _multiply_every(self, y, n, axis) -> np.float64 | np.ndarray:
        if len(y.shape) == 1 and len(y) % n == 0:
            if axis == 0:
                y = y.reshape(len(y), 1)
            else:
                y = y.reshape(1, len(y))

        # Ensure the axis length is divisible by i
        if y.shape[axis] % n != 0:
            raise ValueError(f"The length of the specified axis ({y.shape[axis]}) is not divisible by {n}.")

        # Reshape and sum
        if axis == 0:
            # Group rows
            y = y.reshape(-1, n, y.shape[1]).prod(axis=1)
        elif axis == 1:
            # Group columns
            y = y.reshape(y.shape[0], -1, n).prod(axis=2)

        return y    
    
    def _convolve_interleaved(self, y, axis) -> np.float64 | np.ndarray:
        # Write code that convolves array y1 with y2, where y1 is every 1st element, and y2 every 2nd, along the specified axis.
        if axis == 1:
            if len(y.shape) == 1:
                y1, y2 = y[0::2], y[1::2]
                y = np.convolve(y1, y2)
            else:
                raise Exception("Not yet implemented")
        else:
            raise Exception("Not yet implemented")

        return y    

class ModifierChain:
    """
    A chain of modifier operations to be performed one after the other.
    """
    def __init__(self, modifiers: list[Modifier | str | tuple[str, dict]] = None):
        """Initialize a modifier chain using a list of modifiers or the args that would create them.

        Args:
            modifiers (list[Modifier | str | tuple[str, dict]]): The list of modifiers. Either pass the Modifier class directly in each list element,
            or just the modifier mode, or pass the arguments used to create the modifier class in (str, dict) format.
        """
        if modifiers is None:
            modifiers = []

        self.modifiers = []
        if modifiers:
            for modifier in modifiers:
                if type(modifier) == Modifier:
                    self.modifiers.append(deepcopy(modifier))
                else:
                    if type(modifier) == tuple:
                        operation, kwargs = modifier
                        self.modifiers.append(Modifier(operation, **kwargs))
                    else:
                        self.modifiers.append(Modifier(modifier))

    def __call__(self, y):
        for modifier in self.modifiers:
            y = modifier(y)

        if type(y) == np.ndarray and y.size == 1:
            y = np.float64(y)
        
        return y