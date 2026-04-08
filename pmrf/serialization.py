"""
Functions for model loading and saving.
"""

import os
from pathlib import Path
from typing import BinaryIO, Any

# Assuming parax is installed and exposes save/load at the top level
# Adjust this import if parax hides them inside parax.io
import parax

import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pandas

# Tell jsonpickle how to handle numpy arrays and pandas dataframes safely
jsonpickle_numpy.register_handlers()
jsonpickle_pandas.register_handlers()

def save(target: str | os.PathLike | BinaryIO, tree: Any):
    """
    Save a ParamRF Model (or any Parax PyTree) to a file.
    
    Automatically appends the '.prf' (ParamRF) extension to the target path 
    if no file extension is provided. This enforces the ParamRF file convention 
    while delegating the underlying serialization to Parax.
    
    Parameters
    ----------
    target : str, os.PathLike, or BinaryIO
        The path to the saved file or an open file-like object.
    tree : Any
        The PyTree containing the ParamRF model to save.
    """
    # Only manipulate the path if it's a string or Path-like object
    if isinstance(target, (str, os.PathLike)):
        target_path = Path(target)
        
        # If the path has no suffix (extension), append '.prf'
        if not target_path.suffix:
            target_path = target_path.with_suffix('.prf')
            
        target = target_path

    # Delegate to the underlying Parax save function
    parax.save(target, tree)


def load(source: str | os.PathLike | BinaryIO) -> Any:
    """
    Load a ParamRF Model (or any Parax PyTree) from a file.
    
    Delegates to Parax's loading logic. If the provided path does not exist 
    and lacks an extension, this function will automatically check if a file 
    with the same name and a '.prf' extension exists.
    
    Parameters
    ----------
    source : str, os.PathLike, or BinaryIO
        The path to the saved file or an open file-like object containing the data.
        
    Returns
    -------
    Any
        The deserialized PyTree (e.g., pmrf.Model).
    """
    if isinstance(source, (str, os.PathLike)):
        source_path = Path(source)
        
        # Quality-of-Life: If "my_model" doesn't exist, try "my_model.prf"
        if not source_path.exists() and not source_path.suffix:
            prf_path = source_path.with_suffix('.prf')
            if prf_path.exists():
                source = prf_path
        else:
            source = source_path

    # Delegate to the underlying Parax load function
    return parax.load(source)


__all__ = ['save', 'load']