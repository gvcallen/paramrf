from datetime import datetime
from typing import Union, Sequence, Callable
from numbers import Number
from dataclasses import Field

import jax.numpy as jnp
from equinox import field as base_field
           
def field(
    *,
    derived: bool = False,
    **kwargs,
):    
    metadata = dict(kwargs.pop('metadata', {}))
    if 'derived' in metadata:
        raise Exception("Cannot use metadata with `derived` already set.")
    metadata['derived'] = derived
    
    if derived:
        kwargs['init'] = False
    
    return base_field(metadata=metadata, **kwargs)

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)

def update_dict_with_alias(original: dict, updates: dict, alias_map: dict) -> None:
    # Build prefix lookup trie (flattened since prefixes are strings)
    # Sort prefixes by length (longest first) to match the most specific prefix first
    sorted_aliases = sorted(alias_map.items(), key=lambda x: -len(x[0]))

    for key in original:
        for orig_prefix, update_prefix in sorted_aliases:
            if key.startswith(orig_prefix):
                aliased_key = update_prefix + key[len(orig_prefix):]
                if aliased_key in updates:
                    original[key] = updates[aliased_key]
                break
        # if no prefix matched, keep the original value
        
class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)
