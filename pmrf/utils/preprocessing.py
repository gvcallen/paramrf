from typing import Any

def remove_constant_params(list_of_params: list[dict[str, Any]]):
    # 1. Identify all unique parameter keys present in the networks
    all_keys = set()
    for params in list_of_params:
        all_keys.update(params.keys())

    # 2. Find keys that have the same value across all networks
    keys_to_remove = []
    for key in all_keys:
        # Get values for this key from every network (None if missing)
        values = [params.get(key) for params in list_of_params]
        
        # Check if all values are identical
        # Using set(values) works for hashable types (ints, floats, strings)
        if len(set(values)) == 1:
            keys_to_remove.append(key)

    # 3. Purge the static keys from every network
    for params in list_of_params:
        for key in keys_to_remove:
            if key in params:
                del params[key]   
