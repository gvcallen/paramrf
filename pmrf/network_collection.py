from copy import deepcopy

import numpy as np
import skrf as rf
from typing import List, Optional, Callable, Iterable
import pandas as pd

class NetworkCollection:
    """
    A clean container for heterogeneous scikit-rf Networks.
    - List-like for ordering
    - Dict-like lookup by network.name
    - Optional per-item metadata
    """

    def __init__(self, networks: Iterable[rf.Network] | None = None, *, name: str | None = None, params: dict = None):
        self.networks: List[rf.Network] = []
        self.params = params
        self.name = name
        if networks:
            for ntwk in networks:
                self.add(ntwk)

    # -----------------------------------------------------------
    # Core API
    # -----------------------------------------------------------

    def add(self, ntwk: rf.Network):
        """Add a Network. name must be unique."""
        if not isinstance(ntwk, rf.Network):
            raise TypeError("Only scikit-rf Networks may be added")

        if ntwk.name is None:
            raise ValueError("Network must have a 'name' attribute before adding")
        
        if ntwk.name in self.keys():
            raise ValueError(f"Network with name {ntwk.name} already exists")

        self.networks.append(ntwk)

    def __add__(self, other: "NetworkCollection") -> "NetworkCollection":
        """
        Combine two collections into a new one.
        Networks with duplicate names are auto-renamed.
        """
        if not isinstance(other, NetworkCollection):
            raise TypeError("Can only add another NetworkCollection.")

        new = NetworkCollection(name=self.name + ' + ' + other.name, params=self.params | other.params)
        for ntwk in self:
            new.add(ntwk)
        for ntwk in other:
            new.add(ntwk)
        return new    

    def __getitem__(self, key):
        """Index by integer or string name."""
        if isinstance(key, int):
            return self.networks[key]
        elif isinstance(key, str):
            for ntwk in self.networks:
                if ntwk.name == key:
                    return ntwk
            raise KeyError(f"No network named '{key}'")
        else:
            raise TypeError("Key must be int or str")

    def __len__(self):
        return len(self.networks)

    def __iter__(self):
        return iter(self.networks)
    
    def keys(self):
        return [ntwk.name for ntwk in self.networks]
        

    # -----------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------

    def filter(self, predicate: Callable[[rf.Network, dict], bool]):
        """Return a new NetworkCollection of items where predicate(ntwk, params) is True."""
        out = NetworkCollection()
        for ntwk in self.networks:
            params = ntwk.params
            if predicate(ntwk, params):
                out.add(ntwk, **params)
        return out

    def apply(self, fn: Callable[[rf.Network], rf.Network],
              names: Optional[Iterable[str]] = None):
        """Apply a function to selected networks in-place."""
        targets = names if names else [ntwk.name for ntwk in self.networks]

        for ntwk in self.networks:
            if ntwk.name in targets:
                new_ntwk = fn(ntwk)
                if not isinstance(new_ntwk, rf.Network):
                    raise TypeError("apply() must return a Network")
                # preserve name & metadata
                self.networks[self.networks.index(ntwk)] = new_ntwk

    def copy(self):
        return deepcopy(self)

    def names(self):
        return [ntwk.name for ntwk in self.networks]

    def summary(self):
        """Readable dataset summary."""
        lines = [f"NetworkCollection: {len(self)} networks\n"]
        for ntwk in self.networks:
            f = ntwk.frequency.f
            lines.append(
                f"- {ntwk.name}: {ntwk.nports}-port, "
                f"{f[0]/1e9:.2f}-{f[-1]/1e9:.2f} GHz, "
            )
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, rf.Network]:
        return {ntwk.name: ntwk for ntwk in self.networks}    

    def to_dataframe(self):
        """Convert to a pandas DataFrame for ML or metadata analysis."""
        rows = []
        for ntwk in self.networks:
            row = {"name": ntwk.name, "network": ntwk}
            rows.append(row)
        return pd.DataFrame(rows)
    # -----------------------------------------------------------
    # Frequency Utilities
    # -----------------------------------------------------------

    @property
    def frequency(self) -> rf.Frequency:
        frequency = None
        for ntwk in self.networks:
            if frequency is None:
                frequency = ntwk.frequency.copy()
            else:
                if frequency != ntwk.frequency:
                    raise Exception('"frequency" called on NetworkCollection but not all networks have the same frequency')
        return frequency    

    def frequency_ranges(self):
        return {ntwk.name: (ntwk.f[0], ntwk.f[-1], len(ntwk.f))
                for ntwk in self.networks}

    def common_frequency(self, mode="intersection", npoints=None):
        freqs = [ntwk.frequency.f for ntwk in self.networks]
        f_starts = [f[0] for f in freqs]
        f_stops  = [f[-1] for f in freqs]

        if mode == "preserve":
            return None

        if mode == "intersection":
            fmin = max(f_starts)
            fmax = min(f_stops)
            if fmin >= fmax:
                raise ValueError("No overlapping frequency region available.")
            if npoints is None:
                npoints = min(len(f) for f in freqs)
            return np.linspace(fmin, fmax, npoints)

        if mode == "union":
            fmin = min(f_starts)
            fmax = max(f_stops)
            if npoints is None:
                npoints = max(len(f) for f in freqs)
            return np.linspace(fmin, fmax, npoints)

        if mode in ("min_npoints", "max_npoints"):
            if mode == "min_npoints":
                n = min(len(f) for f in freqs)
            else:
                n = max(len(f) for f in freqs)
            fmin = max(f_starts)
            fmax = min(f_stops)
            if fmin >= fmax:
                raise ValueError("No overlapping region.")
            return np.linspace(fmin, fmax, n)

        raise ValueError(f"Unknown mode '{mode}'")

    def interpolate_to(self, frequency_vector):
        new = NetworkCollection()
        for ntwk in self.networks:
            ntwk_i = ntwk.copy()
            ntwk_i.interpolate_self(frequency_vector)
            new.add(ntwk_i, **self._meta[ntwk.name])
        return new

    def interpolate_self(self, frequency_vector):
        for i, ntwk in enumerate(self.networks):
            ntwk_i = ntwk.copy()
            ntwk_i.interpolate_self(frequency_vector)
            self.networks[i] = ntwk_i

    def interpolate(self, mode="intersection", npoints=None):
        f_vec = self.common_frequency(mode=mode, npoints=npoints)
        if f_vec is not None:
            self.interpolate_self(f_vec)
        return f_vec