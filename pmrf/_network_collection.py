import skrf as rf
from typing import List, Dict, Optional, Callable, Iterable
import pandas as pd


class NetworkCollection:
    """
    A clean container for heterogeneous scikit-rf Networks.
    - List-like for ordering
    - Dict-like lookup by network.name
    - Optional per-item metadata
    """

    def __init__(self, networks: Optional[Iterable[rf.Network]] = None):
        self._nets: List[rf.Network] = []
        self._meta: Dict[str, dict] = {}
        if networks:
            for ntwk in networks:
                self.add(ntwk)

    # -----------------------------------------------------------
    # Core API
    # -----------------------------------------------------------

    def add(self, ntwk: rf.Network, **metadata):
        """Add a Network. name must be unique."""
        if not isinstance(ntwk, rf.Network):
            raise TypeError("Only scikit-rf Networks may be added")

        if not ntwk.name:
            raise ValueError("Network must have a 'name' attribute before adding")

        if ntwk.name in self._meta:
            raise KeyError(f"A Network with name '{ntwk.name}' already exists")

        self._nets.append(ntwk)
        self._meta[ntwk.name] = metadata

    def __getitem__(self, key):
        """Index by integer or string name."""
        if isinstance(key, int):
            return self._nets[key]
        elif isinstance(key, str):
            for ntwk in self._nets:
                if ntwk.name == key:
                    return ntwk
            raise KeyError(f"No network named '{key}'")
        else:
            raise TypeError("Key must be int or str")

    def metadata(self, key: str):
        """Return metadata for a network by name."""
        return self._meta[key]

    def __len__(self):
        return len(self._nets)

    def __iter__(self):
        return iter(self._nets)

    # -----------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------

    def filter(self, predicate: Callable[[rf.Network, dict], bool]):
        """Return a new NetworkCollection of items where predicate(ntwk, meta) is True."""
        out = NetworkCollection()
        for ntwk in self._nets:
            meta = self._meta[ntwk.name]
            if predicate(ntwk, meta):
                out.add(ntwk, **meta)
        return out

    def apply(self, fn: Callable[[rf.Network], rf.Network],
              names: Optional[Iterable[str]] = None):
        """Apply a function to selected networks in-place."""
        targets = names if names else [ntwk.name for ntwk in self._nets]

        for ntwk in self._nets:
            if ntwk.name in targets:
                new_ntwk = fn(ntwk)
                if not isinstance(new_ntwk, rf.Network):
                    raise TypeError("apply() must return a Network")
                # preserve name & metadata
                self._meta[new_ntwk.name] = self._meta.pop(ntwk.name)
                self._nets[self._nets.index(ntwk)] = new_ntwk

    def names(self):
        return [ntwk.name for ntwk in self._nets]

    def summary(self):
        """Readable dataset summary."""
        lines = [f"NetworkCollection: {len(self)} networks\n"]
        for ntwk in self._nets:
            f = ntwk.frequency.f
            meta = self._meta[ntwk.name]
            lines.append(
                f"- {ntwk.name}: {ntwk.nports}-port, "
                f"{f[0]/1e9:.2f}-{f[-1]/1e9:.2f} GHz, "
                f"metadata={list(meta.keys())}"
            )
        return "\n".join(lines)

    def to_dataframe(self):
        """Convert to a pandas DataFrame for ML or metadata analysis."""
        rows = []
        for ntwk in self._nets:
            row = {"name": ntwk.name, "network": ntwk}
            row.update(self._meta[ntwk.name])
            rows.append(row)
        return pd.DataFrame(rows)
