import glob
import os
import zipfile
import glob
from copy import deepcopy
from typing import List, Optional, Callable, Iterable, Union, Dict

import numpy as np
import pandas as pd
import skrf as rf

class NetworkCollection:
    """
    A container for a collection of scikit-rf Networks.

    Unlike :class:`~skrf.networkSet.NetworkSet`, which is designed for 
    statistical analysis of a set of networks with identical topology and 
    frequency (e.g., Monte Carlo analysis), a :class:`NetworkCollection` 
    is a general-purpose container. It supports networks with different 
    frequencies, port counts, and names.

    It behaves like a list (ordered) and a dictionary (lookup by name).

    Parameters
    ----------
    networks : list of :class:`~skrf.network.Network`, optional
        A list of Networks to initialize the collection. 
    name : str, optional
        A name for the collection. 
    params : dict, optional
        A dictionary of parameters associated with the collection itself.

    Attributes
    ----------
    networks : list
        The list of Network objects.
    name : str
        The name of the collection.
    """
    def __init__(self, networks: Optional[Iterable[rf.Network]] | str | None = None, 
                 name: Optional[str] = None, params: Optional[dict] = None):
        if isinstance(networks, str):
            networks = [rf.Network(f) for f in sorted(glob.glob(os.path.join(networks, "*.s*p")))]
        
        self.networks: List[rf.Network] = []
        self.params = params or {}
        self.name = name

        if networks:
            for ntwk in networks:
                self.add(ntwk)

    def __repr__(self):
        return f"<NetworkCollection: {self.name or 'Unnamed'} ({len(self)} networks)>"

    def __str__(self):
        return self.summary()

    def __len__(self):
        return len(self.networks)

    def __iter__(self):
        return iter(self.networks)

    def __getitem__(self, key: Union[int, str, slice]) -> Union[rf.Network, "NetworkCollection"]:
        """
        Get a network by index or name.
        
        Parameters
        ----------
        key : int, str, or slice
            If int, returns the network at that index.
            If str, returns the network with that name.
            If slice, returns a new NetworkCollection.
        """
        if isinstance(key, (int, slice)):
            item = self.networks[key]
            if isinstance(item, list):
                # Handle slice returning a list of networks
                return NetworkCollection(item, name=self.name, params=self.params)
            return item
        elif isinstance(key, str):
            for ntwk in self.networks:
                if ntwk.name == key:
                    return ntwk
            raise KeyError(f"No network named '{key}' in this collection.")
        else:
            raise TypeError(f"Key must be int, str, or slice, not {type(key)}")

    def __add__(self, other: "NetworkCollection") -> "NetworkCollection":
        """
        Concatenate two collections into a new one.
        
        Handles name collisions by appending an incrementing suffix 
        (e.g., `_1`, `_2`) to the incoming networks.
        """
        if not isinstance(other, NetworkCollection):
            return NotImplemented

        # Determine new name
        if self.name == other.name:
            new_name = self.name
        elif not self.name or not other.name:
            new_name = self.name or other.name
        else:
            new_name = f"{self.name} + {other.name}"

        # Merge params
        new_params = self.params.copy()
        new_params.update(other.params)

        new_collection = NetworkCollection(name=new_name, params=new_params)

        # Add all networks
        for ntwk in self.networks:
            new_collection.add(ntwk.copy())

        for ntwk in other.networks:
            ntwk_copy = ntwk.copy()
            # Resolve collisions
            base_name = ntwk_copy.name
            counter = 1
            while ntwk_copy.name in new_collection.keys():
                ntwk_copy.name = f"{base_name}_{counter}"
                counter += 1
            new_collection.add(ntwk_copy)

        return new_collection

    # -----------------------------------------------------------
    # Constructors / IO
    # -----------------------------------------------------------
    
    @classmethod
    def from_dir(cls, directory: str, pattern: str = "*.s*p", 
                 name: Optional[str] = None) -> "NetworkCollection":
        """
        Create a NetworkCollection from a directory of Touchstone files.

        Parameters
        ----------
        directory : str
            Path to the directory.
        pattern : str, optional
            Glob pattern to match files. Default is "*.s*p".
        name : str, optional
            Name of the collection. Defaults to directory name.

        Returns
        -------
        NetworkCollection
        """
        directory = os.path.expanduser(directory)
        if name is None:
            name = os.path.basename(os.path.abspath(directory))
            
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        networks = [rf.Network(f) for f in files]
        return cls(networks, name=name)

    @classmethod
    def from_zip(cls, zip_file: str, name: Optional[str] = None) -> "NetworkCollection":
        """
        Create a NetworkCollection from a zip file of Touchstone files.
        """
        if name is None:
            name = os.path.basename(zip_file)
            
        networks = []
        with zipfile.ZipFile(zip_file, 'r') as z:
            for filename in sorted(z.namelist()):
                if filename.lower().endswith(('.s1p', '.s2p', '.s3p', '.s4p', '.snp')):
                    # skrf can read from file-like objects usually, but sometimes 
                    # requires a wrapper. Reading into BytesIO is safest.
                    from io import BytesIO
                    data = BytesIO(z.read(filename))
                    data.name = filename # Network needs a filename for extension guessing
                    try:
                        networks.append(rf.Network(data, name=os.path.splitext(filename)[0]))
                    except Exception:
                        continue
        return cls(networks, name=name)

    def write(self, directory: str = ".") -> None:
        """
        Write all networks in the collection to Touchstone files in the specified directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ntwk in self.networks:
            ntwk.write_touchstone(dir=directory)

    # -----------------------------------------------------------
    # Management
    # -----------------------------------------------------------

    def add(self, ntwk: rf.Network) -> None:
        """
        Add a Network to the collection.
        
        Parameters
        ----------
        ntwk : skrf.Network
            The network to add. Its `.name` attribute must be set and unique 
            within the collection.
            
        Raises
        ------
        ValueError
            If the network has no name or the name already exists.
        TypeError
            If ntwk is not a scikit-rf Network.
        """
        if not isinstance(ntwk, rf.Network):
            raise TypeError(f"Expected scikit-rf Network, got {type(ntwk)}")

        if ntwk.name is None:
            raise ValueError("Network must have a 'name' attribute.")
        
        # Check uniqueness. (Linear scan is acceptable for typical collection sizes)
        if any(n.name == ntwk.name for n in self.networks):
            raise ValueError(f"Network with name '{ntwk.name}' already exists.")

        self.networks.append(ntwk)

    def keys(self) -> List[str]:
        """Return a list of network names."""
        return [ntwk.name for ntwk in self.networks]

    def copy(self) -> "NetworkCollection":
        """Return a deep copy of the collection."""
        return deepcopy(self)

    def filter(self, func: Callable[[rf.Network], bool]) -> "NetworkCollection":
        """
        Return a new NetworkCollection containing only networks where `func(ntwk)` is True.

        Parameters
        ----------
        func : callable
            A function that accepts a Network and returns a boolean.
        
        Returns
        -------
        NetworkCollection
        """
        subset = [n for n in self.networks if func(n)]
        return NetworkCollection(subset, name=self.name, params=self.params)

    def apply(self, func: Callable[[rf.Network], rf.Network]) -> None:
        """
        Apply a function to every network in the collection in-place.
        
        Parameters
        ----------
        func : callable
            A function that accepts a Network and returns a Network.
        """
        for i, ntwk in enumerate(self.networks):
            res = func(ntwk)
            if not isinstance(res, rf.Network):
                raise TypeError("Function passed to apply() must return a Network.")
            self.networks[i] = res

    def to_dict(self) -> Dict[str, rf.Network]:
        """
        Convert the collection to a dictionary {name: network}.

        Returns
        -------
        dict
            A dictionary where keys are network names and values are the Network objects.
        """
        return {ntwk.name: ntwk for ntwk in self.networks}            

    def to_dataframe(self, attrs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert the collection summary to a pandas DataFrame.
        
        Parameters
        ----------
        attrs : list of str, optional
            List of attributes to extract from each network (e.g. ['name', 'nports']).
            Defaults to name, ports, and frequency range.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for ntwk in self.networks:
            if attrs:
                row = {attr: getattr(ntwk, attr) for attr in attrs}
            else:
                f = ntwk.frequency
                row = {
                    "name": ntwk.name,
                    "nports": ntwk.nports,
                    "f_start": f.f[0] if len(f) > 0 else np.nan,
                    "f_stop": f.f[-1] if len(f) > 0 else np.nan,
                    "npoints": len(f),
                }
            # Add network params if they exist
            if hasattr(ntwk, 'params'):
                row.update(ntwk.params)
            rows.append(row)
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """Return a string summary of the collection."""
        header = f"NetworkCollection: {self.name or 'Unnamed'} ({len(self)} networks)"
        lines = [header, "-" * len(header)]
        for ntwk in self.networks:
            f = ntwk.frequency
            # Format frequency nicely
            if len(f.f) > 0:
                f_str = f"{f.f[0]/1e9:.2f}-{f.f[-1]/1e9:.2f} GHz"
            else:
                f_str = "No Freq"
            lines.append(f"{ntwk.name:<20} | {ntwk.nports} ports | {f_str}")
        return "\n".join(lines)

    # -----------------------------------------------------------
    # Frequency Operations
    # -----------------------------------------------------------

    def common_frequency(self, mode: str = "intersection", npoints: Optional[int] = None) -> np.ndarray:
        """
        Compute a common frequency vector for the collection.

        Parameters
        ----------
        mode : {'intersection', 'union', 'min_npoints', 'max_npoints'}, optional
        npoints : int, optional
        """
        if not self.networks:
            raise ValueError("Collection is empty.")

        freqs = [n.f for n in self.networks]
        starts = [f[0] for f in freqs]
        stops = [f[-1] for f in freqs]
        points = [len(f) for f in freqs]

        if mode == "intersection":
            f_start, f_stop = max(starts), min(stops)
            if npoints is None:
                npoints = min(points)
        elif mode == "union":
            f_start, f_stop = min(starts), max(stops)
            if npoints is None:
                npoints = max(points)
        elif mode == "min_npoints":
            f_start, f_stop = max(starts), min(stops)
            npoints = min(points)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        if f_start >= f_stop:
            raise ValueError("Frequency ranges do not overlap.")

        unit = self.networks[0].frequency.unit
        scaler = rf.Frequency.multiplier_dict[unit.lower()]
        f_scaled = np.linspace(f_start, f_stop, npoints) / scaler
        return rf.Frequency.from_f(f_scaled, unit=unit)

    def _resolve_frequency(self, freq_or_mode: Union[rf.Frequency, np.ndarray, str], 
                           npoints: Optional[int] = None) -> rf.Frequency:
        """Helper to resolve user input into a skrf.Frequency object."""
        if isinstance(freq_or_mode, str):
            return self.common_frequency(mode=freq_or_mode, npoints=npoints)
        elif isinstance(freq_or_mode, rf.Frequency):
            return freq_or_mode
        else:
            return rf.Frequency.from_f(freq_or_mode, unit='Hz')

    def interpolate(self, freq_or_mode: Union[rf.Frequency, np.ndarray, str] = "intersection", 
                    npoints: Optional[int] = None, **kwargs) -> "NetworkCollection":
        """
        Return a new NetworkCollection where all networks are interpolated to a common frequency.

        Parameters
        ----------
        freq_or_mode : skrf.Frequency, np.ndarray, or str
            If 'intersection' or 'union', calculates the common vector.
            If a vector/Frequency object, uses that explicitly.
        npoints : int, optional
            Number of points (if using mode string).
        kwargs : 
            Passed to `Network.interpolate`.

        Returns
        -------
        NetworkCollection
        """
        freq_obj = self._resolve_frequency(freq_or_mode, npoints)
        new_coll = self.copy()
        new_coll.interpolate_self(freq_obj, **kwargs)
        return new_coll

    def interpolate_self(self, freq_or_mode: Union[rf.Frequency, np.ndarray, str] = "intersection", 
                         npoints: Optional[int] = None, **kwargs) -> None:
        """
        Interpolate all networks in the collection in-place to a common frequency.

        Parameters
        ----------
        freq_or_mode : skrf.Frequency, np.ndarray, or str
            If 'intersection' or 'union', calculates the common vector.
            If a vector/Frequency object, uses that explicitly.
        npoints : int, optional
            Number of points (if using mode string).
        kwargs : 
            Passed to `Network.interpolate_self`.
        """
        freq_obj = self._resolve_frequency(freq_or_mode, npoints)
        
        for ntwk in self.networks:
            ntwk.interpolate_self(freq_obj, **kwargs)