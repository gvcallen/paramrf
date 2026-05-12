import dataclasses
from typing import Any, Callable, Generic, Literal, TypeVar, cast

# ==========================================
# 1. Type Definitions
# ==========================================
TRoot = TypeVar("TRoot")
PathOp = Literal["attr", "item"]
PathStep = tuple[PathOp, Any]


# ==========================================
# 2. The Traversal Class (Multi-Target)
# ==========================================
class Traversal(Generic[TRoot]):
    """
    Represents a multi-target focus within an immutable PyTree.

    Iteratively applies mutations across all targets, ensuring 
    rebuild invariants (`__init__` and `__post_init__`) are 
    maintained at every step.

    Parameters
    ----------
    tree : TRoot
        The root immutable object (e.g., Equinox module or dataclass).
    base_path : list[tuple[Literal["attr", "item"], Any]]
        The shared path from the root to the divergence point.
    sub_paths : list[list[tuple[Literal["attr", "item"], Any]]]
        A list of diverging paths, one for each targeted element.
    """
    def __init__(self, tree: TRoot, base_path: list[PathStep], sub_paths: list[list[PathStep]]) -> None:
        self._tree = tree
        self._base_path = base_path
        self._sub_paths = sub_paths

    def __getattr__(self, name: str) -> "Traversal[TRoot]":
        """
        Broadens the traversal by appending an attribute access 
        to every currently focused target.

        Parameters
        ----------
        name : str
            The name of the attribute to access on all targets.

        Returns
        -------
        Traversal[TRoot]
            A new Traversal focused one level deeper.
        """
        new_sub_paths = [path + [("attr", name)] for path in self._sub_paths]
        return Traversal(self._tree, self._base_path, new_sub_paths)

    def __getitem__(self, key: Any) -> "Traversal[TRoot]":
        """
        Broadens the traversal by appending an item/index access 
        to every currently focused target. Strings are treated as attributes.

        Parameters
        ----------
        key : Any
            The index, dictionary key, or attribute string to access.

        Returns
        -------
        Traversal[TRoot]
            A new Traversal focused one level deeper.
        """
        op: PathOp = "attr" if isinstance(key, str) else "item"
        new_sub_paths = [path + [(op, key)] for path in self._sub_paths]
        return Traversal(self._tree, self._base_path, new_sub_paths)

    def apply(self, func: Callable[[Any], Any]) -> TRoot:
        """
        Applies a function to all selected targets sequentially.

        Parameters
        ----------
        func : Callable[[Any], Any]
            The transformation function to apply to each target.

        Returns
        -------
        TRoot
            A new instance of the root tree with all targets updated.
        """
        current_tree = self._tree
        for path in self._sub_paths:
            # Rebuild sequentially to prevent overwriting parallel mutations
            current_tree = Lens(current_tree, self._base_path + path).apply(func)
        return current_tree

    def set(self, value: Any) -> TRoot:
        """
        Sets all selected targets to a specific value.

        Parameters
        ----------
        value : Any
            The new value to assign to all targets.

        Returns
        -------
        TRoot
            A new instance of the root tree with the updated values.
        """
        return self.apply(lambda _: value)

    def get(self) -> list[Any]:
        """
        Extracts all focused values.

        Returns
        -------
        list[Any]
            A list containing the values of all currently focused targets.
        """
        return [Lens(self._tree, self._base_path + path).get() for path in self._sub_paths]
    
    
    def filter(self, predicate: Callable[[Any], bool]) -> "Traversal[TRoot]":
        """
        Filters the currently focused targets, keeping only those that 
        match the condition.

        Parameters
        ----------
        predicate : Callable[[Any], bool]
            A function that returns True to keep a target, or False to drop it.

        Returns
        -------
        Traversal[TRoot]
            A new Traversal focused only on the targets that passed the filter.
        """
        filtered_paths = []
        for path in self._sub_paths:
            # Temporarily fetch the value at this specific path to test it
            target_val = Lens(self._tree, self._base_path + path).get()
            if predicate(target_val):
                filtered_paths.append(path)
                
        return Traversal(self._tree, self._base_path, filtered_paths)


# ==========================================
# 3. The Lens Class (Single-Target)
# ==========================================
class Lens(Generic[TRoot]):
    """
    A fluent interface for mutating immutable Equinox PyTrees.

    Rebuilds the tree from the bottom up to ensure `__init__` and 
    `__post_init__` are triggered for all parent nodes.

    Parameters
    ----------
    tree : TRoot
        The root immutable object to be mutated.
    path : list[tuple[Literal["attr", "item"], Any]], optional
        The current traversal path from the root, by default None.
    """
    _tree: TRoot
    _path: list[PathStep]

    def __init__(self, tree: TRoot, path: list[PathStep] | None = None) -> None:
        self._tree = tree
        self._path = path if path is not None else []

    def __getattr__(self, name: str) -> "Lens[TRoot]":
        """
        Focuses the lens on a named attribute.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        Lens[TRoot]
            A new Lens focused on the specified attribute.
        """
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return Lens(self._tree, self._path + [("attr", name)])

    def __getitem__(self, key: Any) -> "Lens[TRoot]":
        """
        Focuses the lens on a collection item, dictionary key, or attribute.

        If the key is a string, it is treated as an attribute access 
        (equivalent to `getattr`). Otherwise, it is treated as an item access.

        Parameters
        ----------
        key : Any
            The index, key, or attribute name.

        Returns
        -------
        Lens[TRoot]
            A new Lens focused on the specified item or attribute.
        """
        op: PathOp = "attr" if isinstance(key, str) else "item"
        return Lens(self._tree, self._path + [(op, key)])

    # --- The Bridge to Traversal ---
    
    def select(self, *names: str) -> Traversal[TRoot]:
        """
        Branches the Lens into a Traversal targeting multiple attributes.

        Parameters
        ----------
        *names : str
            A variable number of attribute names to target.

        Returns
        -------
        Traversal[TRoot]
            A Traversal object focused on the specified attributes.

        Examples
        --------
        >>> new_model = model.at.select('source_a', 'source_b').set(new_val)
        """
        sub_paths: list[list[PathStep]] = [[("attr", name)] for name in names]
        return Traversal(self._tree, self._path, sub_paths)

    def each(self) -> Traversal[TRoot]:
        """
        Transforms a focus on a collection into a Traversal of its elements.

        Returns
        -------
        Traversal[TRoot]
            A Traversal object focused on every item in the target collection.

        Raises
        ------
        TypeError
            If the current focus is not a dictionary, list, or tuple.

        Examples
        --------
        >>> new_model = model.at.sources.each().apply(lambda x: x * 2)
        """
        target = self._get_target()
        sub_paths: list[list[PathStep]] = []
        
        if isinstance(target, dict):
            sub_paths = [[("item", key)] for key in target.keys()]
        elif isinstance(target, (list, tuple)):
            sub_paths = [[("item", i)] for i in range(len(target))]
        else:
            raise TypeError(f"Cannot iterate over {type(target).__name__} with .each()")
            
        return Traversal(self._tree, self._path, sub_paths)

    def filter(self, predicate: Callable[[Any], bool]) -> Traversal[TRoot]:
        """
        Traverses all attributes of the current focus that match a condition.

        Parameters
        ----------
        predicate : Callable[[Any], bool]
            A function that takes an attribute value and returns True if it 
            should be included in the Traversal.

        Returns
        -------
        Traversal[TRoot]
            A Traversal object focused on all matching attributes.

        Examples
        --------
        >>> is_active = lambda s: getattr(s, 'is_active', False)
        >>> new_model = model.at.filter(is_active).apply(freeze)
        """
        target = self._get_target()
        sub_paths: list[list[PathStep]] = []
        
        for attr_name in dir(target):
            if not attr_name.startswith('_'):
                try:
                    val = getattr(target, attr_name)
                    if predicate(val):
                        sub_paths.append([("attr", attr_name)])
                except Exception:
                    pass  # Skip properties that error on read
                    
        return Traversal(self._tree, self._path, sub_paths)

    # --- Core Rebuild Logic ---

    def _get_target(self) -> Any:
        """Traverses the recorded path to return the current target node."""
        curr: Any = self._tree
        for op_type, val in self._path:
            if op_type == "attr":
                curr = getattr(curr, cast(str, val))
            elif op_type == "item":
                curr = curr[val]
        return curr

    def _rebuild(self, new_leaf: Any) -> TRoot:
        """Rebuilds the tree from the bottom up to maintain invariants."""
        if not self._path:
            return cast(TRoot, new_leaf)

        nodes = [self._tree]
        curr: Any = self._tree
        for op, key in self._path[:-1]:
            if op == "attr":
                curr = getattr(curr, cast(str, key))
            elif op == "item":
                curr = curr[key]
            nodes.append(curr)

        current_value = new_leaf
        
        for (op, key), parent in zip(reversed(self._path), reversed(nodes)):
            if op == "attr":
                if dataclasses.is_dataclass(parent):
                    current_value = dataclasses.replace(parent, **{key: current_value})
                elif hasattr(parent, "_replace"): 
                    current_value = parent._replace(**{key: current_value})
                else:
                    raise TypeError(
                        f"Cannot bottom-up rebuild {type(parent).__name__}. "
                        "Target must be an eqx.Module, dataclass, or NamedTuple."
                    )
            elif op == "item":
                if isinstance(parent, list):
                    new_parent_list = parent.copy()
                    new_parent_list[key] = current_value
                    current_value = new_parent_list
                elif isinstance(parent, dict):
                    new_parent_dict = parent.copy()
                    new_parent_dict[key] = current_value
                    current_value = new_parent_dict
                elif isinstance(parent, tuple):
                    new_parent_tuple = list(parent)
                    new_parent_tuple[key] = current_value
                    current_value = type(parent)(new_parent_tuple)
                else:
                    raise TypeError(f"Cannot bottom-up rebuild item for {type(parent).__name__}.")

        return cast(TRoot, current_value)

    # --- Manipulation & Retrieval Methods ---

    def get(self) -> Any:
        """
        Extracts the currently focused value.

        Returns
        -------
        Any
            The value at the end of the Lens path.
        """
        return self._get_target()

    def set(self, value: Any) -> TRoot:
        """
        Sets the focused target to a specific value.

        Parameters
        ----------
        value : Any
            The new value to assign.

        Returns
        -------
        TRoot
            A new instance of the root tree with the updated value.
        """
        return self._rebuild(value)

    def apply(self, func: Callable[[Any], Any]) -> TRoot:
        """
        Applies a transformation function to the focused target.

        Parameters
        ----------
        func : Callable[[Any], Any]
            The function to transform the current value.

        Returns
        -------
        TRoot
            A new instance of the root tree with the updated value.
        """
        return self._rebuild(func(self._get_target()))