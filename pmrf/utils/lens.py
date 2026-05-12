import dataclasses
from typing import Any, Callable, Generic, Literal, TypeVar, cast

# 1. Type Definitions
TRoot = TypeVar("TRoot")
PathOp = Literal["attr", "item"]
PathStep = tuple[PathOp, Any]


# 2. The Traversal Class (Multi-Target)
class Traversal(Generic[TRoot]):
    """
    Represents a multi-target focus within an immutable PyTree.
    Iteratively applies mutations across all targets, ensuring 
    rebuild invariants are maintained at every step.
    """
    def __init__(self, tree: TRoot, base_path: list[PathStep], steps: list[PathStep]) -> None:
        self._tree = tree
        self._base_path = base_path
        self._steps = steps  # Now holds generic path steps ("attr", name) or ("item", key)

    def apply(self, func: Callable[[Any], Any]) -> TRoot:
        """Applies a function to all selected targets sequentially."""
        current_tree = self._tree
        for step in self._steps:
            # We instantiate a temporary Lens on the *updated* tree 
            # to ensure we don't overwrite previous mutations.
            step_path = self._base_path + [step]
            current_tree = Lens(current_tree, step_path).apply(func)
        return current_tree

    def set(self, value: Any) -> TRoot:
        """Sets all selected targets to a specific value."""
        return self.apply(lambda _: value)

    def get(self) -> list[Any]:
        """Extracts all focused values as a list."""
        return [Lens(self._tree, self._base_path + [step]).get() for step in self._steps]


# 3. The Lens Class (Single-Target)
class Lens(Generic[TRoot]):
    """
    A fluent interface for mutating immutable Equinox PyTrees.
    Rebuilds the tree from the bottom up to ensure __init__ and 
    __post_init__ are triggered for all parent nodes.
    """
    _tree: TRoot
    _path: list[PathStep]

    def __init__(self, tree: TRoot, path: list[PathStep] | None = None) -> None:
        self._tree = tree
        self._path = path if path is not None else []

    def __getattr__(self, name: str) -> "Lens[TRoot]":
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return Lens(self._tree, self._path + [("attr", name)])

    def __getitem__(self, key: Any) -> "Lens[TRoot]":
        return Lens(self._tree, self._path + [("item", key)])

    # --- The Bridge to Traversal ---
    
    def select(self, *names: str) -> Traversal[TRoot]:
        """
        Branches the Lens into a Traversal targeting multiple attributes.
        Example: self.at.select('source_a', 'source_b').apply(func)
        """
        steps: list[PathStep] = [("attr", name) for name in names]
        return Traversal(self._tree, self._path, steps)

    def each(self) -> Traversal[TRoot]:
        """
        Transforms a focus on a collection into a Traversal of its elements.
        Example: self.at.sources.each().apply(func)
        """
        target = self._get_target()
        steps: list[PathStep] = []
        
        if isinstance(target, dict):
            steps = [("item", key) for key in target.keys()]
        elif isinstance(target, (list, tuple)):
            steps = [("item", i) for i in range(len(target))]
        else:
            raise TypeError(f"Cannot iterate over {type(target).__name__} with .each()")
            
        return Traversal(self._tree, self._path, steps)

    def filter(self, predicate: Callable[[Any], bool]) -> Traversal[TRoot]:
        """
        Traverses all attributes of the current focus that match a condition.
        Example: self.at.filter(lambda s: s.is_active).apply(func)
        """
        target = self._get_target()
        steps: list[PathStep] = []
        
        for attr_name in dir(target):
            if not attr_name.startswith('_'):
                try:
                    val = getattr(target, attr_name)
                    if predicate(val):
                        steps.append(("attr", attr_name))
                except Exception:
                    pass # Skip properties that error on read
                    
        return Traversal(self._tree, self._path, steps)

    # --- Core Rebuild Logic ---

    def _get_target(self) -> Any:
        curr: Any = self._tree
        for op_type, val in self._path:
            if op_type == "attr":
                curr = getattr(curr, cast(str, val))
            elif op_type == "item":
                curr = curr[val]
        return curr

    def _rebuild(self, new_leaf: Any) -> TRoot:
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
        """Extracts the focused value."""
        return self._get_target()

    def set(self, value: Any) -> TRoot:
        return self._rebuild(value)

    def apply(self, func: Callable[[Any], Any]) -> TRoot:
        return self._rebuild(func(self._get_target()))