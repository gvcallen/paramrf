from typing import Any, Callable
import equinox as eqx

class Lens:
    """
    A fluent interface for mutating immutable Equinox PyTrees.
    Records attribute and item access to dynamically build a selector 
    function for equinox.tree_at.
    """
    def __init__(self, tree: Any, path: list[tuple[str, Any]] = None):
        self._tree = tree
        self._path = path or []

    def __getattr__(self, name: str) -> 'Lens':
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return Lens(self._tree, self._path + [('attr', name)])

    def __getitem__(self, key: Any) -> 'Lens':
        return Lens(self._tree, self._path + [('item', key)])

    def _get_target(self) -> Any:
        """Traverses the recorded path to return the current target node."""
        curr = self._tree
        for op_type, val in self._path:
            if op_type == 'attr':
                curr = getattr(curr, val)
            elif op_type == 'item':
                curr = curr[val]
        return curr

    def _get_selector(self) -> Callable[[Any], Any]:
        """Compiles the recorded path into a lambda function for eqx.tree_at."""
        path = self._path
        def selector(tree):
            curr = tree
            for op_type, val in path:
                if op_type == 'attr':
                    curr = getattr(curr, val)
                elif op_type == 'item':
                    curr = curr[val]
            return curr
        return selector

    def __dir__(self) -> list[str]:
        """
        Powers Jupyter/IPython autocomplete (IntelliSense). 
        Returns Lens methods AND the current target node's attributes.
        """
        # Start with the Lens's native methods (set, apply, add, etc.)
        attributes = set(super().__dir__())
        
        try:
            # Resolve the current node in the tree
            target = self._get_target()
            
            # Add the target node's attributes to the autocomplete list
            # We filter out dunder methods to keep the dropdown clean
            attributes.update(
                attr for attr in dir(target) 
                if not attr.startswith('__')
            )
        except Exception:
            # If resolution fails mid-typing (e.g., incomplete array index), 
            # just fail gracefully and return the standard Lens methods.
            pass
            
        return sorted(list(attributes))

    # --- Modular Manipulation Methods ---

    def set(self, value: Any) -> Any:
        return eqx.tree_at(self._get_selector(), self._tree, value)

    def apply(self, func: Callable[[Any], Any]) -> Any:
        return eqx.tree_at(self._get_selector(), self._tree, replace_fn=func)

    def add(self, value: Any) -> Any:
        return self.apply(lambda x: x + value)

    def multiply(self, value: Any) -> Any:
        return self.apply(lambda x: x * value)