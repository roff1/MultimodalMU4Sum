"""Generic name -> callable registry with decorator-based registration."""
from typing import Callable, Dict


class Registry:
    """Name -> callable map used to plug in processors, downloaders, metrics, unlearning methods."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str):
        """Return a decorator that registers the object under name; raise ValueError on duplicates."""
        def decorator(obj):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str):
        """Return the object registered under name; raise KeyError listing the available names."""
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        """Return the registered names in sorted order."""
        return sorted(self._items)
