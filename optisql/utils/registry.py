from typing import Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"Registry has no entry named '{name}'")
        return self._items[name]

    def items(self) -> Dict[str, T]:
        return dict(self._items)
