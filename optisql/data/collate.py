from typing import Any, Callable


class BatchCollator:
    """Simple collator that defers tensorization to the model/processor."""

    def __init__(self, processor: Callable[..., Any]) -> None:
        self.processor = processor

    def __call__(self, batch: list[Any]) -> Any:
        return self.processor(batch)
