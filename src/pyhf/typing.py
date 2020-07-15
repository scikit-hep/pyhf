import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

# from typing import Any, Iterable
from . import get_backend

tensorlib, _ = get_backend()


class PDF(Protocol):
    def expected_data(self) -> tensorlib.tensor:
        ...

    def log_prob(self, value: float) -> float:
        ...
