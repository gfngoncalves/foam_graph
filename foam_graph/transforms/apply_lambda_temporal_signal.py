import torch
from collections import deque
from itertools import islice
from typing import Callable, Iterable
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def _sliding_window(iterable, n):
    """Return a sliding window of width *n* over *iterable*.

        >>> list(sliding_window(range(6), 4))
        [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]

    If *iterable* has fewer than *n* items, then nothing is yielded:

        >>> list(sliding_window(range(3), 4))
        []

    For a variant with more features, see :func:`windowed`.
    """
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


class ApplyLambdaTemporalSignal(object):
    def __init__(
        self,
        attr: str,
        func: Callable[[Iterable[Data]], torch.Tensor],
        n_past: int = 0,
        n_future: int = 0,
    ):
        self.func = func
        self.n_past = n_past
        self.n_future = n_future
        self.attr = attr

    def __call__(self, data):
        n = 1 + self.n_past + self.n_future
        y = [
            self.func(data_window).detach().numpy()
            for data_window in _sliding_window(data, n)
        ]
        n_out = -self.n_future + 1 if self.n_future >= 1 else None

        data_out = StaticGraphTemporalSignal(
            data.edge_index,
            data.edge_weight,
            data.features[self.n_past : n_out],
            data.targets[self.n_past : n_out],
            **{
                key: getattr(data, key)[self.n_past : n_out]
                for key in data.additional_feature_keys
            }
        )
        data_out.__setattr__(self.attr, y)
        return data_out

    def __repr__(self):
        return "{}(attr={}, n_past={}, n_future={})".format(
            self.__class__.__name__, self.attr, self.n_past, self.n_future
        )
