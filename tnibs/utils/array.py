from typing import Iterable, List
import numpy as np
import pandas as pd
import polars as pl
import torch

# utils for working with array types

def to_nplist(x, dim=1, copy=True):
    x = x.numpy() if isinstance(x, torch.Tensor) else x
    return x.flatten() if copy else x.view(-1)


def to_list(x, force=False, level=1):
    def k_level(obj, level):
        # this is dumb, we are adding a way to force non "iterable" into a list of a specific depth, but its just a utility.
        test = obj
        while level > 0 and isinstance(test, list):
            if len(test) == 0:
                level -= 1
                break
            test = test[0]
            level -= 1
        for _ in range(level):
            obj = [obj]
        return obj
    
    if force:
        return k_level(x, level)

    # ignore the level parameter
    if isinstance(x, pl.Series):
        return x.to_list()
    elif isinstance(x, pd.Series):
        return x.to_list()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, torch.Tensor):
        return x.tolist()
    elif isinstance(x, Iterable) and not isinstance(x, str):
        return list(x)
    else:
        return k_level(x, level)


def to_tensors(arr, **kwargs):
    """Coerce into Tensor"""
    if isinstance(arr, torch.Tensor):
        return arr.to(**kwargs)
    kwargs.setdefault("dtype", torch.float32)  # np is 64 but torch is 32
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr, **kwargs)
    if isinstance(arr, pl.DataFrame):
        return arr.to_torch(**kwargs)
    if isinstance(arr, pd.DataFrame):
        return torch.tensor(arr.values, **kwargs)


def row_index(arr, indices):
    """Index rows of various types

    Args:
        arr: array_type or table
        indices

    Returns:
        array_type
    """
    if isinstance(arr, pd.DataFrame):
        return arr.iloc[indices, :].values
    elif isinstance(arr, pl.DataFrame):
        try:
            return arr.to_torch(dtype=pl.Float32)[indices]
        except pl.exceptions.PolarsError:  # mixed requires setting correct pl types i.e. int64, float32, todo: actual error
            return arr.to_torch()[indices]
    elif isinstance(arr, torch.Tensor):
        return arr[indices]
    else:
        return np.array(arr)[indices]

def to_permutation(lst: List[int]):
    """Given a list of indices, gets the permutation associated to their order. The result will be a tuple of len(lst) with indices in range(len(lst))

    Args:
        lst (List):

    """
    sorted_lst = sorted(lst)
    value_to_index = {v: i for i, v in enumerate(sorted_lst)}

    n = len(lst)
    result = [0] * n
    for i in range(n):
        result[value_to_index[lst[i]]] = i

    return result

# unused
def inverse_permute_tensor(lst: List[int], tensor=()):
    """Given a list of indices, computes the inverse permutation and applies it to the tensor.

    Args:
        lst (List)

    Returns:
        _type_: Corresponding inverse permutation
    """
    m = len(tensor)
    lst = [idx if idx >= 0 else m + idx for idx in lst]

    inverse_permutation = [None] * m
    for i in range(len(lst)):
        inverse_permutation[lst[i]] = i
    c = len(lst)
    for i, e in enumerate(inverse_permutation):
        if e is None:
            inverse_permutation[i] = c
            c += 1

    return tuple(inverse_permutation)

