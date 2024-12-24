# secondary utils, incidentally useful. Algorithms or objects.

from functools import reduce
import math


class RunEveryNth:
    def __init__(self, n, func):
        self.n = n
        self.func = func
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.n == 0:
            return self.func(*args, **kwargs)
        return None

def factorize(n):
    for i in range(math.isqrt(n), 0, -1):
        d, r = divmod(n, i)
        if r == 0:
            return (d, i)
        
def mean_of_dicts(dict_list):
    keys = dict_list[0].keys()
    total_dict = reduce(
        lambda acc, val: {key: acc.get(key, 0) + val.get(key, 0) for key in keys},
        dict_list,
    )
    mean_dict = {key: total_dict[key] / len(dict_list) for key in keys}

    return mean_dict
