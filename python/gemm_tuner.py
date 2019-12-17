from collections import namedtuple

import scipy.optimize as sciopt

import tuner
import numpy as np
np.set_printoptions(linewidth=1000, precision=3)

# tuner.get_time_for(
#   cache_size
#   item_rows
#   item_cols
#   wg_rows
#   wg_cols
#   tile_rows
#   tile_cols
#   bank_conf_a
#   bank_conf_b
#   mem_type
#   algo_type
#   trans_a
#   trans_b
#   m
#   k
#   n
#   batch
# )

CompileArgs = namedtuple('CompileArgs', [
    'cache_size',
    'item_rows',
    'item_cols',
    'wg_rows',
    'wg_cols',
    'tile_rows',
    'tile_cols',
    'double_buffer',
    'bank_conf_a',
    'bank_conf_b',
])

AlgoArgs = namedtuple('AlgoArgs', ['mem_type', 'algo_type'])

RuntimeArgs = namedtuple(
    'RuntimeArgs', ['transpose_a', 'transpose_b', 'm', 'k', 'n', 'batch'])

_BLAS_MEM_LOCAL = 0
_BLAS_MEM_NOLOCAL = 1

_BLAS_ALGO_NAIVE = 0
_BLAS_ALGO_STANDARD = 1
_BLAS_ALGO_SKINNY = 2

LocalGemm = AlgoArgs(mem_type=_BLAS_MEM_LOCAL, algo_type=_BLAS_ALGO_STANDARD)
NonLocalGemm = AlgoArgs(mem_type=_BLAS_MEM_NOLOCAL,
                        algo_type=_BLAS_ALGO_STANDARD)

_CACHE_MAP = {
    0: 32,
    1: 64,
    2: 128,
    3: 256,
}

_TILE_MAP = {
    0: 1,
    1: 2,
    2: 4,
    3: 8,
    4: 16,
}
_BOOL_MAP = {
    0: False,
    1: True,
}

LocalGemmBounds = CompileArgs(
    cache_size=(1, 3.99),
    item_rows=(0, 4.99),
    item_cols=(0, 4.99),
    wg_rows=(0, 4.99),
    wg_cols=(0, 4.99),
    tile_rows=(0, 0),
    tile_cols=(0, 0),
    double_buffer=(0, 1.99),
    bank_conf_a=(0, 1.99),
    bank_conf_b=(0, 1.99),
)

NoLocalGemmBounds = CompileArgs(
    cache_size=(0, 0),
    item_rows=(0, 5),
    item_cols=(0, 5),
    wg_rows=(0, 5),
    wg_cols=(0, 5),
    tile_rows=(0, 0),
    tile_cols=(0, 0),
    double_buffer=(0, 0),
    bank_conf_a=(0, 0),
    bank_conf_b=(0, 0),
)


def numeric_to_func_args(tuple_):
    def _map_double(val, map_):
        try:
            return map_[int(val)]
        except KeyError:
            if val < 0:
                return map_[0]
            return max(list(map_.keys()))


    def _get_tile(val):
        return _map_double(val, _TILE_MAP)

    def _get_bool(val):
        return _map_double(val, _BOOL_MAP)

    def _get_cache(val):
        return _map_double(val, _CACHE_MAP)

    cache_size = _get_cache(tuple_[0])
    item_rows = _get_tile(tuple_[1])
    item_cols = _get_tile(tuple_[2])
    wg_rows = _get_tile(tuple_[3])
    wg_cols = _get_tile(tuple_[4])
    tile_rows = _get_tile(tuple_[5])
    tile_cols = _get_tile(tuple_[6])
    double_buffer = _get_bool(tuple_[7])
    bank_conf_a = _get_bool(tuple_[8])
    bank_conf_b = _get_bool(tuple_[9])
    return CompileArgs(
        cache_size,
        item_rows,
        item_cols,
        wg_rows,
        wg_cols,
        tile_rows,
        tile_cols,
        double_buffer,
        bank_conf_a,
        bank_conf_b,
    )


def get_kernel_time(c_args, a_args, r_args):
    args = list(c_args) + list(a_args) + list(r_args)
    return tuner.get_time_for(*args)


class CachedCall:
    def __init__(self, callable_):
        self._callable = callable_
        self._cache = {}

    def __call__(self, *args):
        if args in self._cache:
            return self._cache[args]
        else:
            result = self._callable(*args)
            self._cache[args] = result
            return result


class GetKernelTimeCall:
    def __init__(self, callable_):
        self._callable = callable_

    def __call__(self, *args):
        a_args = args[1]
        r_args = args[2]
        converted_args = numeric_to_func_args(args[0])
        return self._callable(converted_args, a_args, r_args)


def find_config_for(a_args, r_args):
    initial_guess = CompileArgs(2, 2, 2, 3, 3, 0, 0, 0, 0, 0)
    func = GetKernelTimeCall(CachedCall(get_kernel_time))
    res = sciopt.basinhopping(func,
                              initial_guess,
                              stepsize=2.0,
                              minimizer_kwargs={
                                  'args': (a_args, r_args),
                                  'bounds': LocalGemmBounds,
                                  'options': {
                                      'eps': 1.0,
                                      'maxls': 120,
                                      },
                              })
    all_res = func._callable._cache
    for item in all_res:
        time = all_res[item]
        if time < 10000:
            print('{}\n    {}\n\n'.format(item[0], time))

    best_args = numeric_to_func_args(res.x)
    print('Best:\n{}\n{}'.format(best_args, res.fun))
    return res
