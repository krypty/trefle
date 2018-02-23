import ctypes as C
import os
from collections import OrderedDict

import numpy as np
from numpy.ctypeslib import ndpointer

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

_fiseval_wrapper = C.cdll.LoadLibrary(
    os.path.join(PARENT_DIR, "cpp/build/fiseval_wrapper.so"))


def run_func_in_c():
    return _fiseval_wrapper.cffi_hello()


def mul_np_array(arr, scaler):
    f = _fiseval_wrapper.c_mul_np_array

    f.argtypes = [
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        C.c_int, C.c_int]

    f(arr, arr.shape[0], scaler)

    return arr


def _get_np_array_2d_ptr(arr, ctype_type):
    ptr_type = C.POINTER(ctype_type)
    n_rows = len(arr)
    return (ptr_type * n_rows)(*[row.ctypes.data_as(ptr_type) for row in arr])


def _get_array_1d_ptr(arr, ctype_type):
    return (ctype_type * len(arr))(*arr)


def predict_native(ind, observations, n_rules, max_vars_per_rule, n_labels,
                   n_consequents, default_rule_cons, vars_ranges,
                   labels_weights,
                   dc_idx):
    f = _fiseval_wrapper.c_predict

    # observations = np.arange(6).reshape(3, 2).astype(np.float64)
    # print(observations)

    # kwargs = OrderedDict([
    #     ("ind", C.POINTER(C.c_float)),
    #     ("ind_n", C.c_int),
    #     ("observations", ndpointer(dtype=np.float64, ndim=2, flags='C')),
    #     ("observations_n", C.c_int),
    #     ("observations_m", C.c_int),
    # ])

    # f.argtypes = kwargs.values()

    kwargs = OrderedDict([
        ("ind", _get_array_1d_ptr(ind, C.c_float)),
        ("ind_n", len(ind)),
        ("observations", _get_np_array_2d_ptr(observations, C.c_double)),
        ("observations_n", observations.shape[0]),
        ("observations_m", observations.shape[1]),
    ])

    print("shape", observations.shape)
    # res = f(c_ind, len(ind))

    f.restype = C.c_float
    res = f(*kwargs.values())  # cannot pass keyword args to native function
    print("res", res)

    print("py res", np.sum(observations[:, 0]))

    from time import sleep
    sleep(0.1)
    assert False, "trololo"

    predicted_outputs = np.array([])
    return predicted_outputs
