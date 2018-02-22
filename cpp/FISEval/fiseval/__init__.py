import ctypes as C
import os

import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

_fiseval_wrapper = C.cdll.LoadLibrary(
    os.path.join(PARENT_DIR, "cpp/build/fiseval_wrapper.so"))


def run_func_in_c():
    return _fiseval_wrapper.cffi_hello()


def mul_np_array(arr, scaler):
    f = _fiseval_wrapper.c_mul_np_array

    f.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        C.c_int, C.c_int]

    f(arr, arr.shape[0], scaler)

    return arr
