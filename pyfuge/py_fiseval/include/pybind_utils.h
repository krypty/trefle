#ifndef PYBIND_UTILS_H
#define PYBIND_UTILS_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename T>
using py_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T, typename U>
void np_arr1d_to_vec(py_array<T> np_arr, std::vector<U> &arr, size_t n) {
        auto arr_buf = np_arr.request();
        auto ptr_arr = (int *)(arr_buf.ptr);
        arr.assign(ptr_arr, ptr_arr + n);
}

// caller must init `arr` with already n rows
template <typename T, typename U>
void np_arr2d_to_vec2d(py_array<T> np_arr, std::vector<std::vector<U>> &arr) {
        size_t rows = np_arr.shape(0);
        size_t cols = np_arr.shape(1);

        auto arr_buf = np_arr.request();
        auto ptr_arr = (double *)(arr_buf.ptr);

        for (size_t i = 0; i < rows; i++) {
                auto offset = ptr_arr + (i * cols);
                arr[i].assign(offset, offset + cols);
        }
}

template <typename T>
py_array<T> vec2d_to_np_vec2d(const std::vector<std::vector<T>> &arr) {
        const size_t rows = arr.size(), cols = arr[0].size();
        auto np_arr = py_array<T>({rows, cols});
        auto np_arr_raw = np_arr.mutable_unchecked();

        for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                        np_arr_raw(i, j) = arr[i][j];
                }
        }
        return np_arr;
}

#endif // PYBIND_UTILS_H
