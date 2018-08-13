#ifndef FISEVAL_BINDINGS_H
#define FISEVAL_BINDINGS_H

#include "fis.h"
#include "trilv.h"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

using namespace std;

typedef py::array_t<double, py::array::c_style | py::array::forcecast>
    py_array_d;
typedef py::array_t<float, py::array::c_style | py::array::forcecast>
    py_array_f;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> py_array_i;

class FISCocoEvalWrapper {
public:
  FISCocoEvalWrapper(const int n_bits_per_mf, const int n_true_labels,
                     const int n_lv_per_ind)
      : n_bits_per_mf(n_bits_per_mf), n_true_labels(n_true_labels),
        n_lv_per_ind(n_lv_per_ind) {
    cout << "hello from FISCocoEvalWrapper " << n_bits_per_mf << ", "
         << n_true_labels << ", " << n_lv_per_ind << endl;
  }
  py::array_t<double> predict_c(const string &ind_sp1, const string &ind_sp2);

private:
  py::array_t<double> parse_ind_sp1(const string &ind_sp1);

private:
  const int n_bits_per_mf;
  const int n_true_labels;
  const int n_lv_per_ind;
};

PYBIND11_MODULE(pyfuge_c, m) {
  py::class_<FISCocoEvalWrapper>(m, "FISCocoEvalWrapper")
      // match the ctor of FISCocoEvalWrapper
      .def(py::init<const int, const int, const int>())
      .def("bind_predict", &FISCocoEvalWrapper::predict_c,
           "a function that use predict");
}

#endif // FISEVAL_BINDINGS_H
