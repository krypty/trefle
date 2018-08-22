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
  FISCocoEvalWrapper(const int n_vars, const int n_rules,
                     const int n_max_vars_per_rule, const int n_bits_per_mf,
                     const int n_true_labels, const int n_lv_per_ind,
                     const int n_bits_per_ant, const int n_cons,
                     const int n_bits_per_cons, const int n_bits_per_label)
      : n_vars(n_vars), n_rules(n_rules),
        n_max_vars_per_rule(n_max_vars_per_rule), n_bits_per_mf(n_bits_per_mf),
        n_true_labels(n_true_labels), n_lv_per_ind(n_lv_per_ind),
        n_bits_per_ant(n_bits_per_ant), n_cons(n_cons),
        n_bits_per_cons(n_bits_per_cons), n_bits_per_label(n_bits_per_label) {
    cout << "hello from FISCocoEvalWrapper " << n_bits_per_mf << ", "
         << n_true_labels << ", " << n_lv_per_ind << endl;
  }
  py::array_t<double> predict_c(const string &ind_sp1, const string &ind_sp2);

private:
  py::array_t<double> parse_ind_sp1(const string &ind_sp1);

  template <typename T>
  vector<vector<T>> parse_bit_array(
      const string &bitarray, const size_t rows, const size_t cols,
      const size_t n_bits_per_elm,
      const std::function<T(const T, const size_t row, const size_t col)> &);

  // TODO remove me
  static double toto(const double v, const size_t i, const size_t j) {
    return v * 100.0;
  };

  inline size_t modulo_trick(const size_t value, const size_t divisor) {
    return value % divisor;
  }

private:
  const int n_vars;
  const int n_rules;
  const int n_max_vars_per_rule;
  const int n_bits_per_mf;
  const int n_true_labels;
  const int n_lv_per_ind;
  const int n_bits_per_ant;
  const int n_cons;
  const int n_bits_per_cons;
  const int n_bits_per_label;
};

PYBIND11_MODULE(pyfuge_c, m) {
  py::class_<FISCocoEvalWrapper>(m, "FISCocoEvalWrapper")
      // match the ctor of FISCocoEvalWrapper
      .def(py::init<const int, const int, const int, const int, const int,
                    const int, const int, const int, const int, const int>())
      .def("bind_predict", &FISCocoEvalWrapper::predict_c,
           "a function that use predict");
}

#endif // FISEVAL_BINDINGS_H
