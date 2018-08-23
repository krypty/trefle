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
                     const int n_bits_per_cons, const int n_bits_per_label,
                     const int dc_weight, py_array_i np_cons_n_labels)
      : n_vars(n_vars), n_rules(n_rules),
        n_max_vars_per_rule(n_max_vars_per_rule), n_bits_per_mf(n_bits_per_mf),
        n_true_labels(n_true_labels), n_lv_per_ind(n_lv_per_ind),
        n_bits_per_ant(n_bits_per_ant), n_cons(n_cons),
        n_bits_per_cons(n_bits_per_cons), n_bits_per_label(n_bits_per_label),
        dc_weight(dc_weight), cons_n_labels(n_cons, 0) {
    cout << "hello from FISCocoEvalWrapper " << n_bits_per_mf << ", "
         << n_true_labels << ", " << n_lv_per_ind << endl;
    auto cons_n_labels_buf = np_cons_n_labels.request();
    auto ptr_cons_n_labels = (int *)(cons_n_labels_buf.ptr);
    cons_n_labels.assign(ptr_cons_n_labels, ptr_cons_n_labels + n_cons);

    for (int i = 0; i < cons_n_labels.size(); i++) {
      cout << "cons n labels " << cons_n_labels[i] << endl;
    }
  }
  py::array_t<double> predict_c(const string &ind_sp1, const string &ind_sp2);

private:
  py::array_t<double> parse_ind_sp1(const string &ind_sp1);

  template <typename T>
  vector<vector<T>> parse_bit_array(
      const string &bitarray, const size_t rows, const size_t cols,
      const size_t n_bits_per_elm,
      const std::function<T(const size_t v, const size_t row, const size_t col)>
          &post_func);

  template <typename T>
  static T dummy_post_func(const size_t v, const size_t i, const size_t j) {
    // this function is used when there is no need to do any post processing
    return v;
  }

  inline size_t modulo_trick(const size_t value, const size_t divisor) {
    return value % divisor;
  }

  static inline double scale0N(const double v, const size_t n_classes,
                               const double min_v, const double max_v) {
    // This function scales v from [0]
    // assumption v's minimum is 0
    return ((max_v - min_v) / n_classes) * v + min_v;
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
  const int dc_weight;
  vector<int> cons_n_labels;
};

PYBIND11_MODULE(pyfuge_c, m) {
  py::class_<FISCocoEvalWrapper>(m, "FISCocoEvalWrapper")
      // match the ctor of FISCocoEvalWrapper
      .def(py::init<const int, const int, const int, const int, const int,
                    const int, const int, const int, const int, const int,
                    const int, py_array_i>())
      .def("bind_predict", &FISCocoEvalWrapper::predict_c,
           "a function that use predict");
}

#endif // FISEVAL_BINDINGS_H
