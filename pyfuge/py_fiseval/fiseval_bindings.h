#ifndef FISEVAL_BINDINGS_H
#define FISEVAL_BINDINGS_H

#include "default_fuzzy_rule.h"
#include "fis.h"
#include "fuzzy_rule.h"
#include "linguisticvariable.h"
#include "singleton_fis.h"
#include "trilv.h"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>

namespace py = pybind11;

using namespace std;

typedef py::array_t<double, py::array::c_style | py::array::forcecast>
    py_array_d;
typedef py::array_t<float, py::array::c_style | py::array::forcecast>
    py_array_f;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> py_array_i;

template <typename T>
using py_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

class FISCocoEvalWrapper {
public:
  FISCocoEvalWrapper(py_array_d np_X_train, const int n_vars, const int n_rules,
                     const int n_max_vars_per_rule, const int n_bits_per_mf,
                     const int n_true_labels, const int n_bits_per_lv,
                     const int n_bits_per_ant, const int n_cons,
                     const int n_bits_per_cons, const int n_bits_per_label,
                     const int dc_weight, py_array_i np_cons_n_labels,
                     py_array_i np_default_cons)
      : X_train(np_X_train.shape(0)), n_vars(n_vars), n_rules(n_rules),
        n_max_vars_per_rule(n_max_vars_per_rule), n_bits_per_mf(n_bits_per_mf),
        n_true_labels(n_true_labels), dc_idx(n_true_labels),
        n_bits_per_lv(n_bits_per_lv), n_lv_per_ind(1 << n_bits_per_lv),
        n_bits_per_ant(n_bits_per_ant), n_cons(n_cons),
        n_bits_per_cons(n_bits_per_cons), n_bits_per_label(n_bits_per_label),
        dc_weight(dc_weight), cons_n_labels(n_cons, 0) {
    // cout << "hello from FISCocoEvalWrapper " << n_bits_per_mf << ", "
    // << n_true_labels << ", " << n_bits_per_lv << endl;

    np_arr1d_to_vec(np_cons_n_labels, cons_n_labels, n_cons);

    // for (int i = 0; i < cons_n_labels.size(); i++) {
    //   cout << "cons n labels " << cons_n_labels[i] << endl;
    // }

    np_arr2d_to_vec2d(np_X_train, X_train);

    np_arr1d_to_vec(np_default_cons, default_cons, n_cons);
  }
  py::array_t<double> predict_c(const string &ind_sp1, const string &ind_sp2);
  py::array_t<double> predict_c_other(const string &ind_sp1,
                                      const string &ind_sp2,
                                      py_array_d other_X);
  void print_ind(const string &ind_sp1, const string &ind_sp2);

private:
  SingletonFIS extract_fis(const string &ind_sp1, const string &ind_sp2);
  py::array_t<double> predict(const string &ind_sp1, const string &ind_sp2,
                              const vector<vector<double>> &observations);

private:
  vector<LinguisticVariable> parse_ind_sp1(const string &ind_sp1);

  vector<vector<size_t>> extract_sel_vars(const string &ind_sp2,
                                          size_t &offset);
  vector<vector<size_t>> extract_r_lv(const string &ind_sp2, size_t &offset);
  vector<vector<size_t>> extract_r_labels(const string &ind_sp2,
                                          size_t &offset);
  vector<vector<double>> extract_r_cons(const string &ind_sp2, size_t &offset);

  template <typename T>
  vector<vector<T>> parse_bit_array(
      const string &bitarray, const size_t rows, const size_t cols,
      const size_t n_bits_per_elm,
      const std::function<T(const size_t v, const size_t row, const size_t col)>
          &post_func);

  template <typename T>
  inline static T dummy_post_func(const size_t v, const size_t i,
                                  const size_t j) {
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

  bool are_all_labels_dc(const vector<size_t> labels_for_a_rule);

  FuzzyRule
  build_fuzzy_rule(const vector<size_t> &vars_rule_i,
                   const unordered_map<size_t, size_t> &vars_lv_lookup,
                   const vector<LinguisticVariable> &vec_lv,
                   const vector<size_t> &r_labels_ri,
                   const vector<double> cons_ri);

  template <typename T, typename U>
  void np_arr1d_to_vec(py_array<T> np_arr, vector<U> &arr, size_t n) {
    auto arr_buf = np_arr.request();
    auto ptr_arr = (int *)(arr_buf.ptr);
    arr.assign(ptr_arr, ptr_arr + n);
  }

  // caller must init `arr` with already n rows
  template <typename T, typename U>
  void np_arr2d_to_vec2d(py_array<T> np_arr, vector<vector<U>> &arr) {
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
  py_array<T> vec2d_to_np_vec2d(const vector<vector<T>> &arr) {
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

private:
  vector<vector<double>> X_train;
  const int n_vars;
  const int n_rules;
  const int n_max_vars_per_rule;
  const int n_bits_per_mf;
  const int n_true_labels;

  // it has been decided that the dc_idx is the last index
  // in [0, n_true_labels] i.e. [0, n_true_labels-1] for low, medium, ...
  // and {n_true_label} for dc.
  const size_t dc_idx;

  const int n_bits_per_lv;
  const int n_lv_per_ind;
  const int n_bits_per_ant;
  const int n_cons;
  const int n_bits_per_cons;
  const int n_bits_per_label;
  const int dc_weight;
  vector<int> cons_n_labels;
  vector<double> default_cons;
};

PYBIND11_MODULE(pyfuge_c, m) {
  py::class_<FISCocoEvalWrapper>(m, "FISCocoEvalWrapper")
      // match the ctor of FISCocoEvalWrapper
      .def(py::init<py_array_d, const int, const int, const int, const int,
                    const int, const int, const int, const int, const int,
                    const int, const int, py_array_i, py_array_i>())
      .def("bind_predict", &FISCocoEvalWrapper::predict_c,
           "a function that use predict")
      .def("bind_predict", &FISCocoEvalWrapper::predict_c_other,
           "a function that use predict")
      .def("print_ind", &FISCocoEvalWrapper::print_ind,
           "pretty print an individual couple");
}

#endif // FISEVAL_BINDINGS_H
