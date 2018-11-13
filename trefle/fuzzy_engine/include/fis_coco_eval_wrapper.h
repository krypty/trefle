#ifndef FISEVAL_BINDINGS_H
#define FISEVAL_BINDINGS_H

#include "default_fuzzy_rule.h"
#include "fuzzy_rule.h"
#include "json_fis_reader.h"
#include "json_fis_writer.h"
#include "linguisticvariable.h"
#include "observations_scaler.h"
#include "predictions_scaler.h"
#include "pybind_utils.h"
#include "singleton_fis.h"
#include "tff_fis_writer.h"
#include "trefle_fis.h"
#include "trilv.h"
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>

class FISCocoEvalWrapper {
public:
  FISCocoEvalWrapper(py_array<double> np_X_train, const int n_vars,
                     const int n_rules, const int n_max_vars_per_rule,
                     const int n_bits_per_mf, const int n_true_labels,
                     const int n_bits_per_lv, const int n_bits_per_ant,
                     const int n_cons, const int n_bits_per_cons,
                     const int n_bits_per_label, const int dc_weight,
                     py_array<int> np_cons_n_labels,
                     py_array<int> np_n_classes_per_cons,
                     py_array<int> np_default_cons,
                     py_array<double> np_vars_range,
                     py_array<double> np_cons_range)
      : X_train(np_X_train.shape(0)), n_vars(n_vars), n_rules(n_rules),
        n_max_vars_per_rule(n_max_vars_per_rule), n_bits_per_mf(n_bits_per_mf),
        n_true_labels(n_true_labels), dc_idx(n_true_labels),
        n_bits_per_lv(n_bits_per_lv), n_lv_per_ind(1 << n_bits_per_lv),
        n_bits_per_ant(n_bits_per_ant), n_cons(n_cons),
        n_bits_per_cons(n_bits_per_cons), n_bits_per_label(n_bits_per_label),
        dc_weight(dc_weight), n_classes_per_cons(n_cons, 0),
        cons_n_labels(n_cons, 0), vars_range(np_vars_range.shape(0)),
        cons_range(np_cons_range.shape(0)) {

    np_arr1d_to_vec(np_cons_n_labels, cons_n_labels, n_cons);
    np_arr1d_to_vec(np_n_classes_per_cons, n_classes_per_cons, n_cons);

    np_arr2d_to_vec2d(np_X_train, X_train);

    np_arr1d_to_vec(np_default_cons, default_cons, n_cons);

    np_arr2d_to_vec2d(np_vars_range, vars_range);
    np_arr2d_to_vec2d(np_cons_range, cons_range);
  }
  py::array_t<double> predict_c(const string &ind_sp1, const string &ind_sp2);
  py::array_t<double> predict_c_other(const string &ind_sp1,
                                      const string &ind_sp2,
                                      py_array<double> other_X);
  void print_ind(const string &ind_sp1, const string &ind_sp2);

  string to_tff(const string &ind_sp1, const string &ind_sp2);

private:
  SingletonFIS extract_fis(const string &ind_sp1, const string &ind_sp2);
  py::array_t<double>
  predict(const string &ind_sp1, const string &ind_sp2,
          const std::vector<std::vector<double>> &observations);

private:
  std::vector<LinguisticVariable> parse_ind_sp1(const string &ind_sp1);

  std::vector<std::vector<size_t>> extract_sel_vars(const string &ind_sp2,
                                                    size_t &offset);
  std::vector<std::vector<size_t>> extract_r_lv(const string &ind_sp2,
                                                size_t &offset);
  std::vector<std::vector<size_t>> extract_r_labels(const string &ind_sp2,
                                                    size_t &offset);
  std::vector<std::vector<double>> extract_r_cons(const string &ind_sp2,
                                                  size_t &offset);

  template <typename T>
  std::vector<std::vector<T>> parse_bit_array(
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

  bool are_all_labels_dc(const std::vector<size_t> &labels_for_a_rule);

  FuzzyRule
  build_fuzzy_rule(const std::vector<size_t> &vars_rule_i,
                   const std::unordered_map<size_t, size_t> &vars_lv_lookup,
                   const std::vector<LinguisticVariable> &vec_lv,
                   const std::vector<size_t> &r_labels_ri,
                   const std::vector<double> cons_ri);

private:
  std::vector<std::vector<double>> X_train;
  const int n_vars;
  const int n_rules;
  const int n_max_vars_per_rule;
  const int n_bits_per_mf;
  const int n_true_labels;

  // it has been decided that the dc_idx is the last index
  // in [0, n_true_labels] i.e. [0, n_true_labels-1] for low, medium, ...
  // and {n_true_labels} for dc.
  const size_t dc_idx;

  const int n_bits_per_lv;
  const int n_lv_per_ind;
  const int n_bits_per_ant;
  const int n_cons;
  const int n_bits_per_cons;
  const int n_bits_per_label;
  const int dc_weight;
  std::vector<size_t> cons_n_labels;
  std::vector<size_t> n_classes_per_cons;
  std::vector<double> default_cons;
  std::vector<std::vector<double>> vars_range;
  std::vector<std::vector<double>> cons_range;
};

#endif // FISEVAL_BINDINGS_H
