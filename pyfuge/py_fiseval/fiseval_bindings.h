#ifndef FISEVAL_BINDINGS_H
#define FISEVAL_BINDINGS_H

#include "default_fuzzy_rule.h"
#include "fis.h"
#include "fuzzy_rule.h"
#include "json_fis_reader.h"
#include "json_fis_writer.h"
#include "linguisticvariable.h"
#include "observations_scaler.h"
#include "pybind_utils.h"
#include "singleton_fis.h"
#include "tff_fis_writer.h"
#include "trilv.h"
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>

using namespace std;

class PredictionsScaler {
  using vector2d = vector<vector<double>>;

public:
  PredictionsScaler(const vector2d &cons_range,
                    const vector<size_t> &n_labels_per_cons)
      : cons_range{cons_range}, n_labels_per_cons{n_labels_per_cons} {}

  vector2d scale(const vector2d &scaled_predictions) const {
    const size_t n_predictions = scaled_predictions.size();
    const size_t n_cons = scaled_predictions[0].size();

    vector2d predictions(n_predictions, vector<double>(n_cons, 0));

    for (size_t i = 0; i < n_predictions; i++) {
      for (size_t j = 0; j < n_cons; j++) {
        auto var_min = cons_range[j][0];
        auto var_max = cons_range[j][1];
        auto n_labels = n_labels_per_cons[j] - 1;

        predictions[i][j] = scaled_predictions[i][j] / n_labels;
        predictions[i][j] = (var_max - var_min) * predictions[i][j] + var_min;
      }
    }
    return predictions;
  }

private:
  vector2d cons_range;
  vector<size_t> n_labels_per_cons;
};

class TrefleFIS {
public:
  static TrefleFIS from_tff(const string &tff_str) {
    JsonFISReader fis_reader(tff_str);

    auto fis = fis_reader.read();

    auto vars_range = fis_reader.get_vars_range();
    auto cons_range = fis_reader.get_cons_range();
    auto n_labels_per_cons = fis_reader.get_n_labels_per_cons();

    ObservationsScaler observations_scaler(vars_range);
    PredictionsScaler predictions_scaler(cons_range, n_labels_per_cons);

    return TrefleFIS(fis, observations_scaler, predictions_scaler);
  }

  static TrefleFIS from_tff_file(const string &tff_file) {
    // TODO
    cout << "read from_tff_file" << endl;
    string TODO_CHANGE_ME_STR = "";
    return from_tff(TODO_CHANGE_ME_STR);
  }

  py_array<double> predict(py_array<double> &np_observations) {
    vector<vector<double>> observations(np_observations.shape(0));
    np_arr2d_to_vec2d(np_observations, observations);

    observations = observations_scaler.scale(observations);

    auto y_pred = fis.predict(observations);

    y_pred = predictions_scaler.scale(y_pred);
    return vec2d_to_np_vec2d(y_pred);
  }

private:
  TrefleFIS(const SingletonFIS &fis,
            const ObservationsScaler &observations_scaler,
            const PredictionsScaler &predictions_scaler)
      : fis{fis}, observations_scaler{observations_scaler},
        predictions_scaler{predictions_scaler} {};

private:
  SingletonFIS fis;
  ObservationsScaler observations_scaler;
  PredictionsScaler predictions_scaler;
};

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
    // cout << "hello from FISCocoEvalWrapper " << n_bits_per_mf << ", "
    // << n_true_labels << ", " << n_bits_per_lv << endl;

    np_arr1d_to_vec(np_cons_n_labels, cons_n_labels, n_cons);
    np_arr1d_to_vec(np_n_classes_per_cons, n_classes_per_cons, n_cons);

    // for (int i = 0; i < cons_n_labels.size(); i++) {
    //   cout << "cons n labels " << cons_n_labels[i] << endl;
    // }

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

private:
  vector<vector<double>> X_train;
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
  vector<size_t> cons_n_labels;
  vector<size_t> n_classes_per_cons;
  vector<double> default_cons;
  vector<vector<double>> vars_range;
  vector<vector<double>> cons_range;
};

PYBIND11_MODULE(pyfuge_c, m) {
  py::class_<FISCocoEvalWrapper>(m, "FISCocoEvalWrapper")
      // match the ctor of FISCocoEvalWrapper
      .def(py::init<py_array<double>, // X_train
                    const int,        // n_vars
                    const int,        // n_rules
                    const int,        // n_max_vars_per_rule
                    const int,        // n_bits_per_mf
                    const int,        // n_true_labels
                    const int,        // n_bits_per_lv
                    const int,        // n_bits_per_ant
                    const int,        // n_cons
                    const int,        // n_bits_per_cons
                    const int,        // n_bits_per_label
                    const int,        // dc_weight
                    py_array<int>,    // n_classes_per_cons
                    py_array<int>,    // cons_n_labels
                    py_array<int>,    // default_cons
                    py_array<double>, // vars_range
                    py_array<double>  // cons_range
                    >())
      .def("bind_predict", &FISCocoEvalWrapper::predict_c,
           "a function that use predict")
      .def("bind_predict", &FISCocoEvalWrapper::predict_c_other,
           "a function that use predict")
      .def("print_ind", &FISCocoEvalWrapper::print_ind,
           "pretty print an individual couple")
      .def("to_tff", &FISCocoEvalWrapper::to_tff,
           "a function that returns a tff string from a given individual "
           "couple");

  py::class_<TrefleFIS>(m, "TrefleFIS")
      // .def(py::init<int>())
      .def("predict", &TrefleFIS::predict, "predict one or more observations")
      .def("from_tff", &TrefleFIS::from_tff,
           "get a TrefleFIS instance from a tff str")
      .def("from_tff_file", &TrefleFIS::from_tff_file,
           "get a TrefleFIS instance from a tff file");
}

#endif // FISEVAL_BINDINGS_H
