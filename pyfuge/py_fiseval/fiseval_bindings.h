#ifndef FISEVAL_BINDINGS_H
#define FISEVAL_BINDINGS_H

#include "default_fuzzy_rule.h"
#include "fis.h"
#include "fuzzy_rule.h"
#include "json_fis_reader.h"
#include "json_fis_writer.h"
#include "linguisticvariable.h"
#include "singleton_fis.h"
#include "trilv.h"
#include <fstream>
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

class TffFISWriter {
public:
  TffFISWriter(const SingletonFIS &fis, size_t n_true_labels,
               const vector<size_t> &n_cons_per_labels,
               const vector<size_t> &n_classes_per_cons,
               const vector<vector<double>> vars_range,
               const vector<vector<double>> cons_range, const string &filepath)
      : fis{fis}, n_true_labels{n_true_labels},
        n_classes_per_cons{n_classes_per_cons},
        n_cons_per_labels{n_cons_per_labels}, vars_range{vars_range},
        cons_range{cons_range}, filepath{filepath} {};

  virtual void write() {
    JsonFISWriter writer(fis, n_true_labels, n_cons_per_labels,
                         n_classes_per_cons, vars_range, cons_range,
                         json_output);
    writer.write();
    std::ofstream fout(filepath);
    if (fout) {
      fout << json_output;
    } else {
      cerr << "cannot write in " << filepath << endl;
    }
  }

private:
  const SingletonFIS fis;
  const size_t n_true_labels;
  const vector<size_t> n_cons_per_labels;
  const vector<size_t> n_classes_per_cons;
  const vector<vector<double>> vars_range;
  const vector<vector<double>> cons_range;
  const string &filepath;
  string json_output;
};

class ObservationsScaler {
  using vector2d = vector<vector<double>>;
  using map_ranges = unordered_map<size_t, vector<double>>;

public:
  ObservationsScaler(const map_ranges &vars_range) : vars_range{vars_range} {}

  vector2d scale(const vector2d &observations) {
    const size_t n_obs = observations.size();
    const size_t n_vars = observations[0].size();

    vector2d scaled_observations(n_obs, vector<double>(n_vars, 0));

    for (size_t i = 0; i < n_obs; i++) {
      for (size_t j = 0; j < n_vars; j++) {
        auto search = vars_range.find(j);
        // if we have not a range for this variable (i.e. it is not used by
        // the fuzzy system) then we don't apply any normalization.
        if (search == vars_range.end()) {
          scaled_observations[i][j] = observations[i][j];
        } else {
          auto var_min = vars_range[j][0];
          auto var_max = vars_range[j][1];
          scaled_observations[i][j] =
              (observations[i][j] - var_min) / (var_max - var_min);
        }
      }
    }
    return scaled_observations;
  }

private:
  map_ranges vars_range;
};

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

  py_array_d predict(py_array_d &np_observations) {
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
  FISCocoEvalWrapper(py_array_d np_X_train, const int n_vars, const int n_rules,
                     const int n_max_vars_per_rule, const int n_bits_per_mf,
                     const int n_true_labels, const int n_bits_per_lv,
                     const int n_bits_per_ant, const int n_cons,
                     const int n_bits_per_cons, const int n_bits_per_label,
                     const int dc_weight, py_array_i np_cons_n_labels,
                     py_array_i np_n_classes_per_cons,
                     py_array_i np_default_cons, py_array_d np_vars_range,
                     py_array_d np_cons_range)
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
                                      py_array_d other_X);
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
      .def(py::init<py_array_d, // X_train
                    const int,  // n_vars
                    const int,  // n_rules
                    const int,  // n_max_vars_per_rule
                    const int,  // n_bits_per_mf
                    const int,  // n_true_labels
                    const int,  // n_bits_per_lv
                    const int,  // n_bits_per_ant
                    const int,  // n_cons
                    const int,  // n_bits_per_cons
                    const int,  // n_bits_per_label
                    const int,  // dc_weight
                    py_array_i, // n_classes_per_cons
                    py_array_i, // cons_n_labels
                    py_array_i, // default_cons
                    py_array_d, // vars_range
                    py_array_d  // cons_range
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
