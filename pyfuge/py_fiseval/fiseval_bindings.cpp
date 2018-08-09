#include "fis.h"
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

class FISEvalWrapper {
public:
  FISEvalWrapper(int ind_n, py_array_d observations, const int n_rules,
                 const int max_vars_per_rule, const int n_labels,
                 const int n_consequents, py_array_i default_rule_cons,
                 py_array_d vars_ranges, py_array_d labels_weights,
                 const int dc_idx) {
    this->n_consequents = n_consequents;

    this->observations = observations;
    auto buf_observations = observations.request();
    double *observations_ptr = (double *)buf_observations.ptr;
    int observations_r = buf_observations.shape[0];
    int observations_c = buf_observations.shape[1];

    auto buf_default_rule_cons = default_rule_cons.request();
    int *default_rule_cons_ptr = (int *)buf_default_rule_cons.ptr;
    int default_rule_cons_n = buf_default_rule_cons.size;

    auto buf_vars_ranges = vars_ranges.request();
    double *vars_ranges_ptr = (double *)buf_vars_ranges.ptr;
    int vars_ranges_r = buf_vars_ranges.shape[0];
    int vars_ranges_c = buf_vars_ranges.shape[1];

    // FIXME
    // does not do anything but retain the pointer to not delete its content.
    // Will be deleted when this instance is deleting. this->labels_weights =
    // labels_weights;

    auto buf_labels_weights = labels_weights.request();
    auto labels_weights_ptr = (double *)buf_labels_weights.ptr;
    int labels_weights_n = buf_labels_weights.size;

    fiseval = new FISEval(
        ind_n, observations_ptr, observations_r, observations_c, n_rules,
        max_vars_per_rule, n_labels, n_consequents, default_rule_cons_ptr,
        default_rule_cons_n, vars_ranges_ptr, vars_ranges_r, vars_ranges_c,
        labels_weights_ptr, labels_weights_n, dc_idx);
  }

  py::array_t<double> predict_c(py_array_f ind) {
    auto buf_ind = ind.request();

    /// call the native function
    double *y_preds_ptr = fiseval->predict((float *)buf_ind.ptr);

    /// convert the result into a numpy array
    auto y_preds =
        py::array_t<double>({(int)observations.shape(0), n_consequents});
    // get a mutable proxy of the numpy array (faster)
    auto y_preds_raw = y_preds.mutable_unchecked<2>();

    for (size_t i = 0; i < y_preds_raw.shape(0); i++) {
      for (size_t j = 0; j < y_preds_raw.shape(1); j++) {
        y_preds_raw(i, j) = y_preds_ptr[i * n_consequents + j];
      }
    }

    /// return the numpy array
    return y_preds;
  }

  virtual ~FISEvalWrapper() { delete fiseval; }

private:
  FISEval *fiseval;
  py_array_d observations;
  // py_array_d labels_weights;
  int n_consequents;
};

class FISCocoEvalWrapper {
public:
  FISCocoEvalWrapper(const int n_bits_per_mf, const int n_true_labels,
                     const int n_lv_per_ind)
      : n_bits_per_mf(n_bits_per_mf), n_true_labels(n_true_labels),
        n_lv_per_ind(n_lv_per_ind) {
    cout << "hello from FISCocoEvalWrapper " << n_bits_per_mf << ", "
         << n_true_labels << ", " << n_lv_per_ind << endl;
  }
  py::array_t<double> predict_c(const string &ind_sp1, const string &ind_sp2) {
    // double predict_c(const string &ind_sp1, const string &ind_sp2) {
    cout << "ind_sp1 " << ind_sp1 << " (" << ind_sp1.length() << ")" << endl;
    cout << "ind_sp2 " << ind_sp2 << " (" << ind_sp2.length() << ")" << endl;

    // TODO: substr() is creating a new string each time, use c++17's
    // stringview?
    vector<vector<int>> vec_lv;
    vec_lv.reserve(n_lv_per_ind);

    const int n_bits_per_line = n_true_labels * n_bits_per_mf;

    for (int lv_i = 0; lv_i < n_lv_per_ind; lv_i++) {
      // cout << "lv_i " << lv_i << endl;
      // vec_lv[lv_i].reserve(n_true_labels);
      vector<int> vec_lv_i;

      for (int mf_i = 0; mf_i < n_true_labels; mf_i++) {
        // cout << "lv_i " << lv_i << endl;
        const int offset = lv_i * n_bits_per_line + (mf_i * n_bits_per_mf);
        // cout << "offset " << offset << endl;
        const int v = stoi(ind_sp1.substr(offset, n_bits_per_mf), nullptr, 2);
        cout << "v: " << v << endl;
        vec_lv_i.push_back(v);
      }

      vec_lv.push_back(vec_lv_i);
      cout << endl;
    }

    cout << "a" << endl;
    for (const auto &row : vec_lv) {
      for (const auto &col : row) {
        cout << col << ", ";
      }
      cout << endl;
    }

    // TODO: remove me, return instead y_pred
    auto arr = py::array_t<double>({n_lv_per_ind, n_true_labels});
    auto arr_raw = arr.mutable_unchecked<2>();

    for (int i = 0; i < n_lv_per_ind; i++) {
      for (int j = 0; j < n_true_labels; j++) {
        //// normalize in [0,1]
        // arr_raw(i, j) = vec_lv[i][j] / double(1 << n_bits_per_mf);
        arr_raw(i, j) = vec_lv[i][j];
      }
    }
    return arr;
  }

private:
  const int n_bits_per_mf;
  const int n_true_labels;
  const int n_lv_per_ind;
};

PYBIND11_MODULE(pyfuge_c, m) {
  // py::class_<FISEvalWrapper>(m, "FISEvalWrapper")
  //     // match the ctor of FISEvalWrapper
  //     .def(py::init<int, py_array_d, const int, const int, const int, const
  //     int,
  //                   py_array_i, py_array_d, py_array_d, const int>())
  //     .def("bind_predict", &FISEvalWrapper::predict_c,
  //          "a function that use predict");

  py::class_<FISCocoEvalWrapper>(m, "FISCocoEvalWrapper")
      // match the ctor of FISCocoEvalWrapper
      .def(py::init<const int, const int, const int>())
      .def("bind_predict", &FISCocoEvalWrapper::predict_c,
           "a function that use predict");
}
