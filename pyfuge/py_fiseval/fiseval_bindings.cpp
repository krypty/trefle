#include "fis.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

typedef py::array_t<double, py::array::c_style | py::array::forcecast>
                 py_array_d;
typedef py::array_t<float, py::array::c_style | py::array::forcecast>
                 py_array_f;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> py_array_i;

py::array_t<double> predict_c(py_array_f ind, py_array_d observations,
          const int n_rules, const int max_vars_per_rule, const int n_labels,
          const int n_consequents, py_array_i default_rule_cons,
          py_array_d vars_ranges, py_array_d labels_weights,
          const int dc_idx) {

          auto buf_ind = ind.request();
          auto buf_observations = observations.request();
          auto buf_default_rule_cons = default_rule_cons.request();
          auto buf_vars_ranges = vars_ranges.request();
          auto buf_labels_weights = labels_weights.request();

          /// call the native function
          double *y_preds_ptr = predict(
              (float*)buf_ind.ptr, buf_ind.size,
              (double*)buf_observations.ptr, buf_observations.shape[0], buf_observations.shape[1],
              n_rules,
              max_vars_per_rule,
              n_labels,
              n_consequents,
              (int*)buf_default_rule_cons.ptr, buf_default_rule_cons.size,
              (double*)buf_vars_ranges.ptr, buf_vars_ranges.shape[0], buf_vars_ranges.shape[1],
              (double*)buf_labels_weights.ptr, buf_labels_weights.size,
              dc_idx
          );

          /// convert the result into a numpy array
          auto y_preds = py::array_t<double>({(int)observations.shape(0), n_consequents});
          // get a mutable proxy of the numpy array (faster)
          auto y_preds_raw = y_preds.mutable_unchecked<2>();

          for(size_t i = 0; i < y_preds_raw.shape(0); i++){
            for(size_t j = 0; j < y_preds_raw.shape(1); j++){
              y_preds_raw(i, j) = y_preds_ptr[i * n_consequents + j];
            }
          }

          /// return the numpy array
          return y_preds;
        }

PYBIND11_MODULE(pyfuge_c, m) {
  m.doc() = "pyfuge example of binding";
  m.def("bind_predict", &predict_c, "a function that use predict");
}
