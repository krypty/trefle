#ifndef TREFLE_PY_MODULE_H
#define TREFLE_PY_MODULE_H

#include "fiseval_bindings.h"
#include "pybind_utils.h"
#include "trefle_fis.h"
#include <pybind11/pybind11.h>

/**
 * This class is the entrypoint to the C++ classes from Python's
 * point of view.
 * It should contains no logic but only gateways to C++ functionalities.
 */

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
      .def("predict", &TrefleFIS::predict, "predict one or more observations")
      .def("from_tff", &TrefleFIS::from_tff,
           "get a TrefleFIS instance from a tff str")
      .def("from_tff_file", &TrefleFIS::from_tff_file,
           "get a TrefleFIS instance from a tff file");
}

#endif // TREFLE_PY_MODULE_H
