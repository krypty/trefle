#ifndef TREFLE_FIS_H
#define TREFLE_FIS_H

#include "observations_scaler.h"
#include "predictions_scaler.h"
#include "pybind_utils.h"
#include "singleton_fis.h"
#include <string>

class TrefleFIS {
public:
  static TrefleFIS from_tff(const string &tff_str);
  static TrefleFIS from_tff_file(const string &tff_file);

  py_array<double> predict(py_array<double> &np_observations);
  void describe();

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

#endif // TREFLE_FIS_H
