#include "trefle_fis.h"
#include "json_fis_reader.h"
#include "observations_scaler.h"
#include "predictions_scaler.h"
#include <string>
#include <vector>

TrefleFIS TrefleFIS::from_tff(const string &tff_str) {
  JsonFISReader fis_reader(tff_str);

  auto fis = fis_reader.read();

  auto vars_range = fis_reader.get_vars_range();
  auto cons_range = fis_reader.get_cons_range();
  auto n_labels_per_cons = fis_reader.get_n_labels_per_cons();

  ObservationsScaler observations_scaler(vars_range);
  PredictionsScaler predictions_scaler(cons_range, n_labels_per_cons);

  return TrefleFIS(fis, observations_scaler, predictions_scaler);
}

TrefleFIS TrefleFIS::from_tff_file(const string &tff_file) {
  // TODO
  cout << "read from_tff_file" << endl;
  string TODO_CHANGE_ME_STR = "";
  return from_tff(TODO_CHANGE_ME_STR);
}

py_array<double> TrefleFIS::predict(py_array<double> &np_observations) {
  std::vector<std::vector<double>> observations(np_observations.shape(0));
  np_arr2d_to_vec2d(np_observations, observations);

  observations = observations_scaler.scale(observations);

  auto y_pred = fis.predict(observations);

  y_pred = predictions_scaler.scale(y_pred);
  return vec2d_to_np_vec2d(y_pred);
}
