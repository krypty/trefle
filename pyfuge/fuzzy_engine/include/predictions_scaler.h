#ifndef PREDICTIONS_SCALER_H
#define PREDICTIONS_SCALER_H

#include <vector>

class PredictionsScaler {
  using vector2d = std::vector<std::vector<double>>;

public:
  PredictionsScaler(const vector2d &cons_range,
                    const std::vector<size_t> &n_labels_per_cons)
      : cons_range{cons_range}, n_labels_per_cons{n_labels_per_cons} {}

  vector2d scale(const vector2d &scaled_predictions) const {
    const size_t n_predictions = scaled_predictions.size();
    const size_t n_cons = scaled_predictions[0].size();

    vector2d predictions(n_predictions, std::vector<double>(n_cons, 0));

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
  std::vector<size_t> n_labels_per_cons;
};

#endif // PREDICTIONS_SCALER_H
