#ifndef OBSERVATIONS_SCALER_H
#define OBSERVATIONS_SCALER_H

#include <unordered_map>
#include <vector>

class ObservationsScaler {
  using vector2d = std::vector<std::vector<double>>;
  using map_ranges = std::unordered_map<size_t, std::vector<double>>;

public:
  ObservationsScaler(const map_ranges &vars_range) : vars_range{vars_range} {}

  double scale(size_t var_idx, double value) {
    auto var_min = vars_range[var_idx][0];
    auto var_max = vars_range[var_idx][1];
    return (value - var_min) / (var_max - var_min);
  }

  vector2d scale(const vector2d &observations) {
    const size_t n_obs = observations.size();
    const size_t n_vars = observations[0].size();

    vector2d scaled_observations(n_obs, std::vector<double>(n_vars, 0));

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
          scaled_observations[i][j] = scale(j, observations[i][j]);
        }
      }
    }
    return scaled_observations;
  }

private:
  map_ranges vars_range;
};

#endif // OBSERVATIONS_SCALER_H
