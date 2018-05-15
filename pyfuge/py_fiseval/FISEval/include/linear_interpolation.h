#ifndef LINEAR_INTERPOLATION_H
#define LINEAR_INTERPOLATION_H

#include <algorithm>
#include <cmath>
#include <vector>

#define EPSILON 1e-9

class LinearInterpolation {
public:
  static double interp(vector<double> &xs, const vector<double> &ys,
                       const double x) {
    assert(xs.size() == ys.size());
    sort(xs.begin(), xs.end());
    // assert(is_sorted(xs.begin(), xs.end()) && "xs is expected to be sorted");
    // using this image as reference:
    // https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/LinearInterpolation.svg/300px-LinearInterpolation.svg.png

    if (x <= xs[0]) {
      return ys[0];
    } else if (x >= xs[xs.size() - 1]) {
      return ys[ys.size() - 1];
    }

    int idx_low = lower_bound(xs.begin(), xs.end(), x) - xs.begin() - 1;
    int idx_up = min(idx_low + 1, (int)xs.size() - 1);

    double deltaX = xs[idx_up] - xs[idx_low]; // delta >= 0 because xs is sorted
    if (deltaX < EPSILON) {
      // delta is too small, interpolation will lead to zero division error.
      // return the nearest known ys value.
      // note: index of x0 or x1, does not matter because they pretty much the
      // same value
      // coutd << "yolo !!!" << endl;
      // coutd << "deltaX too small, ret " << ys[idx_low] << endl;
      return ys[idx_low];
    }
    double y =
        ys[idx_low] + ((x - xs[idx_low]) * (ys[idx_up] - ys[idx_low]) / deltaX);
    return y;
  }
};

#endif
