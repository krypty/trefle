#ifndef LINEAR_INTERPOLATION_H
#define LINEAR_INTERPOLATION_H

#include <vector>
#include <algorithm>
#include <cmath>

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

    // coutd << "lininterp x:" << x << endl;
    // coutd << "xs: ";
    // for (const auto i : xs)
    //   std::cout << i << ' ';
    // cout << endl;

    // coutd << "ys: ";
    // for (const auto i : ys)
    //   std::cout << i << ' ';
    // cout << endl;

    if (x <= xs[0]) {
      // coutd << "clip to left " << ys[0] << endl;
      return ys[0];
    } else if (x >= xs[xs.size() - 1]) {
      // coutd << "clip to right " << ys[ys.size() - 1] << endl;
      return ys[ys.size() - 1];
    }

    int idx_low = lower_bound(xs.begin(), xs.end(), x) - xs.begin() - 1;
    int idx_up = min(idx_low + 1, (int)xs.size() - 1);
    // int idx_up = upper_bound(xs.begin() + idx_low + 1, xs.end(), x) -
    // xs.begin();

    // coutd << "idx_low " << idx_low << ", " << idx_up << endl;

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
    // coutd << "do interp " << y << endl;
    return y;
  }
};

#endif
