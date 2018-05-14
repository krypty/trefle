#include "catch.hpp"

#include "fis.h"
#include "linear_interpolation.h"

#include "custom_eigen_td.h"
#include <Eigen/Core>

using namespace std;

TEST_CASE("xs and ys should have the same size", "[single-file]") {
  vector<double> xs{1, 2, 3, 4, 5};
  vector<double> ys{10, 20, 30, 40, 50};

  REQUIRE(LinearInterpolation::interp(xs, ys, 3.5) == Approx(35));
  REQUIRE(LinearInterpolation::interp(xs, ys, -100) == Approx(10));
  REQUIRE(LinearInterpolation::interp(xs, ys, 100) == Approx(50));
  REQUIRE(LinearInterpolation::interp(xs, ys, 0) == Approx(10));
}

TEST_CASE("step function ascending", "[single-file]") {
  vector<double> xs{1, 2, 3, 3, 4};
  vector<double> ys{0, 0, 0, 1, 1};

  REQUIRE(LinearInterpolation::interp(xs, ys, 2.9) == Approx(0));
  REQUIRE(LinearInterpolation::interp(xs, ys, 3.0) == Approx(0));
  REQUIRE(LinearInterpolation::interp(xs, ys, 3.1) == Approx(1));
}

TEST_CASE("step function descending", "[single-file]") {
  vector<double> xs{1, 2, 2, 3, 4};
  vector<double> ys{1, 1, 0, 0, 0};

  REQUIRE(LinearInterpolation::interp(xs, ys, 1.9) == Approx(1));
  REQUIRE(LinearInterpolation::interp(xs, ys, 2.0) == Approx(1));
  REQUIRE(LinearInterpolation::interp(xs, ys, 2.1) == Approx(0));
}

TEST_CASE("basic interpolation", "[single-file]") {
  vector<double> xs{1.8, 2.7};
  vector<double> ys{1, 0};

  REQUIRE(LinearInterpolation::interp(xs, ys, 1.7) == Approx(1));
  REQUIRE(LinearInterpolation::interp(xs, ys, 1.8) == Approx(1));
  REQUIRE(LinearInterpolation::interp(xs, ys, 1.9) == Approx(0.888889));
  REQUIRE(LinearInterpolation::interp(xs, ys, 2.6) == Approx(0.111111));
  REQUIRE(LinearInterpolation::interp(xs, ys, 2.7) == Approx(0));
  REQUIRE(LinearInterpolation::interp(xs, ys, 2.8) == Approx(0));
}

// TEST_CASE("lalal", "[single-file]") {

//   RowVector4d rules_act(0.2, 0.67, 0.33, 0.5);

//   // Vector3d ifs_cons(1.0, 0.5, 0);
//   Matrix<double, Dynamic, 3> ifs_cons;
//   ifs_cons.resize(4, 3);
//   ifs_cons << 0, 0, 1, // cons r1
//               1, 0, 1, // cons r2
//               0, 0, 0, // cons r3
//               1, 1, 1; // cons r4

//   cout << rules_act << endl;
//   cout << ifs_cons << endl;

//   cout << "result: " << endl;
//   cout << (rules_act * ifs_cons) / rules_act.sum() << endl;
// }
