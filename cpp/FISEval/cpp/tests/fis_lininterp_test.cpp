#include "../vendor/catch/catch.hpp"

#include "../hpp/fis.h"

using namespace std;

TEST_CASE("lininterp xs and ys should have the same size", "[single-file]") {
  vector<double> xs{1, 2, 3, 4, 5};
  vector<double> ys{10, 20, 30, 40, 50};

  REQUIRE(lininterp(xs, ys, 3.5) == Approx(35));
  REQUIRE(lininterp(xs, ys, -100) == Approx(10));
  REQUIRE(lininterp(xs, ys, 100) == Approx(50));
  REQUIRE(lininterp(xs, ys, 0) == Approx(10));
}

TEST_CASE("step function ascending", "[single-file]") {
  vector<double> xs{1, 2, 3, 3, 4};
  vector<double> ys{0, 0, 0, 1, 1};

  REQUIRE(lininterp(xs, ys, 2.9) == Approx(0));
  REQUIRE(lininterp(xs, ys, 3.0) == Approx(0));
  REQUIRE(lininterp(xs, ys, 3.1) == Approx(1));
}

TEST_CASE("step function descending", "[single-file]") {
  vector<double> xs{1, 2, 2, 3, 4};
  vector<double> ys{1, 1, 0, 0, 0};

  REQUIRE(lininterp(xs, ys, 1.9) == Approx(1));
  REQUIRE(lininterp(xs, ys, 2.0) == Approx(1));
  REQUIRE(lininterp(xs, ys, 2.1) == Approx(0));
}
