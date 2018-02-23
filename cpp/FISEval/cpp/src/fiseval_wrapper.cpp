#include "../hpp/fiseval_wrapper.h"
#include "../hpp/fis.h"
#include "cmath"
#include "iostream"
#include "omp.h"

#define coutd std::cout << "<<C++>> "

class User {
  std::string name;

public:
  User(char *name) : name(name) {}
  User(std::string &name) : name(name) {}

  std::string greet() { return "hello, " + name; }
};

void hello(char *name) {
  User user(name);
  std::cout << user.greet() << std::endl;
}

int main() {
  std::cout << "result : " << mySuperAlgo() << std::endl;
  return 0;
}

int mySuperAlgo() {
  Point *p = new Point();

  const int size = 10000;
  double sinTable[size];

  int out[4];

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    int tid = omp_get_thread_num();

    for (int j = 0; j < size; j++) {
      sinTable[i] = std::sin(2 * M_PI * i / size);
      sinTable[i] = std::sin(2 * M_PI * (i + tid) / size);
    }

    out[i % 4] = 5;
  }

  int res = 0;
  for (int i = 0; i < 4; i++) {
    res += out[i];
  }

  p->afficher();
  delete p;

  return res;
}

extern "C" {
extern int cffi_hello() { return mySuperAlgo(); }

extern void c_mul_np_array(double *in_array, int length, int scaler) {
  mul_np_array(in_array, length, scaler);
}

extern float c_predict(float *ind, int ind_n, double **observations,
                       int observations_n, int observations_m) {
  coutd << observations_n << ", " << observations_m << std::endl;
  coutd << observations[0][0] << ";" << std::endl;
  float sum = 0.0f;

  const int N_THREADS = omp_get_max_threads();

  float mid_sum[N_THREADS];
  std::fill(mid_sum, mid_sum + N_THREADS, 0);
#pragma omp parallel for
  for (int i = 0; i < observations_n; i++) {
    // sum += observations[i][0];
    int tid = omp_get_thread_num();
    mid_sum[tid] += observations[i][0];
  }

  for (int i = 0; i < N_THREADS; i++) {
    sum += mid_sum[i];
  }

  return sum;
}
}
