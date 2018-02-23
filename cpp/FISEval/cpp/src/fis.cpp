#include "../hpp/fis.h"
#include "omp.h"
#include <cmath>
#include <iostream>

#define coutd std::cout << "<<C++>> "

using namespace std;

void mul_np_array(double *in_array, int length, int scaler) {
#pragma omp parallel for
  for (int i = 0; i < length; i++) {
    in_array[i] = in_array[i] * scaler;
  }
}

float predict(float *ind, int ind_n, double **observations, int observations_n,
              int observations_m, int n_rules, int max_vars_per_rules,
              int n_labels, int n_consequents, int *default_rule_cons,
              int default_rule_cons_n, double **vars_range, int vars_range_n,
              int vars_range_m, double *labels_weights, int labels_weights_n) {

  cout << "hello" << std::endl;
  coutd << observations_n << ", " << observations_m << std::endl;
  coutd << observations[0][0] << ";" << std::endl;

  // coutd << "vars_range" << std::endl;
  // coutd << "shape" << vars_range_n << ", " << vars_range_m << std::endl;
  // for (int i = 0; i < vars_range_n; i++) {
  //   coutd << "";
  //   for (int j = 0; j < vars_range_m; j++) {
  //     cout << vars_range[i][j] << "; ";
  //   }
  //   cout << std::endl;
  // }

  // coutd << "observations" << std::endl;
  // coutd << "shape" << observations_n << ", " << observations_m << std::endl;
  // for (int i = 0; i < observations_n; i++) {
  //   coutd << "";
  //   for (int j = 0; j < observations_m; j++) {
  //     cout << observations[i][j] << "; ";
  //   }
  //   cout << std::endl;
  // }

  // coutd << "default rule cons" << std::endl;
  // for (int i = 0; i < default_rule_cons_n; i++) {
  //   coutd << default_rule_cons[i] << std::endl;
  // }

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
