#include "../hpp/fis.h"
#include "omp.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <Eigen/Core>

#define coutd std::cout << "<<C++>> " << std::setprecision(2)

using namespace std;

void mul_np_array(double *in_array, int length, int scaler) {
#pragma omp parallel for
  for (int i = 0; i < length; i++) {
    in_array[i] = in_array[i] * scaler;
  }
}

float predict(float *ind, int ind_n, double *observations, int observations_r,
              int observations_c, int n_rules, int max_vars_per_rules,
              int n_labels, int n_consequents, int *default_rule_cons,
              int default_rule_cons_n, double *vars_range, int vars_range_r,
              int vars_range_c, double *labels_weights, int labels_weights_n) {

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
  // coutd << "shape" << observations_r << ", " << observations_c << std::endl;
  // for (int i = 0; i < observations_r; i++) {
  //   coutd << "";
  //   for (int j = 0; j < observations_c; j++) {
  //     cout << observations[i][j] << "; ";
  //   }
  //   cout << std::endl;
  // }

  // coutd << "default rule cons" << std::endl;
  // for (int i = 0; i < default_rule_cons_n; i++) {
  //   coutd << default_rule_cons[i] << std::endl;
  // }

  int n_true_labels = n_labels - 1;

  int n_obs = observations_r;
  int n_vars = observations_c;

  double *evo_mfs[n_vars];
  for (int i = 0; i < n_vars; i++) {
    evo_mfs[i] = new double[n_true_labels];
  }
  extract_mfs_from_ind(ind, evo_mfs, n_vars, n_true_labels);

  double *evo_ants[n_rules];
  for (int i = 0; i < n_rules; i++) {
    evo_ants[i] = new double[n_vars];
  }

  // offset where the antecedents values begin which is after the evo_mfs values
  float *ind_offset = ind + (n_vars * n_true_labels);
  extract_ants_from_ind(ind_offset, evo_ants, n_rules, n_vars);

  // for (int i = 0; i < n_rules; i++) {
  //   coutd << endl;
  //   for (int j = 0; j < n_vars; j++) {
  //     coutd << evo_ants[i][j] << endl;
  //   }
  // }
  // extract_ind(ind, n_vars, n_labels, n_rules, n_consequents);

  return 12345;
}

void extract_mfs_from_ind(float *ind, double **evo_mfs, int n_vars,
                          int n_labels) {
  array1d_to_2d(ind, evo_mfs, n_vars, n_labels);
}
void extract_ants_from_ind(float *ind, double **evo_ants, int n_rules,
                           int n_vars) {
  array1d_to_2d(ind, evo_ants, n_rules, n_vars);
}

template <typename T, typename U>
void array1d_to_2d(T *in, U **out, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      out[i][j] = in[i * m + j];
    }
  }
}

double omp_sum(double **arr, int arr_n) {
  const int N_THREADS = omp_get_max_threads();
  double sum = 0.0f;

  float mid_sum[N_THREADS];
  std::fill(mid_sum, mid_sum + N_THREADS, 0);

#pragma omp parallel for
  for (int i = 0; i < arr_n; i++) {
    // sum += arr[i][0];
    int tid = omp_get_thread_num();
    mid_sum[tid] += arr[i][0];
  }

  for (int i = 0; i < N_THREADS; i++) {
    sum += mid_sum[i];
  }

  return sum;
}
