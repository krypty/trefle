#include "../hpp/fis.h"
#include "omp.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#define coutd std::cout << "<<C++>> " << std::setprecision(2)

using namespace std;
using namespace Eigen;

void mul_np_array(double *in_array, int length, int scaler) {
#pragma omp parallel for
  for (int i = 0; i < length; i++) {
    in_array[i] = in_array[i] * scaler;
  }
}

int unitfloat2idx(float flt, Map<RowVectorXd> &weights) {
  vector<int> indices;

  // coutd << weights << endl;
  // coutd << weights.maxCoeff() << endl;
  weights = weights / weights.minCoeff();

  for (int i = 0; i < weights.size(); i++) {
    for (int j = 0; j < weights(i); j++) {
      indices.push_back(i);
    }
  }

  int vec_n = indices.size();
  int idx = flt * vec_n;
  int safe_idx = max(0, min(idx, vec_n - 1));
  int to_ret = indices[int(safe_idx)];
  // coutd << "to ret " << to_ret << endl;
  return to_ret;
}

MatrixXi evo_ants2ifs_ants(const Map<MatXf> &evo_ants,
                           Map<RowVectorXd> &vec_labels_weights) {
  const auto unitfloat2idx_ants = [&](float v) {
    return unitfloat2idx(v, vec_labels_weights);
  };
  coutd << "before" << evo_ants << endl;
  // MatrixXf ifs_ants;
  // ifs_ants = evo_ants.unaryExpr(unitfloat2idx_ants);
  MatrixXi ifs_ants(evo_ants.rows(), evo_ants.cols());
  ifs_ants = evo_ants.unaryExpr(unitfloat2idx_ants);

  coutd << "after" << endl << ifs_ants << endl;
  return ifs_ants;
}

float predict(float *ind, int ind_n, double *observations, int observations_r,
              int observations_c, int n_rules, int max_vars_per_rules,
              int n_labels, int n_consequents, int *default_rule_cons,
              int default_rule_cons_n, double *vars_range, int vars_range_r,
              int vars_range_c, double *labels_weights, int labels_weights_n,
              int dc_idx) {

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

  // EXTRACT NEW INDIVIDUAL
  // ind is a float array that is the individual which represents a FIS.
  Map<MatXf> evo_mfs(ind, n_vars, n_true_labels);
  // coutd << "evo_mfs" << endl;
  // coutd << evo_mfs << endl;

  // offset where the antecedents values begin which is after the evo_mfs values
  float *ind_offset_ants = ind + (n_vars * n_true_labels);
  Map<MatXf> evo_ants(ind_offset_ants, n_rules, n_vars);
  // coutd << "evo_ants" << endl;
  // coutd << evo_ants << endl;

  float *ind_offset_cons = ind_offset_ants + (n_rules * n_vars);
  Map<MatXf> evo_cons(ind_offset_cons, n_rules, n_consequents);
  // coutd << "evo_cons" << endl;
  // cout << evo_cons << endl;

  // CONVERT EVOLUTION ANTS TO IFS ANTS
  // MatXf ifs_ants =
  Map<RowVectorXd> vec_labels_weights(labels_weights, labels_weights_n);
  MatrixXi ifs_ants = evo_ants2ifs_ants(evo_ants, vec_labels_weights);
  coutd << "ifs_ants" << endl;
  cout << ifs_ants << endl;
  // CONVERT EVOLUTION MFS TO IFS MFS (i.e. in_values)

  // CONVERT EVOLUTION CONS TO IFS CONS
  const auto binarize_mat = [&](float v) {
    return v >= 0.5 ? 1.0f : 0.0f;
  }; // TODO: extract this into .h
  MatXf ifs_cons = evo_cons.unaryExpr(binarize_mat);

  MatrixXd mf_values_eye = MatrixXd::Identity(n_labels, n_labels - 1);
  // set DC row to 1. This will neutralize the effect of AND_min
  mf_values_eye.row(n_labels - 1).setOnes();
  // cout << mf_values_eye;

  // Map<MatXd> mat_obs(observations, observations_r, observations_c);
  // for(int i = 0; i < observations_r; i++){
  //   coutd << mat_obs.row(i) << endl;
  //   coutd << "yolo" << endl;
  // }

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
void array1d_to_2d(T *in, U **out, int n, int ifs_ants) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < ifs_ants; j++) {
      out[i][j] = in[i * ifs_ants + j];
    }
  }
}

double omp_sum(double **arr, int arr_n) {
  const int N_THREADS = omp_get_max_threads();
  double sum = 0.0f;

  vector<float> mid_sum(N_THREADS);
  std::fill(mid_sum.begin(), mid_sum.end() - 1, 0);

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
