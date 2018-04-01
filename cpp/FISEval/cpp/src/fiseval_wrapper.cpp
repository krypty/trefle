#include "../hpp/fiseval_wrapper.h"
#include "../hpp/custom_eigen_td.h"
#include "../hpp/fis.h"
#include "cmath"
#include "iostream"
#include "omp.h"
#include <Eigen/Core>
using namespace Eigen;
using namespace std;

#define coutd std::cout << "<<C++>> "

int mySuperAlgo() {
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

    cout << "sintable" << sinTable[0] << endl;

    out[i % 4] = 5;
  }

  int res = 0;
  for (int i = 0; i < 4; i++) {
    res += out[i];
  }

  return res;
}

extern "C" {
extern int cffi_hello() { return mySuperAlgo(); }

extern void c_mul_np_array(double *in_array, int length, int scaler) {
  mul_np_array(in_array, length, scaler);
}

extern double* c_predict(float *ind, int ind_n, double *observations,
                       int observations_r, int observations_c, int n_rules,
                       int max_vars_per_rules, int n_labels, int n_consequents,
                       int *default_rule_cons, int default_rule_cons_n,
                       double *vars_range, int vars_range_r, int vars_range_c,
                       double *labels_weights, int labels_weights_n, int dc_idx) {

  // Map<VectorXf> mf(ind,ind_n);

  // coutd << "float*" << std::endl;
  // for(int i = 0; i < ind_n; i++){
  //   cout << ind[i] << " ";
  // }
  // cout << std::endl;

  // coutd << "vector" << std::endl;
  // coutd << mf << std::endl;

  // for (int i = 0; i < observations_n; i++) {
  //   coutd << "new line" << std::endl;
  //   for (int j = 0; j < observations_m; j++) {
  //     coutd << observations[i * observations_m + j] << ", " << std::endl;
  //   }
  //   coutd << "------" << std::endl;
  // }

  // TODO double** --> double*
  // Map<Matrix<double, Dynamic, Dynamic, RowMajor>> mf(
  // Map<MatXd> mf(observations, observations_r, observations_c);
  // coutd << mf << std::endl;
  // coutd << "rows: " << mf.rows() << ", cols= " << mf.cols() << std::endl;
  // mf = mf * -1;
  // coutd << mf << endl;

  return predict(ind, ind_n, observations, observations_r, observations_c,
                 n_rules, max_vars_per_rules, n_labels, n_consequents,
                 default_rule_cons, default_rule_cons_n, vars_range,
                 vars_range_r, vars_range_c, labels_weights, labels_weights_n, dc_idx);
}
}
