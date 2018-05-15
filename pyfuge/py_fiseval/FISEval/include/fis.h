#ifndef FIS_H
#define FIS_H

#include "custom_eigen_td.h"
#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

class FISEval {
public:
  FISEval(int ind_n, double *observations, int observations_r,
          int observations_c, const int n_rules, int max_vars_per_rules,
          int n_labels, int n_consequents, int *default_rule_cons,
          int default_rule_cons_n, double *vars_range, int vars_range_r,
          int vars_range_c, double *labels_weights, int labels_weights_n,
          int dc_idx);

  virtual ~FISEval();

  double *predict(float *ind);

private:
  static MatrixXi evo_ants2ifs_ants(const Map<MatXf> &evo_ants,
                                    Map<RowVectorXd> &vec_labels_weights);

  static MatrixXd evo_mfs2ifs_mfs(const Map<MatXf> &evo_mfs,
                                  Map<MatXd> &mat_vars_ranges);

  static int unitfloat2idx(float flt, Map<RowVectorXd> &weights);

  // members
  int ind_n;

  Map<MatXd> mat_obs;
  int observations_r;
  int observations_c;

  int n_rules;
  int max_vars_per_rules;
  int n_labels;
  int n_true_labels;
  int n_consequents;
  int n_vars;

  Map<RowVectorXi> mat_default_rule_cons_i;
  RowVectorXd mat_default_rule_cons;
  Map<MatXd> mat_vars_ranges;

  Map<RowVectorXd> vec_labels_weights;

  int dc_idx;
  MatrixXd mf_values_eye;
};
#endif
