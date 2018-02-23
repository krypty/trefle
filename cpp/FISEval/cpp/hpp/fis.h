#ifndef FIS_H
#define FIS_H

#include <iostream>
using namespace std;

void mul_np_array(double *arr, int arr_n, int scaler);

float predict(float *ind, int ind_n, double **observations, int observations_n,
              int observations_m, int n_rules, int max_vars_per_rules,
              int n_labels, int n_consequents, int *default_rule_cons,
              int default_rule_cons_n, double **vars_range, int vars_range_n,
              int vars_range_m, double *labels_weights, int labels_weights_n);

#endif
