#ifndef FIS_H
#define FIS_H

#include <iostream>
using namespace std;

void mul_np_array(double *arr, int arr_n, int scaler);

float predict(float *ind, int ind_n, double *observations, int observations_r,
              int observations_c, int n_rules, int max_vars_per_rules,
              int n_labels, int n_consequents, int *default_rule_cons,
              int default_rule_cons_n, double *vars_range, int vars_range_r,
              int vars_range_c, double *labels_weights, int labels_weights_n);

double omp_sum(double **arr, int arr_n);
void extract_mfs_from_ind(float *ind, double **evo_mfs, int n_vars,
                          int n_labels);

void extract_ants_from_ind(float *ind, double **evo_ants, int n_vars,
                           int n_labels);

template <typename T, typename U>
void array1d_to_2d(T *in, U **out, int n, int m);
#endif
