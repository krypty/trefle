#include "fis.h"
#include "linear_interpolation.h"
#include "omp.h"
#include <cmath>
#include <iomanip>
#include <iostream>

#define coutd std::cout << "<<C++>> " << std::setprecision(3)
using namespace std;
using namespace Eigen;

int FISEval::unitfloat2idx(float flt, Map<RowVectorXd> &weights) {
  vector<int> indices;

  RowVectorXd weights_normed = weights / weights.minCoeff();

  for (int i = 0; i < weights_normed.size(); i++) {
    for (int j = 0; j < weights_normed(i); j++) {
      indices.push_back(i);
    }
  }

  int vec_n = indices.size();
  int idx = flt * vec_n;
  int safe_idx = max(0, min(idx, vec_n - 1));
  int to_ret = indices[int(safe_idx)];
  return to_ret;
}

MatrixXi FISEval::evo_ants2ifs_ants(const Map<MatXf> &evo_ants,
                                    Map<RowVectorXd> &vec_labels_weights) {
  const auto unitfloat2idx_ants = [&](float v) {
    return unitfloat2idx(v, vec_labels_weights);
  };

  MatrixXi ifs_ants(evo_ants.rows(), evo_ants.cols());
  ifs_ants = evo_ants.unaryExpr(unitfloat2idx_ants).cast<int>();

  return ifs_ants;
}

MatrixXd FISEval::evo_mfs2ifs_mfs(const Map<MatXf> &evo_mfs,
                                  Map<MatXd> &mat_vars_ranges) {
  int rows = evo_mfs.rows();
  int cols = evo_mfs.cols();
  MatrixXd ifs_mfs(rows, cols);

  // FIXME output ifs_mfs is wrong !
  for (int i = 0; i < rows; i++) {
    VectorXd row_i = ifs_mfs.row(i);

    for (int j = 0; j < cols; j++) {
      row_i(j) = evo_mfs(i, j) * mat_vars_ranges(i, 0) + mat_vars_ranges(i, 1);
    }

    // sort MF per row to always have increasing p0,p1,pN
    sort(row_i.data(), row_i.data() + row_i.size());
    ifs_mfs.row(i) = row_i;
  }

  return ifs_mfs;
}

FISEval::FISEval(int ind_n, double *observations, int observations_r,
                 int observations_c, const int n_rules, int max_vars_per_rules,
                 int n_labels, int n_consequents, int *default_rule_cons,
                 int default_rule_cons_n, double *vars_range, int vars_range_r,
                 int vars_range_c, double *labels_weights, int labels_weights_n,
                 int dc_idx)
    : mat_obs(observations, observations_r, observations_c),
      mat_default_rule_cons_i(default_rule_cons, default_rule_cons_n),
      mat_vars_ranges(vars_range, vars_range_r, vars_range_c),
      vec_labels_weights(labels_weights, labels_weights_n) {

  this->ind_n = ind_n;

  this->observations_r = observations_r;
  this->observations_c = observations_c;

  this->n_rules = n_rules;
  this->max_vars_per_rules = max_vars_per_rules;
  this->n_labels = n_labels;
  this->n_true_labels = n_labels - 1;
  this->n_consequents = n_consequents;
  this->n_vars = observations_c;
  this->mat_default_rule_cons = mat_default_rule_cons_i.cast<double>();

  this->mf_values_eye = MatrixXd::Identity(n_labels, n_labels - 1);
  this->dc_idx = dc_idx;

  // set DC row to 1. This will neutralize the effect of AND_min
  this->mf_values_eye.row(dc_idx).setOnes();
}

FISEval::~FISEval() {}

double *FISEval::predict(float *ind) {
  // EXTRACT NEW INDIVIDUAL
  // ind is a float array that is the individual which represents a FIS.
  Map<MatXf> evo_mfs(ind, n_vars, n_true_labels);

  // offset where the antecedents values begin which is after the evo_mfs
  // values
  float *ind_offset_ants = ind + (n_vars * n_true_labels);
  Map<MatXf> evo_ants(ind_offset_ants, n_rules, n_vars);

  float *ind_offset_cons = ind_offset_ants + (n_rules * n_vars);
  Map<MatXf> evo_cons(ind_offset_cons, n_rules, n_consequents);

  // CONVERT EVOLUTION ANTS TO IFS ANTS
  MatrixXi ifs_ants = evo_ants2ifs_ants(evo_ants, vec_labels_weights);

  // CONVERT EVOLUTION MFS TO IFS MFS
  MatrixXd ifs_mfs = evo_mfs2ifs_mfs(evo_mfs, mat_vars_ranges);

  // CONVERT EVOLUTION CONS TO IFS CONS
  const auto binarize_mat = [&](float v) { return v >= 0.5 ? 1.0 : 0.0; };
  MatrixXd ifs_cons = evo_cons.unaryExpr(binarize_mat).cast<double>();

  // add default rule consequents to ifs_cons
  ifs_cons.conservativeResize(ifs_cons.rows() + 1, NoChange);
  ifs_cons.row(ifs_cons.rows() - 1) = mat_default_rule_cons;

  // create a row-major matrix here because we will returned it to Python
  MatXd defuzzified_outputs(observations_r, n_consequents);
  defuzzified_outputs.setConstant(NAN);

  const int N_ANTECEDENTS = ifs_ants.cols();

// EVALUATE
#pragma omp parallel for
  for (int i = 0; i < observations_r; i++) {
    const VectorXd obs = mat_obs.row(i);

    // a rule can be implicated in [0, 1]
    RowVectorXd rules_activations(n_rules + 1); // +1 is for default rule
    rules_activations.setConstant(0.0);

    for (int ri = 0; ri < n_rules; ri++) {
      VectorXi ants_ri = ifs_ants.row(ri);

      if ((ants_ri.array() == dc_idx).all()) {
        // ignore rule with all antecedents set to DONT_CARE
        // coutd << "ignored rule " << ri << " because all ants are DC" << endl;
        continue;
      }

      // an antecedent can be implicated in [0, 1]
      double min_antecedent_implication = 1.0;

      // coutd << "fuz ants" << endl;
      for (int ai = 0; ai < N_ANTECEDENTS; ai++) {
        // ai is the i-th antecedent. Each rule has as many antecedents
        // as there are variables, so the ai is also the i-th variable.
        // An unused variables is set to dc_idx (DONT_CARE)
        // coutd << "rule " << ri << ", ant " << ai << ", mf_values idx "
        // << ifs_ants(ai) << endl;
        VectorXd in_values = ifs_mfs.row(ai);
        VectorXd mf_values = mf_values_eye.row(ants_ri(ai));

        // TODO: implement an Eigen version of lininterp
        vector<double> xs(in_values.data(),
                          in_values.data() + in_values.size());
        vector<double> ys(mf_values.data(),
                          mf_values.data() + mf_values.size());

        const double fuzzified_x = LinearInterpolation::interp(xs, ys, obs(ai));
        min_antecedent_implication =
            min(min_antecedent_implication, fuzzified_x);
      }
      rules_activations(ri) = min_antecedent_implication;
    }
    // compute default rule activation
    rules_activations(rules_activations.size() - 1) =
        1 - rules_activations.maxCoeff();

    VectorXd defuzz_out =
        (rules_activations * ifs_cons) / rules_activations.sum();

    defuzzified_outputs.row(i) = defuzz_out;
  }

  // END EVALUATE

  // for some unknown reason, we cannot simple return the myMatrix.data() to
  // the caller (Python)
  // Guesses:
  // 1. NULL ptr is returned because once this function is finished, myMatrix
  // is deleted
  // 2. there are some kind of padding (strides?) when manipulating Eigen
  // matrices
  // 3. there is a problem of indexing (rowmajor vs columnmajor)
  // 4. there is a problem of iterating from one elm to another because the
  // step size is wrong (void* instead of double* ? padding ?)
  double *predictions_outputs = new double[observations_r * n_consequents];
  for (int i = 0; i < observations_r; i++) {
    for (int j = 0; j < n_consequents; j++) {
      predictions_outputs[i * n_consequents + j] = defuzzified_outputs(i, j);
    }
  }

  return predictions_outputs;
}
