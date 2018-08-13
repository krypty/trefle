#include "fiseval_bindings.h"

py::array_t<double> FISCocoEvalWrapper::predict_c(const string &ind_sp1,
                                                  const string &ind_sp2) {
  // double predict_c(const string &ind_sp1, const string &ind_sp2) {
  cout << "ind_sp1 " << ind_sp1 << " (" << ind_sp1.length() << ")" << endl;
  cout << "ind_sp2 " << ind_sp2 << " (" << ind_sp2.length() << ")" << endl;

  /// Parse ind_sp1
  // const vector<LinguisticVariable> vec_lv = parse_ind_sp1(ind_sp1);
  auto vec_lv = parse_ind_sp1(ind_sp1);

  return vec_lv;

  /// Parse ind_sp2
  // TODO: parse ind_sp2

  /// Combine ind_sp1 and ind_sp2 to create a FIS
  /*
  auto rules = ...
  SingletonFIS fis(rules, default_rule)
  */

  /// Use this FIS and predict the output given X_train/new_X
  // TODO: create a predict_c(ind_sp1, ind_sp2, new_X) overriding method

  /// Return the y_pred to the caller
  // return y_pred
}

py::array_t<double> FISCocoEvalWrapper::parse_ind_sp1(const string &ind_sp1) {

  // TODO: substr() is creating a new string each time, use c++17's
  // stringview?
  vector<vector<double>> vec_lv;
  vec_lv.reserve(n_lv_per_ind);

  const int n_bits_per_line = n_true_labels * n_bits_per_mf;

  for (int lv_i = 0; lv_i < n_lv_per_ind; lv_i++) {
    // cout << "lv_i " << lv_i << endl;
    // vec_lv[lv_i].reserve(n_true_labels);
    vector<double> vec_lv_i;

    for (int mf_i = 0; mf_i < n_true_labels; mf_i++) {
      // cout << "lv_i " << lv_i << endl;
      const int offset = lv_i * n_bits_per_line + (mf_i * n_bits_per_mf);
      // cout << "offset " << offset << endl;
      double v = stoi(ind_sp1.substr(offset, n_bits_per_mf), nullptr, 2);
      // normalize v
      v /= (1 << n_bits_per_mf) - 1;
      cout << "v: " << v << endl;
      vec_lv_i.push_back(v);
    }

    vec_lv.push_back(vec_lv_i);
    cout << endl;
  }

  cout << "a" << endl;
  int mf_index = 1; // TODO remove me
  for (auto &row : vec_lv) {
    // create each LV
    // MF's points must be increasing
    sort(row.begin(), row.end());
    TriLV lv(row);
    cout << "lv : " << lv << endl;
    cout << "fuzzified to : " << lv.fuzzify(mf_index, 0.967742) << endl;
    for (const auto &col : row) {
      cout << col << ", ";
    }
    cout << endl;
  }

  // TODO: remove me, return instead y_pred
  auto arr = py::array_t<double>({n_lv_per_ind, n_true_labels});
  auto arr_raw = arr.mutable_unchecked<2>();

  for (int i = 0; i < n_lv_per_ind; i++) {
    for (int j = 0; j < n_true_labels; j++) {
      //// normalize in [0,1]
      // arr_raw(i, j) = vec_lv[i][j] / double((1 << n_bits_per_mf)-1);
      arr_raw(i, j) = vec_lv[i][j];
    }
  }
  return arr;
}
