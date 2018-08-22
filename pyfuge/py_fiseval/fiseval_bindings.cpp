#include "fiseval_bindings.h"

py::array_t<double> FISCocoEvalWrapper::predict_c(const string &ind_sp1,
                                                  const string &ind_sp2) {
  // double predict_c(const string &ind_sp1, const string &ind_sp2) {
  cout << "ind_sp1 " << ind_sp1 << " (" << ind_sp1.length() << ")" << endl;
  cout << "ind_sp2 " << ind_sp2 << " (" << ind_sp2.length() << ")" << endl;

  /// Parse ind_sp1
  // const vector<LinguisticVariable> vec_lv = parse_ind_sp1(ind_sp1);
  auto vec_lv = parse_ind_sp1(ind_sp1);

  /// Parse ind_sp2
  // TODO: parse ind_sp2


  const auto dummy_size_t = [](const size_t v, const size_t i, const size_t j) {
    return v * 10;
  };
  const auto dummy_double = [](const double v, const size_t i, const size_t j) {
    return v * 10.0;
  };
  // TODO: post parsing func: modulo_trick(v, n_vars)
  cout << "sel vars" << endl;
  const size_t n_bits_sel_vars = n_rules * n_max_vars_per_rule * n_bits_per_ant;
  cout << "c++ v: " << n_bits_sel_vars << endl;
  size_t offset = 0;
  string sel_vars_bits = ind_sp2.substr(offset, n_bits_sel_vars);
  auto sel_vars =
      parse_bit_array<size_t>(sel_vars_bits, n_rules, n_max_vars_per_rule,
                              n_bits_per_ant, dummy_size_t);

  cout << "r lv" << endl;
  const size_t n_bits_r_lv = n_rules * n_max_vars_per_rule * n_lv_per_ind;

  // TODO: post parsing func:None, no modulo_trick needed since it is already a
  // multiple of 2.
  cout << "c++ v: " << n_bits_r_lv << endl;
  offset += n_bits_sel_vars;
  string r_lv_bits = ind_sp2.substr(offset, n_bits_r_lv);
  auto r_lv = parse_bit_array<size_t>(r_lv_bits, n_rules, n_max_vars_per_rule,
                                      n_lv_per_ind, dummy_size_t);

  // todo add lambda function as paramter of parse bit array to directly convert
  // the parsed number to business logic

  // TODO: post parsing func: modulo_trick + dc_padding
  cout << "r labels" << endl;
  const size_t n_bits_r_labels =
      n_rules * n_max_vars_per_rule * n_bits_per_label;
  cout << "c++ v: " << n_bits_r_labels << endl;
  offset += n_bits_r_lv;
  string r_labels_bits = ind_sp2.substr(offset, n_bits_r_labels);
  auto r_labels =
      parse_bit_array<size_t>(r_labels_bits, n_rules, n_max_vars_per_rule,
                              n_bits_per_label, dummy_size_t);

  // TODO: post parsing func: scaling using cons_range + round/ceil/floor on
  // classification variables
  cout << "r cons" << endl;
  const size_t n_bits_r_cons = n_rules * n_cons * n_bits_per_cons;
  offset += n_bits_r_labels;
  string r_cons_bits = ind_sp2.substr(offset, n_bits_r_cons);
  auto r_cons = parse_bit_array<double>(r_cons_bits, n_rules, n_cons,
                                        n_bits_per_cons, toto);

  /// Combine ind_sp1 and ind_sp2 to create a FIS
  /*
  auto rules = ...
  SingletonFIS fis(rules, default_rule)
  */

  /// Use this FIS and predict the output given X_train/new_X
  // TODO: create a predict_c(ind_sp1, ind_sp2, new_X) overriding method

  /// Return the y_pred to the caller
  // return y_pred

  cout << "predict done" << endl;

  return vec_lv;
}

template <typename T>
vector<vector<T>> FISCocoEvalWrapper::parse_bit_array(
    const string &bitarray, const size_t rows, const size_t cols,
    const size_t n_bits_per_elm,
    const std::function<T(const T, const size_t row, const size_t col)> &func) {
  vector<vector<T>> matrix(rows, vector<T>(cols, 0));

  const size_t n_bits_per_line = cols * n_bits_per_elm;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      const size_t offset = i * n_bits_per_line + (j * n_bits_per_elm);
      T value = stoi(bitarray.substr(offset, n_bits_per_elm), nullptr, 2);
      matrix[i][j] = func(value, i, j);
      // matrix[i][j] = value;
      cout << matrix[i][j] << ",";
    }
    cout << endl;
  }

  return matrix;
}

py::array_t<double> FISCocoEvalWrapper::parse_ind_sp1(const string &ind_sp1) {

  // // TODO: substr() is creating a new string each time, use c++17's
  // // stringview?
  // vector<vector<double>> vec_lv;
  // vec_lv.reserve(n_lv_per_ind);

  const auto dummy_f = [](const double v, const size_t i, const size_t j) {
    return v;
  };

  vector<vector<double>> vec_lv = parse_bit_array<double>(
      ind_sp1, n_lv_per_ind, n_true_labels, n_bits_per_mf, dummy_f);

  cout << "a" << endl;
  size_t mf_index = 1; // TODO remove me
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

  for (size_t i = 0; i < n_lv_per_ind; i++) {
    for (size_t j = 0; j < n_true_labels; j++) {
      //// normalize in [0,1]
      // arr_raw(i, j) = vec_lv[i][j] / double((1 << n_bits_per_mf)-1);
      arr_raw(i, j) = vec_lv[i][j];
    }
  }
  return arr;
}
