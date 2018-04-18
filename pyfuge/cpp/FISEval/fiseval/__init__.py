# import ctypes as C

import numpy as np
import pyfuge_c

np.set_printoptions(precision=2, suppress=True)


# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# _fiseval_wrapper = C.cdll.LoadLibrary(
#     os.path.join(PARENT_DIR, "cpp/build/fiseval_wrapper.so"))


# def _get_np_array_2d_ptr(arr, ctype_type):
#     ptr_type = C.POINTER(ctype_type)
#     n_rows = len(arr)
#     return (ptr_type * n_rows)(*[row.ctypes.data_as(ptr_type) for row in arr])


# def _get_np_array_1d_ptr(arr, ctype_type):
#     # return _get_np_array_2d_ptr(arr, ctype_type)
#     return _get_array_1d_ptr(arr, ctype_type)


# def _get_array_1d_ptr(arr, ctype_type):
#     return (ctype_type * len(arr))(*arr)
#
#
# def _get_np_array_1d_ptr(arr, ctype_type):
#     try:
#         arr.shape[1]
#     except IndexError:
#         assert False, "arr must not be a 1D array !"
#     return arr.ctypes.data_as(C.POINTER(ctype_type))


def predict_native(ind, observations, n_rules, max_vars_per_rule, n_labels,
                   n_consequents, default_rule_cons, vars_ranges,
                   labels_weights,
                   dc_idx):
    y_preds = pyfuge_c.bind_predict(
        np.array(ind, dtype=np.float32), observations,
        n_rules, max_vars_per_rule, n_labels, n_consequents,
        default_rule_cons, vars_ranges, labels_weights,
        dc_idx
    )

    return y_preds

# def predict_native(ind, observations, n_rules, max_vars_per_rule, n_labels,
#                    n_consequents, default_rule_cons, vars_ranges,
#                    labels_weights,
#                    dc_idx):
#     # fix weird bug when dealing with 1D int numpy array
#     default_rule_cons = default_rule_cons.tolist()
#
#     # FIXME use .ravel() to pass 1d array to c++
#     # TODO find a way to avoid copy when passing to c++ (id(a) != id(a.ravel())
#
#     kwargs = OrderedDict([
#         ("ind", _get_array_1d_ptr(ind, C.c_float)),
#         ("ind_n", len(ind)),
#         ("observations", (_get_array_1d_ptr(observations.ravel(), C.c_double))),
#         ("observations_r", observations.shape[0]),
#         ("observations_c", observations.shape[1]),
#         ("n_rules", n_rules),
#         ("max_vars_per_rules", max_vars_per_rule),
#         ("n_labels", n_labels),
#         ("n_consequents", n_consequents),
#         ("default_rule_cons", _get_array_1d_ptr(default_rule_cons, C.c_int)),
#         ("default_rule_cons_n", len(default_rule_cons)),
#         ("vars_range", _get_np_array_1d_ptr(vars_ranges, C.c_double)),
#         ("vars_range_r", vars_ranges.shape[0]),
#         ("vars_range_c", vars_ranges.shape[1]),
#         ("labels_weights",
#          _get_array_1d_ptr(labels_weights.astype(np.float64), C.c_double)),
#         ("labels_weights_n", len(labels_weights)),  # FIXME: same as n_labels ?
#         ("dc_idx", dc_idx)
#     ])
#
#     out_rows = observations.shape[0]
#     out_cols = n_consequents
#
#     # print("ind", ind)
#     # print("observations")
#     # print(observations)
#     # print(observations.shape)
#     # print("n_rules", n_rules)
#     # print("max_vars_per_rule", max_vars_per_rule)
#     # print("n_labels", n_labels)
#     # print("n_consequents", n_consequents)
#     # print("default_rule_cons", default_rule_cons)
#     # print("vars_ranges\n", vars_ranges)
#     # print("labels_weights", labels_weights)
#     # print("dc_idx", dc_idx)
#
#     # print("shape", observations.shape)
#     # res = f(c_ind, len(ind))
#
#     f = _fiseval_wrapper.c_predict
#     f.restype = C.POINTER(C.c_double)
#     res_ptr = f(*kwargs.values())  # cannot pass keyword args to native function
#     # f(*kwargs.values())  # cannot pass keyword args to native function
#
#     ptr = C.cast(res_ptr, C.POINTER(C.c_double * (out_rows * out_cols)))
#     res = np.frombuffer(ptr.contents)
#
#     # print("res", res.reshape(-1, n_consequents))
#     # print("res", res.shape)
#
#     # arr = (C.c_double * 4).from_address(
#     #     C.addressof(res_ptr.contents))
#     # res = np.ndarray(buffer=arr, dtype=np.double, shape=(4,1),
#     #                  order="C")
#     # print("res Python", res)
#
#     # print(res_ptr)
#     # res = np.ctypeslib.as_array(
#     #     res_ptr, shape=(observations.shape[0], n_consequents),
#     # )
#     # res = np.fromiter(res_ptr, dtype=np.double,
#     #                   count=observations.shape[0] * n_consequents).reshape(-1, n_consequents)
#
#     # img_buffer = (C.c_double * (out_rows) *
#     #               (out_cols)).from_address(C.addressof(res_ptr.contents))
#     # res = np.ndarray(buffer=img_buffer, dtype=np.double, shape=(out_rows,out_cols))
#     # print(res)
#
#     # ArrayType = C.c_double * out_rows * out_cols
#     # addr = C.addressof(res_ptr.contents)
#     # a = np.frombuffer(ArrayType.from_address(addr))
#     # print("res", a)
#
#     # arr = (C.c_double * out_rows * out_cols).from_address(
#     #     C.addressof(res_ptr.contents))
#     # res = np.ndarray(buffer=arr, dtype=np.double, shape=(out_rows, out_cols),order="C")
#     # print("res Python", res)
#
#     # res = np.frombuffer(
#     #     (C.c_float * (observations.shape[0] * n_consequents)).from_address(
#     #         res_ptr), np.double)
#
#     # print("res\n", res)
#
#     # buf = np.ones(shape=(out_rows, out_cols))
#     # arr = np.frombuffer(res_ptr, dtype=C.c_double, count=out_cols * out_rows,
#     #                     offset=0)
#     # print(arr)
#     # print("py res", np.sum(observations[:, 0]))
#
#     # n_obs = observations.shape[0]
#     # n_vars = observations.shape[1]
#     # evo_mfs, evo_ants, evo_cons = IFSUtils.extract_ind_new(ind, n_vars,
#     #                                                        n_labels, n_rules,
#     #                                                        n_consequents)
#     # print("evo_mfs shape", evo_mfs.shape)
#     # print("evo_mfs", evo_mfs)
#     # print("evo_ants shape", evo_ants.shape)
#     # print("evo_ants", evo_ants)
#     #
#     # print("evo_cons shape", evo_cons.shape)
#     # print("evo_cons", evo_cons)
#     #
#     # ifs_mfs = IFSUtils.evo_mfs2ifs_mfs_new(evo_mfs, vars_ranges)
#     # print("ifs mfs")
#     # print(ifs_mfs)
#     #
#     # ifs_ants = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)
#     # print("ifs_ants")
#     # print(ifs_ants)
#     #
#     # ifs_evo = IFSUtils.evo_cons2ifs_cons(evo_cons)
#     # print("ifs_evo")
#     # print(ifs_evo)
#
#     # from time import sleep
#     # sleep(0.1)
#     # assert False, "trololo"
#
#     predicted_outputs = res.reshape(-1, n_consequents)
#     return predicted_outputs
