import numpy as np

from pyfuge.evo.examples.evo_wine_classifier import load_wine_dataset
from pyfuge.evo.helpers import ifs_utils
from pyfuge.evo.helpers.native_ind_evaluator import NativeIndEvaluator


def simple_predict():
    import random
    random.seed(10)
    np.random.seed(10)

    np.set_printoptions(precision=6)

    ##
    ## LOAD DATASET
    ##
    ds_train, ds_test = load_wine_dataset(test_size=0.3)

    ##
    ## EXPERIMENT PARAMETERS
    ##
    # n_vars = ds_train.N_VARS
    n_rules = 4
    n_max_vars_per_rule = 2  # FIXME: don't ignore it
    mf_label_names = ["LOW", "HIGH", "DC"]
    default_rule_output = [0, 1, 0]  # [class_0, class_1, class_2]
    labels_weights = np.array([1, 1, 6])

    vars_range = np.empty((ds_train.X.shape[1], 2))
    vars_range[:, 0] = ds_train.X.ptp(axis=0)
    vars_range[:, 1] = ds_train.X.min(axis=0)

    n_labels = len(mf_label_names)

    # FIXME
    assert len(mf_label_names) == len(labels_weights)

    ind = [0.4014568272507354, 0.02086371261509734, 0.530756810210902,
           0.3093307757344431, 0.08247854740559168, 0.7019633839142663,
           0.926562433327638, 0.15737845617605617, 0.4787422533535671,
           0.0610712568686953, 0.4691174302744965, 0.04202259295859678,
           0.485411080870507, 0.1947562322392462, 0.2796546885297241,
           0.19321534983890798, 0.8936604011385352, 0.5079196233861747,
           0.14807002370423372, 0.49884630907319505, 0.8970395134423204,
           0.14810283014260428, 0.20260527436666653, 0.3993438824173391,
           0.8359652950866331, 0.5277703928781871, 0.16569094659560835,
           0.04154800624837074, 0.07500165639093082, 0.6088183675731145,
           0.20260527436666653, 0.2751761234518699, 0.26322505730724277,
           0.7404770294423513, 0.5423469084949141, 0.7850582352913449,
           0.33320955037131816, 0.800045802482336, 0.20144613206956785,
           0.5118653116546974, 0.9138359393486287, 0.5751809741545238,
           0.39534799576075363, 0.49472705633579606, 0.9345531513137126,
           0.07917707524907425, 0.7888612066117836, 0.7138797819170863,
           0.35577606365064884, 0.4456129199456278, 0.12103564578689219,
           0.2655354079951163, 0.03078118276387254, 0.10843043323231338,
           0.3993438824173391, 0.3499308379303727, 0.4114438718635758,
           0.06286530731930406, 0.7407613066437472, 0.4678698477895328,
           0.012701678061071364, 0.41970911924220033, 0.576052857622314,
           0.16732654271926128, 0.9357197249689737, 0.9280274704509104,
           0.36221890694318704, 0.22951369290821, 0.5079343214900371,
           0.02206681338701333, 0.11537646855553418, 0.207852814358936,
           0.20668270683981682, 0.572672161174031, 0.9018400559075261,
           0.22146349642572494, 0.5120046294389962, 0.7788752662194598,
           0.6864063822585972, 0.44139806968187045, 0.2666588261678885,
           0.08368468785634153, 0.027077398976345424, 0.5470223993105408,
           0.21373491313776627, 0.7295233734199151, 0.3420745660412098,
           0.07917707524907425, 0.14810283014260428, 0.5386148763532943]

    from time import time
    N = 100

    ind_evaluator = NativeIndEvaluator(len(ind), observations=ds_test.X,
                                       n_rules=n_rules,
                                       max_vars_per_rule=n_max_vars_per_rule,
                                       n_labels=n_labels,
                                       n_consequents=len(default_rule_output),
                                       default_rule_cons=np.array(
                                           default_rule_output),
                                       vars_ranges=vars_range,
                                       labels_weights=labels_weights)

    t0 = time()
    for _ in range(N):
        predicted_outputs = ind_evaluator.predict_native(ind)
    tCPP = time() - t0
    tCPP /= N

    t0 = time()
    for _ in range(N):
        py_predicted_outputs = ifs_utils.IFSUtils.predict(
            ind=ind,
            observations=ds_test.X,
            n_rules=n_rules,
            max_vars_per_rule=n_max_vars_per_rule,
            n_labels=n_labels,
            n_consequents=len(default_rule_output),
            default_rule_cons=np.array(default_rule_output),
            vars_ranges=vars_range,
            labels_weights=labels_weights,
        )
    tPy = time() - t0
    tPy /= N

    print("time c++    {:.3f} ms".format(tCPP * 1000))
    print("time python {:.3f} ms".format(tPy * 1000))
    print("speed up    {:.1f}".format(tPy / tCPP))

    is_close = np.allclose(predicted_outputs, py_predicted_outputs)
    print("is close C++/Python", is_close)

    # print(predicted_outputs)
    # print(py_predicted_outputs)


if __name__ == '__main__':
    simple_predict()
