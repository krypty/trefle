from evo.Evo import SimpleEAExperiment, PFDataset, FitnessEvaluator, IFS, \
    Ind2IFS


def Iris2PFDataset():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    fname = r"../fuzzy_systems/examples/iris/iris.data"
    iris_dataset = pd.read_csv(fname, sep=",",
                               names=["SL", "SW", "PL", "PW", "OUT"])

    X_names = ["SL", "SW", "PL", "PW"]
    X = iris_dataset[X_names].values

    # e.g. "Iris-Setosa" -> "Setosa"
    y = iris_dataset[["OUT"]].apply(axis=1, func=lambda x: x[0][5:]).values

    # print(X.shape)
    # print(y.shape)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return (PFDataset(X_train, y_train, X_names, y_names),
            PFDataset(X_test, y_test, X_names, y_names))


if __name__ == '__main__':
    class YoloFitnessEvaluator(FitnessEvaluator):

        @staticmethod
        def _compute_metric(y_pred, y_true):
            return -((y_pred - y_true) ** 2).mean(axis=None)

        def eval(self, ifs: IFS, dataset: PFDataset):
            X = dataset.X
            y_true = dataset.y
            y_preds = []
            for i, x in enumerate(X):
                # FIXME: replace dict-like prediction by array-based solution
                out = fis.predict(
                    {iris_dataset.columns[i]: xi for i, xi in enumerate(x)}
                )

                y_true_class = dataset.y_names[y_true[i]]
                y_preds.append(out[y_true_class])

            return self._compute_metric(y_preds, y_true)


    class YoloSimpleEAInd2IFS(Ind2IFS):
        # import numpy as np
        # DC_WEIGHT = 3
        # N_ANTS_LABELS = 4  # L, M, H and DC
        # arr = np.arange(N_ANTS_LABELS)
        #
        # repeat_shape = np.ones_like(N_ANTS_LABELS)
        # repeat_shape[-1] = DC_WEIGHT # [1 1 1 ... DC_WEIGHT]
        # arr.repeat(repeat_shape)

        def __init__(self, n_vars, n_rules, n_max_var_per_rule,
                     mf_label_names):
            assert n_max_var_per_rule <= n_vars

            super(YoloSimpleEAInd2IFS, self).__init__()

            # note: multi-objective is not supported ATM
            n_consequent_per_rule = 1

            mf_label_names = ["LOW", "MEDIUM", "HIGH"]
            mf_labels = len(mf_label_names)
            self._ind_len = 0
            self._ind_len += mf_labels * n_vars  # [pl0, pm0, ph0, pl1, pm1, ph1,..]
            self._ind_len += n_max_var_per_rule * n_rules  # [a0r0, a1r0, a2r0, a0r1...]
            self._ind_len += n_consequent_per_rule * n_rules

        def convert(self, ind):
            pass


    ##
    ## TRAINING PHASE
    ##
    ds_train, ds_test = Iris2PFDataset()

    yolo_ind_2_ifs = YoloSimpleEAInd2IFS(
        n_vars=ds_train.N_VARS,
        n_rules=3,
        n_max_var_per_rule=2,
        mf_label_names=["LOW", "MEDIUM", "FABULOUS"]
    )

    SimpleEAExperiment(
        dataset=ds_train,
        ind2ifs=yolo_ind_2_ifs,
        fitevaluator=YoloFitnessEvaluator(),
        # individual_length=yolo_ind_2_ifs.IND_LENGTH,
        default_rule_output=None  # TODO
    )

    ##
    ## VALIDATION PHASE
    ##
