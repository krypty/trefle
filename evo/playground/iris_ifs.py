from evo.dataset.pf_dataset import PFDataset
from evo.experiment.base.simple_experiment import SimpleEAExperiment
from evo.experiment.pyfuge_simple_ea_ind2ifs import PyFUGESimpleEAInd2IFS
from evo.fitness_evaluator.pyfuge_fitness_evaluator import \
    PyFUGEFitnessEvaluator


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
    y = pd.get_dummies(y).values
    # print(y[0])
    y_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)

    return (PFDataset(X_train, y_train, X_names, y_names),
            PFDataset(X_test, y_test, X_names, y_names))


if __name__ == '__main__':
    from time import time

    # from datetime import datetime
    #
    # print(str(datetime.now()))
    t0 = time()
    tick = lambda: print((time() - t0) * 1000)

    ##
    ## TRAINING PHASE
    ##
    ds_train, ds_test = Iris2PFDataset()

    pyfuge_ind_2_ifs = PyFUGESimpleEAInd2IFS(
        n_vars=ds_train.N_VARS,
        n_rules=3,
        n_max_var_per_rule=4,
        mf_label_names=["LOW", "MEDIUM", "HIGH", "DC"],
        default_rule_output=[0, 0, 1],  # [setosa, versicolor, virginica]
        dataset=ds_train,
    )

    SimpleEAExperiment(
        dataset=ds_train,
        ind2ifs=pyfuge_ind_2_ifs,
        fitevaluator=PyFUGEFitnessEvaluator(),
        N_POP=300,
        N_GEN=5
    )

    ##
    ## VALIDATION PHASE
    ##

    tick()
