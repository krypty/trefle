import pandas as pd
import numpy as np


class PFDataset:
    # add a version number in the case where serialization happen
    __dataset_version = 1

    # TODO: add helper function to
    # - transform categorical classes to numerical classes
    # - create pd.get_dummies(y)

    def __init__(self, X, y, X_names=None, y_names=None):
        """
        PyFUGE-formatted dataset. You might create a class or method that
        convert your dataset into this format in order to use PyFUGE.

        :param X: features/variables as an np.ndarray (NxM) where
        N is the number of cases/observations and M is the number of variables
        per observation. Invalid values must be set as np.nan

        :param y: labels/classes for each observation represented as an
        np.ndarray (Nx1) where N is the number of observations. Each class must
        be represented by a unique number. WARNING: should work great for
        numerical and categorical ordinal variables but not for categorical
        non-ordinal classes.

        :param X_names: name of the observations, if any. Might be useful to
        retrieve a particular observation afterwards
        :param y_names: name of the classes
        """
        self.X = X
        # self.y = y
        self.X_names = X_names
        self.y_names = y_names

        self.N_OBS = X.shape[0]
        self.N_VARS = X.shape[1]
        self.N_CLASSES = np.unique(y).shape[0]

        # print(y.shape[1])

        if y.ndim == 1:
            self.y = pd.get_dummies(y).values
        else:
            self.y = y

        # TODO make a Pandas DF ?
