# Trefle — A scikit-learn compatible classifier using interpretable fuzzy systems

Trefle is a **scikit-learn compatible** estimator implementing the [FuzzyCoCo algorithm](#fuzzycoco-algorithm) that uses a cooperative coevolution algorithm to find and build interpretable fuzzy systems.

Here is a basic example using Wisconsin Breast Cancer Dataset, a binary classification problem, from scikit-learn:
```python
import random
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from trefle.fitness_functions.output_thresholder import round_to_cls
from trefle.trefle_classifier import TrefleClassifier

np.random.seed(0)
random.seed(0)

# Load dataset
data = load_breast_cancer()

# Organize our data
X = data["data"]
y = data["target"]
y = np.reshape(y, (-1, 1))  # output needs to be at least 1 column wide

# Split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Declare the fitness function we want to use. Fitness value: higher is better.
def fit(y_true, y_pred):
    # y_pred are floats in [0, n_classes-1]. To use accuracy metric we need
    # to binarize the output using round_to_cls()
    y_pred_bin = round_to_cls(y_pred, n_classes=2)
    return accuracy_score(y_true, y_pred_bin)

# Initialize our classifier
clf = TrefleClassifier(
    n_rules=4,
    n_classes_per_cons=[2],  # a list where each element indicates the number of classes a consequent has. Specify 0 if one consequent is a continuous (regression) value.
    n_labels_per_mf=3,
    default_cons=[0],  # the default rule will use the class 0
    n_max_vars_per_rule=3,
    n_generations=20,
    fitness_function=fit,
    verbose=False,
)

# Train our classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict_classes(X_test)

clf.print_best_fuzzy_system()

# Evaluate accuracy
score = accuracy_score(y_test, y_pred)
print("Score on test set: {:.3f}".format(score))
```

This will output the fuzzy system:

```
IF v0 is low AND v5 is medium AND v16 is low THEN [0]
IF v25 is high AND v9 is high AND v14 is medium THEN [0]
IF v6 is high THEN [0]
IF v21 is low AND v23 is low THEN [1]
ELSE [0]
Variables definition
v0: -0.366, -0.347, -0.343,
v5: 0.155, 2.03, 2.03,
v6: 0.0756, 0.151, 1.36,
v9: 5.06, 11.2, 16.6,
v14: 5.89, 34.2, 37.2,
v16: 0.0815, 0.652, 1.06,
v21: -0.299, -0.294, -0.294,
v23: -0.0555, -0.0553, -0.0553,
v25: 0.193, 0.568, 0.631,

Score on test set: 0.910
```


If you have never heard of fuzzy systems before you can basically think of them as a set of rules giving a prediction after they have been evaluated. For example "IF temperature is HIGH and sunshine is MEDIUM THEN tourists is MEDIUM".

## Installation

Start using Trefle today with pip :-)

```
pip install trefle
```

## Examples of use

See other examples in the **examples** folder.

* [Binary problem](examples/01_binary_problem.py)
* [Multiclass problem](examples/02_multiclass_problem.py)
* [Multiclass one-hot problem](examples/03_multiclass_one_hot_problem.py)
* [Regression problem](examples/04_regression_problem.py)


## Cool features

* Support classification (binary and multiclass), regression and mixed (i.e. both classification and regression) problems
* Fully compatible scikit-learn estimator
    * Use it like a regular estimator
    * Support [GridSearch](docs/COOL_FEATURES.md#grid-search)
* [Fuzzy systems parameters](docs/COOL_FEATURES.md#trefleclassifier-parameters) are customizable e.g. the number of rules, the number of linguistic labels per rule,...
* [Evolutionary parameters](docs/COOL_FEATURES.md#trefleclassifier-parameters) are customizable e.g. number of generations, population size, hall of fame size,...
* [Custom fitness function](docs/COOL_FEATURES.md#custom-fitness-function)
* [Import and Export](docs/COOL_FEATURES.md#import-and-export) the best fuzzy system for future use in an interoperable format
* Fine tune your best fuzzy system using the companion library [LFA Toolbox](https://github.com/krypty/lfa_toolbox). Add or remove a fuzzy rule to increase either the performance or interpretability of the fuzzy system. Or tweak the membership functions.
* The fuzzy engine is implemented in C++14 allowing Trefle to be quite fast and use all the CPU cores
* Last but not least, Trefle is a recursive acronym like GNU which is cool. It stands for **T**refle is a **R**evised and **E**volutionary-based **F**uzzy **L**ogic **E**ngine. And trefle also means clover in French.

## What are fuzzy logic and FuzzyCoco algorithm?

### FuzzyCoCo algorithm

The following sentences are drawn from the PhD thesis "Coevolutionary Fuzzy Modeling" by Carlos Andrés PEÑA REYES that you can find [here](https://infoscience.epfl.ch/record/33110?ln=en).

>Fuzzy CoCo is a novel approach that combines two methodologies - fuzzy systems and coevolutionary algorithms - so as to automatically produce accurate and interpretable systems. The approach is based on two elements: (1) a system model capable of providing both accuracy and human understandability, and (2) an algorithm that allows to build such a model from available information.

In short, as a user this algorithm will give you a model that is interpretable and accurate (i.e. you can see how the model works) given a classification or a regression problem. From this model you can read the features that it extracted.

### How it works?

1. Load dataset
2. Configure experiment i.e. the number of rules, the number of generations and other fuzzy systems or evolutionary parameters
3. Create two initial populations (also called "species"; one for the fuzzy rules and the other for the
variables definition). Both represent individuals as a list of bits.
4. Run evolutionary algorithm. It will perform the following steps.
    1. Select
    2. Crossover
    3. Mutate
    4. Evaluate by combining individuals from a population with representatives
    of the other population to form a fuzzy system
    5. Save the best couple (i.e. the combination of an individual from pop1 and one from pop2)
    6. Repeat these steps until max generations is reached
5. Retrieve best individual couple i.e. the best fuzzy system
6. Optionally use the [LFA Toolbox](https://github.com/krypty/lfa_toolbox) to visualize or fine tune it

# Deployment and Tests

Both are available in the `docs` folder.

# Build from sources

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

# Where is the doc?!

There is no documentation like a Sphinx one. Start by looking in the [docs](docs) folder or directly in the source code of `TrefleClassifier`.

# Credits

* [Gary Marigliano](https://github.com/krypty)
* Carlos Andrés PEÑA REYES
* [CI4CB Team](http://iict-space.heig-vd.ch/cpn/)

![ci4cb_logo](assets/img/ci4cb_logo.png)
![heigvd_logo](assets/img/heigvd_logo.png)
