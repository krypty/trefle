# Trefle — A scikit-learn compatible classifier using interpretable fuzzy systems

Trefle is a **scikit-learn compatible** estimator implementing the [FuzzyCoCo algorithm](#fuzzycoco-algorithm) that use a cooperative coevolution algorithm to find and build interpretable fuzzy systems.

TODO: basic usage using iris



If you have never heard of fuzzy systems before you can basically think of them as a set of rules giving a prediction after they have been evaluated.

## Installation

Start using Trefle today with pip !

```
pip install trefle
```

## Examples of use

TODO: examples of binary, multilabel, regression problem. Example of one-hot problem


## Cool features

* Support classification (binary and multiclass), regression and mixed (i.e. both classification and regression) problems
* Fully compatible scikit-learn estimator
    * Use it like a regular estimator
    * Support GridSearch
* Fuzzy systems parameters are customizable e.g. the number of rules, the number of linguistic labels per rule,...
* Evolutionary paramters are customizable e.g. number of generations, population size, hall of fame size,...
* Custom fitness function
* Import and Export the best fuzzy system for future use in an interoperable format
* Fine tune your best fuzzy system using the companion library [LFA Toolbox](https://github.com/iict/lfa_toolbox). Add or remove a fuzzy rule to increase either the performance or interpretability of the fuzzy system. Or tweak the membership functions.
* The fuzzy engine is implemented in C++14 allowing Trefle to be quite fast and use all the CPU cores
* Last but not least, Trefle is a recursive accronym like GNU which is cool. It stands for **T**refle is a **R**evised and **E**volutionary-based **F**uzzy **L**ogic **E**ngine. And trefle also means clover in French.

## What are fuzzy logic and FuzzyCoco algorithm?

### FuzzyCoCo algorithm

The following sentences are drawn from the PhD thesis "Coevolutionary Fuzzy Modeling" by Carlos Andrés PEÑA REYES that you can find [here](https://infoscience.epfl.ch/record/33110?ln=en).

>Fuzzy CoCo is a novel approach that combines two methodologies - fuzzy systems and coevolutionary algorithms - so as to automatically produce accurate and interpretable systems. The approach is based on two elements: (1) a system model capable of providing both accuracy and human understandability, and (2) an algorithm that allows to build such a model from available information.

In short, as a user this algorithm will give you a model that is interpretable and accurate (i.e. you can see how the model works) given a classification or a regression problem. From this model you can read the features that it extracted.

### How it works?

1. Load dataset
2. Configure experiment i.e. the number of rules, the number of generations and other fuzzy systems or evolutionary parameters
3. Create an initial population of fuzzy systems represented by a list of numbers
4. Run evolutionary algorithm. It will perform the following steps.
    1. Select
    2. Crossover
    3. Mutate
    4. Evaluate
    5. Repeat these steps until max generations is reached
5. Retrieve best individual i.e. the best fuzzy system
6. Use the fuzzy toolkit to visualize it

### Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from trefle.evo.skfuge.scikit_fuge import FugeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

# Organize our data
X = data['data']
y = data['target']

# Split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Initialize our classifier
clf = FugeClassifier(n_rules=2, n_generations=200, pop_size=300,
                     n_labels_per_mf=3, dont_care_prob=0.9, verbose=True)

# Train our classifier
model = clf.fit(X_train, y_train)

# Make predictions
preds = clf.predict(X_test)

fis = clf.get_best_fuzzy_system()
print(fis)

# Use the Fuzzy Toolkit part to view the resulting fuzzy system
FISViewer(fis).show()

# Evaluate accuracy
print("Score:")
print(accuracy_score(y_test, preds))
```

Other examples can be found in [pyfuge/evo/examples](pyfuge/evo/examples).

# Deployment and Tests

Both documentations are available in the `docs` folder.


# Credits

* Carlos Andrés PEÑA REYES
* Gary Marigliano
* [CI4CB Team](http://iict-space.heig-vd.ch/cpn/)
