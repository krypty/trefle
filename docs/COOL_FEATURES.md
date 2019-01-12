# Cool features of Trefle

This document showcases some cool features of Trefle.

## Grid Search

Since Trefle is compatible with scikit-learn you can perform a grid search
using `GridSearchCV`. You can see an example of that in the file
[/examples/grid_search_example.py](/examples/grid_search_example.py).

## TrefleClassifier parameters

`TrefleClassifier` class has a lot of parameters you can tune. Fortunately, most
of them use default values that "are good enough" for small problems.

There are two kinds of parameters, fuzzy systems and evolutionary parameters.
These parameters are listed in `TrefleClassifier` class and are detailed in
the classes [`CocoIndividual`](/trefle/evo/experiment/coco/coco_individual.py)
and [`CocoExperiment`](/trefle/evo/experiment/base/coco_experiment.py).

## Custom fitness function

You can change the fitness function to make Trefle generate models that match
your criteria. For example you might want to have models that have a high
sensibility.
Check [/examples/grid_search_example.py](/examples/grid_search_example.py) out.


## Import and export

