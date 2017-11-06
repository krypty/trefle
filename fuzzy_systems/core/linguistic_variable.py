import numpy as np


class FreeShapeMF:
    """
    This class implements a membership function with free shape.
    The shape of the function is defined by giving a vector of input values
    and a the vector of corresponding membership values.
    Calling this class with a float number as parameter returns the
    degree of activation of the membership function for that value computed
    using interpolations between the two nearest known values.
    """

    def __init__(self, input_values, membership_values):
        """
        Two parameters needed:
        input_values: vector of input values
        membership_values: vector of membership values
        """
        assert len(input_values) == len(membership_values)
        for i in np.arange(1, len(input_values)):
            assert input_values[i - 1] <= input_values[i]
        self.input_values = input_values
        self.membership_values = membership_values

    def __call__(self, value):
        if value <= self.input_values[0]:
            return self.membership_values[0]
        elif value >= self.input_values[-1]:
            return self.membership_values[-1]
        else:
            # FIXME: do not interpolate, return the nearest input_value's
            # result if the caller wants more precision, he should create a
            # MF with more inputs/mf values
            i = 1
            while value > self.input_values[i]:
                i = i + 1
            i_p = (value - self.input_values[i - 1]) / float(
                self.input_values[i] - self.input_values[i - 1])
            return i_p * (
                self.membership_values[i] - self.membership_values[i - 1]) + \
                   self.membership_values[i - 1]

    def apply_to(self, input_values):
        return list(map(self, input_values))


class BaseLinguisticVariable:
    def __str__(self):
        to_return = self.name + '\n'
        for name, mf in self.membership_functions.items():
            to_return += name + ': ' + str(
                zip(mf.input_values, mf.membership_values)) + '\n'
        return to_return

    def get_linguistic_value(self, name):
        assert name in self.level_names
        return self.membership_functions[name].apply_to(self.input_values)

    def fuzzify(self, value):
        self.input_value = value
        self.membership_values = dict()
        for name, mf in self.membership_functions.items():
            self.membership_values[name] = mf(self.input_value)
        return self.membership_values


class LinguisticVariable(BaseLinguisticVariable):
    """
    This class implements a linguistic variable with some constraints
    in the definition in its membership functions.
    Linguistic values are shaped as trapezoidal membership functions,
    each membership function is defined by a a series of 2 or 4 points:

    1 _____p[0].  _ _ _ _ _ _      1 _ _ _  p[1].________.p[2] _ _ _ _temperature.fuzzify(19)
                \\                              /          \\
                 \\                            /            \\
                  \\                          /              \\
                   \\                        /                \\
    0 _ _ _ _ _ _ _ \\._______      0 _____./_ _ _ _ _ _ _ _ _ \\.______
                      p[1]                  p[0]                   p[3]

    Therefore, linguistic values are defined at initialization by specifying
    their transition points:

    1 _______   t[1]_______   t[3]___  t[5]______ . . .
             \   /         \   /     \  /
              \ /           \ /       \ /
               X             X         X
              / \           / \       / \
    0 _______/   \_________/  \_____/   \_________ . . .
           t[0]          t[2]     t[4]

    """

    def __init__(self, name, v_min, v_max, transitions, level_names=None,
                 res=0.1):
        """
        Four parameters needed:
        name: name of the variable
        v_min: minimum input value
        v_max: maximum input value
        transitions: list of values defining the starting and ending points of the linguistic values
        level_names: optional, name, or list of names of the linguistic values
        res: resolution
        """
        self.name = name
        assert v_min < v_max
        self.v_min = v_min
        self.v_max = v_max
        self.resolution = res
        self.input_values = np.arange(self.v_min, self.v_max, self.resolution)
        self.__set_transitions(transitions)
        self.__set_level_names(level_names)
        self.membership_functions = dict()
        self.membership_functions[self.level_names[0]] = FreeShapeMF(
            [self.transitions[0], self.transitions[1]],
            [1.0, 0.0])
        for i in np.arange(1, len(self.level_names) - 1):
            self.membership_functions[self.level_names[i]] = FreeShapeMF(
                [self.transitions[(i * 2) - 2],
                 self.transitions[(i * 2) - 1],
                 self.transitions[(i * 2) + 0],
                 self.transitions[(i * 2) + 1]],
                [0.0, 1.0, 1.0, 0.0])
        self.membership_functions[self.level_names[-1]] = FreeShapeMF(
            [self.transitions[-2], self.transitions[-1]],
            [0.0, 1.0])
        self.input_value = None
        self.membership_values = dict()

    def __set_transitions(self, transitions):
        n_transitions = len(transitions)
        assert n_transitions >= 2
        assert n_transitions % 2 == 0
        assert self.v_min <= transitions[0]
        assert transitions[-1] <= self.v_max
        for i in np.arange(1, n_transitions):
            assert transitions[i - 1] <= transitions[i]
        self.transitions = transitions

    def __set_level_names(self, level_names):
        if level_names is None:
            level_names = 'V_level_'
        if isinstance(level_names, str):
            level_names = [level_names]
        if len(level_names) == 1:
            if not level_names[0].endswith('_level_'):
                level_names[0] += '_level_'
            level_names = [level_names[0] + str(i) for i in
                           np.arange((len(self.transitions) / 2) + 1)]
        assert len(level_names) == (len(self.transitions) / 2) + 1
        self.level_names = level_names
