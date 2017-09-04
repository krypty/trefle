import numpy as np


class FuzzyRule:
    """
    This class implements a fuzzy rule.
    A single type of operation per rule is allowed. Yo can choose between:
    AND - minimum
    AND - product
    AND - bounded product
    OR  - maximum
    OR  - probabilistic sum
    OR  - bounded sum
    """

    __fuzzy_operations_names = {
        'AND_min': 'AND',
        'AND_prod': 'AND',
        'AND_boundprod': 'AND',
        'OR_max': 'OR',
        'OR_probsum': 'OR',
        'OR_boundsum': 'OR'
    }
    __fuzzy_operations = {
        'AND_min': np.min,
        'AND_prod': np.prod,
        'AND_boundprod': lambda x: np.max([0, np.sum(x) - 1]),
        'OR_max': np.max,
        'OR_probsum': lambda x: np.sum(x) - np.prod(x),
        'OR_boundsum': lambda x: np.min([1, np.sum(x)])
    }
    __fuzzy_implication = {
        'MIN': np.minimum,
        'PROD': np.multiply
    }

    def __init__(self, operation, antecedent, consequent, implication):
        """
        Three parameters are needed:
        operation: the fuzzy operation to perform
        antecedent: a list of tuples [(linguistic_variable, linguistic_value),...] defining the input fuzzy condition
        consequent: a tuple (linguistic_variable, linguistic_value) defining the output fuzzy assignement
        """
        assert operation in self.__fuzzy_operations.keys()
        assert implication in self.__fuzzy_implication.keys()
        self.operation = operation
        self.antecedent = antecedent
        self.consequent = consequent
        self.implication = implication
        self.antecedent_activation = 0.0
        self.consequent_activation = np.zeros(len(consequent[0].input_values))

    def __str__(self):
        to_return = 'Fuzzy rule:\n\tIF '
        for i, pair in enumerate(self.antecedent):
            to_return += pair[0].name + ' IS ' + pair[1]
            if i < (len(self.antecedent) - 1):
                to_return += ' ' + \
                             self.__fuzzy_operations_names[self.operation] + ' '
        to_return += '\n\tTHEN ' + \
                     self.consequent[0].name + ' is ' + self.consequent[1]
        to_return += '\n\tAntecedent activation: ' + \
                     str(self.antecedent_activation)
        return to_return

    def compute_antecedent_activation(self, input_values):
        """
        This function computes the activation of the antecedent of the rule.
        The first step is the fuzzification of the input values. Then, the activation
        is computed by applying the fuzzy operation to the values of the  membership functions.
        """
        temp = []
        for pair in self.antecedent:
            val = input_values.get(pair[0].name)
            if val is not None:
                membership_values = pair[0].fuzzify(val)
                temp.append(membership_values[pair[1]])
        if len(temp) == 0:
            self.antecedent_activation = 0.0
        else:
            self.antecedent_activation = self.__fuzzy_operations[
                self.operation](temp)
        return self.antecedent_activation

    def compute_consequent_activation(self):
        """
        This function applies the causal implication operator in order to compute
        the activation of the rule's consequent.
        """
        self.consequent_activation = self.consequent[0].get_linguistic_value(
            self.consequent[1])
        self.consequent_activation = self.__fuzzy_implication[self.implication](
            self.antecedent_activation, self.consequent_activation)
        return self.consequent_activation
