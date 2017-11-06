import numpy as np

from fuzzy_systems.core.fuzzy_rule import FuzzyRule
from fuzzy_systems.core.linguistic_variable import LinguisticVariable


class FIS:
    """
    This class implements a Fuzzy Inference System (FIS).
    Available aggregators are:
    - OR_max: maximum
    - OR_probsum: probabilistic sum
    - OR_boundsum: bounded sum
    Available defuzzifiers are:
    - COA: center of areas
    - MOM: minimum of maximums
    """

    @staticmethod
    def __mom(v, m):
        i = np.argmax(m)
        y_inf = v[min(i)]
        y_sup = v[max(i)]
        return (y_inf + y_sup) / 2.0

    __fuzzy_aggregators = {
        'OR_max': np.maximum,
        'OR_probsum': lambda x, y: np.add(x, y) - np.multiply(x, y),
        'OR_boundsum': lambda x, y: np.minimum(1, np.add(x, y))
    }

    __fuzzy_defuzzifiers = {
        'COA': lambda v, m: np.sum(np.multiply(v, m)) / np.sum(m),
        'MOM': __mom
    }

    def __init__(self, rules, aggregator='OR_max', defuzzifier='COA'):
        """
        Three parameters are needed:
        rules: a list of objects of type FuzzyRule containing the rules of the system
        aggregator: the fuzzy operator to be used to aggregate the rules outputs
        defuzzifier: the defuzzifier function to use
        """
        self.rules = rules
        self.input_variables = set()
        for r in self.rules:
            for a in r.antecedent:
                self.input_variables.add(a[0])
            self.output_variable = r.consequent[0]

        assert aggregator in self.__fuzzy_aggregators.keys()
        self.aggregator = self.__fuzzy_aggregators[aggregator]

        assert defuzzifier in self.__fuzzy_defuzzifiers.keys()
        self.defuzzifier = defuzzifier

        self.input_values = dict()
        self.fuzzified_output = np.zeros(len(self.output_variable.input_values))
        self.defuzzified_output = 0.0

    def __str__(self):
        to_return = 'Input variables:\n'
        to_return += '\t' + \
                     str([i_v.name for i_v in self.input_variables]) + '\n'
        to_return += 'Output variables:\n'
        to_return += '\t' + self.output_variable.name + '\n'
        to_return += 'Rules:\n'
        for r in self.rules:
            to_return += '\t' + str(r) + '\n'
        return to_return

    def compute_antecedent_activations(self, input_values):
        """
        This function computes the activation of the antecedent of all rules.
        """
        self.input_values = input_values
        for r in self.rules:
            r.compute_antecedent_activation(input_values)

    def compute_consequent_activations(self):
        """
        This function computes the activation of the consequent of all rules.
        """
        for r in self.rules:
            r.compute_consequent_activation()

    def aggregate(self):
        """
        This function performs the aggregation of the rules outputs
        """
        self.fuzzified_output = np.zeros(len(self.output_variable.input_values))
        for r in self.rules:
            # GARY perform aggregation two by two
            self.fuzzified_output = self.aggregator(
                self.fuzzified_output, r.consequent_activation)

    def defuzzify(self):
        """
        This function defuzzifies the fuzzified_output of the system
        """
        self.defuzzified_output = self.__fuzzy_defuzzifiers[self.defuzzifier](
            self.output_variable.input_values, self.fuzzified_output)
        return self.defuzzified_output


def main():
    temperature = LinguisticVariable('temperature', 0, 35,
                                        [17, 20, 26, 29],
                                        ['cold', 'warm', 'hot'], res=1.0)

    sunshine = LinguisticVariable('sunshine', 0, 100, [30, 50, 50, 100],
                                     ['cloudy', 'partsunny', 'sunny'],
                                     res=1.0)

    tourists = LinguisticVariable('tourists', 0, 100, [0, 50, 50, 100],
                                     ['low', 'medium', 'high'], res=1.0)

    rule_1 = FuzzyRule('OR_max',
                          [(temperature, 'hot'), (sunshine, 'sunny')],
                          (tourists, 'high'), 'MIN')

    rule_2 = FuzzyRule('AND_min',
                          [(temperature, 'warm'), (sunshine, 'partsunny')],
                          (tourists, 'medium'),
                          'MIN')

    rule_3 = FuzzyRule('OR_max',
                          [(temperature, 'cold'), (sunshine, 'cloudy')],
                          (tourists, 'low'), 'MIN')

    # print(rule_1)
    # print(rule_2)
    # print(rule_3)

    tourist_prediction_example = FIS([rule_1, rule_2, rule_3])

    # print(tourist_prediction_example)

    input_values = {'temperature': 19, 'sunshine': 60}
    tourist_prediction_example.compute_antecedent_activations(input_values)
    # print(tourist_prediction_example)

    # print("comp cons act")
    tourist_prediction_example.compute_consequent_activations()

    # print("aggr")
    tourist_prediction_example.aggregate()

    print("defuz", tourist_prediction_example.defuzzify())


def tip_problem_hector():
    quality = LinguisticVariable('quality', 0, 10,
                                    [0, 5],
                                    ['poor', 'average', 'good'], res=1.0)

    service = LinguisticVariable('service', 0, 10,
                                    [0, 5],
                                    ['poor', 'average', 'good'], res=1.0)

    tip = LinguisticVariable('tip', 0, 25,
                                [0, 13, 15],
                                ['low', 'medium', 'high'], res=1.0)

    rule_1 = FuzzyRule('OR_max', [(quality, 'poor'), (service, 'poor')],
                          (tip, 'low'), 'MIN')

    rule_2 = FuzzyRule('AND_min', [(service, 'average')],
                          (tip, 'medium'), 'MIN')

    rule_3 = FuzzyRule('OR_max', [(service, 'good'), (quality, 'good')],
                          (tip, 'high'), 'MIN')

    # print(rule_1)
    # print(rule_2)
    # print(rule_3)

    tourist_prediction_example = FIS([rule_1, rule_2, rule_3])

    # print(tourist_prediction_example)

    input_values = {'quality': 6.5, 'service': 9.8}
    tourist_prediction_example.compute_antecedent_activations(input_values)
    # print(tourist_prediction_example)

    # print("comp cons act")
    tourist_prediction_example.compute_consequent_activations()

    # print("aggr")
    tourist_prediction_example.aggregate()

    print("defuz", tourist_prediction_example.defuzzify())


if __name__ == '__main__':
    # main()
    tip_problem_hector()
