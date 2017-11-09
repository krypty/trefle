from fuzzy_systems.core.membership_functions.free_shape_mf import FreeShapeMF


class SingletonMF(FreeShapeMF):
    """
    Syntactic sugar to create a singleton output fuzzy set
    """
    def __init__(self, in_value):
        super(SingletonMF, self).__init__(mf_values=[1], in_values=[in_value])
