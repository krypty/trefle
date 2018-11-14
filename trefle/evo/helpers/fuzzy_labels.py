class LElem:
    def __init__(self, parent, value, is_dc=False):
        self._parent = parent
        self._value = value
        self._is_dc = is_dc

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return self._parent == other._parent and self._value == other._value

    @property
    def value(self):
        return self._value

    def is_dc(self):
        return self._is_dc


class LabelEnum:
    pass


class Label2(LabelEnum):
    _name = "L2"
    LOW = LElem(_name, 0)
    HIGH = LElem(_name, 1)
    DC = LElem(_name, 2, is_dc=True)

    @staticmethod
    def len():
        return 2


class Label3(LabelEnum):
    _name = "L3"
    LOW = LElem(_name, 0)
    MEDIUM = LElem(_name, 1)
    HIGH = LElem(_name, 2)
    DC = LElem(_name, 3, is_dc=True)

    @staticmethod
    def len():
        return 3


class Label4(LabelEnum):
    _name = "L4"
    LOW = LElem(_name, 0)
    MEDIUM = LElem(_name, 1)
    HIGH = LElem(_name, 2)
    VERY_HIGH = LElem(_name, 3)
    DC = LElem(_name, 4, is_dc=True)

    @staticmethod
    def len():
        return 4


class Label5(LabelEnum):
    _name = "L5"
    VERY_LOW = LElem(_name, 0)
    LOW = LElem(_name, 1)
    MEDIUM = LElem(_name, 2)
    HIGH = LElem(_name, 3)
    VERY_HIGH = LElem(_name, 4)
    DC = LElem(_name, 5, is_dc=True)

    @staticmethod
    def len():
        return 5


class Label6(LabelEnum):
    _name = "L6"
    VERY_LOW = LElem(_name, 0)
    LOW = LElem(_name, 1)
    MEDIUM = LElem(_name, 2)
    HIGH = LElem(_name, 3)
    VERY_HIGH = LElem(_name, 4)
    VERY_VERY_HIGH = LElem(_name, 5)
    DC = LElem(_name, 6, is_dc=True)

    @staticmethod
    def len():
        return 6


class Label7(LabelEnum):
    _name = "L7"
    VERY_VERY_LOW = LElem(_name, 0)
    VERY_LOW = LElem(_name, 1)
    LOW = LElem(_name, 2)
    MEDIUM = LElem(_name, 3)
    HIGH = LElem(_name, 4)
    VERY_HIGH = LElem(_name, 5)
    VERY_VERY_HIGH = LElem(_name, 6)
    DC = LElem(_name, 7, is_dc=True)

    @staticmethod
    def len():
        return 7


class Label8(LabelEnum):
    _name = "L8"
    VERY_VERY_LOW = LElem(_name, 0)
    VERY_LOW = LElem(_name, 1)
    LOW = LElem(_name, 2)
    MEDIUM = LElem(_name, 3)
    HIGH = LElem(_name, 4)
    VERY_HIGH = LElem(_name, 5)
    VERY_VERY_HIGH = LElem(_name, 6)
    VERY_VERY_VERY_HIGH = LElem(_name, 7)
    DC = LElem(_name, 8, is_dc=True)

    @staticmethod
    def len():
        return 8


class Label9(LabelEnum):
    _name = "L9"
    VERY_VERY_VERY_LOW = LElem(_name, 0)
    VERY_VERY_LOW = LElem(_name, 1)
    VERY_LOW = LElem(_name, 2)
    LOW = LElem(_name, 3)
    MEDIUM = LElem(_name, 4)
    HIGH = LElem(_name, 5)
    VERY_HIGH = LElem(_name, 6)
    VERY_VERY_HIGH = LElem(_name, 7)
    VERY_VERY_VERY_HIGH = LElem(_name, 8)
    DC = LElem(_name, 9, is_dc=True)

    @staticmethod
    def len():
        return 9


class Label10(LabelEnum):
    _name = "L10"
    VERY_VERY_VERY_LOW = LElem(_name, 0)
    VERY_VERY_LOW = LElem(_name, 1)
    VERY_LOW = LElem(_name, 2)
    LOW = LElem(_name, 3)
    MEDIUM = LElem(_name, 4)
    HIGH = LElem(_name, 5)
    VERY_HIGH = LElem(_name, 6)
    VERY_VERY_HIGH = LElem(_name, 7)
    VERY_VERY_VERY_HIGH = LElem(_name, 8)
    VERY_VERY_VERY_VERY_HIGH = LElem(_name, 9)
    DC = LElem(_name, 10, is_dc=True)

    @staticmethod
    def len():
        return 10


class Label11(LabelEnum):
    _name = "L11"
    VERY_VERY_VERY_VERY_LOW = LElem(_name, 0)
    VERY_VERY_VERY_LOW = LElem(_name, 1)
    VERY_VERY_LOW = LElem(_name, 2)
    VERY_LOW = LElem(_name, 3)
    LOW = LElem(_name, 4)
    MEDIUM = LElem(_name, 5)
    HIGH = LElem(_name, 6)
    VERY_HIGH = LElem(_name, 7)
    VERY_VERY_HIGH = LElem(_name, 8)
    VERY_VERY_VERY_HIGH = LElem(_name, 9)
    VERY_VERY_VERY_VERY_HIGH = LElem(_name, 10)
    DC = LElem(_name, 11, is_dc=True)

    @staticmethod
    def len():
        return 11


if __name__ == "__main__":

    def toto(label):
        print(label)
        print("is zero", label.value == 0)
        print(isinstance(label, Label2))
        print("is dc", label.is_dc())
        print("")
        print("")
        print("")

    toto(Label2.LOW)
    toto(Label2.HIGH)
    toto(Label3.DC)
    toto(Label2.DC)
    print(Label2.DC)

    print("must be false", Label3.LOW == Label2.LOW)
    print("must be false", Label3.LOW == Label2.LOW)
    print("must be true", Label3.LOW == Label3.LOW)

    print(Label2.LOW)
    print(Label2.HIGH)
    print(Label2.DC)
    print(Label3.DC)

    # print(Label3.__members__.values())
    # print(Label9.__members__.values())

    a = Label4.DC
    print("issss", a.is_dc())
    print("issss", a.value)

    # print(len(Label5))
    # print(len(a.__class__))
    # print(a.len())
