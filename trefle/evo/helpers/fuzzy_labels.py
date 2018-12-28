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

    def __len__(self):
        return self._parent.len()

    def __repr__(self):
        return "{} from class {}".format(self.value, self._parent.__name__)


class LabelEnum:
    pass


class Label2(LabelEnum):
    def LOW(self=None):
        return LElem(Label2, value=0)

    def HIGH(self=None):
        return LElem(Label2, value=1)

    def DC(self=None):
        return LElem(Label2, value=2, is_dc=True)

    @staticmethod
    def len():
        return 2


class Label3(LabelEnum):
    def LOW(self=None):
        return LElem(Label3, value=0)

    def MEDIUM(self=None):
        return LElem(Label3, value=1)

    def HIGH(self=None):
        return LElem(Label3, value=2)

    def DC(self=None):
        return LElem(Label3, value=3, is_dc=True)

    @staticmethod
    def len():
        return 3


class Label4(LabelEnum):
    def LOW(self=None):
        return LElem(Label4, value=0)

    def MEDIUM(self=None):
        return LElem(Label4, value=1)

    def HIGH(self=None):
        return LElem(Label4, value=2)

    def VERY_HIGH(self=None):
        return LElem(Label4, value=3)

    def DC(self=None):
        return LElem(Label4, value=4, is_dc=True)

    @staticmethod
    def len():
        return 4


class Label5(LabelEnum):
    def VERY_LOW(self=None):
        return LElem(Label5, value=0)

    def LOW(self=None):
        return LElem(Label5, value=1)

    def MEDIUM(self=None):
        return LElem(Label5, value=2)

    def HIGH(self=None):
        return LElem(Label5, value=3)

    def VERY_HIGH(self=None):
        return LElem(Label5, value=4)

    def DC(self=None):
        return LElem(Label5, value=5, is_dc=True)

    @staticmethod
    def len():
        return 5


class Label6(LabelEnum):
    def VERY_LOW(self=None):
        return LElem(Label6, value=0)

    def LOW(self=None):
        return LElem(Label6, value=1)

    def MEDIUM(self=None):
        return LElem(Label6, value=2)

    def HIGH(self=None):
        return LElem(Label6, value=3)

    def VERY_HIGH(self=None):
        return LElem(Label6, value=4)

    def VERY_VERY_HIGH(self=None):
        return LElem(Label6, value=5)

    def DC(self=None):
        return LElem(Label6, value=6, is_dc=True)

    @staticmethod
    def len():
        return 6


class Label7(LabelEnum):
    def VERY_VERY_LOW(self=None):
        return LElem(Label7, value=0)

    def VERY_LOW(self=None):
        return LElem(Label7, value=1)

    def LOW(self=None):
        return LElem(Label7, value=2)

    def MEDIUM(self=None):
        return LElem(Label7, value=3)

    def HIGH(self=None):
        return LElem(Label7, value=4)

    def VERY_HIGH(self=None):
        return LElem(Label7, value=5)

    def VERY_VERY_HIGH(self=None):
        return LElem(Label7, value=6)

    def DC(self=None):
        return LElem(Label7, value=7, is_dc=True)

    @staticmethod
    def len():
        return 7


class Label8(LabelEnum):
    def VERY_VERY_LOW(self=None):
        return LElem(Label8, value=0)

    def VERY_LOW(self=None):
        return LElem(Label8, value=1)

    def LOW(self=None):
        return LElem(Label8, value=2)

    def MEDIUM(self=None):
        return LElem(Label8, value=3)

    def HIGH(self=None):
        return LElem(Label8, value=4)

    def VERY_HIGH(self=None):
        return LElem(Label8, value=5)

    def VERY_VERY_HIGH(self=None):
        return LElem(Label8, value=6)

    def VERY_VERY_VERY_HIGH(self=None):
        return LElem(Label8, value=7)

    def DC(self=None):
        return LElem(Label8, value=8, is_dc=True)

    @staticmethod
    def len():
        return 8


class Label9(LabelEnum):
    def VERY_VERY_VERY_LOW(self=None):
        return LElem(Label9, value=0)

    def VERY_VERY_LOW(self=None):
        return LElem(Label9, value=1)

    def VERY_LOW(self=None):
        return LElem(Label9, value=2)

    def LOW(self=None):
        return LElem(Label9, value=3)

    def MEDIUM(self=None):
        return LElem(Label9, value=4)

    def HIGH(self=None):
        return LElem(Label9, value=5)

    def VERY_HIGH(self=None):
        return LElem(Label9, value=6)

    def VERY_VERY_HIGH(self=None):
        return LElem(Label9, value=7)

    def VERY_VERY_VERY_HIGH(self=None):
        return LElem(Label9, value=8)

    def DC(self=None):
        return LElem(Label9, value=9, is_dc=True)

    @staticmethod
    def len():
        return 9


class Label10(LabelEnum):
    def VERY_VERY_VERY_LOW(self=None):
        return LElem(Label10, value=0)

    def VERY_VERY_LOW(self=None):
        return LElem(Label10, value=1)

    def VERY_LOW(self=None):
        return LElem(Label10, value=2)

    def LOW(self=None):
        return LElem(Label10, value=3)

    def MEDIUM(self=None):
        return LElem(Label10, value=4)

    def HIGH(self=None):
        return LElem(Label10, value=5)

    def VERY_HIGH(self=None):
        return LElem(Label10, value=6)

    def VERY_VERY_HIGH(self=None):
        return LElem(Label10, value=7)

    def VERY_VERY_VERY_HIGH(self=None):
        return LElem(Label10, value=8)

    def VERY_VERY_VERY_VERY_HIGH(self=None):
        return LElem(Label10, value=9)

    def DC(self=None):
        return LElem(Label10, value=10, is_dc=True)

    @staticmethod
    def len():
        return 10


class Label11(LabelEnum):
    def VERY_VERY_VERY_VERY_LOW(self=None):
        return LElem(Label11, value=0)

    def VERY_VERY_VERY_LOW(self=None):
        return LElem(Label11, value=1)

    def VERY_VERY_LOW(self=None):
        return LElem(Label11, value=2)

    def VERY_LOW(self=None):
        return LElem(Label11, value=3)

    def LOW(self=None):
        return LElem(Label11, value=4)

    def MEDIUM(self=None):
        return LElem(Label11, value=5)

    def HIGH(self=None):
        return LElem(Label11, value=6)

    def VERY_HIGH(self=None):
        return LElem(Label11, value=7)

    def VERY_VERY_HIGH(self=None):
        return LElem(Label11, value=8)

    def VERY_VERY_VERY_HIGH(self=None):
        return LElem(Label11, value=9)

    def VERY_VERY_VERY_VERY_HIGH(self=None):
        return LElem(Label11, value=10)

    def DC(self=None):
        return LElem(Label11, value=11, is_dc=True)

    @staticmethod
    def len():
        return 11


if __name__ == "__main__":
    print(Label3.LOW().value)
    print(len(Label2.LOW()))

    toto = Label5

    assert issubclass(toto, LabelEnum)
