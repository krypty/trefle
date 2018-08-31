from enum import Enum, unique


@unique
class LabelEnum(Enum):
    """
    Do not use this class directly.

    members must be increasing and a DC member is mandatory and must be the last
    member.
    """

    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def is_dc(self):
        return self.name == "DC"


class Label2(LabelEnum):
    LOW = ()
    HIGH = ()
    DC = ()


class Label3(LabelEnum):
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    DC = ()


class Label4(LabelEnum):
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_HIGH = ()
    DC = ()


class Label5(LabelEnum):
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    DC = ()


class Label6(LabelEnum):
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_VERY_HIGH = ()
    DC = ()


class Label7(LabelEnum):
    VERY_VERY_LOW = ()
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_VERY_HIGH = ()
    DC = ()


class Label8(LabelEnum):
    VERY_VERY_LOW = ()
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_VERY_HIGH = ()
    VERY_VERY_VERY_HIGH = ()
    DC = ()


class Label9(LabelEnum):
    VERY_VERY_VERY_LOW = ()
    VERY_VERY_LOW = ()
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_VERY_HIGH = ()
    VERY_VERY_VERY_HIGH = ()
    DC = ()


class Label10(LabelEnum):
    VERY_VERY_VERY_LOW = ()
    VERY_VERY_LOW = ()
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_VERY_HIGH = ()
    VERY_VERY_VERY_HIGH = ()
    VERY_VERY_VERY_VERY_HIGH = ()
    DC = ()


class Label11(LabelEnum):
    VERY_VERY_VERY_VERY_LOW = ()
    VERY_VERY_VERY_LOW = ()
    VERY_VERY_LOW = ()
    VERY_LOW = ()
    LOW = ()
    MEDIUM = ()
    HIGH = ()
    VERY_VERY_HIGH = ()
    VERY_VERY_VERY_HIGH = ()
    VERY_VERY_VERY_VERY_HIGH = ()
    DC = ()


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

    print(Label3.__members__.values())
    print(Label9.__members__.values())

    a = Label4.DC
    print("issss", a.is_dc())
    print("issss", a.value)
