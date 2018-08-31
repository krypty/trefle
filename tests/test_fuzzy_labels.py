from pyfuge.evo.helpers.fuzzy_labels import Label3, Label2, Label4


def test_first_label_must_be_0():
    assert Label3.LOW.value == 0


def test_two_labels_from_different_classes_must_not_be_equal():
    assert Label3.LOW != Label2.LOW
    assert Label3.LOW != Label2.HIGH
    assert Label3.LOW == Label3.LOW
    assert Label4.VERY_HIGH != Label3.DC


def test_label_named_DC_must_return_true_using_is_dc():
    a = Label4.DC
    assert a.is_dc()
