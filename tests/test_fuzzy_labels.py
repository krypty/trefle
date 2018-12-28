from trefle.evo.helpers.fuzzy_labels import Label3, Label2, Label4


def test_first_label_must_be_0():
    assert Label3.LOW().value == 0


def test_two_labels_from_different_classes_must_not_be_equal():
    assert Label3.LOW() != Label2.LOW()
    assert Label3.LOW() != Label2.HIGH()
    assert Label3.LOW() == Label3.LOW()
    assert Label4.VERY_HIGH() != Label3.DC()


def test_length_of_different_fuzzy_labels_are_different():
    assert len(Label3.LOW()) != len(Label4.LOW())


def test_length_of_same_fuzzy_labels_are_equal():
    assert len(Label3.LOW()) == len(Label3.HIGH())


def test_label_named_DC_must_return_true_using_is_dc():
    a = Label4.DC()
    assert a.is_dc()
