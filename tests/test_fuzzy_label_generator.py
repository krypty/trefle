import pytest

from pyfuge.evo.helpers.fuzzy_labels_generator import generate_labels


def test_generate_less_than_2_labels_should_fail():
    with pytest.raises(ValueError):
        generate_labels(n_labels=1)


def test_generate_2_labels():
    assert generate_labels(n_labels=2) == ("low", "high")


def test_generate_4_labels():
    assert generate_labels(n_labels=4) == ("very low", "low", "medium", "high")


def test_generate_many_labels():
    assert generate_labels(n_labels=8) == (
        "3 very low",
        "very very low",
        "very low",
        "low",
        "medium",
        "high",
        "very high",
        "very very high",
    )
