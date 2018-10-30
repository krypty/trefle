def generate_labels(n_labels):
    def _generate_more_than_two_labels(_n_labels):
        def get_labels(n, prefix, labels=None):
            if labels is None:
                _labels = []
            else:
                _labels = labels

            # base recursion case
            if n == -1:
                return _labels

            # other cases
            if n > 2:
                new_label = "{} very {}".format(n, prefix)
            else:
                new_label = n * "very " + prefix

            _labels.append(new_label)

            return get_labels(n - 1, prefix, _labels)

        k = _n_labels - 1  # minus one for the "medium" label
        k1 = k // 2
        k = k - k1
        return (
            *get_labels(k - 1, "low"),
            "medium",
            *reversed(get_labels(k1 - 1, "high")),
        )

    if n_labels < 2:
        raise ValueError("cannot generate less than 2 labels")
    if n_labels == 2:
        return ("low", "high")
    else:
        return _generate_more_than_two_labels(n_labels)
