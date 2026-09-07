import math

from sampling_comparison.summarize import percentile, truth


def test_percentile_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert math.isclose(percentile([0.0, 10.0], 0.95), 9.5)


def test_truth_is_strict_and_case_insensitive():
    assert truth("True")
    assert truth("yes")
    assert not truth("False")
    assert not truth("")

