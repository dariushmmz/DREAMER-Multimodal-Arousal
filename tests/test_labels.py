import pytest

from stress_classification.labels import score_to_three_class


@pytest.mark.parametrize(
    ("score", "expected"), [(1, 0), (2, 0), (3, 1), (4, 2), (5, 2)]
)
def test_three_class_mapping(score, expected):
    assert score_to_three_class(score) == expected


@pytest.mark.parametrize("score", [0, 6, -1])
def test_three_class_mapping_rejects_out_of_range(score):
    with pytest.raises(ValueError):
        score_to_three_class(score)

