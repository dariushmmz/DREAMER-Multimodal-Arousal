import numpy as np
import pytest

from stress_classification.splits import SubjectSplit, assert_no_group_overlap


def test_subject_split_is_disjoint_and_complete():
    split = SubjectSplit(train=(1, 2), validation=(3,), test=(4,))
    split.validate(range(1, 5))
    masks = split.masks(np.array([1, 2, 3, 4]))
    assert masks["train"].tolist() == [True, True, False, False]
    assert masks["validation"].tolist() == [False, False, True, False]
    assert masks["test"].tolist() == [False, False, False, True]


def test_subject_split_rejects_leakage():
    with pytest.raises(ValueError, match="leakage"):
        SubjectSplit(train=(1, 2), validation=(2, 3), test=(4,))


def test_explicit_group_check_rejects_overlap():
    with pytest.raises(AssertionError):
        assert_no_group_overlap({"train": [1, 2], "test": [2, 3]})

