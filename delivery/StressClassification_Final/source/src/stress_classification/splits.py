"""Subject-aware split definitions and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


def _ids(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


@dataclass(frozen=True)
class SubjectSplit:
    """Immutable train/validation/test subject partition."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    name: str = "subject_independent_holdout_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", _ids(self.train))
        object.__setattr__(self, "validation", _ids(self.validation))
        object.__setattr__(self, "test", _ids(self.test))
        self.validate()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Iterable[int]], name: str | None = None) -> "SubjectSplit":
        return cls(
            train=tuple(values["train"]),
            validation=tuple(values["validation"]),
            test=tuple(values["test"]),
            name=name or str(values.get("name", "subject_independent_holdout_v1")),
        )

    @property
    def all_subjects(self) -> tuple[int, ...]:
        return tuple(sorted(self.train + self.validation + self.test))

    def validate(self, expected_subjects: Iterable[int] | None = None) -> None:
        partitions = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        for name, values in partitions.items():
            if not values:
                raise ValueError(f"{name} subject set is empty")
            if min(values) < 1:
                raise ValueError("subject identifiers must be 1-based positive integers")
        names = tuple(partitions)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                shared = partitions[left] & partitions[right]
                if shared:
                    raise ValueError(f"subject leakage between {left} and {right}: {sorted(shared)}")
        if expected_subjects is not None:
            expected = {int(value) for value in expected_subjects}
            actual = set(self.all_subjects)
            if actual != expected:
                raise ValueError(
                    f"split coverage mismatch; missing={sorted(expected-actual)}, "
                    f"unexpected={sorted(actual-expected)}"
                )

    def partition_for(self, subject_id: int) -> str:
        value = int(subject_id)
        if value in self.train:
            return "train"
        if value in self.validation:
            return "validation"
        if value in self.test:
            return "test"
        raise KeyError(f"subject {value} is not assigned to this split")

    def masks(self, subject_ids: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(subject_ids, dtype=np.int32)
        return {
            "train": np.isin(values, self.train),
            "validation": np.isin(values, self.validation),
            "test": np.isin(values, self.test),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def assert_no_group_overlap(partitions: Mapping[str, Iterable[int]]) -> None:
    """Raise if any group occurs in more than one named partition."""

    owners: dict[int, str] = {}
    for name, groups in partitions.items():
        for group in groups:
            value = int(group)
            if value in owners:
                raise AssertionError(
                    f"group {value} occurs in both {owners[value]} and {name}"
                )
            owners[value] = name

