# test_mood_machine.py
"""
Regression tests for the Mood Machine.

These tests protect the behaviors that already work and document the
behaviors that are still broken. Tests for known limitations are marked
with `xfail`, so the suite stays green while honestly recording what
each model cannot do yet.

Run with:
    python -m pytest test_mood_machine.py -v
"""

from collections import Counter

import pytest
from sklearn.metrics import accuracy_score

from dataset import SAMPLE_POSTS, TRUE_LABELS
from evaluate import accuracy_by_category
from ml_experiments import predict_single_text, train_ml_model
from mood_analyzer import MoodAnalyzer
from test_cases import CATEGORIES, TEST_CASES

VALID_LABELS = {"positive", "negative", "neutral", "mixed", "sarcasm"}
EXPECTED_CATEGORIES = {
    "clear", "negation", "sarcasm", "slang", "emoji", "mixed", "neutral",
}


@pytest.fixture(scope="module")
def analyzer() -> MoodAnalyzer:
    return MoodAnalyzer()


@pytest.fixture(scope="module")
def ml_model():
    """Train the ML model once and share it across tests."""
    return train_ml_model(SAMPLE_POSTS, TRUE_LABELS)


def cases_in(category: str):
    return [case for case in TEST_CASES if case["category"] == category]


# ---------------------------------------------------------------------
# Data integrity: the datasets must be usable at all.
# ---------------------------------------------------------------------

def test_sample_data_is_aligned():
    """Every post needs exactly one label."""
    assert len(SAMPLE_POSTS) == len(TRUE_LABELS)


def test_all_labels_are_valid():
    for label in TRUE_LABELS:
        assert label in VALID_LABELS, f"unexpected label: {label}"
    for case in TEST_CASES:
        assert case["label"] in VALID_LABELS, f"unexpected label: {case['label']}"


def test_test_set_covers_every_category():
    """Each phenomenon needs enough cases to say something meaningful."""
    counts = Counter(case["category"] for case in TEST_CASES)
    for category in EXPECTED_CATEGORIES:
        assert counts[category] >= 3, (
            f"category '{category}' has {counts[category]} cases; "
            "add at least 3"
        )
    assert set(counts) == EXPECTED_CATEGORIES == set(CATEGORIES)


# ---------------------------------------------------------------------
# Rule based model: behaviors that must keep working.
# ---------------------------------------------------------------------

def test_rule_based_clear_cases(analyzer):
    for case in cases_in("clear"):
        assert analyzer.predict_label(case["text"]) == case["label"], case["text"]


def test_rule_based_neutral_cases(analyzer):
    for case in cases_in("neutral"):
        assert analyzer.predict_label(case["text"]) == case["label"], case["text"]


def test_rule_based_negation(analyzer):
    """'not' flips the mood of the word that follows it."""
    for case in cases_in("negation"):
        assert analyzer.predict_label(case["text"]) == case["label"], case["text"]


def test_trailing_not_does_not_crash(analyzer):
    """Regression test: 'not' at the end used to raise IndexError."""
    assert analyzer.predict_label("I am not") in VALID_LABELS
    assert analyzer.predict_label("not") in VALID_LABELS


def test_rule_based_predictions_are_valid_labels(analyzer):
    for case in TEST_CASES:
        assert analyzer.predict_label(case["text"]) in VALID_LABELS


# ---------------------------------------------------------------------
# Rule based model: known limitations, expected to fail (xfail).
# If a limitation gets fixed, the test reports XPASS so you notice.
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", cases_in("sarcasm"),
    ids=[case["text"][:40] for case in cases_in("sarcasm")],
)
@pytest.mark.xfail(
    reason="rule based model cannot express a 'sarcasm' label", strict=False
)
def test_rule_based_sarcasm(analyzer, case):
    assert analyzer.predict_label(case["text"]) == case["label"]


@pytest.mark.parametrize(
    "case", cases_in("emoji"),
    ids=[case["text"][:40] for case in cases_in("emoji")],
)
@pytest.mark.xfail(
    reason="preprocess() strips or ignores emojis", strict=False
)
def test_rule_based_emoji(analyzer, case):
    assert analyzer.predict_label(case["text"]) == case["label"]


@pytest.mark.parametrize(
    "case", cases_in("mixed"),
    ids=[case["text"][:40] for case in cases_in("mixed")],
)
@pytest.mark.xfail(
    reason="predict_label() never returns 'mixed'", strict=False
)
def test_rule_based_mixed(analyzer, case):
    assert analyzer.predict_label(case["text"]) == case["label"]


@pytest.mark.parametrize(
    "case", cases_in("slang"),
    ids=[case["text"][:40] for case in cases_in("slang")],
)
@pytest.mark.xfail(
    reason="starter word lists do not cover slang", strict=False
)
def test_rule_based_slang(analyzer, case):
    assert analyzer.predict_label(case["text"]) == case["label"]


# ---------------------------------------------------------------------
# ML model: sanity checks and the train/test gap.
# ---------------------------------------------------------------------

def test_ml_model_fits_training_data(ml_model):
    """A model that cannot fit its training data is broken, not honest."""
    vectorizer, model = ml_model
    preds = [predict_single_text(t, vectorizer, model) for t in SAMPLE_POSTS]
    assert accuracy_score(TRUE_LABELS, preds) >= 0.9


def test_ml_predictions_are_valid_labels(ml_model):
    vectorizer, model = ml_model
    for case in TEST_CASES:
        assert predict_single_text(case["text"], vectorizer, model) in VALID_LABELS


# ---------------------------------------------------------------------
# The evaluation harness itself.
# ---------------------------------------------------------------------

def test_accuracy_by_category_counts_correctly():
    cases = [
        {"text": "a", "label": "positive", "category": "x"},
        {"text": "b", "label": "negative", "category": "x"},
        {"text": "c", "label": "neutral", "category": "y"},
    ]
    accuracies = accuracy_by_category(cases, ["positive", "positive", "neutral"])
    assert accuracies == {"x": 0.5, "y": 1.0}
