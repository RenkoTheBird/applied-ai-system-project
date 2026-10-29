# evaluate.py
"""
Evaluation harness for the Mood Machine.

This script measures how well each model *really* performs by running
both classifiers on the held-out TEST_CASES in test_cases.py. These
posts are never used to train the ML model, so the numbers here show
generalization instead of memorization.

What it reports for each model:
  - Overall accuracy on the held-out test set
  - Per-category accuracy (clear, negation, sarcasm, slang, ...)
  - Precision / recall / F1 for every label
  - A side-by-side comparison of every single prediction
  - Confusion matrices saved to confusion_matrices.png

Usage:
    python evaluate.py
"""

from collections import Counter
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)

from dataset import SAMPLE_POSTS, TRUE_LABELS
from ml_experiments import predict_single_text, train_ml_model
from mood_analyzer import MoodAnalyzer
from test_cases import TEST_CASES

# Fixed label order so both models' reports line up for comparison.
LABELS = ["positive", "negative", "neutral", "mixed", "sarcasm"]


def accuracy_by_category(
    cases: List[Dict[str, str]],
    preds: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy separately for each test case category.

    This is the key metric for the Mood Machine: overall accuracy can
    look fine while the model fails completely on sarcasm or emoji.
    """
    totals: Counter = Counter()
    correct: Counter = Counter()
    for case, pred in zip(cases, preds):
        category = case["category"]
        totals[category] += 1
        if pred == case["label"]:
            correct[category] += 1
    return {cat: correct[cat] / totals[cat] for cat in sorted(totals)}


def evaluate_model(
    name: str,
    predict: Callable[[str], str],
    cases: List[Dict[str, str]],
) -> List[str]:
    """
    Run one model over the test cases and print a full report.

    `predict` is any function that takes a text string and returns a
    mood label, so this works for both the rule based model and the
    ML model.

    Returns the list of predicted labels.
    """
    trues = [case["label"] for case in cases]
    preds = [predict(case["text"]) for case in cases]

    num_correct = sum(p == t for p, t in zip(preds, trues))
    print(f"\n{'=' * 60}")
    print(f"{name}: held-out test results")
    print(f"{'=' * 60}")
    print(
        f"Overall accuracy: {accuracy_score(trues, preds):.2f} "
        f"({num_correct}/{len(cases)})"
    )

    print("\nAccuracy by category:")
    category_counts = Counter(case["category"] for case in cases)
    for category, accuracy in accuracy_by_category(cases, preds).items():
        n = category_counts[category]
        print(f"  {category:<12} {accuracy:.2f}  ({int(round(accuracy * n))}/{n})")

    print("\nClassification report:")
    print(classification_report(trues, preds, labels=LABELS, zero_division=0))

    return preds


def print_comparison(
    cases: List[Dict[str, str]],
    rule_preds: List[str],
    ml_preds: List[str],
) -> None:
    """
    Print every test case with both models' predictions side by side.
    Wrong predictions are marked with an 'x'.
    """
    print(f"\n{'=' * 60}")
    print("Side-by-side comparison (x = wrong)")
    print(f"{'=' * 60}")
    header = f"{'text':<47} {'true':<9} {'rule':<9} {'ml':<9}"
    print(header)
    print("-" * len(header))
    for case, rule_pred, ml_pred in zip(cases, rule_preds, ml_preds):
        text = case["text"]
        if len(text) > 44:
            text = text[:41] + "..."
        true = case["label"]
        rule_mark = " " if rule_pred == true else "x"
        ml_mark = " " if ml_pred == true else "x"
        print(
            f"{text:<47} {true:<9} "
            f"{rule_pred + ' ' + rule_mark:<9} {ml_pred + ' ' + ml_mark:<9}"
        )


def save_confusion_matrices(
    trues: List[str],
    rule_preds: List[str],
    ml_preds: List[str],
    path: str = "confusion_matrices.png",
) -> None:
    """
    Save a side-by-side confusion matrix figure for both models.

    Rows are the true labels, columns are the predicted labels. A
    perfect model has all its counts on the diagonal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, preds, title in zip(
        axes,
        [rule_preds, ml_preds],
        ["Rule based", "ML (logistic regression)"],
    ):
        ConfusionMatrixDisplay.from_predictions(
            trues, preds, labels=LABELS, ax=ax, colorbar=False
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Confusion matrices on held-out TEST_CASES")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nConfusion matrices saved to {path}")


def main() -> None:
    print("Evaluating both models on held-out TEST_CASES")
    print(f"({len(TEST_CASES)} cases across {len(set(c['category'] for c in TEST_CASES))} categories)")

    # The rule based model needs no training.
    analyzer = MoodAnalyzer()
    rule_preds = evaluate_model("Rule based", analyzer.predict_label, TEST_CASES)

    # The ML model trains on SAMPLE_POSTS, then is tested on posts it
    # has never seen.
    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
    ml_preds = evaluate_model(
        "ML (logistic regression)",
        lambda text: predict_single_text(text, vectorizer, model),
        TEST_CASES,
    )

    # Show the gap between training accuracy and test accuracy. A big
    # gap means the model memorized the training data (overfitting).
    train_preds = [
        predict_single_text(text, vectorizer, model) for text in SAMPLE_POSTS
    ]
    train_acc = accuracy_score(TRUE_LABELS, train_preds)
    test_acc = accuracy_score([c["label"] for c in TEST_CASES], ml_preds)
    print(f"\nML model: training accuracy = {train_acc:.2f}, "
          f"held-out accuracy = {test_acc:.2f}")
    print("A large gap means the model memorized the training examples "
          "instead of learning patterns that generalize.")

    print_comparison(TEST_CASES, rule_preds, ml_preds)

    trues = [case["label"] for case in TEST_CASES]
    save_confusion_matrices(trues, rule_preds, ml_preds)


if __name__ == "__main__":
    main()
