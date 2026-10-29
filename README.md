# The Mood Machine

A mood classifier for short, social-media-style text. The project ships **two** classifiers — a hand-written rule-based model and a scikit-learn logistic-regression model — plus a held-out evaluation harness and a pytest regression suite that compare how each model handles real language phenomena like negation, sarcasm, slang, emoji, and mixed emotions.

## Project origin

This repo began as `ai110-module3tinker-themoodmachine-starter`, a starter lab for the AI110 course (Module 3 "Tinker"). The lab's goal was hands-on experience with how basic classifiers work, where they break, and how small modeling choices affect accuracy and fairness. The starter shipped a rule-based `MoodAnalyzer` skeleton with TODOs, a 12-post labeled dataset, and an optional scikit-learn extension; this final version completes the rule-based implementation, expands the dataset to 63 labeled posts across 7 language categories, and adds a held-out evaluation harness, a regression test suite, a model card, and system diagrams.

## Architecture overview

```plaintext
├── dataset.py         # Word lists (POSITIVE/NEGATIVE/MIXED) + 63 labeled SAMPLE_POSTS/TRUE_LABELS
├── mood_analyzer.py   # Rule-based MoodAnalyzer: preprocess -> score_text -> predict_label -> explain
├── ml_experiments.py  # ML model: CountVectorizer (bag of words) -> LogisticRegression
├── main.py            # Entry point: rule-based eval on SAMPLE_POSTS + batch demo + interactive loop
├── test_cases.py      # 27 held-out TEST_CASES grouped by category (never used for training)
├── evaluate.py        # Harness: runs both models on TEST_CASES, reports metrics + confusion matrices
├── test_mood_machine.py # Pytest regression suite (python -m pytest -v)
├── model_card.md      # Filled-out model card comparing both models
└── diagrams/          # system_diagram.md (mermaid) + confusion_matrices.png
```

**Rule-based pipeline** (`mood_analyzer.py`): `preprocess` strips, lowercases, tokenizes, and removes punctuation; `score_text` adds +1 per positive word, −1 per negative word, flips the following word after `"not"`, and lets mixed words pull the score toward 0; `predict_label` maps score > 0 to `positive`, < 0 to `negative`, 0 to `neutral`.

**ML pipeline** (`ml_experiments.py`): a `CountVectorizer` converts text to bag-of-words features and a `LogisticRegression` classifier is trained on `SAMPLE_POSTS`/`TRUE_LABELS`. It can emit all five labels, including `mixed` and `sarcasm`, which the rule-based model cannot produce.

**Evaluation** (`evaluate.py`): runs both models over the held-out `TEST_CASES` and reports overall and per-category accuracy, per-label precision/recall/F1, a side-by-side prediction table, the train/test accuracy gap, and saves confusion matrices to `diagrams/confusion_matrices.png`. See `diagrams/system_diagram.md` for the full component diagram.

## Setup and running the code

Requires Python 3.10+ (developed on 3.14). Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the rule-based model (dataset evaluation, batch demo, then an interactive prompt — type `quit` or press Enter to exit):

```bash
python main.py
```

Train and try the ML model:

```bash
python ml_experiments.py
```

Evaluate both models on the held-out test set:

```bash
python evaluate.py
```

## Sample interactions

Interactive rule-based session (`python main.py`):

```text
=== Interactive Mood Machine (rule based) ===
Type a sentence to analyze its mood.
Type 'quit' or press Enter on an empty line to exit.

You: I love this class so much
Model: positive
You: I am not happy about this
Model: negative
You: great, my car broke down again
Model: positive          <- sarcasm fools the rule-based model
You: quit
Goodbye from the Mood Machine.
```

Held-out evaluation (`python evaluate.py`, abridged):

```text
Rule based: held-out test results
Overall accuracy: 0.52 (14/27)
  clear 1.00 | negation 1.00 | neutral 1.00 | slang 0.25 | sarcasm/emoji/mixed 0.00

ML (logistic regression): held-out test results
Overall accuracy: 0.67 (18/27)
  mixed 1.00 | negation 1.00 | emoji 0.75 | slang 0.75 | clear 0.50 | sarcasm/neutral 0.33

ML model: training accuracy = 1.00, held-out accuracy = 0.67
```

## Design decisions

- **Two-model comparison.** The rule-based model makes each modeling choice explicit and inspectable; the ML model shows what a learned bag-of-words classifier picks up from the same data. Keeping both lets failures be compared directly.
- **Negation by token consumption.** In `score_text`, `"not"` inspects the next token, flips its polarity, and marks it consumed so it is not double-counted. This handles "not happy" → negative and "not bad" → positive, and a trailing `"not"` is guarded so it cannot crash (covered by a regression test).
- **Mixed words dampen the score.** `MIXED_WORDS` ("but", "kinda", "so-so", ...) pull a nonzero score one step toward 0 rather than voting positive or negative — a compromise, since `predict_label` still cannot emit a `mixed` label.
- **Three labels for rules, five for ML.** The rule-based model only returns `positive`/`negative`/`neutral`; the ML model can also return `mixed` and `sarcasm` because those labels exist in `TRUE_LABELS`.
- **Strictly held-out test set.** `TEST_CASES` never appear in `SAMPLE_POSTS`, so `evaluate.py` measures generalization rather than memorization — which is exactly what exposes the ML model's 1.00-train vs. 0.67-test overfitting gap.
- **`xfail` for known limitations.** Rather than hiding failures, the test suite marks sarcasm, emoji, mixed, and slang cases as expected failures; a fix turns them into visible XPASS results.

## Testing summary

`python -m pytest test_mood_machine.py -v` — 25 tests: **11 passed, 13 xfailed, 1 xpassed**.

- **Passing:** dataset/label alignment and validity, category coverage (≥3 cases per category), rule-based correctness on clear, neutral, and negation cases, the trailing-`"not"` regression, valid-label guarantees, ML training-set fit (≥0.9 accuracy), and the `accuracy_by_category` harness itself.
- **Expected failures (`xfail`):** rule-based handling of sarcasm (3), emoji (4), mixed (3), and slang (3) — documented limitations of a lexicon-based scorer.
- **XPASS:** one slang case ("highkey stressed about finals") passes unexpectedly — it contains "stressed", a known negative word.

`python evaluate.py` on the 27 held-out cases: rule-based accuracy **0.52** (perfect on clear/negation/neutral, zero on sarcasm/emoji/mixed), ML accuracy **0.67** (perfect on mixed/negation, weak on neutral/sarcasm, and it even misses half the clear cases — a sign of overfitting to the small training set).

## Reflection and docs

- `model_card.md` — completed model card covering data, behavior, limitations, and improvement ideas.
- `diagrams/system_diagram.md` — mermaid diagram of components, data flow, and the testing suite.
- `diagrams/confusion_matrices.png` — side-by-side confusion matrices for both models on the held-out set.
