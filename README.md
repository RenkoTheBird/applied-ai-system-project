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
├── diagrams/          # system_diagram.md (mermaid) + confusion_matrices.png
└── logs/              # Captured run transcripts (see "Reproducible execution evidence")
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

## Reproducible execution evidence

The `logs/` directory holds real transcripts captured in `.venv` (Python 3.14.6, scikit-learn 1.9.1, macOS arm64, 2026-10-29). Each file starts with the command that produced it; for the two interactive programs, the piped stdin lines are listed at the top as `stdin>` entries, since a piped session does not echo typed input back.

| Log file | Command that produced it |
| --- | --- |
| `logs/rule_based_session.txt` | `python main.py` (inputs piped to stdin) |
| `logs/ml_session.txt` | `python ml_experiments.py` (inputs piped to stdin) |
| `logs/evaluation.txt` | `python evaluate.py` |
| `logs/test_suite.txt` | `python -m pytest test_mood_machine.py -v` |

### Sample command executions

```bash
# Rule-based model: dataset eval + batch demo + interactive session
python main.py <<'EOF'
I love this class so much
I am not happy about this
great, my car broke down again
just passed my driving test :)
I am happy about the job but sad to leave my friends
the meeting is at 3pm
I am not
quit
EOF

# ML model: trains on SAMPLE_POSTS, prints training eval, then interactive session
python ml_experiments.py <<'EOF'
I love this class so much
great, my car broke down again
I am happy about the job but sad to leave my friends
the meeting is at 3pm
quit
EOF

python evaluate.py                        # held-out metrics + confusion_matrices.png
python -m pytest test_mood_machine.py -v  # regression/guardrail suite
```

### Example inputs and outputs

Interactive tail of `logs/rule_based_session.txt` — each `Model:` line answers the `stdin>` lines in order:

```text
You: Model: positive    <- "I love this class so much"      (clear positive)
You: Model: negative    <- "I am not happy about this"      (negation handled)
You: Model: positive    <- "great, my car broke down again" (sarcasm fools it)
You: Model: neutral     <- "just passed my driving test :)" (emoji ignored)
You: Model: negative    <- "I am happy about the job but sad to leave my friends"
You: Model: neutral     <- "the meeting is at 3pm"          (neutral)
You: Model: neutral     <- "I am not"                       (trailing "not": no crash)
You: Goodbye from the Mood Machine.                         ("quit")
```

ML model on similar inputs (`logs/ml_session.txt`) — it emits labels the rule-based model cannot:

```text
You: ML model: positive   <- "I love this class so much"
You: ML model: sarcasm    <- "great, my car broke down again"
You: ML model: mixed      <- "I am happy about the job but sad to leave my friends"
You: ML model: positive   <- "the meeting is at 3pm"        (neutral missed)
```

Held-out evaluation excerpt (`logs/evaluation.txt`):

```text
============================================================
Rule based: held-out test results
============================================================
Overall accuracy: 0.52 (14/27)

Accuracy by category:
  clear        1.00  (6/6)
  emoji        0.00  (0/4)
  mixed        0.00  (0/3)
  negation     1.00  (4/4)
  neutral      1.00  (3/3)
  sarcasm      0.00  (0/3)
  slang        0.25  (1/4)

============================================================
ML (logistic regression): held-out test results
============================================================
Overall accuracy: 0.67 (18/27)

Accuracy by category:
  clear        0.50  (3/6)
  emoji        0.75  (3/4)
  mixed        1.00  (3/3)
  negation     1.00  (4/4)
  neutral      0.33  (1/3)
  sarcasm      0.33  (1/3)
  slang        0.75  (3/4)

ML model: training accuracy = 1.00, held-out accuracy = 0.67
```

`logs/rule_based_session.txt` also records that the rule-based model scores only **0.30 on the 63-post `SAMPLE_POSTS`** training data — expected, since it can never emit the `sarcasm` (9 posts) or `mixed` (10 posts) labels that make up ~30% of that dataset.

### Reliability and guardrail results

Tail of `logs/test_suite.txt` — the suite is green while honestly recording known limitations:

```text
test_mood_machine.py::test_rule_based_clear_cases PASSED
test_mood_machine.py::test_rule_based_neutral_cases PASSED
test_mood_machine.py::test_rule_based_negation PASSED
test_mood_machine.py::test_trailing_not_does_not_crash PASSED
test_mood_machine.py::test_rule_based_predictions_are_valid_labels PASSED
test_mood_machine.py::test_rule_based_sarcasm[...] XFAIL  (3 cases)
test_mood_machine.py::test_rule_based_emoji[...]   XFAIL  (4 cases)
test_mood_machine.py::test_rule_based_mixed[...]   XFAIL  (3 cases)
test_mood_machine.py::test_rule_based_slang[...]   XFAIL x3 / XPASS x1
test_mood_machine.py::test_ml_model_fits_training_data PASSED
test_mood_machine.py::test_ml_predictions_are_valid_labels PASSED
test_mood_machine.py::test_accuracy_by_category_counts_correctly PASSED
================== 11 passed, 13 xfailed, 1 xpassed in 0.62s ==================
```

Guardrails the suite enforces:

- **Output contract:** both models may only return labels in `{positive, negative, neutral, mixed, sarcasm}` (`test_*_predictions_are_valid_labels`).
- **Crash guard:** a trailing `"not"` cannot index past the token list (`test_trailing_not_does_not_crash`; visible live in the session log: `"I am not" -> neutral`).
- **Data integrity:** `SAMPLE_POSTS`/`TRUE_LABELS` length alignment, all labels valid, and every category covered by at least 3 held-out cases.
- **Honest-limitation sentinels:** sarcasm/emoji/mixed/slang tests are marked `xfail`, so a future fix surfaces as `XPASS` rather than a silent pass; one slang case already XPASSes because it happens to contain the known negative word "stressed".

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

The project showed me that AI is good and working through and testing a variety of solutions, but the onus is still on the developer to ask for and understand them.