# Mood Machine — System Diagram

Overview of the main components, data flow, and testing suite.

```mermaid
flowchart TD
    %% ============ DATA ============
    subgraph Data["Data (dataset.py / test_cases.py)"]
        WL["Word lists<br/>POSITIVE_WORDS<br/>NEGATIVE_WORDS<br/>MIXED_WORDS"]
        TRAIN["SAMPLE_POSTS + TRUE_LABELS<br/>56 labeled posts<br/>(clear, negation, sarcasm,<br/>slang, emoji, mixed, neutral)"]
        HELDOUT["TEST_CASES<br/>27 held-out posts<br/>grouped by category"]
    end

    %% ============ RULE-BASED MODEL ============
    subgraph Rule["Rule-based model (mood_analyzer.py)"]
        PRE["preprocess(text)<br/>strip → lowercase → tokenize<br/>→ remove punctuation"]
        SCORE["score_text(tokens)<br/>+1 positive / −1 negative<br/>'not' flips next word<br/>mixed words pull score to 0"]
        PRED["predict_label(score)<br/>&gt;0 positive<br/>&lt;0 negative<br/>=0 neutral"]
        EXPL["explain(text)<br/>human-readable rationale"]
        PRE --> SCORE --> PRED
        SCORE -.-> EXPL
    end

    %% ============ ML MODEL ============
    subgraph ML["ML model (ml_experiments.py)"]
        VEC["CountVectorizer<br/>bag-of-words features"]
        LR["LogisticRegression<br/>(max_iter=1000)"]
        MLPRED["predict_single_text(text)"]
        VEC --> LR --> MLPRED
    end

    %% ============ ENTRY POINTS ============
    subgraph Entry["Entry points"]
        MAIN["python main.py<br/>rule-based eval on SAMPLE_POSTS<br/>+ batch demo + interactive loop"]
        MLEXp["python ml_experiments.py<br/>train → training accuracy<br/>+ interactive loop"]
        EVAL["python evaluate.py<br/>evaluation harness"]
    end

    %% ============ EVALUATION OUTPUTS ============
    subgraph Metrics["Evaluation outputs (evaluate.py)"]
        ACC["Overall + per-category accuracy"]
        PRF["Precision / recall / F1<br/>per label"]
        TABLE["Side-by-side<br/>prediction table"]
        GAP["Train vs. held-out<br/>accuracy gap (overfitting check)"]
        PNG["confusion_matrices.png"]
    end

    %% ============ TESTING ============
    subgraph Tests["Testing suite (test_mood_machine.py — pytest)"]
        T1["Data integrity<br/>posts/labels aligned, valid labels,<br/>category coverage ≥3 each"]
        T2["Rule-based passing tests<br/>clear, neutral, negation,<br/>trailing-'not' regression, valid labels"]
        T3["Known limitations (xfail)<br/>sarcasm, emoji, mixed, slang"]
        T4["ML sanity checks<br/>training accuracy ≥ 0.9,<br/>valid prediction labels"]
        T5["Harness check<br/>accuracy_by_category()"]
    end

    %% ============ FLOWS ============
    WL -->|defaults via constructor| PRE
    TRAIN -->|fit on texts + labels| VEC
    HELDOUT --> EVAL
    HELDOUT --> Tests

    MAIN --> Rule
    MLEXp --> ML
    EVAL -->|MoodAnalyzer.predict_label| Rule
    EVAL -->|train_ml_model then<br/>predict_single_text| ML

    Rule --> Metrics
    ML --> Metrics

    TEST_INPUT["Text input<br/>(dataset / user / test case)"] --> PRE
    TEST_INPUT --> MLPRED

    Tests -.->|imports & validates| Data
    Tests -.->|imports & validates| Rule
    Tests -.->|imports & validates| ML
    Tests -.->|imports accuracy_by_category| EVAL
```

## Reading the diagram

- **`dataset.py`** supplies both the lexicons for the rule-based model and the training set for the ML model. **`test_cases.py`** is strictly held out — the ML model never trains on it.
- The **rule-based pipeline** is `preprocess → score_text → predict_label`, producing `positive` / `negative` / `neutral`. It cannot emit `mixed` or `sarcasm`, which is why those tests are marked `xfail`.
- The **ML pipeline** is `CountVectorizer → LogisticRegression`, trained on `SAMPLE_POSTS`/`TRUE_LABELS`.
- **`evaluate.py`** is the shared harness: it runs both models over `TEST_CASES` and reports per-category accuracy, per-label precision/recall/F1, a side-by-side table, the train/test gap, and confusion matrices.
- **`test_mood_machine.py`** guards working behavior (passing tests), documents known failures (`xfail`), and sanity-checks data integrity, the ML model, and the harness itself.
