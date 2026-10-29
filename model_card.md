# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I compared both the rule-based model and the logistic regression ML model; with the LR ML model being significantly more effective.

**Intended purpose:**  
The model is attempting to classify short text messages as moods like positive, negative, neutral, mixed, or sarcasm.

**How it works (brief):**  
The rule based version scores text based upon the presence of positive, negative, and mixed words, with simple negation handling.
For the ML version, it uses logistic regression, which scores the likelihood of the text belonging to one of the mood categories with little overhead.



## 2. Data

**Dataset description:**  
There are 63 posts in SAMPLE_POSTS, chosen to encompass modern speech along with positive, neutral, mixed, negative, and sarcastic moods. A separate set of 27 posts lives in `test_cases.py` (TEST_CASES) and is used only for evaluation, never for training.

**Labeling process:**  
Each post was hand-labeled as positive, negative, neutral, mixed, or sarcasm. Labels were chosen to encompass common mannerisms of speech, including sarcasm, which is its own mood. Sometimes, sarcastic sentences can be interpreted as positive.

**Important characteristics of your dataset:**  
The dataset includes sarcasm, slang, emojis, negation, and several ambiguous sentences spread across 7 categories.

**Possible issues with the dataset:**  
A rule-based dataset can't capture the inherent ambiguity of language and double meanings of words. The dataset is also small (63 posts), so the ML model can memorize it instead of generalizing.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Positive words add +1 to the score and negative words subtract 1. "Not" flips the polarity of the word that follows it and consumes it so it isn't counted twice. Mixed words pull a nonzero score one step closer to 0. The final score maps to positive (>0), negative (<0), or neutral (0).

**Strengths of this approach:**  
It is perfect on held-out clear, negation, and neutral cases (1.00 per-category accuracy), and every decision is inspectable.

**Weaknesses of this approach:**  
The approach is poor at detecting mixed sentences or sarcasm (it sees the text as positive). It can never emit the `mixed` or `sarcasm` labels, and it ignores emojis and slang — scoring 0 on those held-out categories.

## 4. How the ML Model Works (if used)

**Features used:**  
Bag-of-words features from scikit-learn's CountVectorizer feeding a LogisticRegression classifier (max_iter=1000).

**Training data:**  
The model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
The model fits the training set perfectly (1.00 accuracy) but reaches only 0.67 on held-out posts. It handles some sarcasm, slang, emoji, and mixed cases that the rule-based model cannot express, but still misses some clear and neutral posts.

**Strengths and weaknesses:**  
LR learns patterns quickly and can express all five labels, but with only 63 training posts it memorizes — the 1.00 train vs. 0.67 test gap is textbook overfitting.

## 5. Evaluation

**How you evaluated the model:**  
`evaluate.py` ran both models on the 27 held-out TEST_CASES. Rule-based: 0.52 overall (perfect on clear, negation, and neutral; zero on sarcasm, emoji, and mixed). LR-based: 0.67 overall (perfect on mixed and negation) — and 1.00 on its own training data, showing overfitting rather than true skill. Full metrics and confusion matrices are produced by `evaluate.py` and saved to `confusion_matrices.png`.

**Examples of correct predictions:**  
All below are for the LR based model on held-out posts.

"the food was great but the service was awful" -> predicted=mixed, true=mixed
"I just love being stuck in the rain without an umbrella" -> predicted=sarcasm, true=sarcasm
"not bad for a first try" -> predicted=positive, true=positive (both models got this one)

The first pairs "great" with "awful" — genuinely mixed. In the second, nobody loves being rained on; this is sarcasm. In the third, "not" flips "bad", and both the rule-based negation handling and the ML model get it right.

**Examples of incorrect predictions:**  
The rule-based model reads sarcasm literally — "Oh great, another homework assignment due tomorrow" -> positive — and scores every emoji case as neutral. The ML model misses half the clear cases ("I had an amazing day with my friends" -> negative) and calls two of the three neutral posts positive ("the meeting is at 3pm" -> positive).

## 6. Limitations
The dataset is still small (63 training posts), so the ML model's perfect training score is memorization, not skill — the held-out accuracy (0.67) is the honest number. The rule-based model structurally cannot express 2 of the 5 labels (mixed, sarcasm), and neither model treats emojis as meaningful tokens.

## 7. Ethical Considerations
There is frequent Western bias with regard to analyzing messages, especially with how LLMs handle English versus all other languages.

## 8. Limitations and Biases

**What are the limitations or biases in the system?**

- **Label bias.** All 63 training labels and 27 test labels were assigned by one person. Sarcasm is treated as its own mood, but many sarcastic posts can also read as negative — a different labeler would draw that boundary differently, and both models inherit those choices as ground truth.
- **Language and register bias.** Every post is short, casual, first-person English in a student register (classes, exams, homework). The slang vocabulary ("lowkey", "ngl", "no cap", "mid") reflects one demographic and one moment in time, so the models will do worse on other dialects, languages, and formal text than the numbers suggest.
- **Lexicon bias.** The rule-based model only knows ~10 positive, ~10 negative, and 8 mixed words. Any emotional word outside those lists contributes nothing, so unknown-but-emotional text is systematically pulled toward "neutral" — a built-in false negative for expressiveness.
- **Structural bias.** `predict_label` can only return positive/negative/neutral, so the rule-based model is incapable of being correct on mixed or sarcastic posts by construction (0.00 on both held-out categories). Neither model can express uncertainty or assign more than one mood to a post.
- **Emoji blindness.** `preprocess` keeps emoji as unknown tokens, so 🥲 and 💀 carry no mood signal for either model.
- **Overfitting as bias.** With only 63 training posts, logistic regression memorizes the sample (1.00 train vs. 0.67 held-out). Its "learned" patterns are biased toward whatever word combinations happened to appear in the dataset rather than general mood structure.

## 9. Potential for Misuse

**Could the AI be misused, and how would you prevent that?**

Yes. A mood classifier over people's writing is a small-scale emotion-inference tool, and the failure modes matter even at 0.67 accuracy:

- **Emotional surveillance.** Running the classifier on employee chats, student posts, or customer messages to score mood without consent. Prevention: only deploy where users are told their text is being mood-analyzed, and show every label as a guess — the rule-based model's `explain()` output should travel with the prediction.
- **High-stakes decisions.** Using the mood label for hiring, discipline, ad targeting, or mental-health triage. A sarcastic post read literally produces false confidence; a slang-heavy post read as neutral produces false negatives — and both errors land hardest on the groups whose speech the dataset didn't cover. Prevention: keep a human in the loop for any consequential use, and never let this label be the sole signal.
- **Unequal enforcement.** Auto-flagging "negative" posts would disproportionately mislabel sarcastic and slang-heavy writing, effectively penalizing informal dialects. Prevention: audit per-category accuracy before deployment — the current numbers (sarcasm at 0.00–0.33) already disqualify those labels for enforcement use.
- **Gaming the score.** The rule-based scorer is fully inspectable, so anyone who reads the word lists can stuff positive words to fake a cheerful score. Prevention: treat the score as an educational demo, not a trust signal; anything gating a real decision would need an opaque model plus adversarial testing.

## 10. Surprises During Reliability Testing

**What surprised you while testing the AI's reliability?**

- **The ML model failed the easy cases.** It scored 1.00 on its own training posts but only 0.50 on held-out "clear" posts — it called "I had an amazing day with my friends" negative. The "smarter" model was less reliable than a word counter on the simplest inputs; the 1.00-train vs. 0.67-test gap made overfitting concrete rather than theoretical.
- **A pass for the wrong reason.** "highkey stressed about finals" was marked `xfail` but XPASSed — the model ignored "highkey" entirely and was right only because "stressed" happened to be in the negative list. A correct label did not mean correct understanding.
- **Bag-of-words invents mood where there is none.** The ML model labeled "the meeting is at 3pm" positive — it hallucinates sentiment from word-shape coincidences even when a post carries no mood signal.
- **The negation trick overdelivered.** Consuming the token after "not" handled both "not happy" -> negative and "not bad" -> positive, scoring a perfect 1.00 on the held-out negation category — better than expected for a few lines of logic. It still only understands "not", though; "don't" and "never" are invisible to it.
- **Sarcasm was a total wall.** I expected the rule-based model to miss sarcasm, but 0.00 across the whole category — every sarcastic post scored positive — showed this isn't a rare edge case; a large slice of real speech is structurally invisible to a lexicon scorer.
- **The ML model "learned" emoji without reading them.** It hit 0.75 on emoji posts, but the vectorizer can't really use 💀 — it must be keying on surrounding words like "best" and "monday". Right answers, probably wrong evidence.

## 11. Ideas for Improvement
More labeled data is the biggest lever, both for ML generalization and for closing the train/test gap. Near term: add emoji and slang entries to the lexicons and let `predict_label` emit "mixed". Longer term: a transformer model would help the model identify sentiment instead of sentence content.

## 12. AI Collaboration Details
The AI helped identify which features would be best to add; in this case, the comprehensive testing and evaluation system. It also helped me add many more sample posts quickly, which turned out to substantially improve the model's accuracy. I did reject the suggestion that the testing system could be combined with some sort of specialized model with an external dataset, as this would have been convoluted for this project's scope.
