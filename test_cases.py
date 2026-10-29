"""
Held-out test cases for the Mood Machine evaluation harness.

IMPORTANT: none of these posts appear in SAMPLE_POSTS, so the ML model
has never seen them during training. They exist only to measure how
well each model generalizes to new text.

Each case is a dict with:
  - "text":     the post to classify
  - "label":    the human-assigned mood label
  - "category": the language phenomenon the case is testing
"""

TEST_CASES = [
    # -----------------------------------------------------------------
    # Clear sentiment: the easy cases both models should get right.
    # -----------------------------------------------------------------
    {"text": "I had an amazing day with my friends",
     "label": "positive", "category": "clear"},
    {"text": "This is the best news ever, so happy",
     "label": "positive", "category": "clear"},
    {"text": "What a wonderful surprise, I love it",
     "label": "positive", "category": "clear"},
    {"text": "I feel awful and exhausted today",
     "label": "negative", "category": "clear"},
    {"text": "That movie was boring and terrible",
     "label": "negative", "category": "clear"},
    {"text": "I hate waiting in long lines",
     "label": "negative", "category": "clear"},

    # -----------------------------------------------------------------
    # Negation: "not" flips the meaning of the word that follows it.
    # -----------------------------------------------------------------
    {"text": "I am not excited about the exam",
     "label": "negative", "category": "negation"},
    {"text": "This is not good at all",
     "label": "negative", "category": "negation"},
    {"text": "I am not happy with the results",
     "label": "negative", "category": "negation"},
    {"text": "not bad for a first try",
     "label": "positive", "category": "negation"},

    # -----------------------------------------------------------------
    # Sarcasm: the words say one thing, the meaning is the opposite.
    # -----------------------------------------------------------------
    {"text": "Oh great, another homework assignment due tomorrow",
     "label": "sarcasm", "category": "sarcasm"},
    {"text": "I just love being stuck in the rain without an umbrella",
     "label": "sarcasm", "category": "sarcasm"},
    {"text": "Wow fantastic, my flight got delayed again",
     "label": "sarcasm", "category": "sarcasm"},

    # -----------------------------------------------------------------
    # Slang: informal words that the starter word lists do not know.
    # -----------------------------------------------------------------
    {"text": "lowkey proud of myself ngl",
     "label": "positive", "category": "slang"},
    {"text": "This party is gonna be lit no cap",
     "label": "positive", "category": "slang"},
    {"text": "that exam was mid tbh",
     "label": "negative", "category": "slang"},
    {"text": "highkey stressed about finals",
     "label": "negative", "category": "slang"},

    # -----------------------------------------------------------------
    # Emoji: mood carried by symbols instead of words.
    # -----------------------------------------------------------------
    {"text": "just passed my driving test :)",
     "label": "positive", "category": "emoji"},
    {"text": "feeling sick today :(",
     "label": "negative", "category": "emoji"},
    {"text": "another monday 💀",
     "label": "negative", "category": "emoji"},
    {"text": "best concert ever 😂🔥",
     "label": "positive", "category": "emoji"},

    # -----------------------------------------------------------------
    # Mixed: positive and negative feelings in the same post.
    # -----------------------------------------------------------------
    {"text": "I am happy about the job but sad to leave my friends",
     "label": "mixed", "category": "mixed"},
    {"text": "excited for the trip but nervous about flying",
     "label": "mixed", "category": "mixed"},
    {"text": "the food was great but the service was awful",
     "label": "mixed", "category": "mixed"},

    # -----------------------------------------------------------------
    # Neutral: no mood signal at all.
    # -----------------------------------------------------------------
    {"text": "the meeting is at 3pm",
     "label": "neutral", "category": "neutral"},
    {"text": "I walked to the store this morning",
     "label": "neutral", "category": "neutral"},
    {"text": "water boils at 100 degrees",
     "label": "neutral", "category": "neutral"},
]

CATEGORIES = sorted({case["category"] for case in TEST_CASES})
