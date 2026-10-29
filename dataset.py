"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - MIXED_WORDS: starter list of mixed words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# Targeted fix: Mixed words list
MIXED_WORDS = [
    "mixed",
    "so-so",
    "kind of",
    "kinda",
    "sort of",
    "sorta",
    "somewhat",
    "but",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "I don't understand this",
    "I am not sure about this",
    "I did great on the exam",
    "Just feeling so-so",
    "I absolutely love getting stuck in traffic",
    "Really feeling this one ngl",

    # ------------------------------------------------------------------
    # Expanded set: clear sentiment
    # ------------------------------------------------------------------
    "best birthday ever, thanks everyone",
    "my team won the championship game",
    "the concert last night was incredible",
    "got an A on my chemistry final",
    "she said yes!",
    "pizza night with the whole family was perfect",
    "my phone died right before the interview",
    "ugh, I failed my driving test",
    "nothing is going right this week",
    "I missed the bus and was late to work",
    "the wifi went out during my presentation",
    "my roommate ate my leftovers again",

    # ------------------------------------------------------------------
    # Expanded set: negation
    # ------------------------------------------------------------------
    "I do not feel good about this decision",
    "the movie was not great honestly",
    "not excited for monday",
    "this is not a bad idea",
    "not gonna lie, that was amazing",

    # ------------------------------------------------------------------
    # Expanded set: sarcasm
    # ------------------------------------------------------------------
    "great, my car broke down again",
    "love it when the prof assigns homework on friday",
    "oh wonderful, it's raining on my day off",
    "yes, because that worked so well last time",
    "thanks for spoiling the ending, really appreciated it",
    "sure, I totally wanted to wait in line for two hours",
    "oh yeah, standing in line at the dmv is my favorite hobby",
    "fantastic, the printer is out of ink again",

    # ------------------------------------------------------------------
    # Expanded set: slang
    # ------------------------------------------------------------------
    "this new album slaps fr fr",
    "that fit is fire no cap",
    "the vibes at the party were immaculate",
    "ngl that test cooked me",
    "lowkey want to skip class tomorrow",

    # ------------------------------------------------------------------
    # Expanded set: emoji
    # ------------------------------------------------------------------
    "we won the game 🎉",
    "exams are finally over 😌",
    "my dog is sick 🥲",
    "that exam tho 💀",
    "rip my wallet, tuition is due 💸",

    # ------------------------------------------------------------------
    # Expanded set: mixed
    # ------------------------------------------------------------------
    "excited to graduate but sad to leave everyone",
    "the movie had great acting but a boring plot",
    "proud of the win but exhausted from training",
    "I like my classes but the homework is killing me",
    "part of me wants to go, part of me wants to stay home",
    "the food was amazing but way too expensive",
    "glad the semester is over but dreading finals week",
    "new apartment is nicer but the commute is worse",

    # ------------------------------------------------------------------
    # Expanded set: neutral
    # ------------------------------------------------------------------
    "the library closes at nine",
    "I have two classes on tuesdays",
    "the grocery store is on fifth street",
    "my appointment got moved to thursday",
    "the bus comes every fifteen minutes",
    "the exam covers chapters four through six",
    "my sister lives in chicago",
    "class registration opens next week",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "neutral",   # "I don't understand this"
    "neutral",   # "I am not sure about this"
    "positive",  # "I did great on the exam"
    "mixed",     # "Just feeling so-so"
    "sarcasm",   # "I absolutely love getting stuck in traffic"
    "positive",  # "Really feeling this one ngl"

    # Expanded set: clear sentiment
    "positive",  # "best birthday ever, thanks everyone"
    "positive",  # "my team won the championship game"
    "positive",  # "the concert last night was incredible"
    "positive",  # "got an A on my chemistry final"
    "positive",  # "she said yes!"
    "positive",  # "pizza night with the whole family was perfect"
    "negative",  # "my phone died right before the interview"
    "negative",  # "ugh, I failed my driving test"
    "negative",  # "nothing is going right this week"
    "negative",  # "I missed the bus and was late to work"
    "negative",  # "the wifi went out during my presentation"
    "negative",  # "my roommate ate my leftovers again"

    # Expanded set: negation
    "negative",  # "I do not feel good about this decision"
    "negative",  # "the movie was not great honestly"
    "negative",  # "not excited for monday"
    "positive",  # "this is not a bad idea"
    "positive",  # "not gonna lie, that was amazing"

    # Expanded set: sarcasm
    "sarcasm",   # "great, my car broke down again"
    "sarcasm",   # "love it when the prof assigns homework on friday"
    "sarcasm",   # "oh wonderful, it's raining on my day off"
    "sarcasm",   # "yes, because that worked so well last time"
    "sarcasm",   # "thanks for spoiling the ending, really appreciated it"
    "sarcasm",   # "sure, I totally wanted to wait in line for two hours"
    "sarcasm",   # "oh yeah, standing in line at the dmv is my favorite hobby"
    "sarcasm",   # "fantastic, the printer is out of ink again"

    # Expanded set: slang
    "positive",  # "this new album slaps fr fr"
    "positive",  # "that fit is fire no cap"
    "positive",  # "the vibes at the party were immaculate"
    "negative",  # "ngl that test cooked me"
    "negative",  # "lowkey want to skip class tomorrow"

    # Expanded set: emoji
    "positive",  # "we won the game 🎉"
    "positive",  # "exams are finally over 😌"
    "negative",  # "my dog is sick 🥲"
    "negative",  # "that exam tho 💀"
    "negative",  # "rip my wallet, tuition is due 💸"

    # Expanded set: mixed
    "mixed",     # "excited to graduate but sad to leave everyone"
    "mixed",     # "the movie had great acting but a boring plot"
    "mixed",     # "proud of the win but exhausted from training"
    "mixed",     # "I like my classes but the homework is killing me"
    "mixed",     # "part of me wants to go, part of me wants to stay home"
    "mixed",     # "the food was amazing but way too expensive"
    "mixed",     # "glad the semester is over but dreading finals week"
    "mixed",     # "new apartment is nicer but the commute is worse"

    # Expanded set: neutral
    "neutral",   # "the library closes at nine"
    "neutral",   # "I have two classes on tuesdays"
    "neutral",   # "the grocery store is on fifth street"
    "neutral",   # "my appointment got moved to thursday"
    "neutral",   # "the bus comes every fifteen minutes"
    "neutral",   # "the exam covers chapters four through six"
    "neutral",   # "my sister lives in chicago"
    "neutral",   # "class registration opens next week"
]

# The starter list has been expanded beyond the original 12 posts.
# If you add more posts and labels, keep these requirements in mind:
#
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
