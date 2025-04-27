import pandas as pd
import random
import itertools
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Define vocabulary
topics = ["QA", "DILR", "VARC"]
positive_adjectives = ["confident", "comfortable", "strong", "easy", "great", "excellent", "solid", "prepared", "aced"]
negative_adjectives = ["stuck", "struggling", "hard", "difficult", "weak", "confused", "overwhelmed", "lost", "tough"]
verbs = ["feel", "find", "think", "am", "doing", "performing"]
modifiers = ["really", "very", "a bit", "somewhat", "totally", ""]
patterns = [
    "I {verb} {modifier} {adjective} about {topic}",
    "{topic} is {modifier} {adjective}",
    "I’m {modifier} {adjective} with {topic}",
    "I {verb} {topic} is {modifier} {adjective}",
    "I need {modifier} more practice in {topic}",
    "I’m {modifier} good at {topic}",
    "I’m finding {topic} {modifier} {adjective}",
    "{topic} problems are {modifier} {adjective}",
    "I {verb} {modifier} prepared for {topic}",
    "My {topic} skills are {modifier} {adjective}"
]

# Seed examples for realism
seed_examples = [
    ("I’m confident in QA", "positive", "QA"),
    ("DILR is hard", "negative", "DILR"),
    ("VARC is easy", "positive", "VARC"),
    ("I’m stuck on DILR", "negative", "DILR"),
    ("QA is going great", "positive", "QA"),
    ("I find VARC overwhelming", "negative", "VARC"),
    ("I need more practice in QA", "negative", "QA"),
    ("I’m really good at DILR", "positive", "DILR"),
    ("VARC feels a bit tough", "negative", "VARC"),
    ("I aced QA today", "positive", "QA")
]

# Generate template-based feedback
def generate_template_feedback(n=800):
    feedback_list = []
    for _ in range(n):
        topic = random.choice(topics)
        sentiment = random.choice(["positive", "negative"])
        adjective = random.choice(positive_adjectives if sentiment == "positive" else negative_adjectives)
        verb = random.choice(verbs)
        modifier = random.choice(modifiers)
        pattern = random.choice(patterns)
        
        # Skip incompatible patterns
        if "need more practice" in pattern and sentiment == "positive":
            continue
        if "I’m good at" in pattern and sentiment == "negative":
            continue
        
        feedback = pattern.format(verb=verb, modifier=modifier, adjective=adjective, topic=topic)
        feedback_list.append((feedback, sentiment, topic))
    return feedback_list

# Augment seed examples
def augment_seed_examples(seeds, n=200):
    augmented = seeds.copy()
    for feedback, sentiment, topic in seeds:
        if sentiment == "positive":
            neg_feedback = feedback.replace("easy", "not easy").replace("confident", "not confident").replace("great", "not great").replace("good", "not good").replace("aced", "struggling with")
            augmented.append((neg_feedback, "negative", topic))
        elif sentiment == "negative":
            pos_feedback = feedback.replace("hard", "easy").replace("stuck", "confident").replace("overwhelmed", "manageable").replace("tough", "straightforward")
            augmented.append((pos_feedback, "positive", topic))
    return list(itertools.islice(itertools.cycle(augmented), n))

# Generate and save dataset
template_feedback = generate_template_feedback(800)
augmented_seeds = augment_seed_examples(seed_examples, 200)
all_feedback = random.sample(template_feedback + augmented_seeds, 1000)
df = pd.DataFrame(all_feedback, columns=["feedback", "sentiment", "topic"])
df.to_csv("data/feedback.csv", index=False)

# Verify counts
print("Sentiment counts:", df['sentiment'].value_counts().to_dict())
print("Topic counts:", df['topic'].value_counts().to_dict())
print("Sample rows:\n", df.head(10))