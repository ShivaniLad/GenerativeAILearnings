import spacy
from spacy.matcher import Matcher, PhraseMatcher

nlp = spacy.load('en_core_web_sm')

# phrase matching and vocabulary
matcher = Matcher(nlp.vocab)

# matching following patterns
# there are various token attributes to determine matching rules. like LOWER, IS_PUNCT, LENGTH, IS_ALPHA, etc
# 1. SolarPower
pattern1 = [{'LOWER': 'solarpower'}]

# 2. Solar-power
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]

# 3. Solar power
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]

matcher.add('SolarPower', [pattern1, pattern2, pattern3])

doc = nlp("The Solar Power industry continues to grow a solarpower increases. Solar-power is amazing.")
ans = matcher(doc)

for match_id, start, end in ans:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)

# Phrase matcher object
p_matcher = PhraseMatcher(nlp.vocab)

# reading file
with open("../data/reaganomics.txt") as f:
    doc = nlp(f.read())

phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics']

phrase_pattern = [nlp(text) for text in phrase_list]
print(phrase_pattern)

# key :  is a str id for what we are matching
p_matcher.add('Matcher', phrase_pattern)
matches = p_matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)
