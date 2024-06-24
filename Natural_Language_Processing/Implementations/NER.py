import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')


def show_ents(doc):
    if doc.ents:
        print("Token \t\t\t\t Labels \t Explained Label" + "\n" + '-' * 60)
        for ent in doc.ents:
            print(f"{ent.text:{20}} {ent.label_:{10}} {str(spacy.explain(ent.label_))}")
    else:
        print("No entities found.")


doc1 = nlp("A 5km to NYC or St.Louis in U.S. cab ride costs $10.30.")
print("\n" + doc1.text + "\n" + '-' * 60)
show_ents(doc1)

doc2 = nlp("Hi, how are you?")
print("\n" + doc2.text + "\n" + '-' * 60)
show_ents(doc2)

doc3 = nlp("Tesla to build a U.K. factory for 6 million dollar")
print("\n" + doc3.text + "\n" + '-' * 60)
show_ents(doc3)

# add new custom token entity into the vocab entity list available
# Span(doc, start, end, label)
org = doc3.vocab.strings['ORG']
new_entity = Span(doc3, 0, 1, label=org)
doc3.ents = list(doc3.ents) + [new_entity]

# after updating entity list
print("\nAfter updating entity list")
show_ents(doc3)

# matching and adding multiple phrase in entity
doc4 = nlp("Our country created a brand a new vacuum cleaner."
           "This new vacuum-cleaner is best.")

print("\n" + doc4.text + "\n" + '-' * 60)
show_ents(doc4)

# matching two different phrase patter
matcher = PhraseMatcher(nlp.vocab)
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']

phrase_pattern = [nlp(text) for text in phrase_list]

matcher.add('new product', phrase_pattern)

found_matches = matcher(doc4)
print(found_matches)

# using the above found matches start and end values now we can grab the tokens and then add them as new entities
prod = doc4.vocab.strings["PRODUCT"]

new_entity = [Span(doc4, match[1], match[2], label=prod) for match in found_matches]
doc4.ents = list(doc4.ents) + new_entity

# after adding new entities
show_ents(doc4)

""" Similarly we can do it for money 
    for example we have a doc where we have some tokens like $1 or 1 dollar.
    for this we have MONEY entity
"""
