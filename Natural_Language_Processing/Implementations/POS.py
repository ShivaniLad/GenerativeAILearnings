import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

string = '"Hi, how\'re you?"'
doc1 = nlp(string)

"""
INTJ : Interjection : an independent word2
SCONJ : Subordinate conjunction : a word2 or phrase that links a dependent clause to an independent clause
AUX : Auxiliary word2 / verb 
PRON : Pronoun
PUNCT : Punctuation


dep_ : relation between tokens / Syntactic dependency
 It extracts the information on what one word2 means to another grammatically; whether it is a subject, an auxiliary verb, or a root, etc

tag_ : detailed aprt of speech 
"""

# POS, Dependency, tags
print("Token \t\t Part of speech \t\t Dependency \t\t Explained Dependency \t\t Tags \t\t Explained Tag")
for token in doc1:
    print(f"{token.text:{15}} {token.pos_:{23}} {token.dep_:{20}} {str(spacy.explain(token.dep_)):{23}} {token.tag_:{10}} {str(spacy.explain(token.tag_))}")


# another example for POS
doc2 = nlp("I read books on NLP.")
word2 = doc2[1]
print('\n', doc2)
print(f"{word2.text:{10}} {word2.pos_:{10}} {word2.tag_:{10}} {str(spacy.explain(word2.tag_))}")

doc3 = nlp("I read a book on NLP.")
word3 = doc3[1]
print('\n', doc3)
print(f"{word3.text:{10}} {word3.pos_:{10}} {word3.tag_:{10}} {str(spacy.explain(word3.tag_))}")

# count pos in a document
doc = nlp("The quick brown fox jumped over the lazy dog's back")
pos_count = doc.count_by(spacy.attrs.POS)
print('\nCount POS : ', pos_count)

# to know what the POS the numeric data represent
print(doc.vocab[84].text)

for k, v in sorted(pos_count.items()):
    print(f"{k} {doc.vocab[k].text:{5}} {v}")


"""we can do the same for other functions even such as:
    tags ---> attrs.TAG
    dependency ---> attrs.DEP
"""


# visualization using spacy
# dep = dependency, ent = entity
# render to run locally
# options dict do have more options like color, bg, font, etc
displacy.serve(doc, style="dep", options={'distance': 130}, auto_select_port=True)

doc2 = nlp("A 5km to N.Y. or St.Louis in U.S. cab ride costs $10.30.")
options = {'ents': ['GPE'], 'distance': 130}
displacy.serve(doc2, style="ent", options=options, auto_select_port=True)
