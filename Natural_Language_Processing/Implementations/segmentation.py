import spacy
from spacy import Language

nlp = spacy.load('en_core_web_sm')

doc1 = nlp("First sentence. Second sentence. Third sentence.")
# doc.sents is a generator
print("Normal Sentence Tokenization")
for sent in doc1.sents:
    print(sent)


# we cannot slice doc sent directly as it is a generator object if we want to slice doc sents then
print("\nSlicing Sentence : ", list(doc1.sents)[0])

# custom segmentation
doc2 = nlp("First sentence; Second sentence. Third sentence.")

for sent in doc2.sents:
    print(sent)


# Add segmentation rule
# decorators helps us to modify the behaviour of a function or class
@Language.component("token_component")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            # token.i = index of token
            doc[token.i+1].is_sent_start = True
    return doc


nlp.add_pipe("token_component", before='parser')
print(nlp.pipe_names)

doc3 = nlp("First sentence; Second sentence. Third sentence.")

for sent in doc3.sents:
    print(sent)

