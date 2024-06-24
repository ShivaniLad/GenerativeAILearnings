import spacy

# download the model : python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

doc = nlp("sandwich burger car tiger human wheat")
for token in doc:
    print(token)
    print(token.text, "Vector : ", token.has_vector, "OOV : ", token.is_oov)

base_token = nlp("bread")

# similarity between base token and the tokens in doc
for token in doc:
    print(f"{token.text} --- {base_token.text} : {token.similarity(base_token)}")
