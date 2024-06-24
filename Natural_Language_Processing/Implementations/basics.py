# importing required libraries
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from nltk.stem.porter import PorterStemmer


# we need to download the models first
"""
as we are using english language we will download the english language models
spacy provides 4 different models which we can download as per our use.
1. en_core_web_sm : small models with no static word2 vectors
2. en_core_web_md : medium model with 514k keys and 20k unique vectors (1 vector with 300 dimensions)
3. en_core_web_lg : large model with 514k keys and 514k unique vectors (1 vector with 300 dimensions)
4. en_core_web_trf : transformer pipeline

command to download these models
python -m spacy download <model_name (en_core_web_sm)>
"""
nlp = spacy.load('en_core_web_sm')

# word2 tokenization
# stores a sentence as doc
# slice of this doc is as type span in spacy
string = 'today is a generative AI sessions going on.'
doc = nlp(string)
print(type(doc))

# Stemming vs Lemmatization
stemmer = PorterStemmer()
print("Token \t Stemmed Word \t Lemmatized word2 \t Stop Word")
print('-'*60)
for token in doc:
    print(f"{token.text:{15}} {stemmer.stem(token.text):{15}} {token.lemma_:<{20}} {token.is_stop}")

# printing for one token
print('Token slicing : ', type(doc[1:4]))


# sentence tokenization
doc2 = nlp("First sentence. Second sentence. Third sentence.")
for sentence in doc2.sents:
    print(sentence)
    break

# check sentence start
print(doc2[3].is_sent_start)


# tokens when emails and website like data in our data
doc3 = nlp("We are here to help you! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com.")

for token in doc3:
    print(token.text)


doc4 = nlp("A 5km to NYC or St.Louis in U.S. cab ride costs $10.30.")
for token in doc4:
    print(token.text)


# vocab length for our english model
print("Vocab :  -----------------------")
print(len(doc4.vocab))




