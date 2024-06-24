# importing required libraries
import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')

# tokenization
example_string = """
Muad'Dib learned rapidly because his first training was in how to learn.And the first lesson of all was the basic trust that he could learn.
It's shocking to find how many people do not believe they can learn, and how many more believe learning to be difficult.
"""

# sentence tokenizer
print(f"Sentence Tokenizer :\n{sent_tokenize(example_string)}\n")

# word2 tokenizer
print(f"Word Tokenizer :\n{word_tokenize(example_string)}\n")

# removing punctuations from sentence using regex
sentence = re.sub(r"[.',]", " ", example_string)
print(sentence)

# now again tokenizing the sentence
words = word_tokenize(sentence)
print(f"Word Tokenizer :\n{words}\n")

# removing stop words
stopwords = set(stopwords.words("english"))
filtered_words = [word for word in words if word.lower() not in stopwords]
print(f"Filtered List after removing stop words :\n{filtered_words}")

# stemming : reduces the words to their root
"""
Porter Stemmer : writing --> write
Lancaster Stemmer : writing --> writ
Snowball Stemmer : writing --> write
"""
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print(f"Stemmed words :\n{stemmed_words}\n")

# lemmatization : reduces words to their core meaning, but it will give you a complete English word2 that makes sense on its own instead of just a fragment of a word2 like 'discoveri'
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print(f"Lemmatized words :\n{lemmatized_words}\n")

# bag of words
count_vector = CountVectorizer()
vectorised_data = count_vector.fit_transform(lemmatized_words).toarray()
# print(f"Count Vectorized Data :\n{vectorised_data}")
print(f"Vocabulary : \n{count_vector.vocabulary_}\n")

# ngram bow
count_ngram_vector = CountVectorizer(ngram_range=(1, 3))
count_ngram_vector.fit(lemmatized_words)
print(f"NGram Vocabulary : \n{count_ngram_vector.vocabulary_}\n")

# tf-idf
tfidf_vector = TfidfVectorizer()
tfidf_vectorised_data = tfidf_vector.fit_transform(lemmatized_words)
print(f"Tfidf Vocabulary : \n{tfidf_vector.vocabulary_}")

# let's print the idf of each word2:
all_feature_names = tfidf_vector.get_feature_names_out()

for word in all_feature_names:
    # let's get the index in the vocabulary
    indx = tfidf_vector.vocabulary_.get(word)

    # get the score
    idf_score = tfidf_vector.idf_[indx]

    print(f"{word} : {idf_score}")


