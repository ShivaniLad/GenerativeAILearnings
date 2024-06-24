# importing required libraries
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


# loading the dataset
data = pd.read_csv("moviereviews.csv", sep="\t")
print(data.head())
print(f"Shape : {data.shape}")
reviews = data['review']

# cleaning the data
# null values in dataset
print(f"Null Values \n{data.isna().sum()}")

# drop null values rows
data.dropna(inplace=True)
print(f"Null Values \n{data.isna().sum()}")
print(f"Shape : {data.shape}")

# remove records with empty string type values
# str.isspace() helps to know is the string empty or not
# itertuples() iter rows in df returns (index, columns_in_dataframe)
blanks = []

for idx, lbl, rev in data.itertuples():
    if rev.isspace():
        blanks.append(idx)

print("\nIndexes with empty string kind of data.")
print(blanks)

data.drop(blanks, axis=0, inplace=True)
print(f"Shape : {data.shape}")


# remove stop words using spacy library
# load spacy language model
nlp = spacy.load("en_core_web_sm")

# tokenization
data['review'] = data['review'].apply(lambda x: nlp(x))

# removing stop words
data['review'] = data['review'].apply(lambda x: [token for token in x if not token.is_stop])

# removing punctuations
data['review'] = data['review'].apply(lambda x: [token for token in x if not token.is_punct])

# removing the punctuations leftover from the doc
data['review'] = data['review'].apply(lambda x: [token for token in x if token.text not in ['\n', '|', '>']])

# converting list of tokens to string list
data['review'] = data['review'].apply(lambda x: " ".join([val.text for val in x]))
print(data['review'].head())

# split the data
X = data['review']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# create pipelines
"""
Pipeline allows us to sequentially apply a list of transformers to preprocess the data if desired, conclude the sequence with a final predictor for predictive modeling.
pass list of tuples; 
    Pipeline([('variable_name', function/class_name), t2, t3, ...])
"""
# we will create a pipeline for vectorization and our model
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('dtc', DecisionTreeClassifier())])

# fit the model
text_clf.fit(X_train, y_train)

# make predictions
predictions = text_clf.predict(X_test)

# accuracy score
print("\nAccuracy Score \n")
print(accuracy_score(y_test, predictions))

# classification report
print("\nClassification Report \n")
print(classification_report(y_test, predictions))

# trying to make predictions with some custom data
myreview = "I loved the movie."
print(text_clf.predict([myreview]))
