import numpy as np
from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence1 = "There was a fly drinking from my soup."
sentence2 = "To become a commercial pilot, he had to fly for 1500 hours."


"""
model(**encoded_inputs)
returns last_hidden_state, pooler_output

last_hidden_state : represents the final state of every token in input sequence

pooler_output : represents the final state of whole sentence   

"""


def predict(sent):
    encoded_inputs = tokenizer(sent, return_tensors='pt')
    
    return model(**encoded_inputs)[0]


token1 = tokenizer.tokenize(sentence1)
token2 = tokenizer.tokenize(sentence2)

emb1 = predict(sentence1)
emb2 = predict(sentence2)

# each embedding word is with 768 dimension 
print(emb1.shape)
print(emb2.shape)

out1 = emb1[0, token1.index("fly"), :].detach()
out2 = emb2[0, token2.index("fly"), :].detach()

print(cosine(out1, out2))
