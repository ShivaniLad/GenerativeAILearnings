import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.special import softmax

model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# we already have a token named MASK alike PAD, CLS, etc., in out vocab
mask = tokenizer.mask_token
pad = tokenizer.pad_token

# we use mask token to let the model predict the masked word and let the model know that it has to predict this particular word.
sentence = f"I want to {mask} pizza tonight."

tokens = tokenizer.tokenize(sentence)

encoded_inputs = tokenizer(sentence, return_tensors='pt')

"""
outputs logits

- Logits are the role and normalized predictions bert gave us.
- It can be said as the probability scores
- The model assigns these scores to the each possible words in the vocabulary.
- Basically ranking them for the best ones that could fill in the masked position.

"""
outputs = model(**encoded_inputs)
logits = outputs.logits.detach().numpy()[0]

"""

returns an array of size 28996 which is the len of our vocab.
This means that it tries to fit each and every word in the vocab and assigns some confidence to each word in the vocab.

""" 
mask_logits = logits[tokens.index(mask) + 1]

# transfer the values of logits into actual probabilities
confidence_scores = softmax(mask_logits)

# returns array of indices of sorted array, [::-1] --> reverse the array
np.argsort(confidence_scores)[::-1]

# checking around confidence scores
count = 1
for i in np.argsort(confidence_scores):
    if count <= 10:
        print(confidence_scores[i].round())
        count += 1
    else:
        break

print("Tokens \t    Confidence Score")
print("-"*50)
for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]
    # print(f"{pred_token:10} {score}")
    print(sentence.replace(mask, pred_token))
