import torch
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import BertForQuestionAnswering, BertTokenizerFast
from scipy.special import softmax


context = "The giraffe is a large African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes have been thought of as one species, Giraffa camelopardalis, with nine subspecies. Most recently, researchers proposed dividing them into up to eight extant species due to new research into their mitochondrial and nuclear DNA, and individual species can be distinguished by their fur coat patterns. Seven other extinct species of Giraffa are known from the fossil record. "

question = "what are dogs?"

model_name = "deepset/bert-base-cased-squad2"


tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# We put questions and context both together while tokenization because the model will need to see both together in order to get answer. We are doing this here beacuse we have our custom knowledge from which the model needs to answer.
inputs = tokenizer(question, context, return_tensors="pt")

# kwargs in Python is a special syntax that allows you to pass a keyworded, variable-length argument dictionary to a function.

"""

- With torch.no_grad() method is like a loop in which every tensor in that loop will have a requires_grad set to False.  
- It means that the tensors with gradients currently attached to the current computational graph are now detached from the current graph and no longer we will be able to compute the gradients with respect to that tensor. 
- Until the tensor is within the loop it is detached from the current graph. 
- As soon as the tensor defined with gradient is out of the loop, it is again attached to the current graph. 
- This method disables the gradient calculation which reduces the memory consumption for computations.

-- returns start_logits and end_logits which contains the tokens at start of the answer and end of the answer which creates a span of tokens that refers to answer from our original context that we passed to the model.

- So this this is known as an extractive approach.

- As we are taking span of tokens so our answer will always be a substring of our original context.

- start_logits and end_logits are the probability of the words.


"""
with torch.no_grad():
    output = model(**inputs)


start_scores, end_scores = softmax(output.start_logits)[0], softmax(output.end_logits)[0]

# arranging the scores into DataFrame
scores_df = pd.DataFrame({
    "Token Position": list(range(len(start_scores))) * 2,
    "Score": list(start_scores) + list(end_scores),
    "Score Type": ["Start"] * len(start_scores) + ["End"] * len(end_scores)
})

# plotting the above scores
px.bar(scores_df, x="Token Position", y="Score", color="Score Type", barmode="group", title="Start and End Scores for Tokens")


# answers with 3 highest probabilities
start_idx = np.argsort(start_scores)[::-1]
end_idx = np.argsort(end_scores)[::-1]

for s_id, e_id in zip(start_idx[:3], end_idx[:3]):
    answer_ids = inputs.input_ids[0][s_id:e_id + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    print(answer_tokens)



### Part 2
# defining a function that includes above steps and helps us to predict the answer
def predict_answer(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**inputs)

    start_scores, end_scores = softmax(output.start_logits)[0], softmax(output.end_logits)[0]

    # Answer with hightest probability
    start_idx = np.argmax(start_scores)
    end_idx = np.argmax(end_scores)

    confidence = (start_scores[start_idx] + end_scores[end_idx]) / 2

    answer_ids = inputs.input_ids[0][start_idx:end_idx + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer != tokenizer.cls_token:
        return answer, confidence
    elif answer != '':
        return None, confidence
    else:
        return None, confidence


predict_answer(context, "about giraffe")
predict_answer(context, "who are dogs?")


### new context
context = """
This article is about the beverage. For other uses, see Coffee (disambiguation).
Coffee
Espresso latte and black filtered coffee
Type	Usually hot, can be iced
Country of origin 	Yemen[1]
Introduced	15th century
Color	Black, dark brown, light brown, beige
Flavor	Distinctive, somewhat bitter
Ingredients	Roasted coffee beans

Coffee is a beverage brewed from roasted coffee beans. Darkly colored, bitter, and slightly acidic, coffee has a stimulating effect on humans, primarily due to its caffeine content. It has the highest sales in the world market for hot drinks.[2]

The seeds of the Coffea plant's fruits are separated to produce unroasted green coffee beans. The beans are roasted and then ground into fine particles typically steeped in hot water before being filtered out, producing a cup of coffee. It is usually served hot, although chilled or iced coffee is common. Coffee can be prepared and presented in a variety of ways (e.g., espresso, French press, caffè latte, or already-brewed canned coffee). Sugar, sugar substitutes, milk, and cream are often added to mask the bitter taste or enhance the flavor.

Though coffee is now a global commodity, it has a long history tied closely to food traditions around the Red Sea. The earliest credible evidence of coffee drinking as the modern beverage appears in modern-day Yemen in southern Arabia in the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to how it is now prepared for drinking.[3] The coffee beans were procured by the Yemenis from the Ethiopian Highlands via coastal Somali intermediaries, and cultivated in Yemen. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe.

The two most commonly grown coffee bean types are C. arabica and C. robusta.[4] Coffee plants are cultivated in over 70 countries, primarily in the equatorial regions of the Americas, Southeast Asia, the Indian subcontinent, and Africa. As of 2023, Brazil was the leading grower of coffee beans, producing 35% of the world's total. Green, unroasted coffee is traded as an agricultural commodity. Despite coffee sales reaching billions of dollars worldwide, farmers producing coffee beans disproportionately live in poverty. Critics of the coffee industry have also pointed to its negative impact on the environment and the clearing of land for coffee-growing and water use. The global coffee industry is massive and worth $495.50 billion as of 2023.[5] Brazil, Vietnam, and Colombia are the top exporters of coffee beans as of 2023. 
"""

# this line will give us an error as the len of our context is higher than the len of the corpus of our model
len(tokenizer.tokenize(context))

# to avoid this we can add the truncation = True and max_length, of our context  but the drawback here is that it will miss out the data 
predict_answer(context, "which are the most commonly grown coffee bean types?")


# in solution to above problem we can do is split the sentences
sentences = context.split('\n')


def chunk_sentences(senences, chunk_size, stride):
    chunks = []
    num_sentences = len(senences)

    for i in range(0, num_sentences, chunk_size - stride):
        print(i)
        print(i , "-", (i + chunk_size))
        chunk = sentences[i: i + chunk_size]
        print(chunk)
        chunks.append(chunk)

    return chunks


chunked_sentences = chunk_sentences(sentences, 3, 1)

questions = ['What is coffee?', "Which are the most commonly grown coffee bean types?", "How much does coffee industry worth globaly?"]

answers = {}

for chunk in chunked_sentences:
    context = '\n'.join(chunk)
    for question in questions:
        answer, score = predict_answer(context, question)

        if answer:
            if question not in answers:
                answers[question] = (answer, score)

            else:
                if score > answers[question][1]:
                    answers[question] = (answer, score)


# now this works way better than before


tokenizer.cls_token
