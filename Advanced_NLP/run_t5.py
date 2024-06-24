import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

# load the dataset
data = pd.read_csv("product_reviews.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)
dataset = Dataset.from_pandas(data)

# load the model and tokenizer
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset = dataset.shuffle(42).select(range(10000)).class_encode_column('rating').train_test_split(test_size=0.2, stratify_by_column='rating')


# preprocess the data
def preprocess_data(example):
    example['prompt'] = [f"review: {product_title}, {rating} Stars" for product_title, rating in zip(example['product_title'], example['rating'])]
    example['response'] = [f'{title} {text}' for title, text in zip(example['title'], example['text'])]

    inputs = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example['response'], padding="max_length", truncation=True, max_length=128)

    inputs.update({'labels': targets['input_ids']})

    return inputs


train_dataset = dataset['train'].map(preprocess_data, batched=True)
test_dataset = dataset['test'].map(preprocess_data, batched=True)

custom_model = T5ForConditionalGeneration.from_pretrained("t5_models/13062024/checkpoint-2001")


def generate_output(text):
    inputs = tokenizer("review: " + text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    output = custom_model.generate(inputs['input_ids'], max_length=128, no_repeat_ngram_size=3, num_beams=6, early_stopping=True)
    review = tokenizer.decode(output[0], skip_special_tokens=True)

    return review


random_values = test_dataset.shuffle(42).select(range(10))['product_title']

print("RESPONSE : ", generate_output(random_values[0] + ", 2 Stars"))
