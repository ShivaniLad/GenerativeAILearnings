import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorWithPadding

# loading the data
data = pd.read_csv("product_reviews.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)
dataset = Dataset.from_pandas(data)

# load the model and tokenizer
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset = dataset.shuffle(42).select(range(10000)).class_encode_column('rating').train_test_split(test_size=0.2, stratify_by_column='rating')

train_dataset = dataset['train']
test_dataset = dataset['test']


# preprocess the data
def preprocess_data(example):
    example['prompt'] = [f"review: {product_title}, {rating} Stars" for product_title, rating in zip(example['product_title'], example['rating'])]
    example['response'] = [f'{title} {text}' for title, text in zip(example['title'], example['text'])]

    inputs = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example['response'], padding="max_length", truncation=True, max_length=128)

    inputs.update({'labels': targets['input_ids']})

    return inputs


train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir='./t5_models',
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()
