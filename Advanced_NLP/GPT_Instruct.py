from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset.to_pandas().sample(20)


def preprocess(example):
    example['prompt'] = f"{example['instruction']} {example['input']} {example['output']}"

    return example


# preprocessing the data
dataset = dataset.map(preprocess, remove_columns=['output', 'instruction', 'input'])

# train test split
dataset = dataset.shuffle(seed=42).select(range(100000)).train_test_split(test_size=0.2)

train_dataset = dataset['train']
test_dataset = dataset['test']

# here we will be using simple DialoGPT model which can we be a good start for the general conversation
model_name = 'microsoft/DialoGPT-medium'

# tokenizing the data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(data):
    tokenized_data = data.map(
        lambda x: tokenizer(
            x['prompt'],
            truncation=True, 
            max_length=128
        ), 
        batched=True, 
        remove_columns=['prompt']
    )

    return tokenized_data


train_dataset = tokenize(train_dataset)
test_dataset = tokenize(test_dataset)

model = AutoModelForCausalLM.from_pretrained(model_name)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_model()
