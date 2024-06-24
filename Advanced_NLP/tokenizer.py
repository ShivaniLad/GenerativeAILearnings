import pandas as pd
from transformers import BertModel, AutoTokenizer

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence = "When life gives you lemons, don't make lemonade."
tokens = tokenizer.tokenize(sentence)

vocab = tokenizer.vocab
vocab_df = pd.DataFrame({"tokens": vocab.keys(), "token_id": vocab.values()})
vocab_df = vocab_df.sort_values(by='token_id').set_index("token_id")

token_ids = tokenizer.encode(sentence)

list(zip(tokens, token_ids[1:-1]))

tokenizer.decode(token_ids)

sentence2 = sentence.replace("don't ", '')

tokenizer_out = tokenizer([sentence, sentence2], padding=True)

tokenizer.decode(tokenizer_out['input_ids'][0])
