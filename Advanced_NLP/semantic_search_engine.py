import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

dataset = load_dataset("multi_news", split="test")
df = dataset.to_pandas().sample(2000, random_state=42)

"""

SBERT : Sentence Transformers

- lightweight and powerful
- It can be used to compute embeddings using Sentence Transformer models or to calculate similarity scores using Cross-Encoder models. 
- This unlocks a wide range of applications, including 
    - semantic search, 
    - semantic textual similarity
    - paraphrase mining.

- They provide multiple models to achieve above functionalities.
- Check it on its official site https://sbert.net/
- Check Pretrained original models available on : https://sbert.net/docs/sentence_transformer/pretrained_models.html#original-models

- Normal tokenizer gives embeddings on single words while the Sentence transformers and specially trained to give single embedding representation for the entire sentence.
- They are designed fro comparing the sentence similarities.

"""
model = SentenceTransformer("all-MiniLM-L6-v2") 

passage_embeddings = model.encode(df['summary'].to_list())
len(passage_embeddings[0])

query = "Find some articles about technology and artificial intelligence."


def find_relevant_news(query:str):
    query_embedding = model.encode(query)

    # find similarity between two embedding vectors.
    # here we are applying one-many similarity computation
    similarities = util.cos_sim(query_embedding, passage_embeddings)

    # top 3 similarities
    top_indices = torch.topk(similarities.flatten(), k=3).indices

    top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indices]

    return top_relevant_passages


find_relevant_news("Natural Disasters ")
find_relevant_news("Law enforcement and police")
