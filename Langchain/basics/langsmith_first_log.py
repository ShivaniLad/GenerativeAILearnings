import os
from openai import AzureOpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
# os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

# autotrace llm calls
client = wrap_openai(client=AzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version="2024-02-01",
    api_key=os.getenv('OPENAI_API_KEY')
))


@traceable
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="employee-chatbot"
    )
    return result.choices[0].message.content


pipeline("hello world!")
