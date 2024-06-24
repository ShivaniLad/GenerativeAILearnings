import os
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import (
    SystemMessage,
    trim_messages,
    HumanMessage, AIMessage
)
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# ollama llm
model = ChatOllama(model='gemma')

# Managing Conversation History
"""
    LangChain comes with a few built-in helpers for managing a list of messages.
    1. Trim Messages : reduce number of messages we're sending to the model.
    2. Filter Messages
    3. Merge consecutive messages of same type
"""

trimmer = trim_messages(
    max_tokens=60,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)


messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all the questions to the best of your knowledge and ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

print(trimmer.invoke(messages))

# To use it in our chain, we just need to run the trimmer before we pass the messages input to our prompt.
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="Hi! I am Bob.")],
        "language": "German"
    }
)
