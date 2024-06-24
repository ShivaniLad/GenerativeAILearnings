import os
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

model = ChatOllama(model='gemma')
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


"""
    Prompt Templates help to turn raw user information into a format that the LLM can work with.
    First, adding in a system message. To do this, we will create a ChatPromptTemplate. We will utilize MessagesPlaceholder to pass all the messages in.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all the questions to the best of your knowledge and ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_chain = prompt | model

response = prompt_chain.invoke(
    {
        "messages": [
            HumanMessage(content="What is my name?.")
        ]
    }
)

print(response.content)

with_prompt_message_history = RunnableWithMessageHistory(prompt_chain, get_session_history)
prompt_config = {
    "configurable": {"session_id": "abc2"}
}

prompt_response = with_prompt_message_history.invoke(
    [HumanMessage(content="What is my name?")],
    config=prompt_config,
)

print(prompt_response.content)
