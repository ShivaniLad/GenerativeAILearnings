import os
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# ollama llm
llm = Ollama(model='gemma')
model = ChatOllama(model='gemma')

print(llm.invoke([HumanMessage(content="Hi! I'm Harry.")]))
print(llm.invoke([HumanMessage(content="What's my name?")]))

# it doesn't take the conversation history into the model until now.
# carrying conversation history

# sending list of messages all together
llm.invoke([
    HumanMessage(content="Hi, I'm Harry."),
    AIMessage(content="Hi Harry! How can I assist you today?"),
    HumanMessage(content="What's my name?"),
])
# now we are getting a proper response


# directly passing the raw message
chain = RunnableParallel({"output_msg": model})

"""
 Message History : 
    - we can use message history class to wrap our model
    - track input and output
    - store in some datastore
    - A key part here is the function we pass into as the get_session_history. 
    - This function is expected to take in a session_id and return a Message History object. 
    - This session_id is used to distinguish between separate conversations, and should be passed in as part of the config when calling the new chain.
    - BaseChatMessageHistory : Abstract base class for storing chat message history.
    - RunnableWithMessageHistory : wraps another Runnable and manages the chat message history for it; it is responsible for reading and updating the chat message history.
"""

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    output_messages_key="output_msg"
)

# we need to pass session_id into the runnable everytime
config = {
    "configurable": {
        "session_id": "abc"
    }
}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I am Harry.")],
    config=config,
)

print(response)
