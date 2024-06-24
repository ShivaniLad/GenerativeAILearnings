import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


# streamlit framework
st.title("Langchain Demo")
input_text = st.text_input("Search the topic here.")

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. PLease response to the user queries."),
        ("user", "Question:{question}")
    ]
)
