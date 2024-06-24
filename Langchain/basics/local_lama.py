import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. PLease response to the user queries."),
        ("user", "Question:{question}")
    ]
)

# streamlit framework
st.title("Langchain Demo")
input_text = st.text_input("Search the topic here.")

# ollama llm
llm = Ollama(model='llama2')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

output = None

if input_text:
    output = chain.invoke({'question': input_text})
    st.write(output)
