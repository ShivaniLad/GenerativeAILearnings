import requests
import streamlit as st
from langserve import RemoteRunnable

"""
    # using langserve python SDK
    ollama = RemoteRunnable("http://localhost:8000/openai")
    essay_chain = RemoteRunnable("http://localhost:8000/essay")
    
    essay_chain.invoke({"topic": "parrots"})
    
    prompt = [
        SystemMessage(content='Act like either a cat or a parrot.')
        HumanMessage(content='Hello')
    ]
"""


# using requests
def get_openchat_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={'input': {'topic': input_text}})

    return response.json()['output']


def get_llama_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={'input': {'topic': input_text}})

    return response.json()['output']


# streamlit framework
st.title('Langchain Demo with LLAMA2 API')
input_txt = st.text_input("Write an essay on")
input_txt1 = st.text_input("Write an poem on")

if input_txt:
    st.write(get_openchat_response(input_txt))

if input_txt1:
    st.write(get_llama_response(input_txt1))
