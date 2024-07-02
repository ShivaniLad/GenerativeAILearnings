# import required libraries
import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS

load_dotenv()

# load groq api
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    print("in if")
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    st.title("ChatGroq Demo")
    llm = ChatGroq(model_name='mixtral-8x7b-32768')

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on provided context only.
        Please provide the most accurate response based on the context and question.
        {context}
        <context>
        Questions:{input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.text_input("Input your prompt here...")

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        print("Response time : ", (time.process_time() - start))
        st.write(response['answer'])

else:
    st.write("Problem running the page")
