import uvicorn
import os
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server using runnable interfaces"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} with 50 words.")

add_routes(
    app,
    prompt | llm,
    path="/essay"
)  


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
