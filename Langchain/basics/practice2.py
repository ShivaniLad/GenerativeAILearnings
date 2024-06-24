import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# ollama llm
llm = Ollama(model='llama2')

messages = [
    SystemMessage(content="Translate the following from English to Spanish."),
    HumanMessage(content="Hi!"),
]

results = llm.invoke(messages)
print(results)

# output parser
parser = StrOutputParser()

print(parser.invoke(results))


# prompt template
# 1. create a string to format to be system message
system_template = "Translate the following into {language}: "

# 2. create prompt template. Combination of system_template + simpler template for where to put the text
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}")
    ]
)

# the input to this prompt_template is a dictionary
result = prompt_template.invoke({"language": "italian", "text": "hi"})
print(result.to_messages())


# Chaining together the components with LCEL (LangChain Expression Language)
chain = prompt_template | llm | parser
result = chain.invoke({
    "language": "Hindi",
    "text": "How are you?"
})

print(result)
