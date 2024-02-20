# LangChain supports many other chat models. Here, we're using Ollama


# https://python.langchain.com/docs/integrations/chat/ollama_functions
# https://python.langchain.com/docs/integrations/chat/ollama


from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.retrievers import ArxivRetriever
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
import os

import dotenv

dotenv.load_dotenv()

 
OLLMA_BASE_URL = os.getenv("OLLMA_BASE_URL")


# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(
    model="mistral:instruct",
    base_url= OLLMA_BASE_URL
    )

from langchain_experimental.llms.ollama_functions import OllamaFunctions

# model = OllamaFunctions(model="mistral")
model = OllamaFunctions(
    model="mistral:instruct",
    base_url= OLLMA_BASE_URL
    )


model = model.bind(
    functions=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    function_call={"name": "get_current_weather"},
)

from langchain.schema import HumanMessage

output = model.invoke("what is the weather in Boston?")

x=0
