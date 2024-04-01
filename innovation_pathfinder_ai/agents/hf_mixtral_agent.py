from langchain_community.llms import HuggingFaceEndpoint
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
# Import things that are needed generically
from langchain.tools.render import render_text_description
import os
from dotenv import load_dotenv
from innovation_pathfinder_ai.structured_tools.structured_tools import (
    arxiv_search, get_arxiv_paper, google_search, wikipedia_search, knowledgeBase_search, memory_search
)

from langchain.prompts import PromptTemplate
from innovation_pathfinder_ai.templates.react_json_with_memory import template_system
from innovation_pathfinder_ai.utils import logger
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".cache.db"))
logger = logger.get_console_logger("hf_mixtral_agent")

config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

# Load the model from the Hugging Face Hub
llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                          temperature=0.1,
                          max_new_tokens=1024,
                          repetition_penalty=1.2,
                          return_full_text=False)

tools = [
    memory_search,
    knowledgeBase_search,
    arxiv_search,
    wikipedia_search,
    # google_search,
    # get_arxiv_paper,
]

prompt = PromptTemplate.from_template(
    template=template_system
)
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)


async def generate_response_with_streaming(chat_history, input_text):
    """
    Generates response with character-by-character streaming.

    Args:
        chat_history (list): List of tuples containing user input and response.
        input_text (str): User's current input text.

    Yields:
        str: Characters generated one by one for streaming.
    """
    for char in llm.generate_text(
        prompt=prompt.apply(input_text=input_text, chat_history=chat_history),
        stop=["\nObservation"],
        max_length=1024,
        temperature=0.1,
        repetition_penalty=1.2,
        return_past_key_values=True,
    )["generated_text"]:
        yield char

# Define the agent using the custom function
chat_model_with_stop = llm.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# Instantiate AgentExecutor with a custom function for post-processing
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,  # Cap number of iterations
    # max_execution_time=60,  # Timeout at 60 sec
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)
