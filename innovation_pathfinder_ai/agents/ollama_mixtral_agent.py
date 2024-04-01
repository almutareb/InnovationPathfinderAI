# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
# Import things that are needed generically
from typing import List, Dict
from langchain.tools.render import render_text_description
import os
import dotenv
from innovation_pathfinder_ai.structured_tools.structured_tools import (
    arxiv_search, get_arxiv_paper, google_search
)

# hacky and should be replaced with a database
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)


dotenv.load_dotenv()

OLLMA_BASE_URL = os.getenv("OLLMA_BASE_URL")


# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(
    model="mistral:instruct",
    base_url= OLLMA_BASE_URL
    )


tools = [
    arxiv_search,
    google_search,
    get_arxiv_paper,
    ]


prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)


# define the agent
chat_model_with_stop = llm.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    # handle_parsing_errors=True #prevents error
    )

    
if __name__ == "__main__":
    
    input = agent_executor.invoke(
        {
            "input": "How to generate videos from images using state of the art macchine learning models; Using the axriv retriever  " +
            "add the urls of the papers used in the final answer using the metadata from the retriever please do not use '`' " + 
            "please use the `download_arxiv_paper` tool  to download any axriv paper you find" + 
            "Please only use the tools provided to you"
            # f"Please prioritize the newest papers this is the current data {get_current_date()}"
        }
    )

    # input_1 = agent_executor.invoke(
    #     {
    #         "input": "I am looking for a text to 3d model; Using the axriv retriever  " +
    #         "add the urls of the papers used in the final answer using the metadata from the retriever"
    #         # f"Please prioritize the newest papers this is the current data {get_current_date()}"
    #     }
    # )
    
    # input_2 = agent_executor.invoke(
    #     {
    #         "input": "I am looking for a text to 3d model; Using the google search tool " +
    #         "add the urls in the final answer using the metadata from the retriever, also provid a summary of the searches"
    #         # f"Please prioritize the newest papers this is the current data {get_current_date()}"
    #     }
    # )

    x = 0 # for debugging purposes