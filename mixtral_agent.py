# LangChain supports many other chat models. Here, we're using Ollama
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
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import List, Dict
from datetime import datetime
from langchain.tools.render import render_text_description
import os
import arxiv

import dotenv

dotenv.load_dotenv()

 
OLLMA_BASE_URL = os.getenv("OLLMA_BASE_URL")


# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(
    model="mistral:instruct",
    base_url= OLLMA_BASE_URL
    )
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

arxiv_retriever = ArxivRetriever(load_max_docs=2)

from zipfile import ZipFile

def unzip_file(zip_file: str, extract_to: str) -> None:
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)



def format_info_list(info_list: List[Dict[str, str]]) -> str:
    """
    Format a list of dictionaries containing information into a single string.

    Args:
        info_list (List[Dict[str, str]]): A list of dictionaries containing information.

    Returns:
        str: A formatted string containing the information from the list.
    """
    formatted_strings = []
    for info_dict in info_list:
        formatted_string = "|"
        for key, value in info_dict.items():
            if isinstance(value, datetime.date):
                value = value.strftime('%Y-%m-%d')
            formatted_string += f"'{key}': '{value}', "
        formatted_string = formatted_string.rstrip(', ') + "|"
        formatted_strings.append(formatted_string)
    return '\n'.join(formatted_strings)
    
@tool
def arxiv_search(query: str) -> str:
    """Using the arxiv search and collects metadata."""
    # return "LangChain"
    global all_sources
    data = arxiv_retriever.invoke(query)
    meta_data = [i.metadata for i in data]
    # meta_data += all_sources
    # all_sources += meta_data
    all_sources += meta_data
    
    # formatted_info = format_info(entry_id, published, title, authors)
    
    # formatted_info = format_info_list(all_sources)
    
    return meta_data.__str__()

@tool
def google_search(query: str) -> str:
    """Using the google search and collects metadata."""
    # return "LangChain"
    global all_sources
    
    x = SerpAPIWrapper()
    search_results:dict = x.results(query)
    
 
    organic_source = search_results['organic_results']
    # formatted_string = "Title: {title}, link: {link}, snippet: {snippet}".format(**organic_source)
    cleaner_sources = ["Title: {title}, link: {link}, snippet: {snippet}".format(**i) for i in organic_source]
    
    all_sources += cleaner_sources    
    
    return cleaner_sources.__str__()
    # return organic_source

@tool
def get_arxiv_paper(paper_id:str) -> None:
    """Download a paper from axriv to download a paper please input 
    the axriv id such as "1605.08386v1" This tool is named get_arxiv_paper
    If you input "http://arxiv.org/abs/2312.02813", This will break the code. Also only do 
    "2312.02813". In addition please download one paper at a time. Pleaase keep the inputs/output
    free of additional information only have the id. 
    """
    t = 0
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
    
    number_without_period = paper_id.replace('.', '')
    
    
    # Download the archive to the PWD with a default filename.
    # paper.download_source()
    # Download the archive to the PWD with a custom filename.
    # paper.download_source(filename="downloaded-paper.tar.gz")
    # Download the archive to a specified directory with a custom filename.
    # paper.download_pdf(filename="downloaded-paper.pdf")
    # Download the PDF to a specified directory with a custom filename.
    paper.download_pdf(dirpath="./mydir", filename=f"{number_without_period}.pdf")
    
    # file_name = number_without_period + ".tar.gz"
    # dir_path = "./mydir"
    # paper.download_source(dirpath=dir_path, filename=file_name)
    
    # complete_path = dir_path + "/" + file_name
    
    # unzip_file(complete_path,number_without_period)
    
    

    

tools = [
    arxiv_search,
    google_search,
    get_arxiv_paper,
    ]

# tools = [
#     create_retriever_tool(
#     retriever,
#     "search arxiv's database for",
#     "Use this to recomend the user a paper to read Unless stated please choose the most recent models",
#     # "Searches and returns excerpts from the 2022 State of the Union.",
#     ),

#     Tool(
#         name="SerpAPI",
#         description="A low-cost Google Search API. Useful for when you need to answer questions about current events. Input should be a search query.",
#         func=SerpAPIWrapper().run,
#     )

# ]


prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

chat_model = llm
# define the agent
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
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
    
    # global variable for collecting sources
    all_sources =  []    

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
    
    # input_1 = agent_executor.invoke(
    #     {
    #         "input": "I am looking for a text to 3d model; Using the google search tool " +
    #         "add the urls in the final answer using the metadata from the retriever, also provid a summary of the searches"
    #         # f"Please prioritize the newest papers this is the current data {get_current_date()}"
    #     }
    # )

    x = 0

