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
from langchain.tools.render import render_text_description
import os

import dotenv

dotenv.load_dotenv()

 
OLLMA_BASE_URL = os.getenv("OLLMA_BASE_URL")


# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(
    model="mistral",
    base_url= OLLMA_BASE_URL
    )
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print(chain.invoke({"topic": "Space travel"}))

retriever = ArxivRetriever(load_max_docs=2)

tools = [
    create_retriever_tool(
    retriever,
    "search arxiv's database for",
    "Use this to recomend the user a paper to read Unless stated please choose the most recent models",
    # "Searches and returns excerpts from the 2022 State of the Union.",
    ),

    Tool(
        name="SerpAPI",
        description="A low-cost Google Search API. Useful for when you need to answer questions about current events. Input should be a search query.",
        func=SerpAPIWrapper().run,
    )

]



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
    handle_parsing_errors=True #prevents error
    )

# agent_executor.invoke(
#     {
#         "input": "Who is the current holder of the speed skating world record on 500 meters? What is her current age raised to the 0.43 power?"
#     }
# )

# agent_executor.invoke(
#     {
#         "input": "what are large language models and why are they so expensive to run?"
#     }
# )

# agent_executor.invoke(
#     {
#         "input": "How to generate videos from images using state of the art macchine learning models"
#     }
# )


agent_executor.invoke(
    {
        "input": "How to generate videos from images using state of the art macchine learning models; Using the axriv retriever  " +
        "add the urls of the papers used in the final answer using the metadata from the retriever"
        # f"Please prioritize the newest papers this is the current data {get_current_date()}"
    }
)

