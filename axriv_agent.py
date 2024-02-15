import sys
import os
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.retrievers import ArxivRetriever

config = load_dotenv(".env")

# Retrieve the Hugging Face API token from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# S3_LOCATION


try:
    # Load the model from the Hugging Face Hub
    llm = model_id = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={
        "temperature": 0.1,         # Controls randomness in response generation (lower value means less random)
        "max_new_tokens": 1024,     # Maximum number of new tokens to generate in responses
        "repetition_penalty": 1.2,  # Penalty for repeating the same words (higher value increases penalty)
        "return_full_text": False   # If False, only the newly generated text is returned; if True, the input is included as well
    })
    print("Model loaded successfully from Hugging Face Hub.")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {e}", file=sys.stderr)

retriever = ArxivRetriever(load_max_docs=2)
docs = retriever.get_relevant_documents(query="1605.08386")
docs[0].metadata  # meta-information of the Document


# Use the agent constructor to create the agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool

from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

# llm = ChatOpenAI(temperature=0)
tool = create_retriever_tool(retriever, "arxiv_retriever", "Retrieves scientific articles from Arxiv")
from langchain.agents import create_openai_functions_agent
# create_openai_tools_agent()
tools = [tool]

# agent = create_openai_tools_agent(llm, [tool])
agent = create_openai_functions_agent(llm, tools, prompt)

# Test the agent by invoking it with a query
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "What's the paper 1605.08386 about?"})
result["output"]
x = 0