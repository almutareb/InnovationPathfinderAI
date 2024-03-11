# HF libraries
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
import gradio as gr
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
# Import things that are needed generically
from typing import List, Dict
from langchain.tools.render import render_text_description
import os
from dotenv import load_dotenv
from innovation_pathfinder_ai.structured_tools.structured_tools import (
    arxiv_search, get_arxiv_paper, google_search, wikipedia_search
)

# hacky and should be replaced with a database
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)
from innovation_pathfinder_ai.utils import collect_urls
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# message_history = ChatMessageHistory()
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
                          return_full_text=False
    )


tools = [
    arxiv_search,
    wikipedia_search,
    google_search,
#    get_arxiv_paper,
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
    max_iterations=6,       # cap number of iterations
    #max_execution_time=60,  # timout at 60 sec
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    )

    
if __name__ == "__main__":
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, ""

    def bot(history):
        response = infer(history[-1][0], history)
        sources = collect_urls(all_sources)
        src_list = '\n'.join(sources)
        response_w_sources = response['output']+"\n\n\n Sources: \n\n\n"+src_list
        history[-1][1] = response_w_sources
        return history

    def infer(question, history):
        query =  question
        result = agent_executor.invoke(
            {
                "input": question,
            }
        )
        return result

    def vote(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)


    css="""
    #col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
    """

    title = """
    <div style="text-align: center;max-width: 700px;">
        <p>Hello Dave, how can I help today?<br />
    </div>
    """

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tab("Google|Wikipedia|Arxiv"):
            with gr.Column(elem_id="col-container"):
                gr.HTML(title)
                with gr.Row():
                    question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
                chatbot = gr.Chatbot([], elem_id="chatbot")
                chatbot.like(vote, None, None)
                clear = gr.Button("Clear")
            question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(debug=True)


    x = 0 # for debugging purposes