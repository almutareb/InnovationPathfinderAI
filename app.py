from fastapi import FastAPI
import gradio as gr
from hf_mixtral_agent import agent_executor
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)
from innovation_pathfinder_ai.utils.utils import extract_urls
from innovation_pathfinder_ai.utils import logger

from innovation_pathfinder_ai.utils.utils import (
    generate_uuid    
)
from langchain_community.vectorstores import Chroma

import chromadb
from configparser import ConfigParser
import dotenv
import os

dotenv.load_dotenv()
config = ConfigParser()
config.read('innovation_pathfinder_ai/config.ini')

logger = logger.get_console_logger("app")

app = FastAPI()

def initialize_chroma_db() -> Chroma:
    collection_name = config.get('main', 'CONVERSATION_COLLECTION_NAME')
    
    client = chromadb.PersistentClient()
    
    collection = client.get_or_create_collection(
    name=collection_name,
    )
    
    return collection



if __name__ == "__main__":
    
    current_id = generate_uuid()
    
    db = initialize_chroma_db()
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, ""

    def bot(history):
        response = infer(history[-1][0], history)
        sources = extract_urls(all_sources)
        src_list = '\n'.join(sources)
        response_w_sources = response['output']+"\n\n\n Sources: \n\n\n"+src_list
        history[-1][1] = response_w_sources
        return history

    def infer(question, history):
        query =  question
        result = agent_executor.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )
        
        db.add(
            ids=[current_id],
            documents=[result['output']],
            metadatas=[
                {
                    "query":query,
                    "sources":result['sources'].__str__()
                }
            ]
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

    demo.queue().launch(debug=True, share=True)

    x = 0 # for debugging purposes
    app = gr.mount_gradio_app(app, demo, path="/")