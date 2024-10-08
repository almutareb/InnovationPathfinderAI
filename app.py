from fastapi import FastAPI
import gradio as gr
from gradio.themes.base import Base
from innovation_pathfinder_ai.agents.hf_mixtral_agent import agent_executor
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)
from innovation_pathfinder_ai.utils.utils import extract_urls
from innovation_pathfinder_ai.utils import logger

from innovation_pathfinder_ai.utils.utils import (
    generate_uuid    
)
from langchain_chroma import Chroma

import chromadb
import dotenv
import os

dotenv.load_dotenv()
persist_directory = os.getenv('VECTOR_DATABASE_LOCATION')

logger = logger.get_console_logger("app")

app = FastAPI()

def initialize_chroma_db() -> Chroma:
    collection_name = os.getenv('CONVERSATION_COLLECTION_NAME')
    
    client = chromadb.PersistentClient(
        path=persist_directory
        )
    
    collection = client.get_or_create_collection(
    name=collection_name,
    )
    
    return collection



if __name__ == "__main__":
    
    db = initialize_chroma_db()
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, ""

    def bot(history):
        response = infer(history[-1][0], history)
        sources = extract_urls(all_sources)
        src_list = '\n'.join(sources)
        current_id = generate_uuid()
        db.add(
            ids=[current_id],
            documents=[response['output']],
            metadatas=[
                {
                    "human_message":history[-1][0],
                    "sources": 'Internal Knowledge Base From: \n\n' + src_list
                }
            ]
        )
        if not sources:
            response_w_sources = response['output']+"\n\n\n Sources:  \n\n\n Internal knowledge base"
        else:
            response_w_sources = response['output']+"\n\n\n Sources: \n\n\n"+src_list
        history[-1][1] = response_w_sources
        all_sources.clear()
        return history

    def infer(question, history):
        query =  question
        result = agent_executor.invoke(
            {
                "input": question,
                "chat_history": history
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
    <div style="text-align:left;">
        <p>Hello Human, I am your AI knowledge research assistant. I can explore topics across ArXiv, Wikipedia and use Google search.<br />
    </div>
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="AlfredAI - AI Knowledge Research Assistant") as demo:
       # with gr.Tab("Google|Wikipedia|Arxiv"):
            with gr.Column(elem_id="col-container"):
                gr.HTML(title)
                with gr.Row():
                    question = gr.Textbox(label="Question", 
                                          placeholder="Type your question and hit Enter",)
                chatbot = gr.Chatbot([], 
                                     elem_id="AI Assistant",
                                     bubble_full_width=False,
                                     avatar_images=(None, "./innovation_pathfinder_ai/assets/avatar.png"),
                                     height=480,)
                chatbot.like(vote, None, None)
                clear = gr.Button("Clear")
            question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            with gr.Accordion("Open for More!", open=False):
                gr.Markdown("Nothing yet...")

    demo.queue()
    demo.launch(debug=True, favicon_path="innovation_pathfinder_ai/assets/favicon.ico", share=True)

    x = 0 # for debugging purposes
    app = gr.mount_gradio_app(app, demo, path="/")