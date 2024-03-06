import gradio as gr
from hf_mixtral_agent import agent_executor
from innovation_pathfinder_ai.source_container.container import (
    all_sources
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
    
    def collect_urls(data_list):
        urls = []
        for item in data_list:
            # Check if item is a string and contains 'link:'
            if isinstance(item, str) and 'link:' in item:
                start = item.find('link:') + len('link: ')
                end = item.find(',', start)
                url = item[start:end if end != -1 else None].strip()
                urls.append(url)
            # Check if item is a dictionary and has 'Entry ID'
            elif isinstance(item, dict) and 'Entry ID' in item:
                urls.append(item['Entry ID'])
        return urls

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