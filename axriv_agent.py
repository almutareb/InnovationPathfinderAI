import sys
import os
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

config = load_dotenv(".env")

# Retrieve the Hugging Face API token from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# S3_LOCATION


try:
    # Load the model from the Hugging Face Hub
    model_id = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={
        "temperature": 0.1,         # Controls randomness in response generation (lower value means less random)
        "max_new_tokens": 1024,     # Maximum number of new tokens to generate in responses
        "repetition_penalty": 1.2,  # Penalty for repeating the same words (higher value increases penalty)
        "return_full_text": False   # If False, only the newly generated text is returned; if True, the input is included as well
    })
    print("Model loaded successfully from Hugging Face Hub.")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {e}", file=sys.stderr)

from langchain.retrievers import ArxivRetriever
retriever = ArxivRetriever(load_max_docs=2)
docs = retriever.get_relevant_documents(query="1605.08386")
docs[0].metadata  # meta-information of the Document
x = 0