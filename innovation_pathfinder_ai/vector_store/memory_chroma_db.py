import chromadb
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from innovation_pathfinder_ai.utils.utils import hash_text

config = load_dotenv(".env")
VECTOR_DATABASE_LOCATION=os.getenv('VECTOR_DATABASE_LOCATION')
chroma_client = chromadb.PersistentClient(VECTOR_DATABASE_LOCATION)
mem_collection = chroma_client.get_or_create_collection(name="agent_mem")

model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"

def add_to_memory(text):
    """
    Add given text to the vector store to extend the internal knowledge base.

    Args:
        text (dict): The new information to store
    """
    docs = text['response']
    doc_metadata = text['sources']
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    all_splits = text_splitter.create_documents(docs)

    # Embed and index
    #embedding = GPT4AllEmbeddings()
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    mem_collection.add(
        documents=all_splits,
        #metadatas=doc_metadata,
        ids=hash_text(docs)
    )


    # # Index
    # vectorstore = Chroma.add_documents(
    #     self,
    #     documents=all_splits,
    #     collection_name=mem_collection,
    #     embedding=embedding_function,
    # )
    # vectorstore.persist()