# got some of the code from 
# https://diptimanrc.medium.com/rapid-q-a-on-multiple-pdfs-using-langchain-and-chromadb-as-local-disk-vector-store-60678328c0df
# https://stackoverflow.com/questions/76482987/chroma-database-embeddings-none-when-using-get
# https://docs.trychroma.com/embeddings/hugging-face?lang=py
# https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
# https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
# https://python.langchain.com/docs/integrations/vectorstores/chroma#update-and-delete
# https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from innovation_pathfinder_ai.utils.utils import (
    generate_uuid    
)
from configparser import ConfigParser
import dotenv
import os

dotenv.load_dotenv()
config = ConfigParser()
config.read('innovation_pathfinder_ai/config.ini')
persist_directory = config.get('main', 'VECTOR_DATABASE_LOCATION')


def read_markdown_file(file_path: str) -> str:
    """
    Read a Markdown file and return its content as a single string.

    Args:
        file_path (str): The path to the Markdown file.

    Returns:
        str: The content of the Markdown file as a single string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def add_markdown_to_collection(
    markdown_file_location:str,
    collection_name:str,
    chunk_size:int,
    chunk_overlap:int,
) -> None:
    """
    Embeds markdown data to a given chroma db collection
    
    markdown_file_location (str): location of markdown file
    collection_name (str) : the collection where the documents will be added
    chunk_size (int) : size of the chunks to be embedded
    chunk_overlap (int) : the ammount of overlappping chunks

    """
    markdown_document = read_markdown_file(markdown_file_location)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)

    client = chromadb.PersistentClient(
         path=persist_directory,
        )


    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(
        name=collection_name,
        )
    
    embed_data = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key= os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    documents_page_content:list = [i.page_content for i in splits]


    for i in range(0, len(splits)):
        data = splits[i]
        collection.add(
            ids=[generate_uuid()], # give each document a uuid
            documents=documents_page_content[i], # contents of document
            embeddings=embed_data.embed_with_retries(documents_page_content[i]),
            metadatas=data.metadata,  # type: ignore
        )
        
def split_by_intervals(s: str, interval: int, overlapped: int = 0) -> list:
    """
    Split a string into intervals of a given length, with optional overlapping.
    
    Args:
        s: The input string.
        interval: The length of each interval.
        overlapped: The number of characters to overlap between intervals. Default is 0.
    
    Returns:
        A list of substrings, each containing 'interval' characters from the input string.
    """
    result = []
    for i in range(0, len(s), interval - overlapped):
        result.append(s[i:i + interval])
    return result


def add_pdf_to_vector_store(
    # vector_store:Chroma.from_documents,
    collection_name,
    pdf_file_location:str,
    text_chunk_size=1000,
    text_chunk_overlap=10,
    ) -> None:
    """
    ## Summary
    given the location of a pdf file this will chunk it's contents
    and store it the given vectorstore
    
    ## Arguments
    collection_name (str) : name of collection to store documents
    pdf_file_location (str) : location of pdf file
    
    ## Return 
    None
    """
    
    documents = []
    
    loader = PyPDFLoader(pdf_file_location)
    
    documents.extend(loader.load())
    
    split_docs:list[Document] = []
    
    for document in documents:
        sub_docs = split_by_intervals(
            document.page_content, 
            text_chunk_size,
            text_chunk_overlap
            )
        
        for sub_doc in sub_docs:
            loaded_doc = Document(sub_doc, metadata=document.metadata)
            split_docs.append(loaded_doc)
        
    
    client = chromadb.PersistentClient(
     path=persist_directory,
    )
    
    collection = client.get_or_create_collection(
    name=collection_name,
    )
    
    embed_data = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_name= "sentence-transformers/all-MiniLM-L6-v2" # added model name for clariity
    )
    
    # create the open-source embedding function
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    documents_page_content:list = [i.page_content for i in split_docs]
    
    
    for i in range(0, len(split_docs)):
        data = split_docs[i]
        
        collection.add(
            ids=[generate_uuid()], # give each document a uuid
            documents=documents_page_content[i], # contents of document
            embeddings=embed_data.embed_with_retries(documents_page_content[i]),
            metadatas=data.metadata,  # type: ignore
        )
    
    
if __name__ == "__main__":
    
    collection_name="ArxivPapers"
    
    client = chromadb.PersistentClient(
     path=persist_directory,
    )
    
    # delete existing collection
    # client.delete_collection(
    # name=collection_name,
    # )
    
    collection = client.get_or_create_collection(
    name=collection_name,
    )
    
    pdf_file_location = "/workspaces/InnovationPathfinderAI/2212.02623.pdf"
    
    add_pdf_to_vector_store(
        collection_name="ArxivPapers",
        pdf_file_location=pdf_file_location,
    )
    
    pdf_file_location = "/workspaces/InnovationPathfinderAI/2402.17764.pdf"
    
    add_pdf_to_vector_store(
        collection_name="ArxivPapers",
        pdf_file_location=pdf_file_location,
    )
    
    #create the cliient using Chroma's library
    client = chromadb.PersistentClient(
     path=persist_directory,
    )
    
    # This is an example collection name
    collection_name="ArxivPapers"
    
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        )
    
    #method of integrating Chroma and Langchain
    vector_db = Chroma(
    client=client, # client for Chroma
    collection_name=collection_name,
    embedding_function=embedding_function,
    )
    
    query = "ai" # your query 
    
    # using your Chromadb as a retriever for langchain
    retriever = vector_db.as_retriever()

    # returning a list of documents
    docs = retriever.get_relevant_documents(query)
    
    # pdf_file_location = "mydir/181000551.pdf"
    # pdf_file_location = "/workspaces/InnovationPathfinderAI/2402.17764.pdf"
    
    
    # example query using Chroma
    
    # results = collection.query(
    # query_texts=["benchmark"],
    # n_results=3,
    # include=['embeddings', 'documents', 'metadatas'],
    # )