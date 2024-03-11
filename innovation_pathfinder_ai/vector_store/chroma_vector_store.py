# got some of the code from 
# https://diptimanrc.medium.com/rapid-q-a-on-multiple-pdfs-using-langchain-and-chromadb-as-local-disk-vector-store-60678328c0df
# https://stackoverflow.com/questions/76482987/chroma-database-embeddings-none-when-using-get
# https://docs.trychroma.com/embeddings/hugging-face?lang=py
# https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide

import PyPDF2
import io
import os
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import dotenv
from tqdm import tqdm
import uuid


dotenv.load_dotenv()


VECTOR_DATABASE_LOCATION = os.getenv("VECTOR_DATABASE_LOCATION")

def generate_uuid() -> str:
    """
    Generate a UUID (Universally Unique Identifier) and return it as a string.

    Returns:
        str: A UUID string.
    """
    return str(uuid.uuid4())

        
def extract_text_from_pdf(file) -> list[str]:
    documents = []
    try:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text= page.extract_text() + "\n"
            documents.append(text)
    except Exception as e:
        print(e)
    finally:
        return documents


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
    
    text_splitter = CharacterTextSplitter(
        chunk_size=text_chunk_size,
        chunk_overlap=text_chunk_overlap,
        )
    
    documents.extend(loader.load())
    
    client = chromadb.PersistentClient(
    # path=persist_directory,
    )
    
    # client.delete_collection(
    # name=collection_name,
    # )
    
    collection = client.get_or_create_collection(
    name=collection_name,
    )
    
    embed_data = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key= os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    
    
    chunked_documents = text_splitter.split_documents(documents)
    
    documents_page_content:list = [i.page_content for i in documents]
    
    
    for i in range(0, len(documents)):
        data = documents[i]
        print(i)
        collection.add(
            ids=[generate_uuid()], # give each document a uuid
            documents=documents_page_content[i], # contents of document
            embeddings=embed_data.embed_with_retries(documents_page_content[i]),
            metadatas=data.metadata,  # type: ignore
        )
    
    
def load_chunk_persist_pdf(
    pdf_folder_path: str = "mydir",
    vector_db_location:str = VECTOR_DATABASE_LOCATION,
    ) -> Chroma:

    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding = HuggingFaceEmbeddings(),
        persist_directory=VECTOR_DATABASE_LOCATION,
    )
    vectordb.persist()
    return vectordb


def load_vector_store(
    vector_store_location=os.getenv("VECTOR_DATABASE_LOCATION"),
    embeddings:chromadb.utils.embedding_functions = HuggingFaceEmbeddings(),
) -> Chroma:
    """
    ## Summary
    get the vector_store 

    ## Arguments
    vector_store_location (str) : the location of the vector store
    embeddings (chromadb.utils.embedding_functions) : the function for embedding the data 
        
    ## Return
    returns the chroma db vector store
    """
    
    db = Chroma(
    persist_directory=vector_store_location, 
    embedding_function=embeddings,
    )
    
    return db


if __name__ == "__main__":
    
    vector_db = load_vector_store()
    # pdf_file_location = "mydir/181000551.pdf"
    # pdf_file_location = "/workspaces/InnovationPathfinderAI/2402.17764.pdf"
    pdf_file_location = "/workspaces/InnovationPathfinderAI/2212.02623.pdf"
    
    add_pdf_to_vector_store(
        collection_name="ArxivPapers",
        pdf_file_location=pdf_file_location,
    )