from langchain_core.vectorstores import VectorStore
from innovation_pathfinder_ai.vector_store.utils import (
    read_markdown_file
)
from innovation_pathfinder_ai.utils.utils import (
    generate_uuid    
)

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import dotenv
import os
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader

# for typing 
from typing import List, Any
from chromadb.api import BaseAPI, ClientAPI
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()

class InnovataionVectorStore(VectorStore):
    pass

class ChromaInnovationVectorStore:
    
    def __init__(
        self,
        chroma_client: ClientAPI,
        chroma_collection: Collection,
        embedding_function: HuggingFaceEmbeddings,
        vector_store_database_location:str = os.getenv('VECTOR_DATABASE_LOCATION'),
        ) -> None:
        self.chroma_client = chroma_client
        self.chroma_collection = chroma_collection
        self.embedding_function = embedding_function
        self.vector_store_database_location = vector_store_database_location
    
    def create_markdown_documents(
        self,
        markdown_file_location:str,
        chunk_size:int,
        chunk_overlap:int,
    ) -> List[Document]:
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

        return splits
    
    def create_documents_from_pdfs(
        self,
        pdf_file_location:str,
        text_chunk_size=1000,
        text_chunk_overlap=10,
        ) -> List[Document]:
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
            sub_docs = self.split_by_intervals(
                document.page_content, 
                text_chunk_size,
                text_chunk_overlap
                )
            
            for sub_doc in sub_docs:
                loaded_doc = Document(sub_doc, metadata=document.metadata)
                split_docs.append(loaded_doc)
            
        return split_docs

    @staticmethod
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