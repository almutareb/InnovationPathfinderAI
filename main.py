import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import WebBaseLoader

load_dotenv()


def chunk_web_data(
    urls: List[str],
    chunk_overlap: Optional[int] = 50,
    tokens_per_chunk: Optional[int] = None,
    model_name: str = os.getenv("EMBEDDING_MODEL"),
    chunk_size: int = 1000,
    ) -> List[Document]:
    """
    ## Summary
    This function is used to chunk webpages
    
    ## Arguments
    urls list[str] : a list of urls  to be chunks
    chunk_size int : the chunking size
    model_name str : the embedding model used to chunk will use the environment default unless overwritten
    tokens_per_chunk int | None : the amount of chunks per token paramter inhereted from `SentenceTransformersTokenTextSplitter`
    chunk_size int : the size of chunks per `Document`

    ## Return
    it may be a List[Document] or None if it is a List[Document] then these chunks will be 
    embedded wiht a different function or method
    """
    
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=tokens_per_chunk,
        model_name=model_name,
        chunk_size=chunk_size
    )
    
    loader = WebBaseLoader(urls)

    data = loader.load_and_split(
        text_splitter=text_splitter
    )
    
    return data


if __name__ == "__main__":
    web_data_load:List[str] = [
        "https://www.gutenberg.org/cache/epub/73268/pg73268-images.html",
        "https://www.gutenberg.org/cache/epub/73269/pg73269-images.html",
        
    ]

    data = chunk_web_data(
        urls=web_data_load
    )


    x = 0 