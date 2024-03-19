from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.retrievers import ArxivRetriever
#from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
#from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
import arxiv
import ast
# hacky and should be replaced with a database
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)
from innovation_pathfinder_ai.utils.utils import (
    parse_list_to_dicts, format_wiki_summaries, format_arxiv_documents, format_search_results
)
from innovation_pathfinder_ai.database.db_handler import (
    add_many
)
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

@tool
def arxiv_search(query: str) -> str:
    """Search arxiv database for scientific research papers and studies. This is your primary online information source.
    always check it first when you search for information, before using any other online tool."""
    global all_sources
    arxiv_retriever = ArxivRetriever(load_max_docs=3)
    data = arxiv_retriever.invoke(query)
    meta_data = [i.metadata for i in data]
    formatted_sources = format_arxiv_documents(data)
    all_sources += formatted_sources
    parsed_sources = parse_list_to_dicts(formatted_sources)
    add_many(parsed_sources)
  
    return data.__str__()

@tool
def get_arxiv_paper(paper_id:str) -> None:
    """Download a paper from axriv to download a paper please input 
    the axriv id such as "1605.08386v1" This tool is named get_arxiv_paper
    If you input "http://arxiv.org/abs/2312.02813", This will break the code. Also only do 
    "2312.02813". In addition please download one paper at a time. Pleaase keep the inputs/output
    free of additional information only have the id. 
    """
    # code from https://lukasschwab.me/arxiv.py/arxiv.html
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
    
    number_without_period = paper_id.replace('.', '')
    
    # Download the PDF to a specified directory with a custom filename.
    paper.download_pdf(dirpath="./downloaded_papers", filename=f"{number_without_period}.pdf")
    
    
@tool
def google_search(query: str) -> str:
    """Search Google for additional results when you can't answer questions using arxiv search or wikipedia search."""
    global all_sources
    
    websearch = GoogleSearchAPIWrapper()
    search_results:dict = websearch.results(query, 3)
    cleaner_sources =format_search_results(search_results)
    parsed_csources = parse_list_to_dicts(cleaner_sources)
    add_many(parsed_csources)
    all_sources += cleaner_sources    
    
    return cleaner_sources.__str__()

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for additional information to expand on research papers or when no papers can be found."""
    global all_sources

    api_wrapper = WikipediaAPIWrapper()
    wikipedia_search = WikipediaQueryRun(api_wrapper=api_wrapper)
    wikipedia_results = wikipedia_search.run(query)
    formatted_summaries = format_wiki_summaries(wikipedia_results)
    all_sources += formatted_summaries
    parsed_summaries = parse_list_to_dicts(formatted_summaries)
    add_many(parsed_summaries)
    
    return wikipedia_results.__str__()

@tool
def memory_search(query:str) -> str:
    """Search the memory vector store for existing knowledge and relevent pervious researches. \
        This is your primary source to start your search with checking what you already have learned from the past, before going online."""
    client = chromadb.PersistentClient(
    # path=persist_directory,
    )

    collection_name="agent_mem"
    #store using envar

    # embedding_function = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    # )
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    

    vector_db = Chroma(
        client=client, # client for Chroma
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    retriever = vector_db.as_retriever()
    docs = retriever.get_relevant_documents(query)

    return docs.__str__()