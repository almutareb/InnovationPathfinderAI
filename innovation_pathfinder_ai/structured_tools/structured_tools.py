from langchain.tools import BaseTool, StructuredTool, tool
from langchain.retrievers import ArxivRetriever
from langchain_community.utilities import SerpAPIWrapper
import arxiv

# hacky and should be replaced with a database
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)

@tool
def arxiv_search(query: str) -> str:
    """Using the arxiv search and collects metadata."""
    # return "LangChain"
    global all_sources
    arxiv_retriever = ArxivRetriever(load_max_docs=2)
    data = arxiv_retriever.invoke(query)
    meta_data = [i.metadata for i in data]
    # meta_data += all_sources
    # all_sources += meta_data
    all_sources += meta_data
    
    # formatted_info = format_info(entry_id, published, title, authors)
    
    # formatted_info = format_info_list(all_sources)
    
    return meta_data.__str__()

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
    paper.download_pdf(dirpath="./mydir", filename=f"{number_without_period}.pdf")
    
    
@tool
def google_search(query: str) -> str:
    """Using the google search and collects metadata."""
    # return "LangChain"
    global all_sources
    
    x = SerpAPIWrapper()
    search_results:dict = x.results(query)
    
 
    organic_source = search_results['organic_results']
    # formatted_string = "Title: {title}, link: {link}, snippet: {snippet}".format(**organic_source)
    cleaner_sources = ["Title: {title}, link: {link}, snippet: {snippet}".format(**i) for i in organic_source]
    
    all_sources += cleaner_sources    
    
    return cleaner_sources.__str__()