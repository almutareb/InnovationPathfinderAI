from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.retrievers import ArxivRetriever
#from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
#from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
import arxiv

# hacky and should be replaced with a database
from innovation_pathfinder_ai.source_container.container import (
    all_sources
)
from innovation_pathfinder_ai.utils import create_wikipedia_urls_from_text

@tool
def arxiv_search(query: str) -> str:
    """Search arxiv database for scientific research papers and studies. This is your primary information source.
    always check it first when you search for information, before using any other tool."""
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
    paper.download_pdf(dirpath="./downloaded_papers", filename=f"{number_without_period}.pdf")
    
    
@tool
def google_search(query: str) -> str:
    """Search Google for additional results when you can't answer questions using arxiv search or wikipedia search."""
    # return "LangChain"
    global all_sources
    
    websearch = GoogleSearchAPIWrapper()
    search_results:dict = websearch.results(query, 5)
    
 
    #organic_source = search_results['organic_results']
    # formatted_string = "Title: {title}, link: {link}, snippet: {snippet}".format(**organic_source)
    cleaner_sources = ["Title: {title}, link: {link}, snippet: {snippet}".format(**i) for i in search_results]
    
    all_sources += cleaner_sources    
    
    return cleaner_sources.__str__()

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for additional information to expand on research papers or when no papers can be found."""
    global all_sources

    api_wrapper = WikipediaAPIWrapper()
    wikipedia_search = WikipediaQueryRun(api_wrapper=api_wrapper)
    wikipedia_results = wikipedia_search.run(query)
    all_sources += create_wikipedia_urls_from_text(wikipedia_results)
    return wikipedia_results