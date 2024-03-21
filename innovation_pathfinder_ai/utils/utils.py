import hashlib
import datetime
import os
import uuid

from innovation_pathfinder_ai.utils import logger

logger = logger.get_console_logger("utils")

def create_wikipedia_urls_from_text(text):
    """
    Extracts page titles from a given text and constructs Wikipedia URLs for each title.
    
    Args:
    - text (str): A string containing multiple sections, each starting with "Page:" followed by the title.
    
    Returns:
    - list: A list of Wikipedia URLs constructed from the extracted titles.
    """
    # Split the text into sections based on "Page:" prefix
    sections = text.split("Page: ")
    # Remove the first item if it's empty (in case the text starts with "Page:")
    if sections[0].strip() == "":
        sections = sections[1:]
    
    urls = []  # Initialize an empty list to store the URLs
    for section in sections:
        # Extract the title, which is the string up to the first newline
        title = section.split("\n", 1)[0]
        # Replace spaces with underscores for the URL
        url_title = title.replace(" ", "_")
        # Construct the URL and add it to the list
        url = f"https://en.wikipedia.org/wiki/{url_title}"
        urls.append(url)
        #print(urls)
    
    return urls

def extract_urls(data_list):
    """
    Extracts URLs from a list of of dictionaries.

    Parameters:
    - formatted_list (list): A list of dictionaries, each containing 'Title:', 'link:', and 'summary:'.

    Returns:
    - list: A list of URLs extracted from the dictionaries.
    """
    urls = []
    print(data_list)
    for item in data_list:
        try:
            # Find the start and end indices of the URL
            lower_case = item.lower()
            link_prefix = 'link: '
            summary_prefix = ', summary:'
            start_idx = lower_case.index(link_prefix) + len(link_prefix)
            end_idx = lower_case.index(summary_prefix, start_idx)
            # Extract the URL using the indices found
            url = item[start_idx:end_idx]
            urls.append(url)
        except ValueError:
            # Handles the case where 'link: ' or ', summary:' is not found in the string
            print("Could not find a URL in the item:", item)
    last_sources = urls[-3:]
    return last_sources

def format_wiki_summaries(input_text):
    """
    Parses a given text containing page titles and summaries, formats them into a list of strings,
    and appends Wikipedia URLs based on titles.
    
    Parameters:
    - input_text (str): A string containing titles and summaries separated by specific markers.
    
    Returns:
    - list: A list of formatted strings with titles, summaries, and Wikipedia URLs.
    """
    # Splitting the input text into individual records based on double newlines
    records = input_text.split("\n\n")
    
    formatted_records_with_urls = []
    for record in records:
        if "Page:" in record and "Summary:" in record:
            title_line, summary_line = record.split("\n", 1)  # Splitting only on the first newline
            title = title_line.replace("Page: ", "").strip()
            summary = summary_line.replace("Summary: ", "").strip()
            # Replace spaces with underscores for the URL and construct the Wikipedia URL
            url_title = title.replace(" ", "_")
            wikipedia_url = f"https://en.wikipedia.org/wiki/{url_title}"
            # Append formatted string with title, summary, and URL
            formatted_record = "Title: {title}, Link: {wikipedia_url}, Summary: {summary}".format(
                title=title, summary=summary, wikipedia_url=wikipedia_url)
            formatted_records_with_urls.append(formatted_record)
        else:
            print("Record format error, skipping record:", record)
    
    return formatted_records_with_urls

def format_arxiv_documents(documents):
    """
    Formats a list of document objects into a list of strings.
    Each document object is assumed to have a 'metadata' dictionary with 'Title' and 'Entry ID',
    and a 'page_content' attribute for content.

    Parameters:
    - documents (list): A list of document objects.

    Returns:
    - list: A list of formatted strings with titles, links, and content snippets.
    """
    formatted_documents = [
        "Title: {title}, Link: {link}, Summary: {snippet}".format(
            title=doc.metadata['Title'],
            link=doc.metadata['Entry ID'],
            snippet=doc.page_content  # Adjust the snippet length as needed
        )
        for doc in documents
    ]
    return formatted_documents

def format_search_results(search_results):
    """
    Formats a list of dictionaries containing search results into a list of strings.
    Each dictionary is expected to have the keys 'title', 'link', and 'snippet'.

    Parameters:
    - search_results (list): A list of dictionaries, each containing 'title', 'link', and 'snippet'.

    Returns:
    - list: A list of formatted strings based on the search results.
    """
    formatted_results = [
        "Title: {title}, Link: {link}, Summary: {snippet}".format(**i)
        for i in search_results
    ]
    return formatted_results

def parse_list_to_dicts(items: list) -> list:
    parsed_items = []
    for item in items:
        # Extract title, link, and summary from each string
        title_start = item.find('Title: ') + len('Title: ')
        link_start = item.find('Link: ') + len('Link: ')
        summary_start = item.find('Summary: ') + len('Summary: ')

        title_end = item.find(', Link: ')
        link_end = item.find(', Summary: ')
        summary_end = len(item)

        title = item[title_start:title_end]
        link = item[link_start:link_end]
        summary = item[summary_start:summary_end]

        # Use the hash_text function for the hash_id
        hash_id = hash_text(link)

        # Construct the dictionary for each item
        parsed_item = {
            "url": link,
            "title": title,
            "hash_id": hash_id,
            "summary": summary
        }
        parsed_items.append(parsed_item)
    return parsed_items

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")

def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it doesn't already exist.

    Args:
    - folder_path (str): The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
def generate_uuid() -> str:
    """
    Generate a UUID (Universally Unique Identifier) and return it as a string.

    Returns:
        str: A UUID string.
    """
    return str(uuid.uuid4())        