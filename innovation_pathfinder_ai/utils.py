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
    
    return urls

def collect_urls(data_list):
    urls = []
    for item in data_list:
        # Check if item is a string and contains 'link:'
        if isinstance(item, str) and 'link:' in item:
            start = item.find('link:') + len('link: ')
            end = item.find(',', start)
            url = item[start:end if end != -1 else None].strip()
            urls.append(url)
        # Check if item is a dictionary and has 'Entry ID'
        elif isinstance(item, dict) and 'Entry ID' in item:
            urls.append(item['Entry ID'])
    last_sources = urls[-3:]
    return last_sources