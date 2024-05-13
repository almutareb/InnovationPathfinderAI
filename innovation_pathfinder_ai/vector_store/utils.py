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