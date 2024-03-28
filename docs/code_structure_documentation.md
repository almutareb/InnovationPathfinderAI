## Functionality Breakdown

### Internal Knowledge Exploration

- **memory_search**: Searches a Chroma vector store containing past findings, utilizing sentence embeddings to identify relevant information.
  
- **knowledgeBase_search**: Similar to `memory_search`, but specifically explores a collection within Chroma potentially holding research papers or relevant information chunks.

### External Information Retrieval

- **arxiv_search**: Retrieves scientific research papers from the Arxiv database based on user queries, stores metadata, and formats papers for display.

- **get_arxiv_paper**: Downloads full PDF content of a specific Arxiv paper based on its ID and saves it to a designated directory.

- **wikipedia_search**: Queries Wikipedia for relevant summaries based on user input, formats and stores retrieved summaries.

- **google_search**: Performs a Google Search using user queries, retrieves relevant results, and stores them for later use.

### Data Storage and Management

- **Chroma Vector Store**: Stores text content embeddings for similarity search (`memory_search`) and Arxiv paper embeddings (`embed_arvix_paper`).

- **Global `all_sources` Variable**: Accumulates retrieved information from various sources (Arxiv, Wikipedia, Google Search).

## Additional Notes

- These tools are integrated with a larger LangChain framework, as indicated by the `@tool` decorator.
  
- Configuration files (`config.ini`) define details like vector store location and collection names.

## Getting Started

1. Install the LangChain framework and required libraries.
  
2. Configure access to the Chroma vector store (likely involves environment variables or configuration files).

3. Refer to the InnovationPathfinderAI project documentation for detailed integration instructions.

## Disclaimer

This documentation provides an overview of the tools' functionalities. 
