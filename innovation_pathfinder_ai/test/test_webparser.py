import os
from typing import List, Optional
import pytest
import sys
import os
import dotenv

dotenv.load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to the Python path
sys.path.append(project_root)

from innovation_pathfinder_ai.vector_store.chroma_vector_store import Document
from innovation_pathfinder_ai.vector_store.chroma_vector_store import chunk_web_data


@pytest.fixture
def mock_web_data():
    # Define mock web data
    urls = ["http://example.com/page1", "http://example.com/page2"]
    return urls


def test_chunk_web_data_with_default_parameters(mock_web_data):
    # Test chunk_web_data with default parameters
    result = chunk_web_data(mock_web_data)
    assert isinstance(result, list)
    assert all(isinstance(doc, Document) for doc in result)


def test_chunk_web_data_with_custom_parameters(mock_web_data):
    # Test chunk_web_data with custom parameters
    chunk_overlap = 20
    tokens_per_chunk = 200
    model_name = os.getenv("EMBEDDING_MODEL")
    chunk_size = 500
    result = chunk_web_data(mock_web_data, chunk_overlap=chunk_overlap, tokens_per_chunk=tokens_per_chunk, model_name=model_name, chunk_size=chunk_size)
    assert isinstance(result, list)
    assert all(isinstance(doc, Document) for doc in result)


def test_chunk_web_data_with_invalid_urls():
    # Test chunk_web_data with invalid URLs
    invalid_urls = ["invalid_url"]
    with pytest.raises(Exception):
        chunk_web_data(invalid_urls)


if __name__ == "__main__":
    pytest.main()
