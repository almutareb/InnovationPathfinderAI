import pytest
import  chromadb
from unittest.mock import Mock, patch
from innovation_pathfinder_ai.vector_store.vector_store_class import ChromaInnovationVectorStore
from innovation_pathfinder_ai.vector_store.utils import read_markdown_file
import os
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

#typing
from langchain_core.documents import Document

@pytest.fixture
def mock_chroma_client():
    # client = chromadb.PersistentClient(
    #  path="./testing",
    # )
    client = chromadb.EphemeralClient()
    return client


@pytest.fixture
def mock_chroma_collection(mock_chroma_client):
    client = mock_chroma_client
    collection_name = "testing"
    collection = client.get_or_create_collection(
        name=collection_name,
        )
    yield collection
    client.delete_collection(collection_name)

@pytest.fixture
def mock_embedding_function():
    embedding_function = SentenceTransformerEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        )
    return embedding_function

@pytest.fixture
def chroma_innovation_vector_store(mock_chroma_client, mock_chroma_collection, mock_embedding_function):
    return ChromaInnovationVectorStore(
        chroma_client=mock_chroma_client,
        chroma_collection=mock_chroma_collection,
        embedding_function=mock_embedding_function
    )
    

def test_create_markdown_documents(chroma_innovation_vector_store, mock_chroma_collection, mock_embedding_function):
    # Mocking dependencies
    mock_read_markdown_file = Mock(return_value="Sample markdown content")
    with patch('innovation_pathfinder_ai.vector_store.utils.read_markdown_file', mock_read_markdown_file):
        # Calling the method to be tested
        splits = chroma_innovation_vector_store.create_markdown_documents("docs/code_structure_documentation.md", 100, 50)
        # Assertions
        # mock_read_markdown_file.assert_called_once_with("docs/code_structure_documentation.md")
        # mock_embedding_function.assert_called_once_with("Sample markdown content")
        assert isinstance(splits, list)
        assert all(isinstance(item, Document) for item in splits)

def test_create_documents_from_pdfs(chroma_innovation_vector_store):
    # Mocking dependencies
    # mock_loader_instance = Mock()
    # mock_loader_instance.load.return_value = [Mock(page_content="Sample PDF content", metadata={})]
    # with patch('your_module.PyPDFLoader', return_value=mock_loader_instance):
        # Calling the method to be tested
    splits = chroma_innovation_vector_store.create_documents_from_pdfs("assets/1706.03762.pdf", 1000, 10)
    # Assertions
    assert isinstance(splits, list)
    assert all(isinstance(item, Document) for item in splits)

def test_create_documents_from_web(chroma_innovation_vector_store):
    # Mocking dependencies
    # mock_loader_instance = Mock()
    # mock_loader_instance.load_and_split.return_value = [Mock(page_content="Sample web content", metadata={})]
    # with patch('your_module.WebBaseLoader', return_value=mock_loader_instance):
        # Calling the method to be tested
    splits = chroma_innovation_vector_store.create_documents_from_web(["http://example.com"], chunk_overlap=50, tokens_per_chunk=None, model_name=os.getenv("EMBEDDING_MODEL"), chunk_size=1000)
    
    # Assertions
    assert isinstance(splits, list)
    assert all(isinstance(item, Document) for item in splits)

def test_create_documents_from_images(chroma_innovation_vector_store):
    # Calling the method to be tested
    result = chroma_innovation_vector_store.create_documents_from_images(["assets/sample_screenshot.png"])
    # Assertions
    # assert result.page_content.startswith("Sample caption")
    # assert "image_location" in result.metadata
    
    assert isinstance(result, list)
    assert all(isinstance(item, Document) for item in result)