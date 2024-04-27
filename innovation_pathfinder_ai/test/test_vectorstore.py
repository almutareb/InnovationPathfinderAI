from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
import unittest
import tempfile
from langchain_community.vectorstores import Chroma
import chromadb
import requests

from innovation_pathfinder_ai.utils.image_processing.image_processing import(
    caption_image
)

from vector_store.chroma_vector_store import (
    add_images
)

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        collection_name="TestImages"
        self.tmp_dir = tempfile.TemporaryDirectory()
        
        client = chromadb.PersistentClient(
            path=self.tmp_dir.name,
            )
        
        collection = client.get_or_create_collection(
        name=collection_name,   
        )
        
        embedding_function = SentenceTransformerEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        )
        
        self.vector_db = Chroma(
        client=client, # client for Chroma
        collection_name=collection_name,
        embedding_function=embedding_function,
        )
        
    def tearDown(self):
        self.tmp_dir.cleanup()
        
    def test_data(self):
        url = "https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png"
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(response.content)  # Write the image content to the temporary file
            temp_file_name = temp_file.name  # Get the temporary file name
            x = add_images(temp_file_name)
            xx = 0
        pass
