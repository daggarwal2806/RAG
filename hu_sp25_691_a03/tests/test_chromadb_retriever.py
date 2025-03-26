import unittest
from classes.chromadb_retriever import ChromaDBRetriever

class TestChromaDBRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = ChromaDBRetriever(vectordb_dir="data/vectordb", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", collection_name="collections")

    def test_query(self):
        results = self.retriever.query("What is the capital of France?")
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()