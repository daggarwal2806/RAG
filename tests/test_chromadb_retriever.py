import unittest
from classes.chromadb_retriever import ChromaDBRetriever

class TestChromaDBRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = ChromaDBRetriever(vectordb_dir="data/test/vectordb", embedding_model_name="sentence-transformers/all-mpnet-base-v2", collection_name="collections")

    def test_query(self):
        results = self.retriever.query("What is the capital of France?")
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()