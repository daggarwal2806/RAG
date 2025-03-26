import unittest
from classes.embedding_loader import EmbeddingLoader

class TestEmbeddingLoader(unittest.TestCase):

    def setUp(self):
        self.loader = EmbeddingLoader(cleaned_text_file_list=["test_cleaned.txt"], cleaned_text_dir="data/cleaned_text", embeddings_dir="data/embeddings", vectordb_dir="data/vectordb", collection_name="collections")

    def test_load_cleaned_text(self):
        text = self.loader._load_cleaned_text("data/cleaned_text/test_cleaned.txt")
        self.assertIsInstance(text, str)

    def test_load_embeddings(self):
        embeddings = self.loader._load_embeddings("data/embeddings/test_cleaned.json")
        self.assertIsInstance(embeddings, list)

if __name__ == "__main__":
    unittest.main()