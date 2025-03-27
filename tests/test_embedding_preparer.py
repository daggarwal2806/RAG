import unittest
from classes.embedding_preparer import EmbeddingPreparer

class TestEmbeddingPreparer(unittest.TestCase):

    def setUp(self):
        self.preparer = EmbeddingPreparer(file_list=["test_cleaned.txt"], input_dir="data/test/cleaned_text", output_dir="data/test/embeddings", embedding_model_name="sentence-transformers/all-mpnet-base-v2")

    def test_generate_embedding(self):
        embedding = self.preparer._generate_embedding("This is a test text.")
        self.assertIsInstance(embedding, list)

if __name__ == "__main__":
    unittest.main()