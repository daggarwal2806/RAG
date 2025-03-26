import unittest
from classes.document_ingestor import DocumentIngestor

class TestDocumentIngestor(unittest.TestCase):

    def setUp(self):
        self.ingestor = DocumentIngestor(file_list=["test.pdf"], input_dir="data/raw_input", output_dir="data/cleaned_text", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_extract_text_from_pdf(self):
        text = self.ingestor._extract_text_from_pdf("data/raw_input/test.pdf")
        self.assertIsInstance(text, str)

    def test_clean_text(self):
        cleaned_text = self.ingestor._clean_text("This is a test text.")
        self.assertIsInstance(cleaned_text, str)

if __name__ == "__main__":
    unittest.main()