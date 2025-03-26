import unittest
from main import step01_ingest_documents, step02_generate_embeddings, step03_store_vectors, step04_retrieve_relevant_chunks, step05_generate_response
from classes.config_manager import ConfigManager

class TestIntegration(unittest.TestCase):

    def setUp(self):
        # Setup any necessary configuration or state
        self.config = ConfigManager("config.json")
        self.args = self.Args()

    class Args:
        input_filename = "all"
        query_args = "Tell me about the US Constitution"
        use_rag = True

    def test_pipeline(self):
        # Run each step of the pipeline
        step01_ingest_documents(self.args)
        step02_generate_embeddings(self.args)
        step03_store_vectors(self.args)
        step04_retrieve_relevant_chunks(self.args)
        response = step05_generate_response(self.args)
        
        # Verify the final response
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "Error: Could not process the query.")
        self.assertIn("US Constitution", response)  # Assuming the response contains "US Constitution"

if __name__ == "__main__":
    unittest.main()