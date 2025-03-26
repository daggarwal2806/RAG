import unittest
from classes.llm_client import LLMClient

class TestLLMClient(unittest.TestCase):

    def setUp(self):
        self.client = LLMClient("http://localhost:1234/v1/completions", "llama-3.2-1b-instruct")

    def test_query(self):
        response = self.client.query("What is the capital of France?")
        self.assertIsInstance(response, str)

if __name__ == "__main__":
    unittest.main()