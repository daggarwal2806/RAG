import unittest
from unittest.mock import MagicMock
from classes.rag_query_processor import RAGQueryProcessor
from classes.llm_client import LLMClient
from classes.chromadb_retriever import ChromaDBRetriever

class TestRAGQueryProcessor(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        # Mock LLMClient
        self.llm_client = MagicMock(spec=LLMClient)
        self.llm_client.query.return_value = "Mocked LLM response"

        # Mock ChromaDBRetriever
        self.retriever = MagicMock(spec=ChromaDBRetriever)
        self.retriever.query.return_value = [
                {"context": "Context about France", "id": "1", "score": 0.9, "text": "France is a country in Europe."}
            ]

        self.processor_rag = RAGQueryProcessor(llm_client=self.llm_client, retriever=self.retriever, use_rag=True)
        self.processor_non_rag = RAGQueryProcessor(llm_client=self.llm_client, retriever=self.retriever, use_rag=False)

    def test_query_with_rag(self):
        query_text = "What is the capital of France?"
        response = self.processor_rag.query(query_text)
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "Error: Could not process the query.")

        # Capture the prompt sent to the LLM
        self.llm_client.query.assert_called_once()
        prompt = self.llm_client.query.call_args[0][0]
        
        expected_prompt = (
            "Please answer the following user query based strictly on the provided constitutional context. "
            "Do not rely on external knowledge, assumptions, or unsupported information. "
            "Your response should be a concise single-paragraph, no longer than 200 words. "
            "Ensure that the response is clear, concise, and directly addresses the query. "
            "Avoid any repetition or redundant information. Focus on providing a unique and comprehensive answer.\n\n"
            "Context:\nContext about France\n\n"
            "User Query:\nWhat is the capital of France?\n\n"
            "Response:"
        )
        print(prompt)
        self.assertEqual(prompt, expected_prompt)

    def test_query_without_rag(self):
        query_text = "What is the capital of France?"
        response = self.processor_non_rag.query(query_text)
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "Error: Could not process the query.")

        # Capture the prompt sent to the LLM
        self.llm_client.query.assert_called_once()
        prompt = self.llm_client.query.call_args[0][0]
        
        expected_prompt = query_text
        self.assertEqual(prompt, expected_prompt)

if __name__ == "__main__":
    unittest.main()