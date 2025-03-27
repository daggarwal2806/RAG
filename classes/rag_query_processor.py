from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
import logging

class RAGQueryProcessor:

    def __init__(self,
                 llm_client: LLMClient,
                 retriever: ChromaDBRetriever,
                 use_rag: bool = False):
        self.use_rag = use_rag
        self.llm_client = llm_client
        self.retriever = retriever if use_rag else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")

    def query(self, query_text: str):
        """
        Processes the query with optional RAG.
        """
        try:
            self.logger.debug(f"Received query: {query_text}")
            context = ""
            final_prompt = query_text
            if self.use_rag:
                self.logger.info("-"*80)
                self.logger.info("Using RAG pipeline...")
                retrieved_docs = self.retriever.query(query_text)
                if not retrieved_docs:
                    logging.info("*** No relevant documents found.")
                else:
                    contexts = []
                    for result in retrieved_docs:
                        context = result.get('context', '')
                        contexts.append(context)
                        logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
                        logging.info(f"Score: {result.get('score', 'N/A')}")
                        doc_text = result.get('text', '')
                        preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
                        logging.info(f"Document: {preview_text}")
                        logging.info(f"Context: {context}")
                    context = "\n".join(contexts)
                self.logger.info("-" * 80)
                final_prompt = (
                    f"Please answer the following user query based strictly on the provided constitutional context. "
                    f"Do not rely on external knowledge, assumptions, or unsupported information. "
                    f"Your response should be a concise single-paragraph, no longer than 200 words. "
                    f"Ensure that the response is clear, concise, and directly addresses the query. "
                    f"Avoid any repetition or redundant information. Focus on providing a unique and comprehensive answer.\n\n"
                    f"Context:\n{context}\n\n"
                    f"User Query:\n{query_text}\n\n"
                    f"Response:"
                )

            self.logger.debug(f"Prompt to LLM: {final_prompt}")
            response = self.llm_client.query(final_prompt)
            # self.logger.debug(f"LLM Response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return "Error: Could not process the query."



# from langchain.chains import RetrievalQA
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# import google.generativeai as genai

# Configure the Gemini AI API key
# genai.configure(api_key="your-key") 

# # List available models and choose one
# models = list(genai.list_models())  # Convert generator to list
# available_model = models[0]  # Choose the first available model or any other suitable model
# print(f"Using model: {available_model}")

# # Initialize the model
# model_name = available_model.name  # Extract the model name as a string
# model = genai.GenerativeModel(model_name)

# class RAGQueryProcessor:

#     def __init__(self,
#                  llm_client: LLMClient,
#                  retriever: ChromaDBRetriever,
#                  use_rag: bool = False):
#         self.use_rag = use_rag
#         self.llm_client = llm_client
#         self.retriever = retriever if use_rag else None
#         self.logger = logging.getLogger(__name__)
#         self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")

#         # Initialize LangChain components
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.vectorstore = Chroma(collection_name="collections", embedding_function=self.embeddings)

#     def query_gemini(self, prompt: str) -> str:
#         """
#         Queries the Gemini AI model with the given prompt.
#         """
#         response = model.generate_content(prompt)
#         return response['text']

#     def query(self, query_text: str):
#         """
#         Processes the query with optional RAG.
#         """
#         try:
#             self.logger.debug(f"Received query: {query_text}")

#             if self.use_rag:
#                 self.logger.info("-" * 80)
#                 self.logger.info("Using RAG pipeline...")
#                 retrieved_docs = self.retriever.query(query_text)

#                 if not retrieved_docs:
#                     self.logger.info("*** No relevant documents found.")
#                     context = ""
#                 else:
#                     contexts = [result["context"] for result in retrieved_docs]
#                     context = "\n".join(contexts)

#                 self.logger.info("-" * 80)

#                 # Construct the final prompt
#                 final_prompt = self.construct_prompt(context, query_text)
#                 self.logger.debug(f"Prompt to LLM: {final_prompt}")
#                 response = self.query_gemini(final_prompt)
#                 self.logger.debug(f"LLM Response: {response}")
#                 return response
#             else:
#                 # Directly query the LLM without RAG
#                 response = self.llm_client.query(query_text)
#                 self.logger.debug(f"LLM Response: {response}")
#                 return response
#         except Exception as e:
#             self.logger.error(f"Error processing query: {e}")
#             return "Error: Could not process the query."

#     def construct_prompt(self, context: str, query_text: str) -> str:
#         """
#         Constructs a better prompt for the LLM.
#         """
#         if context:
#             prompt = (
#                 f"Context:\n{context}\n\n"
#                 f"User Query:\n{query_text}\n\n"
#                 f"Please provide a detailed comparison in one long paragraph, using only the provided context and without using any outside material."
#             )
#         else:
#             prompt = query_text
#         return prompt



# Construct structured prompt
# final_prompt = f"""
# You are an AI assistant answering user queries using retrieved context.
# If the context is insufficient, say 'I don't know'. 

# Context:
# {context if context else "No relevant context found."}

# Question:
# {query_text}
# """

# self.logger.debug(f"Prompt to LLM: {final_prompt}")

# response = self.llm_client.query(final_prompt)
# self.logger.debug(f"LLM Response: {response}")

# return response



# # RAG mode: Retrieve relevant context
# context = ""
# if self.use_rag:
#     self.logger.info("-"*80)
#     self.logger.info("Using RAG pipeline...")
#     retrieved_docs = self.retriever.query(query_text)
#     # context = "\n".join(retrieved_docs[0].get('contexts'))
#     # print(f"retrieved_docs:\n", retrieved_docs)
#
#     if not retrieved_docs:
#         logging.info("*** No relevant documents found.")
#     else:
#         result = retrieved_docs[0]
#         context = result.get('context', '')
#
#         # logging.info(f"Result {idx + 1}:")
#         logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
#         logging.info(f"Score: {result.get('score', 'N/A')}")
#         doc_text = result.get('text', '')
#         preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
#         logging.info(f"Document: {preview_text}")
#         logging.info(f"Context: {context}")
#     self.logger.info("-" * 80)
#
# # Construct the final prompt
# final_prompt = f"Context:\n{context}\n\nUser Query:\n{query_text}" if context else query_text
# self.logger.debug(f"Prompt to LLM: {final_prompt}")
# response = self.llm_client.query(final_prompt)
# self.logger.debug(f"LLM Response: {response}")
# return response
