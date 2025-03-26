import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
from pathlib import Path
import logging
import numpy as np

class ChromaDBRetriever:
    """Retrieves relevant documents from ChromaDB based on a search phrase."""

    def __init__(self, embedding_model_name: str,
                 collection_name: str,
                 vectordb_dir: str,
                 score_threshold: float = 0.5):
        self.vectordb_path = Path(vectordb_dir)
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.score_threshold = score_threshold  # Minimum similarity score for valid results

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ChromaDBRetriever: embedding_model_name: {embedding_model_name}, collection_name: {collection_name}, score_threshold: {score_threshold}")

    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for the input text."""
        try:
            return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            self.logger.error(f"Error generating embedding for text: {e}")
            return []

    def extract_context(self, full_text: str, search_str: str) -> str:
        """
        Extracts the paragraph that contains the search term.
        Falls back to the entire text if no match is found.
        """
        # Extracting the full text as it is rquired for sufficient context and we know that
        # the text is already relevant to the search query
        return full_text
    
        # paragraphs = full_text.split("\n\n")  # Split by paragraph
        # for para in paragraphs:
        #     if search_str.lower() in para.lower():
        #         return para.strip()
        # return full_text[:300]  # Fallback: Return the first 300 characters if no match

    # def extract_countries(self, query: str) -> List[str]:
    #     """Extracts country names from the query using pycountry."""
    #     countries = []
    #     for country in pycountry.countries:
    #         if re.search(r'\b' + re.escape(country.name) + r'\b', query, re.IGNORECASE) or \
    #            re.search(r'\b' + re.escape(country.alpha_2) + r'\b', query, re.IGNORECASE) or \
    #            re.search(r'\b' + re.escape(country.alpha_3) + r'\b', query, re.IGNORECASE):
    #             countries.append(country.name)
    #     return countries

    def query(self, search_phrase: str) -> Dict[str, Any]:
        print("Querying ChromaDB")
        print("Search Phrase:", search_phrase)
        """
        Queries ChromaDB collection and returns structured results.
        Filters low-score matches and prioritizes the most relevant document.
        """

        # Tell me about the constitutions of India, Thailand, Bahamas and Norway
        # Document Retrieval is nor perfect as of now!!
        try:
            # countries = self.extract_countries(search_phrase)
            top_k = 10 # Set top_k based on the number of countries (max 10 at the moment)
            embedding_vector = self.embed_text(search_phrase)
            results = self.collection.query(query_embeddings=[embedding_vector], n_results=top_k)
            print("Results:", results)

            # Parse results
            retrieved_docs = []
            #missing_countries = [] MAYBE FUTURE SCOPE
            # query_words = set(search_phrase.lower().split())

            for doc_id, metadata, distance in zip(results.get("ids", [[]])[0], results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]):
                print("Doc ID:", doc_id)
                if distance < self.score_threshold:
                    continue  # Skip low-confidence matches

                text = metadata.get("text", "")
                extracted_context = self.extract_context(text, search_phrase)
                retrieved_docs.append({
                        "id": doc_id,
                        "score": round(distance, 4),
                        "context": extracted_context,
                        "source": metadata.get("source", "Unknown"),
                    })
    
            retrieved_docs.sort(key=lambda x: x["score"]) 

            # Determine the margin dynamically using standard deviation
            if retrieved_docs:
                scores = np.array([doc["score"] for doc in retrieved_docs])
                min_score = scores[0]
                std_dev = np.std(scores)
                margin = std_dev  # Use standard deviation as the margin

                filtered_docs = [doc for doc in retrieved_docs if doc["score"] <= min_score + margin]
            else:
                filtered_docs = []

            # Sort by score (lower is better in similarity searches)
            # retrieved_docs.sort(key=lambda x: x["score"])

            # return {
            #     "retrieved_docs": filtered_docs,
            #     "missing_countries": {}
            # }

            print("Retrieved Docs:", filtered_docs)

            return filtered_docs
        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {e}")
            return []
        