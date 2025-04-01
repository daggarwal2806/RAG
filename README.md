## RAG

---

# **Retrieval-Augmented Generation (RAG) Pipeline for Constitutional Analysis**

## **Overview**
This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system designed to enhance Large Language Model (LLM) responses by integrating domain-specific knowledge from constitutional documents. The system processes, retrieves, and augments responses using embeddings stored in a vector database. It focuses on comparing, summarizing, and explaining constitutions from multiple countries.

---

## **Features**
- **Document Ingestion:** Processes raw constitutional texts into cleaned chunks.
- **Embedding Generation:** Generates dense vector representations using Sentence-Transformers (`all-mpnet-base-v2`).
- **Vector Storage:** Stores embeddings in ChromaDB for efficient similarity-based retrieval.
- **Context Retrieval:** Retrieves relevant contexts using cosine similarity and statistical filtering methods.
- **Response Augmentation:** Combines retrieved contexts with LLM outputs to generate accurate responses.
- **Testing Framework:** Includes unit tests, integration tests, and performance benchmarks.

---

## **Project Structure**
```
├── data/
│   ├── raw_input/          # Raw constitutional documents
│   ├── test/               # Test data for unit tests
├── classes/
│   ├── document_ingestor.py     # Handles document ingestion
│   ├── embedding_preparer.py    # Generates embeddings
│   ├── chromadb_retriever.py    # Retrieves relevant contexts
│   ├── rag_query_processor.py   # Augments LLM responses
│   ├── llm_client.py            # Interacts with the LLM API
├── tests/                  # Unit, integration and performance tests
├── config.json                  # Configuration file
├── main.py                      # Main entry point for running the pipeline
└── README.md                    # Project documentation
```

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/daggarwal2806/RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the `config.json` file:
   - Specify paths for input/output directories.
   - Set the embedding model name (`sentence-transformers/all-mpnet-base-v2`).
   - Update ChromaDB configuration.

---

## **Usage**
### **Pipeline Execution**
Run the pipeline using the following commands:
1. **Ingest Documents:**
   ```bash
   python main.py step01_ingest --input_filename 'all'
   ```
2. **Generate Embeddings:**
   ```bash
   python main.py step02_generate_embeddings --input_filename 'all'
   ```
3. **Store Vectors:**
   ```bash
   python main.py step03_store_vectors --input_filename 'all'
   ```
4. **Retrieve Relevant Contexts:**
   ```bash
   python main.py step04_retrieve_chunks --query_args 'Compare constitutions of US and Norway'
   ```
5. **Generate Response:**
   ```bash
   python main.py step05_generate_response --query_args 'Compare constitutions of US and Norway' --use_rag
   ```

### **Testing Framework**
Run unit and integration tests:
```bash
python -m unittest discover tests/
```

---

## **Configuration**
The `config.json` file contains key parameters for the pipeline:
```json
{
    "log_level": "debug",
    "raw_input_directory": "data/raw_input",
    "cleaned_text_directory": "data/cleaned_text",
    "embeddings_directory": "data/embeddings",
    "vectordb_directory": "data/vectordb",
    "collection_name": "collections",
    "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
    "llm_api_url": "http://localhost:1234/v1/completions",
    "llm_model_name": "llama-3.2-1b-instruct"
}
```

---

## **Testing and Evaluation**
### **Unit Tests**
Unit tests validate individual components such as `DocumentIngestor`, `EmbeddingPreparer`, `ChromaDBRetriever`, etc.

### **Integration Tests**
Integration tests ensure seamless functionality across all pipeline components.

### **Performance Benchmarks**
Performance tests measure response latency and retrieval accuracy.

---

## **Key Design Decisions**
1. Selection of constitutions as the domain due to their textual nature and manageable scope.
2. Use of MPNet (`sentence-transformers/all-mpnet-base-v2`) for embedding generation to ensure high semantic accuracy.
3. Implementation of statistical filtering (standard deviation) for precise document retrieval.
4. Iterative refinement of query templates to improve response quality.

---

## **Future Improvements**
- Expand document coverage to include all countries and aliases.
- Fine-tune embedding models for constitutional analysis.
- Incorporate multiple documents per country for deeper insights.
- Explore paid tools like LangChain for advanced query generation.
- Optimize response generation by exploring alternative LLMs with lower latency.

---
