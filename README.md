# Contract Advisor RAG: Towards Building A High-Precision Legal Expert LLM APP

# Business Objective
Lizzy AI is an early-stage Israeli startup developing the next-generation contract AI. We leverage Hybrid LLM technology (edge, private cloud, and LLM services) to build the first, fully autonomous, artificial contract lawyer. Our goal is to develop a powerful contract assistant, with the ultimate goal of creating a fully autonomous contract bot capable of drafting, reviewing, and negotiating contracts independently, end-to-end, without human assistance.

# Retrieval Augmented Generation (RAG)
RAG is a technique used in natural language processing (NLP) that combines language modeling with information retrieval to improve the performance of text generation tasks.

### Why RAG?

* Preventing hallucinations: Large language models (LLMs) are powerful, but can sometimes generate outputs that appear correct but are actually incorrect. RAG pipelines can help address this by providing LLMs with factual, retrieved inputs, which can lead to more accurate outputs.
* Working with custom data: Many base LLMs are trained on vast amounts of internet data, which gives them strong language modeling capabilities. However, this broad training can mean they lack specific domain knowledge. RAG systems can supplement LLMs with more targeted data, such as medical information or company documentation, allowing the LLMs to generate outputs that are customized for particular use cases.

# RAG Workflow

### Data Importing and Preparation
* Import and prepare raw data that will form the knowledge base.
* Clean and preprocess the data as necessary, including removing formatting, handling special characters, and tokenizing the text.

### Text Chunking
* Split the prepared data into smaller, manageable chunks.
* Divide the text into segments that are large enough to capture coherent information, but small enough to be processed efficiently.

### Vectorization of Chunks
* Convert the text chunks into vector representations using a text embedding model, such as BERT or Sentence Transformers.

### Vector Storage
* Store the vectorized chunks in a vector database or index, such as FAISS, Pinecone, or Weaviate.

### Query Processing
* Process the user's query or question and prepare it for search.
* Tokenize the query, handle special characters, and remove any unnecessary information.

### Query Vectorization
* Vectorize the query using the same text embedding model used to vectorize the text chunks.

### Vector Similarity Search
* Use the query vector to search for the most relevant chunks in the vector database.

### Context Preparation
* Compile the retrieved chunks into a context that can be used by the language model.

### Prompt Construction
* Use the context to construct a prompt for the language model, including the original query, retrieved context, and any additional instructions or formatting requirements.

### Answer Generation
* Feed the constructed prompt into a generative language model, such as GPT-3 or T5, to generate an answer.

### Post-processing
* Refine the generated answer if necessary, including formatting, adding citations, or removing any unnecessary information.

### Response Delivery
* Deliver the final answer to the user, either through a user interface or API.

### Feedback and Improvement
* Collect user feedback or performance metrics to fine-tune the retrieval process, chunk size, or other parameters, allowing the system to continuously improve its performance and provide more accurate and relevant answers to user queries.