import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
)
from datasets import Dataset

# Add the SQLite and ChromaDB imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

def load_environment_variables():
    load_dotenv()
    return os.getenv('OPENAI_KEY')

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def create_vector_store(documents, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma.from_documents(documents, embeddings)

def create_qa_chain(db, api_key):
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def perform_qa(qa_chain, query):
    return qa_chain({"query": query})

def evaluate_with_ragas(qa_chain, questions):
    def get_contexts(question):
        docs = qa_chain.retriever.get_relevant_documents(question)
        return [doc.page_content for doc in docs]

    def get_answer(question):
        result = qa_chain({"query": question})
        return result['result']

    dataset = Dataset.from_dict({"question": questions})
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_relevancy,
            answer_relevancy,
            faithfulness,
            context_recall,
        ],
        retriever=get_contexts,
        llm=get_answer,
    )
    
    return result

def main():
    # Load environment variables
    openai_api_key = load_environment_variables()

    # Load PDF
    pdf_path = "data/Robinson-Advisory.pdf"
    docs = load_pdf(pdf_path)

    # Split documents
    documents = split_documents(docs)

    # Create vector store
    db = create_vector_store(documents, openai_api_key)

    # Create QA chain
    qa_chain = create_qa_chain(db, openai_api_key)

    # Perform QA
    query = "Who are the parties to the Agreement and what are their defined names?"
    result = perform_qa(qa_chain, query)
    print("Question:", query)
    print("Answer:", result['result'])

    # Evaluate with RAGAS
    evaluation_questions = [
        "Who are the parties to the Agreement and what are their defined names?",
        "What is the purpose of this Agreement?",
        "What are the key terms of the Agreement?",
    ]
    evaluation_result = evaluate_with_ragas(qa_chain, evaluation_questions)
    print("\nRAGAS Evaluation Results:")
    print(evaluation_result)

if __name__ == "__main__":
    main()