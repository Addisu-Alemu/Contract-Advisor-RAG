import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

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

def main():
    # Load environment variables
    openai_api_key = load_environment_variables()

    # Load PDF files in the data folder
    pdf_files = [f for f in os.listdir('/home/addisu-alemu/Desktop/week_11/Contract-Advisor-RAG/data') if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join('/home/addisu-alemu/Desktop/week_11/Contract-Advisor-RAG/data', pdf_file)
        print(f"Processing {pdf_file}...")

        # Load PDF
        docs = load_pdf(pdf_path)
        print("PDF loaded successfully.")

        # Split documents
        documents = split_documents(docs)
        print(f"Documents split into {len(documents)} chunks.")

        # Create vector store
        db = create_vector_store(documents, openai_api_key)
        print("Vector store created.")

        # Create QA chain
        qa_chain = create_qa_chain(db, openai_api_key)
        print("QA chain created.")

    st.title("Lizzy AI")

    # Ask user for a query
    query = st.text_input("Please enter a query: ")

    if query:
        # Perform QA
        result = perform_qa(qa_chain, query)

        # Print the result
        st.write(f"\nQuestion: {query}")
        st.write(f"Answer: {result['result']}")

if __name__ == "__main__":
    main()