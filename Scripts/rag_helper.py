# scripts/langchain_processor.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

class LangChainProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def load_text_data(self, file_path):
        loader = TextLoader(file_path)
        return loader.load()

    def load_pdf_data(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

    def create_chroma_db(self, documents):
        db = Chroma.from_documents(documents, self.embeddings, client_settings={})
        return db

    def perform_similarity_search(self, db, query):
        return db.similarity_search(query)