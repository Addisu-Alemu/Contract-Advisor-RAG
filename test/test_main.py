import unittest
import os
from dotenv import load_dotenv
from unittest.mock import patch
from main import load_environment_variables, load_pdf, split_documents, create_vector_store, create_qa_chain, perform_qa

class TestMain(unittest.TestCase):
    @patch('os.getenv', return_value='your_openai_key')
    def test_load_environment_variables(self, mock_getenv):
        self.assertEqual(load_environment_variables(), 'your_openai_key')

    @patch('PyPDFLoader.load', return_value=['some_pdf_content'])
    def test_load_pdf(self, mock_load):
        pdf_path = 'data/Robinson-Advisory.pdf'
        docs = load_pdf(pdf_path)
        self.assertEqual(docs, ['some_pdf_content'])

    @patch('RecursiveCharacterTextSplitter.split_documents', return_value=['some_split_documents'])
    def test_split_documents(self, mock_split):
        docs = ['some_pdfs']
        documents = split_documents(docs)
        self.assertEqual(documents, ['some_split_documents'])

    @patch('OpenAIEmbeddings.get_vector', return_value=['some_vector'])
    @patch('Chroma.from_documents', return_value=['some_chroma'])
    def test_create_vector_store(self, mock_chroma, mock_get_vector):
        documents = ['some_documents']
        api_key = 'your_openai_key'
        db = create_vector_store(documents, api_key)
        self.assertEqual(db, ['some_chroma'])

    @patch('ChatOpenAI', return_value=['some_llm'])
    @patch('RetrievalQA.from_chain_type', return_value=['some_qa_chain'])
    def test_create_qa_chain(self, mock_qa_chain, mock_llm):
        db = ['some_db']
        api_key = 'your_openai_key'
        qa_chain = create_qa_chain(db, api_key)
        self.assertEqual(qa_chain, ['some_qa_chain'])

    @patch('RetrievalQA', return_value=['some_result'])
    def test_perform_qa(self, mock_result):
        qa_chain = ['some_qa_chain']
        query = 'some_query'
        result = perform_qa(qa_chain, query)
        self.assertEqual(result, ['some_result'])

if __name__ == '__main__':
    unittest.main()