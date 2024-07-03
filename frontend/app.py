import streamlit as st
import pandas as pd

# Load the RAG system
openai_api_key = load_environment_variables()
qa_chain = create_qa_chain(create_vector_store(split_documents(load_pdf("data/Robinson-Advisory.pdf")), openai_api_key), openai_api_key)

# Define a function to perform the RAG query
def rag_query(query):
    result = perform_qa(qa_chain, query)
    return result['result']

# Create a Streamlit app
st.title("RAG Query App")
st.write("Enter a query to get an answer")

# Create a text input for the query
query_input = st.text_input("Query")

# Create a button to submit the query
submit_button = st.button("Submit")

# Create a output area for the result
output_area = st.empty()

# Define a callback function for the submit button
def callback():
    query = query_input.value
    result = rag_query(query)
    output_area.write(f"Answer: {result}")

# Add the callback function to the submit button
submit_button.on_click(callback)

# Run the app
if __name__ == "__main__":
    st.run()