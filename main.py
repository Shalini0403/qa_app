#streamlit ui
import os

import streamlit as st
from utils import extract_text_from_pdf

from generator import generate_text
from retriever import create_faiss_index, get_relevant_documents

DATA_FOLDER = "data/documents"
FAISS_INDEX_FOLDER = "data/faiss_index"

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)

st.title("Simple RAG Application with PDF Upload")
st.write("Retrieve and generate answers from uploaded PDF documents using FAISS and GPT.")

# Upload documents
st.header("Upload Documents")
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Extract text from PDF and save it as a text file
        text = extract_text_from_pdf(uploaded_file)
        with open(os.path.join(DATA_FOLDER, f"{uploaded_file.name}.txt"), "w") as f:
            f.write(text)
    st.success("Documents uploaded and text extracted successfully!")
    st.write("Building FAISS index...")
    create_faiss_index(DATA_FOLDER, FAISS_INDEX_FOLDER)
    st.success("FAISS index created!")

# Query input
st.header("Ask a Question")
query = st.text_input("Enter your query:")
if query:
    st.write("Retrieving relevant documents...")
    relevant_docs = get_relevant_documents(query, FAISS_INDEX_FOLDER)
    st.write("Relevant documents retrieved!")
    st.write("Generating answer...")
    answer = generate_text(query, relevant_docs)
    st.subheader("Answer:")
    st.write(answer)

