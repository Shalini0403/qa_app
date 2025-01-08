# retriever.py
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def convert_query_to_vector(query: str) -> np.ndarray:
    """Converts the query string to a vector using a pre-trained model."""
    query_vector = model.encode([query])[0]  # Get the embedding of the query
    return query_vector

def create_faiss_index(data_folder: str, index_folder: str):
    documents = []
    file_names = []

    # Read all documents
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, "r") as f:
            documents.append(f.read())
            file_names.append(file_name)

    # Convert documents to vectors using the same model as queries
    document_vectors = model.encode(documents)
    
    # Create and save FAISS index
    dimension = document_vectors.shape[1]  # Get dimension from the encoded vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(document_vectors, dtype="float32"))
    
    # Save index with consistent filename
    faiss.write_index(index, os.path.join(index_folder, "faiss_index.index"))

    # Save metadata (document file names)
    with open(os.path.join(index_folder, "metadata.txt"), "w") as meta_file:
        meta_file.write("\n".join(file_names))

def get_relevant_documents(query: str, index_folder: str) -> List[str]:
    """Fetch relevant documents based on the query using the FAISS index."""
    # Convert query to vector
    query_vector = convert_query_to_vector(query)
    
    # Load the FAISS index
    index_file_path = os.path.join(index_folder, "faiss_index.index")
    if not os.path.isfile(index_file_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_file_path}")
    
    # Load metadata
    metadata_path = os.path.join(index_folder, "metadata.txt")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    with open(metadata_path, "r") as meta_file:
        file_names = meta_file.read().splitlines()
    
    # Load the index
    index = faiss.read_index(index_file_path)
    
    # Search the index
    query_vector = query_vector.reshape(1, -1)  # Reshape to 2D array
    distances, indices = index.search(query_vector.astype('float32'), k=3)
    
    # Retrieve content of relevant documents
    documents = []
    for idx in indices[0]:
        if idx < len(file_names):
            doc_path = os.path.join("data/documents", file_names[idx])
            with open(doc_path, "r") as f:
                documents.append(f.read())
    
    return documents