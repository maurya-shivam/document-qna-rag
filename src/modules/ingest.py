import os

import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ups_document_qna"


def ingest_document_to_qdrant(uploaded_file):
    # ------------------------------
    # Save temporary file
    # ------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_file_path = tmp.name

    st.info(f"Processing {uploaded_file.name}...")

    # Load PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    st.write(f"Loaded {len(documents)} pages")

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    st.write(f"Split into {len(chunks)} chunks")

    # Embeddings model
    embeddings = OllamaEmbeddings(model="nomic-embed-text:137m-v1.5-fp16")

    # Compute 1 embedding to get vector size
    test_vector = embeddings.embed_query("test")
    vector_size = len(test_vector)

    # Qdrant client
    client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

    # ---------------------------------------
    # Recreate or create collection safely
    # ---------------------------------------
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass  # collection may not exist

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    # ---------------------------------------
    # Upload documents with vectorstore instance
    # ---------------------------------------
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)

    st.success(f"{uploaded_file.name} successfully ingested into Qdrant.")

    # Cleanup
    os.remove(temp_file_path)