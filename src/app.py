import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

'''
Custom Imports
'''
from src.modules.ingest import ingest_document_to_qdrant

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ups_document_qna"


# ------- Cached Vector Store -------
@st.cache_resource
def get_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:137m-v1.5-fp16")
    client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )


# ------- Cached Gemini LLM -------
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )


def simple_rag(question, vector_store, llm):
    # Create retriever
    # retriever = vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 5, "score_threshold": 0.8}
    # )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.8}
    )
    # 1. Retrieve chunks
    docs = retriever._get_relevant_documents(query=question, run_manager=None)
    print(f"Fetched Document Count: {len(docs)}")
    # 2. Build prompt
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an helpful QnA chatbot, you are proficient in answering user queries from the give context.
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}

Answer:
Note: 
- If the question is not related to the context then tell user the information is not in the context and ask a short follow-up quesiton
- if user is not asking for any information then chat with user if needed with short responses. Be gentle and polite.
"""

    # 3. Call LLM
    response = llm.invoke(prompt)

    return response.content, docs



# ---------------------------------
# -------   STREAMLIT UI   --------
# ---------------------------------

def main():
    st.set_page_config(page_title="Simple RAG with Gemini & Qdrant")
    st.header("Document Q&A (Simple RAG) 💬")

    st.subheader("Upload Document for Context")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Ingesting document... This may take a moment."):
                ingest_document_to_qdrant(uploaded_file)
            st.session_state["document_ingested"] = True
    
    st.markdown("---")
    st.subheader("Ask a Question")

    question = st.text_input("Ask your question:", key="user_question")
    if question and st.session_state.get("document_ingested", False):
        vector_store = get_vector_store()
        llm = get_llm()

        with st.spinner("Thinking..."):
            answer, docs = simple_rag(question, vector_store, llm)

        st.subheader("Answer:")
        print(answer)
        st.write(answer)

        st.subheader("Relevant Sources:")
        for i, doc in enumerate(docs):
            st.write(f"### Source {i+1}")
            st.write(doc.page_content[:200] + "...")
            st.json(doc.metadata)
            st.markdown("---")
    elif question and not st.session_state.get("document_ingested", False):
        st.warning("Please upload and ingest a document first to ask questions.")


if __name__ == "__main__":
    main()
