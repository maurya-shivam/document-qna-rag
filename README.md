# UPS Document Q&A with Gemini and Qdrant

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using Python, LangChain, Gemini (as the LLM), and Qdrant (as the vector database). It includes a Streamlit UI for interactive questioning.

## Features

- PDF document ingestion and text splitting.
- Open-source embedding model (Sentence Transformers) for creating embeddings.
- Qdrant for vector storage and retrieval.
- Gemini LLM for answer generation.
- Streamlit UI for asking questions and displaying answers with sources.

## Setup Instructions

### 1. Prerequisites

- **Python 3.8+**
- **Docker**: Required to run Qdrant. Ensure Docker is installed and running on your system.
- **Google API Key**: For accessing the Gemini LLM.

### 2. Project Setup

1.  **Clone the repository (if applicable) or create the project files.**

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python3 -m venv venv
    # Activate the virtual environment
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    # Install required packages
    pip install -r requirements.txt
    ```
    *(Note: If you prefer `uv`, you can use `uv pip install -r requirements.txt`)*

### 3. Install Ollama and Download Embedding Model

1.  **Install Ollama:**
    Follow the instructions on the [Ollama website](https://ollama.com/download) to install Ollama for your operating system.

2.  **Download the `nomic-embed-text` model:**
    Once Ollama is installed, open your terminal and run:
    ```bash
    ollama pull nomic-embed-text
    ```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your Google API key and Qdrant details:
```
GOOGLE_API_KEY=YOUR_API_KEY
QDRANT_HOST=http://127.0.0.1:6333
QDRANT_API_KEY=77028a63-1b55-4b6f-8a78-bf5b14063633
```
Replace `YOUR_API_KEY` with your actual Google API key. The Qdrant host and API key are provided by the user.

### 5. Run Qdrant Vector Database

Ensure Docker is running. Qdrant should be accessible at `http://127.0.0.1:6333` with the provided API key.

### 6. Ingest Document (Optional, via `ingest.py`)

You can ingest a PDF document using the `ingest.py` script. Place your PDF document in the `./data/` directory. The default filename expected by `ingest.py` is `sample.pdf`. If your file has a different name, update the `PDF_PATH` variable in `ingest.py` accordingly.

To ingest the document and populate the Qdrant vector store:
```bash
python3 ingest.py
```
Alternatively, you can upload and ingest documents directly through the Streamlit UI.

### 7. Run the Streamlit Application

Launch the Streamlit UI:
```bash
streamlit run app.py
```

This will open the application in your web browser. You can now upload PDF documents directly through the UI for ingestion into Qdrant, and then ask questions about the ingested documents.