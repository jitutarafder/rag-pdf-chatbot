# PDF RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) Chatbot built with **Streamlit**, **Sentence-Transformers**, and **Groq Cloud API**.

## Features
- **PDF Upload**: Upload one or multiple PDF documents.
- **Semantic Search**: Text is split into chunks and embedded using `all-MiniLM-L6-v2`.
- **Groq LLM**: Ultra-fast responses powered by Groq models.
- **Source Verification**: View retrieved context and similarity scores for full transparency.

## Setup & Running Locally

1. Clone this repository:
```bash
git clone <your-repo-url>
cd "RAG Chatbot"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API key:
Create a `.env` file based on `.env.example`:
```env
groq_api=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
