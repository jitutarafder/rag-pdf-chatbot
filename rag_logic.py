import io  #input output
import os  # os - file read
import numpy as np
import re   #regular expression
import requests
from sentence_transformers import SentenceTransformer #embadding
from PyPDF2 import PdfReader # pdf read korar jnno


def load_env_file():
    env_path=".env"
    if not os.path.exists(env_path):
        #print('Env file nai')
        return
    with open(env_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            if '=' not in line:
                continue
            key,value=line.split('=',1)
            key=key.strip()
            value=value.strip().strip('"').strip("'")
            if key and os.environ.get(key) is None:
                os.environ[key]=value
load_env_file()

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def read_pdf(file_bytes):
    """Turn pdf byte into plain text"""
    pdf_stream = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned

def split_text(text, chunk_size=400, overlap=100):
    """Split text into smaller overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks

def embed_texts(texts):
    """Create a normalized embedding for a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    for i in range(len(embeddings)):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] = embeddings[i] / norm

    return embeddings

def build_vector_store(uploaded_files, chunk_size=400, overlap=100):
    chunks = []
    source = []
    for uploaded_file in uploaded_files:
        text = read_pdf(uploaded_file.getvalue())
        if not text:
            continue
        file_name = uploaded_file.name
        file_chunk = split_text(text, chunk_size, overlap)
        for index, chunk in enumerate(file_chunk):
            chunks.append(chunk)
            source.append(f"{file_name} chunk {index}")
    if not chunks:
        return {"chunks": [], "embeddings": [], "source": []}
    embedding = embed_texts(chunks)
    return {"chunks": chunks, "embeddings": embedding, "source": source}

def find_similar_chanks(question, store, top_k=4):
    """Return the top matching chunks for a question."""
    if not store or not store.get("chunks"):
        return []
    sources = store.get("source") or store.get("sources") or []
    question_embedding = embed_texts([question])[0]
    scores = store["embeddings"] @ question_embedding
    actual_top_k = min(top_k, len(store["chunks"]))
    top_index = np.argsort(-scores)[:actual_top_k]
    results = []
    for index in top_index:
        src = sources[index] if index < len(sources) else f"chunk {index}"
        results.append(
            {
                "text": store["chunks"][index],
                "source": src,
                "score": float(scores[index]),
            }
        )
    return results

find_similar_chunks = find_similar_chanks

def formet_context(retrieved):
    if not retrieved:
        return ""
    lines = []
    for item in retrieved:
        lines.append(f"source : {item['source']} \n{item['text']}")
    return "\n\n---\n\n".join(lines)

format_context = formet_context

def build_prompt(context, question):
    prompt = (
        "You are an AI assistant that answers questions using ONLY the provided document context. "
        "Do not invent facts. If the answer is not in the context, say that it is not available.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer using only the context above."
    )
    return prompt

def get_secret(key_name, default=None):
    val = os.environ.get(key_name)
    if val:
        return val
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return default

def ask_groq(prompt):
    api_key = (
        get_secret("GROQ_API_KEY")
        or get_secret("groq_api")
        or get_secret("GEMINI_API_KEY")
        or get_secret("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY or groq_api in your .env file or Streamlit Secrets.")
    raw_base_url = get_secret("GROQ_BASE_URL", "https://api.groq.com")
    base_url = raw_base_url.rstrip("/")
    model_name = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")
    
    header = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Try Groq responses endpoint
    endpoint = f"{base_url}/responses" if base_url.endswith("/openai/v1") else f"{base_url}/openai/v1/responses"
    payload = {
        "model": model_name,
        "input": prompt
    }

    try:
        response = requests.post(endpoint, headers=header, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                if 'output_text' in data:
                    return data['output_text']
                if 'output' in data and isinstance(data['output'], list):
                    output_part = []
                    for item in data['output']:
                        if isinstance(item, dict) and "content" in item:
                            content = item["content"]
                            if isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "output_text":
                                        output_part.append(part.get("text", ""))
                                    elif isinstance(part, str):
                                        output_part.append(part)
                            elif isinstance(content, str):
                                output_part.append(content)
                    if output_part:
                        return "".join(output_part).strip()
            return str(data).strip()
    except Exception:
        pass

    # 2. Fallback to standard OpenAI-compatible chat completions
    chat_endpoint = f"{base_url}/chat/completions" if base_url.endswith("/openai/v1") else f"{base_url}/openai/v1/chat/completions"
    chat_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }
    chat_resp = requests.post(chat_endpoint, headers=header, json=chat_payload, timeout=30)
    chat_resp.raise_for_status()
    chat_data = chat_resp.json()
    return chat_data["choices"][0]["message"]["content"].strip()

ask_aitools = ask_groq

                            

                







