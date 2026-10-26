import streamlit as st
#state=memeory of the state
import rag_logic

def init_state():
    if "messages" not in st.session_state:
        st.session_state['messages'] = []
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = {"chunks": [], "embeddings": [], "source": []}
    if "retrieved" not in st.session_state:
        st.session_state["retrieved"] = []
    if "last_processed_files" not in st.session_state:
        st.session_state["last_processed_files"] = None


def main():
    st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
    st.title("PDF RAG Chatbot")
    st.write("Upload PDF files and ask questions. The answer comes from the uploaded documents.")

    init_state()

    with st.sidebar:
        st.header("Settings")
        chunk_size = st.number_input("Chunk size (words)", min_value=100, max_value=1200, value=400, step=50)
        overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=300, value=100, step=10)
        top_k = st.slider("Top chunks to search", min_value=1, max_value=8, value=4)
        show_context = st.checkbox("Show retrieved context", value=True)
        st.info("Ensure GROQ_API_KEY (or groq_api) is set in your .env file.")

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        current_config = (
            tuple((f.name, f.size) for f in uploaded_files),
            chunk_size,
            overlap,
        )
        if st.session_state.get("last_processed_files") != current_config:
            try:
                with st.spinner("Processing PDFs and generating embeddings..."):
                    st.session_state["vector_store"] = rag_logic.build_vector_store(
                        uploaded_files, chunk_size, overlap
                    )
                    st.session_state["last_processed_files"] = current_config
                num_chunks = len(st.session_state["vector_store"]["chunks"])
                st.success(f"Processed {len(uploaded_files)} file(s) and created {num_chunks} chunks.")
            except Exception as e:
                st.error(f"Error reading PDFs: {e}")
        else:
            num_chunks = len(st.session_state["vector_store"]["chunks"])
            st.caption(f"Currently ready with {len(uploaded_files)} file(s) ({num_chunks} chunks).")
    else:
        st.session_state["last_processed_files"] = None
        st.session_state["vector_store"] = {"chunks": [], "embeddings": [], "source": []}
        st.warning("Please upload at least one PDF to ask questions.")

    for message in st.session_state['messages']:
        st.chat_message(message["role"]).write(message["content"])

    question = st.chat_input("Ask a question about your PDFs...")
    if question:
        st.session_state['messages'].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        if not st.session_state["vector_store"]["chunks"]:
            answer = "I need at least one PDF file to answer your question."
        else:
            with st.spinner("Finding relevant text and asking Groq AI..."):
                retrieved = rag_logic.find_similar_chanks(question, st.session_state["vector_store"], top_k=top_k)
                st.session_state["retrieved"] = retrieved

                if not retrieved or retrieved[0]["score"] < 0.15:
                    answer = (
                        "I could not find a good answer in the uploaded documents. "
                        "Try a different question or upload more relevant PDFs."
                    )
                else:
                    context = rag_logic.formet_context(retrieved)
                    prompt = rag_logic.build_prompt(context, question)
                    try:
                        answer = rag_logic.ask_groq(prompt)
                    except Exception as e:
                        answer = f"Groq API error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

    if show_context and st.session_state["retrieved"]:
        with st.expander("Retrieved context"):
            for item in st.session_state["retrieved"]:
                st.markdown(f"**Source:** {item['source']}  \n**Score:** {item['score']:.4f}")
                st.write(item["text"])

    if st.button("Clear chat history"):
        st.session_state["messages"] = []
        st.session_state["retrieved"] = []
        st.success("Chat history cleared.")
        st.rerun()


if __name__ == "__main__":
    main()


    
