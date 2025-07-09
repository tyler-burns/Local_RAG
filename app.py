import streamlit as st
from rag_pipeline import build_knowledge_base, answer_question
import time


st.title("🗂️ Offline RAG Folder Q&A")

folder = st.text_input("📁 Folder path containing your documents:")

if st.button("🔄 Build Knowledge Base"):
    start = time.time()
    with st.spinner("Processing documents..."):
        st.session_state['db'] = build_knowledge_base(folder)
    end = time.time()
    st.success(f"Knowledge base built in {end - start:.4f} seconds!")

question = st.text_input("❓ Ask a question:")

if st.button("🧠 Answer") and 'db' in st.session_state:
    start = time.time()
    with st.spinner("Thinking..."):
        response = answer_question(question, st.session_state['db'])
    st.write("**Answer:**", response)
    end = time.time()
    st.write(f"Answered in {end - start:.4f} seconds.")
