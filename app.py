import streamlit as st
from rag_pipeline import build_knowledge_base, answer_question

st.title("🗂️ Offline RAG Folder Q&A")

folder = st.text_input("📁 Folder path containing your documents:")

if st.button("🔄 Build Knowledge Base"):
    with st.spinner("Processing documents..."):
        st.session_state['db'] = build_knowledge_base(folder)
    st.success("Knowledge base built!")

question = st.text_input("❓ Ask a question:")

if st.button("🧠 Answer") and 'db' in st.session_state:
    with st.spinner("Thinking..."):
        response = answer_question(question, st.session_state['db'])
    st.write("**Answer:**", response)
