import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

def load_documents(folder_path):
    loader = DirectoryLoader(folder_path, loader_cls=UnstructuredFileLoader)
    return loader.load()

def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

def load_llm():
    return LlamaCpp(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
        n_ctx=2048,
        temperature=0.3,
        top_p=0.95,
        n_gpu_layers=0,  # adjust for your GPU, or use 0 for CPU only
    )

def answer_question(query, db):
    llm = load_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa.run(query)

def build_knowledge_base(folder_path):
    docs = load_documents(folder_path)
    return build_vectorstore(docs)
