import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

def load_documents(folder_path):
    supported_docs = []
    skipped_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(full_path)
                    supported_docs.extend(loader.load())
                elif ext == ".txt":
                    loader = TextLoader(full_path, autodetect_encoding=True)
                    supported_docs.extend(loader.load())
                else:
                    skipped_files.append(full_path)
            except Exception as e:
                print(f"[ERROR] Failed to load {full_path}: {e}")
                skipped_files.append(full_path)

    if skipped_files:
        print("\n⚠️ Skipped the following unsupported or failed files:")
        for path in skipped_files:
            print(f" - {path}")

    return supported_docs

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
