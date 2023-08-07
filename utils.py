from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import pickle
import os
import streamlit as st

def load_documents():
    load_dotenv()
    file="/mnt/c/Users/amendez/github/chat_Bluetooth/Files/bluetooth-act.pdf"
    pdf=PyPDFLoader(file)
    loaders=pdf.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,length_function=len)
    return text_splitter.split_documents(loaders)

def load_documents_from_web():
    docs = []
    uploaded_files = st.file_uploader("Upload PDF: ",type=['pdf'])
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load()) 
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,length_function=len)
    return text_splitter.split_documents(docs)

def get_faiss_vectorStore(chunks,embeddings):
    if os.path.exists("test.pkl"):
        with open("test.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store=FAISS.from_documents(chunks, embeddings)
        with open("test.pkl", "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store