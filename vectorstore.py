from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os
import streamlit as st

#Crear Class para vectorstore
class VectorStore:  
    def get_vectorStore():
        load_dotenv()
        file="/mnt/c/Users/amendez/github/chat_Bluetooth/Files/bluetooth-act.pdf"
        pdf=PyPDFLoader(file)
        chunks=pdf.load_and_split()
        
        store_name=file[53:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store=FAISS.from_documents(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            # st.write("Vector store created")
    
    #Otra opcion de VectorStore
    def get_vector_store():
        # Read documents
        load_dotenv()
        file="/mnt/c/Users/amendez/github/chat_Bluetooth/Files/bluetooth-act.pdf"
        # docs = []
        # temp_dir = tempfile.TemporaryDirectory()
        # for file in uploaded_files:
        #     temp_filepath = os.path.join(temp_dir.name, file.name)
        #     with open(temp_filepath, "wb") as f:
        #         f.write(file.getvalue())
        #     loader = PyPDFLoader(temp_filepath)
        #     docs.extend(loader.load())
        pdf=PyPDFLoader(file)
        docs=pdf.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,length_function=len)
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = OpenAIEmbeddings()
        # vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
        vectordb=FAISS.from_documents(splits, embeddings)
        with open(f"test.pkl", "wb") as f:
            pickle.dump(vectordb, f)
        st.write("Vector store created")
    
#Ejectuar la clase
if __name__ == "__main__":
    VectorStore.get_vector_store()
    # VectorStore.get_vectorStore()

