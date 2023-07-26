from llama_index import (VectorStoreIndex,StorageContext,load_index_from_storage,ServiceContext)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings import OpenAIEmbedding
from llama_hub.file.pdf.base import PDFReader
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
import faiss
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


file="/mnt/c/Users/amendez/github/chat_Bluetooth/Files/bluetooth-act.pdf"
load_dotenv()
faiss_index = faiss.IndexFlatIP(1536)
#Load Documents
documents=PDFReader().load_data(file=Path(file))
# embeddings = OpenAIEmbedding()
st.write(documents)

# vector_store = FaissVectorStore(faiss_index=faiss_index)
# service_context = ServiceContext.from_defaults(embed_model=embeddings)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=documents)
st.write(index)
#Save index to disk
# index.storage_context.persist()
#Load index from disk
# vector_store =FaissVectorStore.from_persist_dir("./storage")
# storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
# index = load_index_from_storage(storage_context=storage_context)       
# st.write(index) 
# return index
