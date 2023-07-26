from llama_index import (StorageContext,load_index_from_storage)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
import streamlit as st
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

vector_store =FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3
    )
# configure response synthesizer


# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever
)

st.header("Chat with yor PDF about Bluetooth")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What question do you have about Bluetooth?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # retrieval_handler = PrintRetrievalHandler(st.container())
        # stream_handler=StreamHandler(st.empty())
        # response = qa.run(prompt, callbacks=[retrieval_handler, stream_handler])
        response=query_engine.query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)