from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from supabase.client import create_client, Client
import streamlit as st
import os

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings=OpenAIEmbeddings()

vector_store=SupabaseVectorStore(
    supabase, 
    embeddings,
    table_name="documents",
    query_name="match_documents"
    )

st.write(vector_store.as_retriever(search_type="similarity",search_kwargs={'k': 4}))
