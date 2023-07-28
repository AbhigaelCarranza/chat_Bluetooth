from llama_index import (VectorStoreIndex,StorageContext,ServiceContext,LLMPredictor)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_hub.file.pdf.base import PDFReader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
import openai
import faiss
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

file="/mnt/c/Users/amendez/github/chat_Bluetooth/Files/bluetooth-act.pdf"
faiss_index = faiss.IndexFlatIP(1536)
#Load Documents
documents=PDFReader().load_data(file=Path(file))
# embeddings = OpenAIEmbedding(model="text-embedding-ada-002")
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo"))
# st.write(documents)

vector_store = FaissVectorStore(faiss_index=faiss_index)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=documents,storage_context=storage_context,service_context=service_context)

#Save index to disk
index.storage_context.persist()