from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index import (StorageContext,load_index_from_storage,LLMPredictor,ServiceContext)
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from llama_index.langchain_helpers.agents import (
    LlamaToolkit,
    create_llama_chat_agent,
    IndexToolConfig,
    LlamaIndexTool,
    create_llama_agent
)
from langchain.agents import Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
import os
import openai
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_index():
    vector_store =FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
    index = load_index_from_storage(storage_context=storage_context)
    return index



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
index=load_index()

tools = [
    Tool(
        name="Hardware_QA_System",
        func=lambda q:str(index.as_query_engine().query(q)),
        description="Always use this tool, useful for when you want to answer queries about hardware embedded in Bluetooth devices",
        return_direct=True
    )
]

memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key="input", output_key="output")
llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo", streaming=True)
# memory=ConversationSummaryBufferMemory(llm=llm,return_messages=True)
agent_executor=initialize_agent(tools,llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)


if __name__ == "__main__":
    agent_executor.run(input="My name is Abhigael, I am a software engineer.")
    print(agent_executor.memory)
    agent_executor.run(input="What is my name?.")
    print(agent_executor.memory)