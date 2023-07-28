from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index import (StorageContext,load_index_from_storage,LLMPredictor,ServiceContext)
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
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
import streamlit as st
from dotenv import load_dotenv
import os
import openai
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def clear_memory(memory):
    memory.clear()
    st.session_state.clear()

def streamlit_chatbot(agent_executor,memory):
    st.header("Chat with yor PDF about Bluetooth")

    tools = [
        Tool(
            name="Hardware_QA_System",
            func=lambda q:str(index.as_query_engine().query(q)),
            description="Always use this tool, useful for when you want to answer queries about hardware embedded in Bluetooth devices",
            return_direct=True
        )
    ]

    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True, input_key="input", output_key="output")
    llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo",streaming=True)
    # memory=ConversationSummaryBufferMemory(llm=llm,return_messages=True)
    agent_executor=initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,verbose=True,memory=memory)
    # response=agent_executor.run(input="Can you use SCO or eSCO in LE?")
    # st.write(agent_executor)
    return agent_executor, memory

def streamlit_chatbot():
    tools = [
        Tool(
            name="Hardware_QA_System",
            func=lambda q:str(index.as_query_engine().query(q)),
            description="Always use this tool, useful for when you want to answer queries about hardware embedded in Bluetooth devices",
            return_direct=True
        )
    ]

    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True, input_key="input", output_key="output")
    llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo",streaming=True)
    # memory=ConversationSummaryBufferMemory(llm=llm,return_messages=True)
    agent_executor=initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,verbose=True,memory=memory) 
    
    st.header("Chat with yor PDF about Bluetooth")
    with st.sidebar:
        st.write(agent_executor.memory)
    if "messages" not in st.session_state or st.sidebar.button("Clear message history", on_click=clear_memory, args=(memory,)):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input(placeholder="What question do you have about Bluetooth?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler=StreamHandler(st.empty())
            # stream_handler=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            # try:
            #     response = agent_executor.run(input=prompt, callbacks=[stream_handler])
            # except ValueError as ve:
            #     response = str(ve).strip('Could not parse LLM output: ')
            
            # response = agent_executor.run(prompt)
            response = agent_executor.run(input=prompt, callbacks=[stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})

def load_index():
    vector_store =FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
    index = load_index_from_storage(storage_context=storage_context)
    return index

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    index=load_index()
    
    tools = [
        Tool(
            name="Vector Index",
            func=lambda q:str(index.as_query_engine().query(q)),
            description="useful for when you want to answer queries about Bluetooth context",
            return_direct=True
        )
    ]

    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo",streaming=True)
    agent_executor=initialize_agent(tools=tools,llm=llm,memory=memory,agent="conversational-react-description",verbose=False)

    streamlit_chatbot(agent_executor,memory)

def main_test():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    index=load_index()

    tool_config = IndexToolConfig(
        query_engine=index.as_query_engine(similarity_top_k=3),
        name="Vector Index",
        description="useful for when you want to answer queries about Bluetooth context",
        tool_kwargs={"return_direct": True}
    )
    tool=LlamaIndexTool.from_tool_config(tool_config)

    # st.write(tool)
    toolkit=LlamaToolkit(index_configs=[tool_config])
    
    llm =ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain=create_llama_chat_agent(toolkit,llm,memory,verbose=True)
    # agent_chain=create_llama_agent(toolkit,llm,verbose=True)
    st.write(agent_chain)
    # streamlit_chatbot(agent_chain,memory)
    
if __name__ == "__main__":
    main()
    # main_test()