from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv
import streamlit as st
import pickle
import os

#Sidebar content
with st.sidebar:
    st.title("LLM Bluetooth")
    st.markdown(''' 
    ## Bluetooth
    This app is for the LLM Bluetooth project.
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    
    ''')
    add_vertical_space(2)
    st.write("Follow me on [Linkedin](https://www.linkedin.com/in/abhigaelcarranza/)")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

def clear_memory(memory):
    memory.clear()
    st.session_state.clear()

@st.cache_resource()
def qa_chain(_vectordb):
    # Define retriever
    retriever = _vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, streaming=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )
    return qa, memory

def main():
    load_dotenv()
    st.header("Chat with yor PDF about Bluetooth")
    
    #upload a PDF file
    # with open(f"bluetooth-act.pkl", "rb") as f:
    #     vector_store = pickle.load(f)
    with open(f"test.pkl", "rb") as f:
        vector_store = pickle.load(f)
    
    #Chatbot with streamlit and Langchain
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    qa,memory=qa_chain(vector_store)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What question do you have about Bluetooth?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            full_response = qa.run(prompt,callbacks=[st_callback])
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.button("Clear chat", on_click=clear_memory, args=(memory,)) 

def main_1():
    load_dotenv()
    st.header("Chat with yor PDF about Bluetooth")
    
    with open(f"test.pkl", "rb") as f:
        vector_store = pickle.load(f)

    qa,memory=qa_chain(vector_store)
    
    if "messages" not in st.session_state or st.sidebar.button("Clear message history", on_click=clear_memory, args=(memory,)):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What question do you have about Bluetooth?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # retrieval_handler = PrintRetrievalHandler(st.container())
            # stream_handler=StreamHandler(st.empty())
            stream_handler=StreamlitCallbackHandler(st.container())
            # response = qa.run(prompt, callbacks=[retrieval_handler, stream_handler])
            response = qa.run(prompt, callbacks=[stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})
    
if __name__ == "__main__":
    main_1()