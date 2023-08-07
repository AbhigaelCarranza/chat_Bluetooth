from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from utils import load_documents, get_faiss_vectorStore
import streamlit as st
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
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
    retriever = _vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, streaming=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True, chain_type="stuff"
    )
    return qa, memory

# @st.cache_data()
def streamlit_chatbot(_qa_chain, prompt):
    st.header("Chat with yor PDF about Bluetooth")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            # stream_handler=StreamHandler(st.empty())
            stream_handler=StreamlitCallbackHandler(st.container())
            response = _qa_chain.run(prompt, callbacks=[retrieval_handler, stream_handler])
            # response = _qa_chain.run(prompt, callbacks=[stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})
    
if __name__ == "__main__":
    chunks = load_documents()
    embeddings = OpenAIEmbeddings()
    vectoredb = get_faiss_vectorStore(chunks=chunks, embeddings=embeddings)
    qa,memory=qa_chain(vectoredb)
    
    prompt=st.chat_input(placeholder="What question do you have about Bluetooth?")
    streamlit_chatbot(qa,prompt)
    st.sidebar.button("Clear message history", on_click=clear_memory, args=(memory,))