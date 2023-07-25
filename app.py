from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler

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

def clear_memory(memory):
    memory.clear()
    st.session_state.clear()
    

def main():
    load_dotenv()
    st.header("Chat with yor PDF about Bluetooth")
    
    #upload a PDF file
    with open(f"bluetooth-act.pkl", "rb") as f:
        vector_store = pickle.load(f)
    
    #Chatbot with streamlit and Langchain
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    llm=ChatOpenAI(temperature=0,model_name=st.session_state["openai_model"],streaming=True)
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    qa=ConversationalRetrievalChain.from_llm(llm,vector_store.as_retriever(),memory=memory)

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
if __name__ == "__main__":
    main()