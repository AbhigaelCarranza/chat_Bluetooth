import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client
from utils import load_documents, get_faiss_vectorStore , get_supabase_VectorStore

client = Client()

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

@st.cache_resource(ttl="1h")
def configure_retriever():
    documents=load_documents()
    embeddings = OpenAIEmbeddings()
    # vector_store=get_faiss_vectorStore(documents,embeddings)
    vector_store=get_supabase_VectorStore(embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# tool= create_retriever_tool(
#     configure_retriever(),
#     "search_bluetooth_docs",
#     "Searches and returns documents regarding Bluetooth. so if you are ever asked about Bluetooth you should use this tool."
#     )

tool_1= create_retriever_tool(
    configure_retriever(),
    "search_Leyes-Federales_docs",
    "Searches and returns documents regarding Leyes Federales about Mexico. so if you are ever asked about Leyes Mexicanas you should use this tool."
    )

tools=[tool_1]
llm=ChatOpenAI(temperature=0,streaming=True,model_name="gpt-3.5-turbo")
# message=SystemMessage(
#     content=(
#         "You are a helpful chatbot who is tasked with answering questions about Bluetooth. "
#         "Unless otherwise explicitly stated, it is probably fair to assume that questions are about Bluetooth. "
#         "If there is any ambiguity, you probably assume they are about that."
#     )
# )

message=SystemMessage(
    content=(
        """Eres un abogado de IA que proporciona asesoramiento legal instantáneo y herramientas de creación de documentos para asuntos personales y comerciales.
Puedes responder preguntas legales comunes, redactar contratos y acuerdos, revisar y comparar documentos y realizar investigaciones legales.
Puedes asistir a tus clientes con la gestión de casos, el análisis de documentos, fundamentar casos con la ley, la revisión de contratos, la diligencia debida, la facturación y más. 
Puedes ayudar a tus clientes con derechos del consumidor, multas de aparcamiento, compensación por vuelos, comisiones bancarias, disputas con el propietario y más. 
Utilizas el Inteligencia Artificial para comprender y generar textos legales sobre las leyes Mexicanas. 
Debes presentarte con “Soy Jaime, tu Abogado personalizado”, pero solo al principio de una conversación. 
"""
    )
)

prompt=OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent=OpenAIFunctionsAgent(llm=llm,tools=tools,prompt=prompt)
agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

memory=AgentTokenBufferMemory(llm=llm)
starter_message="Ask me a question about Leyes Mexicanas."

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"]=[AIMessage(content=starter_message)]

def send_feedback(run_id,score):
    client.create_feedback(run_id,"user_score",score=score)
    
for msg in st.session_state.messages:
    if isinstance(msg,AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg,HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)

if prompt:=st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=agent_executor(
            {"input":prompt, "history":st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input":prompt},response)
        st.session_state['messages']=memory.buffer
        run_id=response["__run"].run_id
        
        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
