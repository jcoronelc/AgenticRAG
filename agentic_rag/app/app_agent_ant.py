from config import api_key, base_url, models_llm,  model_llm_embeddings, model_llm_responses, collection_name

import streamlit as st
from crewai import Crew, LLM, Process
from agents.router_agent import create_router_agent
from agents.retriever_agent import create_retriever_agent
from agents.grader_agent import create_grader_agent
from agents.hallucination_agent import create_hallucination_agent
from agents.answer_agent import create_answer_agent
from agents.evaluator_agent import create_evaluator_agent

from tasks.router_task import create_router_task
from tasks.retriever_task import create_retriever_task
from tasks.grader_task import create_grader_task
from tasks.hallucination_task import create_hallucination_task
from tasks.answer_task import create_answer_task
from tasks.evaluator_task import create_evaluator_task

from tools.router_tool import router_tool
from tools.rag_tool import rag_tool
from tools.web_search_tool import web_search_tool

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from openai import OpenAI

SESSION_ID = "1234"
B_INST, E_INST = "<s>[INST]", "[/INST]</s>"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#streamlit run app.py

client = OpenAI(api_key=api_key, base_url=base_url)
llm = LLM(model="lm_studio/" + model_llm_responses, base_url=base_url, api_key=api_key)

# llm = LLM(
#     model="llama3.2:1b",  # Nombre del modelo en Ollama
#     base_url="http://34.46.82.206:11434",  # IP de tu VM en Google Cloud
# )

# Crear agentes

retriever_agent = create_retriever_agent(llm, rag_tool)
# grader_agent = create_grader_agent(llm)
# hallucination_agent = create_hallucination_agent(llm)
evaluator_agent = create_evaluator_agent(llm)
router_agent = create_router_agent(llm, web_search_tool)
answer_agent = create_answer_agent(llm)

# Crear tareas
retriever_task = create_retriever_task(retriever_agent, rag_tool)
# grader_task = create_grader_task(grader_agent, retriever_task)
# hallucination_task = create_hallucination_task(hallucination_agent, grader_task)
evaluator_task = create_evaluator_task(evaluator_agent, retriever_task)
router_task = create_router_task(router_agent, evaluator_task, retriever_task)
answer_task = create_answer_task(answer_agent, router_task)

# Crew de Agentes
# rag_crew = Crew(
#     agents=[retriever_agent, evaluator_agent, decider_agent, answer_agent],
#     tasks=[retriever_task, evaluator_task, decider_task, answer_task],
#     process=Process.sequential, # or Process.hierarchical
#     verbose=True,
#     # memory=True, # memory capabilities
# )

rag_crew = Crew(
    agents=[retriever_agent, evaluator_agent, router_agent, answer_agent],
    tasks=[retriever_task, evaluator_task, router_task, answer_task],
    verbose=True,
    # memory=True, # memory capabilities
)



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """ Obtiene el historial de la sesión """
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def get_response(session_id, question, rag_crew):
    """ Obtiene la respuesta de la cadena completa basada en la consulta. """
    inputs = {"question": question}
    
    try:
        result = rag_crew.kickoff(inputs=inputs)
        response_text = result.raw 
        if response_text is not None:
            return response_text
        else:
            return "No se ha encontrado información"
    except KeyError as e:
        return f"Lo siento, ocurrió un error: {str(e)}"

st.set_page_config(page_title="Chatbot MultiAgente", page_icon="🦜")
st.title("Chatea con :blue[tus datos] 🤖")

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

selected_llm_model = st.sidebar.selectbox("Elige un modelo LLM:", list(models_llm.values()))
    
# Crear el historial de la sesión si no existe
if SESSION_ID not in st.session_state.chat_store:
    st.session_state.chat_store[SESSION_ID] = ChatMessageHistory()
    
if "disabled" not in st.session_state:
    st.session_state.disabled = False

# Mostrar el historial de la conversación
for message in st.session_state.chat_store[SESSION_ID].messages:
    MESSAGE_TYPE = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(MESSAGE_TYPE):
        st.write(message.content)

# Btn para limpiar chat
if st.sidebar.button("🗑 Limpiar chat"):
    st.session_state.chat_store[SESSION_ID] = ChatMessageHistory()
    st.rerun()

def disable_input():
    st.session_state.disabled = True

user_query = st.chat_input("Escribe tu mensaje ✍", disabled=st.session_state.disabled, on_submit=disable_input)

if user_query:
    with st.spinner("Generando respuesta, por favor espera..."):
        response = get_response(SESSION_ID, user_query, rag_crew)
            
        st.session_state.disabled = False
        st.session_state.chat_store[SESSION_ID].add_message(HumanMessage(content=user_query))
        st.session_state.chat_store[SESSION_ID].add_message(AIMessage(content=response))
        st.rerun()
