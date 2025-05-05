from config import api_key, base_url, models_llm, model_llm_embeddings, model_llm_responses, collection_name, collection_name_active, retrieval_method
from agentic_rag_crew.src.agentic_rag_crew.crew import AgenticRagCrew

from crewai import Crew, LLM, Process
from openai import OpenAI
from mem0 import MemoryClient

from embedding.chroma_utils import query_rephrasing_with_crewai

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from datetime import datetime

import streamlit as st
import uuid  
import json
import os


os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
memory_storage_path = r"./data/output/chat/crewai_storage"
    
if not os.path.exists(memory_storage_path):
      os.makedirs(memory_storage_path)

os.environ["OPENAI_API_BASE"] = base_url
os.environ["OPENAI_API_KEY"] = api_key
os.environ["MEM0_API_KEY"] = "m0-bnTCjYzJePVtZDrRmibHDe2zvkiByzrMSTde30OM"      
      
SESSION_ID = "1234"
B_INST, E_INST = "<s>[INST]", "[/INST]</s>"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

client = OpenAI(api_key=api_key, base_url=base_url)
client_memo = MemoryClient()

llm = LLM(model="lm_studio/" + model_llm_responses, base_url=base_url, api_key=api_key)
# embedding_model = LLM(model="lm_studio/" + "All-MiniLM-L6-v2-Embedding-GGUF", base_url=base_url, api_key=api_key)


def save_history():
    chat_data = {
        chat_id: {
        "title": st.session_state.history[chat_id]["title"],
        "messages": [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                } for msg in st.session_state.chat_store[chat_id].messages
                ],
        "created_at": datetime.now().isoformat()
        } for chat_id in st.session_state.history
    }
   
    with open("./data/output/chat/agent_chat_history.json", "w") as f:
        json.dump(chat_data, f, indent=4)
        
def load_history():
    try:
        if os.path.getsize("./data/output/chat/agent_chat_history.json") > 0:
            with open("./data/output/chat/agent_chat_history.json", "r") as f:
                chat_data = json.load(f)
                for chat_id, chat_info in chat_data.items():
                    st.session_state.history[chat_id] = {"title": chat_info["title"]}
                    chat_history = ChatMessageHistory()
                    
                    for msg in chat_info["messages"]:
                        role = msg["role"]
                        content = msg["content"]
                        
                        # Según el rol, creamos el mensaje correspondiente
                        if role == "user":
                            chat_history.add_message(HumanMessage(content=content))
                        elif role == "assistant":
                            chat_history.add_message(AIMessage(content=content))
                        else:
                            print(f"Formato desconocido en mensaje: {msg}")    
                        
                    
                    st.session_state.chat_store[chat_id] = chat_history
                return chat_data
        else:
            return {}
    except FileNotFoundError:
        return {}

def select_chat():
    chat_ids_sorted = sorted(
        st.session_state.history.keys(),
        key=lambda chat_id: st.session_state.history[chat_id]["created_at"],
        reverse=True
    )
    # chat_ids = list(st.session_state.history.keys())
    # chat_ids.sort(reverse=True)
    
    for chat_id in chat_ids_sorted:
        first_message = st.session_state.chat_store[chat_id].messages[0].content if st.session_state.chat_store[chat_id].messages else "No hay mensajes"
        #limitar titulo de chat con la primera query
        summary = " ".join(first_message.split()[:6]) + "..." if len(first_message.split()) >= 6 else first_message
        
        with st.sidebar.container():
            col1, col2 = st.sidebar.columns([4, 1], gap="small")
            with col1:
                # st.markdown(f"<div style='text-align: left; font-size: 15px; padding: 5px;'>{summary}</div>", unsafe_allow_html=True)
                if st.button(f"{summary}", key=chat_id, type='tertiary',  use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    save_history()
                    st.rerun()
                    
            with col2:
                with st.expander("", expanded=False, ):
                    if st.button("❌", key=f"delete_{chat_id}", help="Eliminar chat",  use_container_width=True):
                        del st.session_state.history[chat_id]
                        del st.session_state.chat_store[chat_id]
                        
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.current_chat_id = None
                        
                        save_history()
                        st.rerun()
                

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

def get_response(chat_id, question):
    """ Obtiene la respuesta de la bd vectorial basada en la consulta. """
    
    question = query_rephrasing_with_crewai(question, model_llm_responses)
    user_memory(chat_id) #pasar historial de conversacion a memoria con memo memory
    inputs = {"question": question}
    
    try:
        result = AgenticRagCrew().crew().kickoff(inputs=inputs)
        response_text = result.raw 
        return response_text if response_text else "No se ha encontrado información"
    except KeyError as e:
        return f"Lo siento, ocurrio un error: {str(e)}"

def user_memory(chat_id):
    
    messages = [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        } 
        # for chat_id in st.session_state.history
        for msg in st.session_state.chat_store[chat_id].messages
    ]
    
    if messages:
        client_memo.add(messages, user_id=chat_id)


    
### ------- GUI ----------
st.set_page_config(page_title="Chatbot MultiAgente", page_icon="🦜")
st.title("Chatea con :blue[tus datos] 🤖")

# ---- Inicializacion de claves para la session
if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

if "history" not in st.session_state:
    st.session_state.history = {}

# Cargar historial desde JSON
loaded_history = load_history()
if loaded_history:
    st.session_state.history.update(loaded_history)

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# --- Sidebar: Historial de conversaciones ---
st.sidebar.title("💬  Chats")

# --- Crear nuevo chat ---
if st.sidebar.button("➕ Nuevo Chat"):
    new_chat_id = str(uuid.uuid4())[:8]  
    st.session_state.history[new_chat_id] = {"title": f"Chat {len(st.session_state.history) + 1}"}
    st.session_state.chat_store[new_chat_id] = ChatMessageHistory()
    st.session_state.current_chat_id = new_chat_id
    save_history() 
    st.rerun()
    
select_chat()


# Mostrar el historial de la conversacion
chat_id = st.session_state.current_chat_id if st.session_state.current_chat_id else None


# para crear un chat si no se crea un nuevo chat con el btn
# if not chat_id:
#     new_chat_id = str(uuid.uuid4())[:8]  
#     st.session_state.history[new_chat_id] = {"title": f"Chat {len(st.session_state.history) + 1}"}
#     st.session_state.chat_store[new_chat_id] = ChatMessageHistory()
#     st.session_state.current_chat_id = new_chat_id  
#     save_history()


if chat_id:
    
    rag_crew = Crew(
    agents=[retriever_agent, evaluator_agent, router_agent, answer_agent],
    tasks=[retriever_task, evaluator_task, router_task, answer_task],
    verbose=True,
    process=Process.sequential,
    # memory=True, # memory capabilities
    # memory_config={
    #     "provider": "mem0",
    #     "config": {"user_id": chat_id,
    #                "api_key": "m0-bnTCjYzJePVtZDrRmibHDe2zvkiByzrMSTde30OM"},
    #     "user_memory" : {}
        
    # }
    )
    # rag_crew = Crew(
    # agents=[travel_agent],
    # tasks=[planning_task],
    # verbose=True,
    # memory=True, # memory capabilities
    # memory_config={
    #     "provider": "mem0",
    #     "config": {"user_id": chat_id,
    #                "api_key": "m0-bnTCjYzJePVtZDrRmibHDe2zvkiByzrMSTde30OM"}
    #     # "user_memory" : {}
    # }
    # )

    st.write(f"**Conversación activa:** `{chat_id}`")

    for message in st.session_state.chat_store[chat_id].messages:
        if isinstance(message, AIMessage):
           MESSAGE_TYPE = "assistant"
        elif isinstance(message, HumanMessage):
            MESSAGE_TYPE = "user"
        else:
            MESSAGE_TYPE = "unknown"

        if MESSAGE_TYPE != "unknown":
            with st.chat_message(MESSAGE_TYPE):
                st.markdown(f"{message.content}")

# Entrar mensaje de usuario y seleccionar modelo LLM
with st.container():
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        selected_llm_model = st.selectbox(
            label="Modelo LLM", 
            options=list(models_llm.values()), 
            index=0,
            placeholder="Selecciona el modelo LLM",
            label_visibility="collapsed",
        )
        
    with col2:
        user_query = st.chat_input("Escribe tu mensaje ✍")
        
    with col3: 
        if st.button("🗑 Limpiar"):
            st.session_state.chat_store[chat_id] = ChatMessageHistory()
            save_history() 
            st.rerun()
    
if user_query:
    with st.spinner("Generando respuesta..."):
        response = get_response(chat_id, user_query)
        
        st.session_state.chat_store[chat_id].add_message(HumanMessage(content=user_query))
        st.session_state.chat_store[chat_id].add_message(AIMessage(content=response))

        save_history() 
        st.rerun()

