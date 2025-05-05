from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from config import collection_name_active, persist_directory
import chromadb
import os

class RAGToolInput(BaseModel):
    question: str = Field(..., description="La pregunta directa a buscar")

class RAGTool(BaseTool):
    name: str = "RAG Search Tool"
    description: str = (
        "Busca en una base vectorial (ChromaDB) documentos relevantes para una pregunta del usuario. "
        "Úsala cuando necesites contexto interno antes de responder."
    )
    args_schema: Type[BaseModel] = RAGToolInput

    def _run(self, question: str) -> str:

        if isinstance(question, dict):  # Manejo de retrocompatibilidad
            question = question.get("question") or question.get("description") or str(question)

        if not isinstance(question, str) or not question.strip():
            return "Entrada inválida: la pregunta debe ser una cadena no vacía."
       
        print("RAG TOOL - Question :", question)
        persist_directory = "/home/juancoronel/Desktop/AgenticRAG/agentic_rag/data/output/chroma/persistent_directory"
        # print(persist_directory)
        top_k = 10

        try:
            client_bd = chromadb.PersistentClient(path=persist_directory)
            collection = client_bd.get_collection(collection_name_active)
            results = collection.query(
                query_texts=[question],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved_info = ""
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            # print(f"\nTop {top_k} documentos recuperados:")
            # for i, doc in enumerate(results["documents"][0]):
            #     print(f"\nDocumento {i+1}:\n{doc}")
            #     print("Metadatos:", results["metadatas"][0][i])
            
            for i in range(len(docs)):
                retrieved_info += f"### Documento {i+1}:\n{docs[i]}\n"
                if metas[i]:
                    for key, value in metas[i].items():
                        retrieved_info += f"- {key}: {value}\n"
                retrieved_info += "\n"
            return retrieved_info
        
        except Exception as e:
            print("Error:", str(e))  
            return f"Error querying Chroma DB: {e}"