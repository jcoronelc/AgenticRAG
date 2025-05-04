from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from config import collection_name, persist_directory
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
            question = question.get("question", str(question))
        if not isinstance(question, str) or not question.strip():
            return "Invalid input: Question must be a non-empty string"
       
        print("RAG TOOL - Query:", question) 
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        try:
            client_bd = chromadb.PersistentClient(path=persist_directory)
            collection = client_bd.get_collection(collection_name)
            results = collection.query(
                query_texts=[question],
                n_results=5
            )
            
            if not results["documents"]:
                return "No relevant documents found."
                
            documents = "\n\n".join([doc for doc in results["documents"][0] if doc])
            print("Found documents:", documents)  # Debug
            return documents
        
        except Exception as e:
            print("Error:", str(e))  
            return f"Error querying Chroma DB: {e}"