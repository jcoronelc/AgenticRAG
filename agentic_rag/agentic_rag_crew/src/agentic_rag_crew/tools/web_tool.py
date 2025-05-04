from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from config import tavily_key
from tavily import TavilyClient

class WebSearchToolInput(BaseModel):
    question: str = Field(..., description="La pregunta a buscar en la web")

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool" 
    description: str = (
        "Usa para búsquedas en internet cuando necesites información externa. "
        "Útil para datos no disponibles en los documentos internos."
    )
    args_schema: Type[BaseModel] = WebSearchToolInput
    
    def _run(self, question: str) -> str:
        # Manejo robusto de diferentes formatos de entrada
        if isinstance(question, dict):
            question = question.get("question", str(question))
        
        if not isinstance(question, str) or not question.strip():
            return "Invalid input: Question must be a non-empty string"
       
        print("WEB TOOL - Query:", question) 
        
        try:
            client = TavilyClient(api_key=tavily_key)
            response = client.search(
                query=question,
                search_depth="basic",
                max_results=3,
                include_answer=True,
                include_raw_content=True
            )
            
            if not response.get("results"):
                return "No se encontraron resultados relevantes."
            
            formatted_results = []
            for i, result in enumerate(response["results"][:3], 1):
                content = result.get("content", "Sin contenido disponible")
                url = result.get("url", "URL no disponible")
                formatted_results.append(
                    f"📄 Resultado {i}:\n{content[:500]}...\n🔗 Fuente: {url}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            print("Error:", str(e)) 
            return f"Error querying Web: {e}"