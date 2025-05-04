from config import tavily_key
from tavily import TavilyClient
from crewai.tools import tool
from typing import Union


@tool("web_search_tool")
def web_search_tool(question: Union[str, dict]) -> str:
    """Web Search Tool"""
    if isinstance(question, dict):
        question = question.get("description", "")
    
    try:
        client = TavilyClient(api_key=tavily_key)  
        response = client.search(query = question)        
        # tavily_client = TavilyClient(api_key=tavily_key)  
        # answer = tavily_client.qna_search(query=question)
        # answer = tavily_client.qna_search(query=question)
        results = response["results"]
        first_content = results[0].get("content", "")
        second_content = results[1].get("content", "") if len(results) > 1 else ""
        return f"{first_content} {second_content}".strip()
    except Exception as e:
        return f"Web search failed: {e}"
    
    
    
