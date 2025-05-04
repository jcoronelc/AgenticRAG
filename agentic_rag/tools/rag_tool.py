from config import api_key, base_url, model_llm_embeddings, model_llm_responses, collection_name_active

from crewai.tools import tool
from embedding.chroma_utils import get_embeddings, initialize_chroma, store_document_embeddings, query_chroma_with_llm, call_llm_model
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import pandas as pd
from typing import Union

@tool("rag_tool")
def rag_tool(question: Union[str, dict]) -> str:
    """Search Chroma DB for relevant documents based on a query."""
    if isinstance(question, dict):
        question = question.get("description")
    print(question)
    
    if not isinstance(question, str) or not question.strip():
        return "Invalid input: The 'question' must be a non-empty string."

    persist_directory = "./data/output/chroma/persistent_directory"
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    client_bd = chromadb.PersistentClient(path=persist_directory)
    collection = client_bd.get_collection(collection_name_active)
    print(collection_name_active)

    try:
        results = collection.query(
            query_texts=[question],  
            n_results=3
        )
    except Exception as e:
        return f"Error querying Chroma DB: {e}"

    if not results["documents"]:
        return "No relevant documents found for your query."

    if "documents" in results and results["documents"]:
        documents = "\n\n".join([doc for doc in results["documents"][0] if doc])
        print(documents)
    else:
        documents = "No relevant documents found."
    return documents