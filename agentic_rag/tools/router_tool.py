from crewai.tools import tool

@tool
def router_tool(question):
    """Router Function"""
    if 'UTPL' in question:
        return 'web_search'
    else:
        return 'vectorstore'
