from crewai import Agent


# def create_decider_agent(llm):
#     return Agent(
#         role="Information Synthesizer",
#         goal="Search information",
#         backstory=(
#             "You are a researcher that give a good response"
#             "Use the web search for questions"
#             "Otherwise, use the retrieved context "
#             "Return a final response"
#         ),
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#     )
    


# def create_decider_agent(llm):
#     return Agent(
#         role="Information Synthesizer",
#         goal=(
#             "Decide whether to use a web search or retriever-based information to answer a question, "
#             "and generate a concise and accurate response."
#         ),
#         backstory=(
#             "You are an expert in evaluating information sources and synthesizing data. "
#             "Your role is to decide whether to retrieve information from the web or use pre-existing data "
#             "to provide accurate and concise answers to questions."
#         ),
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#    )
    
def create_router_agent(llm, web_search_tool):
    return Agent(
        role="Router",
        goal=(
            "Decide whether to use a web search or the retrieved documents to answer the question"
        ),
        backstory=(
            "You are an expert in evaluating information sources and synthesizing data. "
            "Your role is to decide whether to retrieve information from the web or use pre-existing data "
            "to provide accurate and concise answers to questions."
        ),
        memory=True,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[web_search_tool],  
        output_format="Thought: I now can give a great answer\nFinal Answer: <final response>"
    )