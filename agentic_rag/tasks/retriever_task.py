from crewai import Task

# def create_retriever_task(agent, router_task, rag_tool):
#     return Task(
#         description="Based on the response from the router task extract information for the question {question} with the help of the respective tool."
#         "Use the web_search_tool to retrieve information from the web in case the router task output is 'websearch'."
#         "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'.",
#         expected_output="You should analyse the output of the 'router_task'"
#         "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
#         "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
#         "Return a claer and consise text as response.",
#         agent=agent,
#         tools=[rag_tool],
#         context=[router_task],
        
#     )

def create_retriever_task(agent, rag_tool):
    return Task(
        description="Extract information for the question {question} with the help of the rag_tool to retrieve information.",
        expected_output=
                         "Use the rag_tool to retrieve information from the vectorstore."
                         "Return a claer and consise text as response..",
        agent=agent,
    )

# def create_retriever_task(agent, rag_tool):
#     return Task(
#         description=(
#             "Using the 'rag_tool', retrieve the top 3 most relevant documents for answering the question: {question}. "
#             "Ensure the retrieved information is directly related to the question and is of high quality."
#             "Perform this task **only once** and do not perform additional searches or modifications."
#         ),
#         expected_output=(
#             "Return a single text containing the most relevant documents retrieved from the Chroma DB. "
#             "The text should be well-structured and directly address the question."
#             "Do not perform any additional searches or modifications after retrieving the documents."
            
#         ),
#         agent=agent,
#         # tools=[rag_tool]
# )