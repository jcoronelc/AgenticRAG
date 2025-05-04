from crewai import Task


# def create_decider_task(web_agent, evaluator_task, retriever_task, web_search_tool, rag_tool):
#     return Task(
#         description=(
#             "Based on the response from 'evaluator_task', determine whether a web search is required for the question '{question}'. "
#             "If 'evaluator_task' indicates that a web search is true, use the 'web_search_tool' to retrieve additional information from the web. "
#             "If 'evaluator_task' indicates that a web search is false, use the 'rag_tool' to generate an answer. "
#             "Ensure the final response is concise and directly addresses the question."
#         ),
#         expected_output=(
#             "A concise and accurate response to the question '{question}'. "
#             "If a web search is required, the response should be based on the information retrieved using the 'web_search_tool'. "
#             "If no web search is required, the response should be based on the information retrieved using the 'rag_tool'. ."
#         ),
#         agent=web_agent,
#         context=[evaluator_task, retriever_task],
#         tools=[web_search_tool, rag_tool]
#     )

def create_router_task(decider_agent, evaluator_task, retriever_task):
    return Task(
        description=(
            "Based on the evaluation from 'evaluator_task', decide whether to use a web search or the retrieved documents "
            "to answer the question: '{question}'. If 'web_search' is true, use the 'web_search_tool' to retrieve additional information. "
            "If 'web_search' is false, use the documents retrieved by 'retriever_task'. "
            "Generate a clear, concise, and well-structured response based on the chosen information source."
        ),
        expected_output=(
            "A clear and well-written response based on the information obtained. "
            "If a web search is required, the response should be based on the information retrieved using the 'web_search_tool'. "
            "If no web search is required, the response should be based on the output of 'retriever_task'."
        ),
        agent=decider_agent,
        context=[evaluator_task, retriever_task],  # This task depends on evaluator_task and retriever_task
    )

# def create_decider_task(decider_agent, evaluator_task, retriever_task, web_search_tool):
#     return Task(
#         description=(
#             "Based on the evaluation from 'evaluator_task', decide whether to use the retrieved information "
#             "or perform a web search to answer the question: '{question}'. "
#             "If the evaluation indicates that a web search is needed, use the 'web_search_tool' to retrieve additional information. "
#             "If the retrieved information is sufficient, use it to generate a response. "
#             "Do not perform additional searches or modifications."
#         ),
#         expected_output=(
#             "A decision on whether to use the retrieved information or perform a web search. "
#             "If a web search is performed, include the additional information retrieved."
#             "Do not perform any additional searches or modifications after use the web search tool."
#         ),
#         agent=decider_agent,
#         context=[evaluator_task], # This task will wait for evaluator_task to complete
#         tools=[web_search_tool]
#     )
# def create_decider_task(web_agent, one_task, retriever_task, web_search_tool):
#     return Task(
#         description=(
#             "Based on the response from 'evaluator_task', determine whether a web search is required for the question '{question}'. "
#             "If output of 'evaluator_task' indicates that Web Search is true, use the 'web_search_tool' to retrieve additional information from the web. "
#             "If output of 'evaluator_task' indicates that Web Search is false, use the response from 'retriever_task' to generate an answer. "
#             "Ensure the final response is concise and directly addresses the question."
#         ),
#         expected_output=(
#             "A concise and accurate response to the question '{question}'. "
#             "If a web search is required, the response should be based on the information retrieved using the 'web_search_tool'. "
#             "If no web search is required, the response should be based on the output of 'retriever_task'."
#         ),
#         agent=web_agent,
#         context=[one_task, retriever_task],
#         tools=[web_search_tool]
#     )

# def create_decider_task(agent, one_task, retriever_task, web_search_tool):
#     return Task(
#         description=
#         "if the response from one_task determines that web search is true, use the web_search_tool to generate an answer for the question {question}"
#         "if the response from one_task determines that web search is false, not use the web_search_tool, and based on the response from retriever_task generate an answer for the question {question}"
#         ,
#         expected_output="You should analyse the output of the 'one_task'"
#                          "Return a text as response.",
                         
#         agent=agent,
#         context=[one_task, retriever_task],
#         tools=[web_search_tool]
#     )
