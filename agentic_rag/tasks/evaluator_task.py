from crewai import Task

def create_evaluator_task(evaluator_agent, retriever_task):
    return Task(
        description=(
            "Evaluate the information retrieved by the 'retriever_task' for the question: '{question}'. "
            "Assess the retrieved documents based on the following criteria:\n"
            "1. **Relevancy (0-1)**: How well do the documents answer the question?\n"
            "2. **Faithfulness (0-1)**: How closely do the documents stick to the provided context?\n"
            "3. **Context Quality (0-1)**: How complete and relevant is the provided context?\n"
            "4. **Web Search (true/false)**: Is a web search needed to augment the context?\n"
            "Provide your evaluation in JSON format."
        ),
        expected_output=(
            "Relevancy :value between 0 and 1"  # Replace with a float value between 0 and 1
            "Faithfulness : value between 0 and 1"  # Replace with a float value between 0 and 1
            "Context Quality:  value between 0 and 1"  # Replace with a float value between 0 and 1
            "Needs Web Search : value true or false"  # Indicate whether web search is needed
            "Answer: final response based on context or web search"
        ),
        agent=evaluator_agent,
        context=[retriever_task] # This task will wait for retriever_task to complete
    )

# def create_evaluator_task(evaluator_agent, retriever_task, web_search_tool):
#     return Task(
#         description=
#         "Use the web_search_tool to retrieve information from the web in case the router task output is 'websearch'."
#         "Based on the response and context retrieved from the 'retriever_task', evaluated the response for the question {question}",
#         expected_output="""You should analyse and evaluate the response based on: 
        
#         1. Relevancy to the question
#         2. Faithfulness to the context
#         3. Context quality and completeness
#         4. Needs web search
        
#         Please grade the following response based on and give me in a JSON format:
#         1. Relevancy (0-1): How well does it answer the question?
#         2. Faithfulness (0-1): How well does it stick to the provided context?
#         3. Context Quality (0-1): How complete and relevant is the provided context?
#         4. Web Search(true or false): Determine only if web search is needed to augment the context
               
#         """,
        
#         agent=evaluator_agent,
#         context=[retriever_task],
#         tools=[web_search_tool]
#     )


# def create_one_task(agent, retriever_task):
#     return Task(
#         description=
#         # "Use the web_search_tool to retrieve information from the web in case the router task output is 'websearch'."
#         "Based on the response and context retrieved from the retriever task, evaluated the response for the question {question}",
#         expected_output="""You should analyse and evaluate the response based on: 
        
#         1. Relevancy to the question
#         2. Faithfulness to the context
#         3. Context quality and completeness
        
#         lease grade the following response based on:
#         1. Relevancy (0-1): How well does it answer the question?
#         2. Faithfulness (0-1): How well does it stick to the provided context?
#         3. Context Quality (0-1): How complete and relevant is the provided context?
           
#         Also determine if web search is needed to augment the context.
        
#         Please grade the response in the following JSON format:

#         {{
#             "Relevancy": <score>,  # Replace with a float value between 0 and 1
#             "Faithfulness": <score>,  # Replace with a float value between 0 and 1
#             "Context Quality": <score>,  # Replace with a float value between 0 and 1
#             "Needs Web Search": <true/false>,  # Indicate whether web search is needed
#             "Explanation": "<explanation>",  # Provide detailed reasoning
#             "Answer": "<final response based on context or web search>"
#         }}""",
        
        

#         agent=agent,
#         context=[retriever_task]
#     )
