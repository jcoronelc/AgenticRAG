from crewai import Task

def create_grader_task(agent, retriever_task):
    return Task(
        description="Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question.",
        expected_output="Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'.",
       
        agent=agent,
        context=[retriever_task],
    )

# def create_grader_task(agent, retriever_task):
#     return Task(
#         description="Based on the response from the retriever task for the question {question} evaluate whether the retrieved content is relevant to the question.",
#         expected_output="""You should analyse and evaluate the response based on: 
        
#         1. Relevancy to the question
#         2. Faithfulness to the context
#         3. Context quality and completeness
        
#         Please grade the following response based on:
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
#         context=[retriever_task],
#     )

