from crewai import Task
from tools.web_search_tool import web_search_tool



# def create_answer_task(agent, hallucination_task):
#     return Task(
#         description=(
#             "Based on the response from the hallucination task for the question {question} evaluate whether the answer is useful to resolve the question."
#     "If the answer is 'yes' return a clear and concise answer."
#     "If the answer is 'no' then perform a 'websearch' and return the response"
#         ),
#         expected_output="Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
#     "Perform a web search using 'web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
#     "Otherwise respond as 'Sorry! unable to find a valid response'.",
#         agent=agent,
#         # context=[hallucination_task]
#         # tools=[web_search_tool],
#     )
    
    
# def create_answer_task(answer_agent, decider_task):
#     return Task(
#          description=(
#             "Use the information gathered from the 'decider_task' for the question '{question}' to generate a clear, "
#             "concise, and well-structured response in Spanish. Ensure the answer is easy to understand and based on "
#             "the provided information. If the information is insufficient or invalid, respond with a friendly message "
#             "indicating that a valid answer could not be found."
#         ),
#         expected_output=(
#             "A clear and well-written response in Spanish, based on the information obtained. "
#             "If there is no valid or sufficient information, respond: 'Sorry, I couldn’t find a valid answer.'"
#         ),
#         agent=answer_agent,
#         context=[decider_task]
#         # tools=[web_search_tool],
#     )

def create_answer_task(answer_agent, decider_task):
    return Task(
        description=(
            "Using the information from 'decider_task', generate a clear, concise, and well-structured response "
            "to the question: '{question}'. Ensure the answer is in Spanish and is based on the provided information. "
            "If the information is insufficient or invalid, respond with a friendly message indicating that a valid answer could not be found."
        ),
        expected_output=(
            "A clear and well-written response in Spanish, based on the information obtained. "
            "If there is no valid or sufficient information, respond: 'Lo siento, no pude encontrar una respuesta válida.'"
        ),
        agent=answer_agent,
        context=[decider_task]   # This task will wait for decider_task to complete
    )