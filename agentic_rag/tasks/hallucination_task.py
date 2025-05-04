from crewai import Task

def create_hallucination_task(agent, grader_task):
    return Task(
        description="Based on the response from the grader task for the question {question} evaluate whether the answer is grounded in / supported by a set of facts.",
        expected_output="Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
    "Respond 'yes' if the answer is in useful and contains fact about the question asked."
    "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'.",
        agent=agent,
        context=[grader_task],
    )
