from crewai import Agent

def create_grader_agent(llm):
    return Agent(
        role='Answer Grader',
        goal='Filter out erroneous retrievals',
        backstory=(
            "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keywords related to the user question, grade it as relevant."
    "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
        ),
        verbose=True,
        memory=True,
        allow_delegation=False,
        llm=llm,
    )
