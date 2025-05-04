from crewai import Agent

def create_hallucination_agent(llm):
    return Agent(
        role="Hallucination Grader",
        goal="Filter out hallucination",
        backstory=(
           "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
        ),
        verbose=True,
        memory=True,
        allow_delegation=False,
        llm=llm,
    )
