from crewai import Agent


# def create_evaluator_agent(llm):
#     return Agent(
#         role="Answer Grader",
#         goal="Evaluate the response ",
#         backstory=(
#             "You are a grader assessing relevance of a retrieved document to a user question." 
#         ),
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#     )

def create_evaluator_agent(llm):
    return Agent(
        role="Answer Grader",
        goal="Evaluate the quality and relevance of the retrieved information to the user's question.",
        backstory=(
            "You are a meticulous evaluator who assesses the quality of retrieved information. "
            "Your role is to grade the relevance, faithfulness, and completeness of the retrieved documents. "
            # "You also determine if additional information from the web is needed to answer the question fully."
        ),
        verbose=True,
        memory=True,
        allow_delegation=False,
        llm=llm,
    )