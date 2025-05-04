from crewai import Agent

def create_answer_agent(llm):
    return Agent(
        role="Answer Writer",
        goal="Generate a clear, concise, and well-structured final answer to the user's question.",
        backstory=(
            "You are a skilled writer who specializes in creating high-quality, coherent, and accurate responses. "
            "Your role is to synthesize information from the retrieved documents or web search results "
            "and present it in a way that is easy to understand and meets the user's needs."
        ),
        memory=True,
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )