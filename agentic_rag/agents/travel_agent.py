from crewai import Agent

def create_travel_agent(llm):
    """Create a travel planning agent with search capabilities"""
   

    return Agent(
        role="Personalized Travel Planner Agent",
        goal="Plan personalized travel itineraries",
        backstory="""You are a seasoned travel planner, known for your meticulous attention to detail.""",
        allow_delegation=False,
        memory=True,
         llm=llm,
    )