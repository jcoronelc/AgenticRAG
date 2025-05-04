from crewai import Task

def create_planning_task(agent):
    """Create a travel planning task"""
    return Task(
        description="Find places to live, eat, and visit in {question}",
        expected_output="A detailed list of places to live, eat, and visit in {question}",
        agent=agent,
    )