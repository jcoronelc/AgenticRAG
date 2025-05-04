from crewai import Agent

def create_rephrasing_agent(llm):
    return Agent(
        role="Experto en optimización de consultas",
        goal="Reformular y mejorar consultas académicas para facilitar la búsqueda en bases de datos.",
        backstory=(
            "Eres un experto en NLP y optimización de consultas para bases de datos académicas. "
            "Tu objetivo es mejorar la claridad y precisión de las preguntas de los usuarios, "
            "manteniendo su significado original y asegurando una mejor recuperación de información."
        ),
        memory=True,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
