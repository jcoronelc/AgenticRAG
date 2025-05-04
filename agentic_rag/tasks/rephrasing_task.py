from crewai import Task


def create_rephrasing_task(rephrasing_agent):
    return Task(
        description=(
            "Corrige la gramática, ortografía y estructura de la siguiente consulta académica: {question}. "
            "Genera una versión alternativa mejorada sin alterar términos técnicos ni identificadores, utilizando sinónimos "
            "Devuelve exclusivamente una única pregunta reformulada en una sola línea en español."
        ),
        expected_output=(
            "Una consulta académica mejorada en una sola línea."
        ),
        agent=rephrasing_agent,
    )