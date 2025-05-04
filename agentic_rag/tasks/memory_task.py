from crewai import Task


def create_memory_task(memory_agent):
    return Task(
        description=(
            """Analiza el siguiente historial de conversación {conversation_log} y extrae el conocimiento fundamental:
            
            Considera:
            1. Conceptos clave mencionados
            2. Relaciones entre entidades
            3. Preguntas recurrentes
            4. Soluciones propuestas
            5. Vacíos de información detectados"""
            
        ),
        expected_output=(
            """Un reporte estructurado con:
            - Listado de conceptos clave con relevancia (0-1)
            - Entidades importantes y sus tipos
            - Relaciones sujeto-verbo-objeto significativas"""
        ),
        agent=memory_agent,
    )