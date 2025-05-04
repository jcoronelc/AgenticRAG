from crewai import Task

def create_rewriting_task(rewriting_agent):
    return Task(
        description=(
            "Dada la pregunta del usuario: '{question}', reformúlala para que sea más precisa y útil en una búsqueda semántica dentro de una base de datos vectorial (como Chroma o FAISS). "
            "No generes una query en SQL ni ningún otro lenguaje estructurado. La salida debe ser una frase en lenguaje natural optimizada para embeddings. "
            "Puedes aplicar reescritura, expansión de términos relevantes y descomposición si la pregunta es compuesta."
        ),
        expected_output=(
            "Una única línea en lenguaje natural que represente la consulta optimizada, sin explicaciones ni formato SQL."
        ),
        agent=rewriting_agent,
    )
