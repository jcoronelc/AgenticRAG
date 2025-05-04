#!/usr/bin/env python
import sys
import warnings

# from agentic_rag_crew.crew import AgenticRagCrew
from crew import AgenticRagCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        # "question": "¿Qué cursos solo fueron dictados en la facultad de 'INGENIERÍA' durante el 202462?"
        # "question": "¿Qué puedes contarme con respecto al programa contabilidad y auditoria?"
        "question": "¿Cuál es la edad, sexo y correo del estudiante cuyo nombre es Ana Juana Caño Muñoz?"
    }
    result = AgenticRagCrew().crew().kickoff(inputs=inputs)
    print("\n\n=== FINAL REPORT ===\n\n")
    print(result.raw)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "question": "¿Cual es el programa academico con mayor numero de inscritos?"
    }
    try:
        AgenticRagCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AgenticRagCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "question": "¿Cual es el programa academico con mayor numero de inscritos?"
    }
    try:
        AgenticRagCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

if __name__ == "__main__":
    run()