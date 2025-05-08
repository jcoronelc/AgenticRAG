import os
import pandas as pd

from config import api_key, base_url, model_llm_embeddings, model_llm_responses, collection_name, collection_name_summary, collection_name_active, collection_name_active_summary, retrieval_method

from discover.processing import Process
from discover.schema import Schema
from discover.formatting import FormattingText
from discover.template import TemplateGenerator
from discover.document import DocumentGeneration

from crewai import Crew
from crewai import LLM

from agents.router_agent import create_router_agent
from agents.retriever_agent import create_retriever_agent
from agents.grader_agent import create_grader_agent
from agents.hallucination_agent import create_hallucination_agent
from agents.answer_agent import create_answer_agent
from agents.evaluator_agent import create_evaluator_agent

from tasks.router_task import create_router_task
from tasks.retriever_task import create_retriever_task
from tasks.grader_task import create_grader_task
from tasks.hallucination_task import create_hallucination_task
from tasks.answer_task import create_answer_task
from tasks.evaluator_task import create_evaluator_task

from tools.router_tool import router_tool
from tools.rag_tool import rag_tool
from tools.web_search_tool import web_search_tool

from embedding.chroma_utils import initialize_chroma, store_document_embeddings, query_chroma_with_llm, call_llm_model
from embedding.pdf_utils import create_pdf, extract_text_from_pdf
from embedding.txt_utils import create_txt, extract_text_from_txt

from openai import OpenAI
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

import time



def main():
    start = time.time()
    client = OpenAI( 
        api_key = api_key,
        base_url = base_url
    )
     
    use_agents = False 
    create_bd = True # Cambia a False para realizar querys a la bd
    save_as_pdf = False  # Cambia a False para guardar en TXT
    use_external_data = True
    format_type = "general" # Formato de datos: detailed o general
    chunk_size = 300 #300
    overlap_size = 75 #75
    top_k = 8
     
    # collection_name = "pdf_agent_embeddings_v6" #collection chroma  (ver read_me)
    output_folder = f"./data/output/documents/documents_{collection_name}"
    non_structured_summary_folder = f'./data/input/non_structured/summaries/{format_type}'

    number_documents = 130000 # nro de docs a generar para embeddings
       
    if create_bd:
        #cargar datos

        integrar_datos_estructurados(client, number_documents, output_folder, save_as_pdf, format_type, chunk_size, overlap_size ) 
        
        integrar_datos_no_estructurados(client, non_structured_summary_folder, format_type, chunk_size, overlap_size)
        

    else:              
        if use_agents:
            #crear agentes

            llm = LLM(model="lm_studio/"+ model_llm_responses, 
                    base_url=base_url, 
                    api_key=api_key)
            
            retriever_agent = create_retriever_agent(llm, rag_tool)
            # grader_agent = create_grader_agent(llm)
            # hallucination_agent = create_hallucination_agent(llm)
            evaluator_agent = create_evaluator_agent(llm)
            router_agent = create_router_agent(llm, web_search_tool)
            answer_agent = create_answer_agent(llm)

            # Crear tareas
            retriever_task = create_retriever_task(retriever_agent, rag_tool)
            # grader_task = create_grader_task(grader_agent, retriever_task)
            # hallucination_task = create_hallucination_task(hallucination_agent, grader_task)
            evaluator_task = create_evaluator_task(evaluator_agent, retriever_task)
            router_task = create_router_task(router_agent, evaluator_task, retriever_task)
            answer_task = create_answer_task(answer_agent, router_task)
            
            rag_crew = Crew(
                agents=[retriever_agent, evaluator_agent, router_agent, answer_agent],
                tasks=[retriever_task, evaluator_task, router_task, answer_task],
                verbose=True,
                # memory=True, # memory capabilities
            )
            # tavily_client = TavilyClient(api_key="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")

            # # Step 2. Executing a Q&A search query
            # answer = tavily_client.qna_search(query="Who is Victor Hugo Saquicela Galarza?")

            # # Step 3. That's it! Your question has been answered!
            # print(answer)

            #inputs ={"question":"¿Como afecta la modalidad (presencial, virtual, etc.) al rendimiento academico de los estudiantes en el periodo academico 202264?"}

        
            inputs ={"question":"¿Como afecta la modalidad (presencial, virtual, etc.) al rendimiento academico de los estudiantes en el periodo academico 202264?"}
            
            try:
                result = rag_crew.kickoff(inputs=inputs)
                print(f"Respuesta: {result}")
            except KeyError as e:
                print(f"KeyError: Missing key {e}. Current inputs: {inputs}")
                raise

        else:
            #question = "Dame una lista de notas del programa Computacion de nivel grado a distancia ?"
            #question = "Cual es el numero total de hombres inscritos en los programas, no necesito por periodo"
            #question = "Dame el numero de hombres inscritos en el programa de computacion"
            question = "Cuantos estudiantes hay en total"
            #question = "Cual es el mejor estudiante del curso 'DOCTRINA SOCIAL DE LA IGLESIA?"
            #question = "Cual es el numero de estudiantes del género 'hombre' que esten en el programa ADMINISTRACION DE EMPRESAS"
            #question = "CUales son los programas nombres que estan alamcenadso ?"
            #question = "¿Cuál es la edad, sexo y correo del estudiante llamado cristina lucía martín medina?"
            #question = "¿Cómo se comparan las notas del bimestre 1 y bimestre 2 para el estudiante con identificador estudiante: 87812?"
            
            response = query_chroma_with_llm(client, question, use_external_data, top_k, retrieval_method )
            print(response)

    end = time.time()
    elapsed = end - start

    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Tiempo total de ejecución: {int(hours)}h {int(minutes)}m {int(seconds)}s")



def integrar_datos_estructurados(client, number_documents, output_folder, save_as_pdf, format_type, chunk_size, overlap_size ):
    df = pd.read_excel('./data/input/structured/calificaciones_completa.xlsx')
        
    process = Process(df)
    df = process.process_data()
        
    #obtener esquema
    schema = Schema(df, number_documents)
    schema, df_sample = schema.get_schema()

    #formateo de datos
    formatting = FormattingText(df_sample)
    formatted_df = formatting.format_data()

    #generacion del template
    template_gen = TemplateGenerator(schema)
    template = template_gen.create_template(format_type)
        #formated_text = template_gen.clean_text(template)

        #generacion de documentos
    doc_gen_agent = DocumentGeneration(template, formatted_df)
    documents = doc_gen_agent.generate_documents()
    os.makedirs(output_folder, exist_ok=True)

    # Crear documentos
    print("Creando documentos...")
    for idx, doc in enumerate(documents, start=1):
        file_extension = "pdf" if save_as_pdf else "txt"
        file_path = os.path.join(output_folder, f"document_{idx}.{file_extension}")

        if save_as_pdf:
            create_pdf(doc, file_path)
        else:
            create_txt(doc, file_path)

    # Procesar documentos
    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)

        if os.path.isfile(file_path):
            if save_as_pdf and file_name.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif not save_as_pdf and file_name.endswith(".txt"):
                text = extract_text_from_txt(file_path)
            else:
                continue  # Ignorar archivos que no coincidan con la configuración
                
            aux_chunk_id = file_name.replace("document_", "")
            chunk_id = aux_chunk_id.replace(".txt", "")
               
            store_document_embeddings(text, chunk_id,  client, collection_name, model_llm_embeddings, chunk_size, overlap_size, format_type)

def integrar_datos_no_estructurados(client, non_structured_summary_folder, format_type, chunk_size, overlap_size):

    print("Procesando datos externos")

    if os.path.exists(non_structured_summary_folder):
        for filename in os.listdir(non_structured_summary_folder):
            file_path = os.path.join(non_structured_summary_folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()


                chunk_id = os.path.splitext(filename)[0]
                store_document_embeddings(raw_text, chunk_id,  client, collection_name_summary, model_llm_embeddings, chunk_size, overlap_size, format_type)

    # if os.path.exists(non_structured_summary_folder):
    #     with open(non_structured_summary_folder, 'r', encoding='utf-8') as f:
    #         raw_text = f.read()
    #         chunk_id = "documento_nonstructured"
    #         store_document_embeddings(raw_text, chunk_id,  client, collection_name, model_llm_embeddings, chunk_size, overlap_size)
       

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('... Proceso finalizado ...')