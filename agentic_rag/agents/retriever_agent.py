from crewai import Agent
from crewai.project import CrewBase, agent

def create_retriever_agent(llm, rag_tool):
    return Agent(
        role="Retriever",
        goal="Use the information retrieved from the vectorstore to answer the question",
        backstory=(
        "You are an assistant for question-answering tasks."
        "Use the information present in the retrieved context to answer the question."
        "You have to provide a clear concise answer."
        ),
        memory=True,
        verbose=True,
        allow_delegation=False,
        tools=[rag_tool],
        llm=llm,
    )

# def create_retriever_agent(llm, rag_tool):
#     return Agent(
#         role="Retriever",
#         goal="Use the information retrieved to answer the question",
#         backstory=(
#              "You are a Helpful Assistant Proficient in Answering concise,factful and to the point answers the question."
#             "Use the information present in the retrieved context to answer the question."
#             "You have to provide a clear concise answer."
#         ),
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#         tools=[rag_tool],
#     )

# def create_retriever_agent(llm, rag_tool):
#     return Agent(
#         role="Retriever",
#         goal="Retrieve the most relevant and accurate information from the Chroma DB to answer the user's question {question}",
#         backstory=(
#             "You are an expert at retrieving information "
#             "Your role is to search the Chroma DB for documents "
#         ),
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#         tools=[rag_tool],
#         max_iter=1,
#         # memory=True,
#         max_retry_limit=1,
#         handle_errors=True,
#         output_format="Thought: I now can give a great answer\nFinal Answer: <retrieved documents>",
#     )


