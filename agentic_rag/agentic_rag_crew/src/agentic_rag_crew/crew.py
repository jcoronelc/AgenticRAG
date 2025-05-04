from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
# Uncomment the following line to use an example of a custom tool
from tools.rag_tool import RAGTool
from tools.web_tool import WebSearchTool
from config import api_key, base_url, model_llm_responses
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
memory_storage_path = r"./data/output/chat/crewai_storage"


# Uncomment the following line to use an example of a knowledge source
# from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class AgenticRagCrew():
	"""AgenticRagCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
 
	def __init__(self):
			self.llm = LLM(
				model=f"lm_studio/{model_llm_responses}",
				base_url=base_url,
				api_key=api_key
			)
 
 
	@before_kickoff # Optional hook to be executed before the crew starts
	def pull_data_example(self, inputs):
		# Example of pulling data from an external API, dynamically changing the inputs
		inputs['extra_data'] = "This is extra data"
		return inputs

	@after_kickoff # Optional hook to be executed after the crew has finished
	def log_results(self, output):
		# Example of logging results, dynamically changing the output
		print(f"Results: {output}")
		return output

	@agent
	def retriever_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['retriever_agent'],
			tools=[RAGTool()], # Example of custom tool, loaded on the beginning of file
			llm = self.llm,
			allow_delegation=False,
			verbose=True
		)

	@agent
	def evaluator_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['evaluator_agent'],
			allow_delegation=False,
			llm = self.llm,
			verbose=True
		)

	@agent
	def web_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['web_agent'],
			tools=[WebSearchTool()],
			llm = self.llm,
			allow_delegation=False,
			verbose=True
		)
  
	@agent
	def answer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['answer_agent'],
			allow_delegation=False,
			llm = self.llm,
			verbose=True
		)

	@task
	def retrieve_task(self) -> Task:
		return Task(
			config=self.tasks_config['retrieve_task']
			# output_file='report.md'
		)
	@task
	def evaluator_task(self) -> Task:
		return Task(
			config=self.tasks_config['context_task'],
			# context=[self.retrieve_task] 
		)
  
	@task
	def web_task(self) -> Task:
		return Task(
			config=self.tasks_config['web_task'],
			# context=[self.evaluator_task] 
		)
  
	@task
	def answer_task(self) -> Task:
		return Task(
			config=self.tasks_config['answer_task'],
			# context=[self.retrieve_task, self.web_task] 
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AgenticRagCrew crew"""
		# You can add knowledge sources here
		# knowledge_path = "user_preference.txt"
		# sources = [
		# 	TextFileKnowledgeSource(
		# 		file_path="knowledge/user_preference.txt",
		# 		metadata={"preference": "personal"}
		# 	),
		# ]

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# memory=True,
			# long_term_memory=LongTermMemory(
 			#          storage=LTMSQLiteStorage(db_path=f"{memory_storage_path}/memory.db")),
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			# knowledge_sources=sources # In the case you want to add knowledge sources
		)
