from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from telegram_bot_crewai.tools.web_search_tool import WebSearchTool
from langchain_groq import ChatGroq
import logging
from typing import Optional
from composio_crewai import ComposioToolSet, Action, App
import os

logger = logging.getLogger(__name__)

composio_toolset = ComposioToolSet(api_key="xgppxxnvbkrc4bdh6svn4r")
tools = composio_toolset.get_tools(actions=[
	'GOOGLECALENDAR_CREATE_EVENT',
	'GOOGLECALENDAR_DELETE_EVENT',
	'GOOGLECALENDAR_FIND_FREE_SLOTS',
	'GOOGLECALENDAR_UPDATE_EVENT'
])

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

model = LLM(
	model="groq/llama-3.3-70b-versatile",
	api_key=groq_api_key,
	max_tokens=1000
)

@CrewBase
class TelegramBotCrew:
	"""Telegram Bot crew for handling queries and appointments"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def manager_agent(self) -> Agent:
		return Agent(
			llm = model,
			allow_delegation = True,
			verbose = True,
			role = "Supervisor",
			goal = "Efficiently manage the crew by delegating the task to the right agent and return to the user the output of the agent when he completed the task",
			backstory = "You are an over-skilled supervisor with a track on project management and guiding teams to success" \
			"You always find the best agent for any given task and make sure that the task is completed with success" \
			"When the agent completed his task, return the output message of the agent to the use"
		)

	@agent
	def calendar_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['calendar_agent'],
			tools=tools,
			llm=model,
			verbose=True,
			allow_delegation=False
		)
	
	@agent
	def research_agent(self) -> Agent:
		return Agent(
			config = self.agents_config['research_agent'],
			tools = [ WebSearchTool() ],
			allow_delegation = False,
			llm = model
		)

	@task
	def handle_query(self) -> Task:
		return Task(
			config=self.tasks_config['handle_query'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Telegram Bot crew with manager-based delegation"""
		try:
			return Crew(
				agents=self.agents,
				tasks=self.tasks,
				manager_llm=ChatGroq(
					temperature=0,
					model="groq/Llama-3.3-70b-Versatile",
					verbose=True,
					max_tokens=1000,
				),
				function_calling_llm = model,
				max_rpm = 5000,
				process=Process.hierarchical,
				verbose=True
			)
		except Exception as e:
			logger.error(f"Error creating crew: {str(e)}")
			raise

	# def format_response(self, response: Optional[str]) -> str:
	# 	"""Format the crew's response for Telegram"""
	# 	if not response:
	# 		return "Sorry, I couldn't process your request."
		
	# 	# Clean up the response by removing any "Final Answer:" prefix
	# 	response = response.replace("Final Answer:", "").strip()
	# 	return response
