from typing import List, Tuple, Any

from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent


class DummyAgent(BaseSingleActionAgent):
    """DummyAgent is an agent that simply repeats the execution of the tool
    by the specified number of times.

    tool: the name of the tool
    count: the number of times for execution

    Example:

    ```
    # create a tool that repeats the python tool execution 3 times
    dummy = DummyAgent(tool=python_tool.name, count=3)
    # create the executor with agent and the tool
    executor = AgentExecutor.from_agent_and_tools(
        agent=dummy,
        tools=[python_tool],
        verbose=True,
    )
    executor.run({'tool_input': 'print("hello world")'})
    ```
    """

    tool: str = ''
    count: int = 3

    @property
    def input_keys(self):
        return ['tool_input']

    @property
    def output_keys(self):
        return []

    def plan(self,
             intermediate_steps: List[Tuple[AgentAction, str]],
             **kwargs: Any):

        if self.count <= 0:
            return AgentFinish({'output': 'Finished execution'},
                               log='Action Finished: ')

        self.count -= 1
        return AgentAction(tool=self.tool,
                           tool_input=kwargs['tool_input'],
                           log='Agent Action: ')

    async def aplan(self,
                    intermediate_steps: List[Tuple[AgentAction, str]],
                    **kwargs: Any):
        raise NotImplementedError("Async not implemented")
