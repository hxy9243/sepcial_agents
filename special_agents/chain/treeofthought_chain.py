from typing import List, Dict, Optional

from langchain.schema import AgentAction, AgentFinish
from langchain.agents.tools import Tool
from langchain.agents import Agent, AgentExecutor
from langchain.callbacks.manager import CallbackManagerForChainRun

from special_agents.agents import GuidedAgent


class TotExecutor(AgentExecutor):

    def __call(self,
               inputs: Dict[str, str],
               run_manager: Optional[CallbackManagerForChainRun],
               ):
        ''' For TotExecutor, instead of returning a list of steps of results,
        return the output (observation) in the format of a tree.

        The final result should contain all the leaves of the results.

        There evaluation and filtering of the results should be left for the
        next step of tools in the pipeline.

        root --- branch 1 --- branch 1.1 -- ...
              |            |- branch 1.2
              |- branch 2 --- branch 2.1
              |- branch 2 --- branch 2.2
        '''

        pass


def new_tot_chain(
        model: Agent,
        tools: List[Tool],
        nbranches: int = 3):
    '''New TreeOfThought Chain executes the agnet execute multiple probable
    outputs of the LLM generation in the format of a tree.
    '''

    agent = GuidedAgent(model=model)

    return TotExecutor(
        agent=agent,
        tools=tools,
    )

