from typing import List, Tuple, Any

import guidance
from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool


PREFIX: str = '''Answer the question with the given
information and tools available.'''

FORMAT_INSTRUCTIONS: str = '''
tools: {{tools_description}}

Think with the following format:
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do and ask:
Any follow-up actions needed: Yes or No
Action: the action to take, should be one of {{tool_names}}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.'''

SUFFIX: str = '''
Begin!
Let's think about the question step by step.
Question: {{question}}{{agent_scratchpad}}
Any follow up actions needed? {{#select 'follow-up'}}Yes{{or}}No{{/select}}
{{#if (== follow-up "Yes")}}
Thought: {{gen 'thought' stop="\\n"}}
Action: {{select 'action' options=tool_names}}
Action Input: {{gen 'tool input' stop="Observation:"}}
{{else}}
Final answer: {{gen 'final answer'}}
{{/if}}
'''


class GuidedAgent(BaseSingleActionAgent):

    tools: List[BaseTool] = []

    @property
    def input_keys(self):
        return ['tools', 'question']

    @property
    def output_keys(self):
        return []

    def plan(self,
             intermediate_steps: List[Tuple[AgentAction, str]],
             **kwargs: Any):
        _guidance = guidance(
            PREFIX + FORMAT_INSTRUCTIONS + SUFFIX,
        )

        tools = kwargs['tools']
        tools_name = [tool.name for tool in tools]
        tools_description = '\n'.join(
            [tool.name + ':' + tool.description for tool in tools])
        question = kwargs['question']
        agent_scratchpad = self._construct_scratchpad(intermediate_steps)

        result = _guidance(
            caching=False,
            tools_description=tools_description,
            tool_names=tools_name,
            question=question,
            agent_scratchpad=agent_scratchpad,)
        if result['follow-up'] == 'No':
            return AgentFinish({'output': result['final answer'].strip()},
                               log='Action Finished: ')
        else:
            action = result['action'].strip()
            action_input = result['tool input'].strip()

            return AgentAction(tool=action,
                               tool_input=action_input,
                               log='Agent Action:')

    def _construct_scratchpad(self,
                              intermediate_steps: List[Tuple[AgentAction, str]]
                              ) -> str:
        thoughts = []
        for action, observation in intermediate_steps:
            thoughts.append(f'{action.log}: {action.tool}\n' +
                            f'Action Input: ```{action.tool_input}```\n'
                            f'Observation: {observation}')
        return '\n'.join(thoughts)

    async def aplan(self,
                    intermediate_steps: List[Tuple[AgentAction, str]],
                    **kwargs: Any):
        raise NotImplementedError("Async not implemented")
