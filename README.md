# Special Agents

Special Agents: a repo experimenting and creating specialized
[Langchain](https://github.com/hwchase17/langchain) agents capable of performing
special tasks.

## Quick Start

To quickly get started, follow these steps to install the Work-In-Progress (WIP) special agents library, preferably in a virtual environment:

```bash
pip3 install langchain guidance
pip3 install git+https://github.com/hxy9243/sepcial_agents
```

Once installed, you can import and run the special agents in your Python project. Here's an example:

```python
from langchain.agents.agent import AgentExecutor
from langchain.tools.python.tool import PythonAstREPLTool

from special_agents.agents import DummyAgent

python_tool = PythonAstREPLTool(description='Python AST tool to execute Python scripts')

tools = [python_tool]

dummy = DummyAgent(
    tool=python_tool.name,
    count=3,
)
executor = AgentExecutor.from_agent_and_tools(
    agent=dummy,
    tools=tools,
    verbose=True,
)
executor.run({'tool_input': 'print("hello world")'})
```

## Examples

For more usage examples and demonstrations, please refer to the [examples](examples) directory.

## Agents

### DummyAgent

The DummyAgent is a versatile agent that repeats the actions of the specified tool a specified number of times. (See examples above.)

It is an excellent tool for exploring and gaining a deeper understanding of Langchain's inner mechanisms. Refer to the example above to see how the DummyAgent can be utilized.

### GuidedAgent

The GuidedAgent utilizes [Microsoft Guidance](https://github.com/microsoft/guidance/tree/main)
library to guide agent outputs, enabling more precise output control and parsing.

## License

This repository is licensed under the Apache 2.0 License.
