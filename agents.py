#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass
from rich.syntax import Syntax
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import json

from .utils import (
    console,
    parse_code_blob,
    parse_json_tool_call,
    truncate_content,
    AgentError,
    AgentParsingError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxIterationsError,
)
from .types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import FinalAnswerTool
from .models import MessageRole
from .monitoring import Monitor
from .prompts import (
    CODE_SYSTEM_PROMPT,
    TOOL_CALLING_SYSTEM_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION,
    SYSTEM_PROMPT_FACTS,
    SYSTEM_PROMPT_FACTS_UPDATE,
    USER_PROMPT_FACTS_UPDATE,
    USER_PROMPT_PLAN_UPDATE,
    USER_PROMPT_PLAN,
    SYSTEM_PROMPT_PLAN_UPDATE,
    SYSTEM_PROMPT_PLAN,
    MANAGED_AGENT_PROMPT,
)
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonInterpreter
from .e2b_executor import E2BExecutor
from .tools import (
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    Tool,
    get_tool_description_with_args,
    Toolbox,
)


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str


class AgentStep:
    pass


@dataclass
class ActionStep(AgentStep):
    agent_memory: List[Dict[str, str]] | None = None
    tool_call: ToolCall | None = None
    start_time: float | None = None
    end_time: float | None = None
    iteration: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    llm_output: str | None = None
    observations: str | None = None
    action_output: Any = None


@dataclass
class PlanningStep(AgentStep):
    plan: str
    facts: str


@dataclass
class TaskStep(AgentStep):
    task: str


@dataclass
class SystemPromptStep(AgentStep):
    system_prompt: str


def format_prompt_with_tools(
    toolbox: Toolbox, prompt_template: str, tool_description_template: str
) -> str:
    tool_descriptions = toolbox.show_tool_descriptions(tool_description_template)
    prompt = prompt_template.replace("{{tool_descriptions}}", tool_descriptions)

    if "{{tool_names}}" in prompt:
        prompt = prompt.replace(
            "{{tool_names}}",
            ", ".join([f"'{tool_name}'" for tool_name in toolbox.tools.keys()]),
        )

    return prompt


def show_agents_descriptions(managed_agents: Dict):
    managed_agents_descriptions = """
You can also give requests to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'request', a long string explaning your request.
Given that this team member is a real human, you should be very verbose in your request.
Here is a list of the team members that you can call:"""
    for agent in managed_agents.values():
        managed_agents_descriptions += f"\n- {agent.name}: {agent.description}"
    return managed_agents_descriptions


def format_prompt_with_managed_agents_descriptions(
    prompt_template,
    managed_agents,
    agent_descriptions_placeholder: Optional[str] = None,
) -> str:
    if agent_descriptions_placeholder is None:
        agent_descriptions_placeholder = "{{managed_agents_descriptions}}"
    if agent_descriptions_placeholder not in prompt_template:
        print("PROMPT TEMPLLL", prompt_template)
        raise ValueError(
            f"Provided prompt template does not contain the managed agents descriptions placeholder '{agent_descriptions_placeholder}'"
        )
    if len(managed_agents.keys()) > 0:
        return prompt_template.replace(
            agent_descriptions_placeholder, show_agents_descriptions(managed_agents)
        )
    else:
        return prompt_template.replace(agent_descriptions_placeholder, "")


YELLOW_HEX = "#d4b702"


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).
    """

    def __init__(
        self,
        tools: Union[List[Tool], Toolbox],
        model: Callable[[List[Dict[str, str]]], str],
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        max_iterations: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbose: bool = False,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[Dict] = None,
        step_callbacks: Optional[List[Callable]] = None,
        planning_interval: Optional[int] = None,
    ):
        if system_prompt is None:
            system_prompt = CODE_SYSTEM_PROMPT
        if tool_parser is None:
            tool_parser = parse_json_tool_call
        self.agent_name = self.__class__.__name__
        self.model = model
        self.system_prompt_template = system_prompt
        self.tool_description_template = (
            tool_description_template
            if tool_description_template
            else DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
        self.max_iterations = max_iterations
        self.tool_parser = tool_parser
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state = {}

        self.managed_agents = {}
        if managed_agents is not None:
            self.managed_agents = {agent.name: agent for agent in managed_agents}

        if isinstance(tools, Toolbox):
            self._toolbox = tools
            if add_base_tools:
                self._toolbox.add_base_tools(
                    add_python_interpreter=(self.__class__ == ToolCallingAgent)
                )
        else:
            self._toolbox = Toolbox(tools, add_base_tools=add_base_tools)
        self._toolbox.add_tool(FinalAnswerTool())

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.logs = []
        self.task = None
        self.verbose = verbose
        self.monitor = Monitor(self.model)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

    @property
    def toolbox(self) -> Toolbox:
        """Get the toolbox currently available to the agent"""
        return self._toolbox

    def initialize_system_prompt(self):
        self.system_prompt = format_prompt_with_tools(
            self._toolbox,
            self.system_prompt_template,
            self.tool_description_template,
        )
        self.system_prompt = format_prompt_with_managed_agents_descriptions(
            self.system_prompt, self.managed_agents
        )

        return self.system_prompt

    def write_inner_memory_from_logs(
        self, summary_mode: Optional[bool] = False
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM.
        """
        memory = []
        for i, step_log in enumerate(self.logs):
            if isinstance(step_log, SystemPromptStep):
                if not summary_mode:
                    thought_message = {
                        "role": MessageRole.SYSTEM,
                        "content": step_log.system_prompt.strip(),
                    }
                    memory.append(thought_message)

            elif isinstance(step_log, PlanningStep):
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": "[FACTS LIST]:\n" + step_log.facts.strip(),
                }
                memory.append(thought_message)

                if not summary_mode:
                    thought_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": "[PLAN]:\n" + step_log.plan.strip(),
                    }
                    memory.append(thought_message)

            elif isinstance(step_log, TaskStep):
                task_message = {
                    "role": MessageRole.USER,
                    "content": "New task:\n" + step_log.task,
                }
                memory.append(task_message)

            elif isinstance(step_log, ActionStep):
                if step_log.llm_output is not None and not summary_mode:
                    thought_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": step_log.llm_output.strip(),
                    }
                    memory.append(thought_message)

                if step_log.tool_call is not None:
                    tool_call_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": str(
                            [
                                {
                                    "id": step_log.tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": step_log.tool_call.name,
                                        "arguments": step_log.tool_call.arguments,
                                    },
                                }
                            ]
                        ),
                    }
                    memory.append(tool_call_message)

                if step_log.tool_call is None and step_log.error is not None:
                    message_content = (
                        "Error:\n"
                        + str(step_log.error)
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                    tool_response_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": message_content,
                    }
                if step_log.tool_call is not None and (
                    step_log.error is not None or step_log.observations is not None
                ):
                    if step_log.error is not None:
                        message_content = (
                            "Error:\n"
                            + str(step_log.error)
                            + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                        )
                    elif step_log.observations is not None:
                        message_content = f"Observation:\n{step_log.observations}"
                    tool_response_message = {
                        "role": MessageRole.TOOL_RESPONSE,
                        "content": f"Call id: {(step_log.tool_call.id if getattr(step_log.tool_call, 'id') else 'call_0')}\n"
                        + message_content,
                    }
                    memory.append(tool_response_message)

        return memory

    def get_succinct_logs(self):
        return [
            {key: value for key, value in log.items() if key != "agent_memory"}
            for log in self.logs
        ]

    def extract_action(self, llm_output: str, split_token: str) -> Tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            llm_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = llm_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!"
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task) -> str:
        """
        This method provides a final answer to the task, based on the logs of the agent's interactions.
        """
        self.input_messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
            }
        ]
        self.input_messages += self.write_inner_memory_from_logs()[1:]
        self.input_messages += [
            {
                "role": MessageRole.USER,
                "content": f"Based on the above, please provide an answer to the following user request:\n{task}",
            }
        ]
        try:
            return self.model(self.input_messages)
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.toolbox).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.toolbox.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg)

        try:
            if isinstance(arguments, str):
                observation = available_tools[tool_name].__call__(
                    arguments, sanitize_inputs_outputs=True
                )
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                observation = available_tools[tool_name].__call__(
                    **arguments, sanitize_inputs_outputs=True
                )
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg)
            return observation
        except Exception as e:
            if tool_name in self.toolbox.tools:
                tool_description = get_tool_description_with_args(
                    available_tools[tool_name]
                )
                error_msg = (
                    f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following:\n{tool_description}"
                )
                raise AgentExecutionError(error_msg)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg)

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        single_step: bool = False,
        additional_args: Optional[Dict] = None,
    ):
        """
        Runs the agent for the given task.

        Args:
            task (`str`): The task to perform.
            stream (`bool`): Wether to run in a streaming way.
            reset (`bool`): Wether to reset the conversation or keep it going from previous run.
            single_step (`bool`): Should the agent run in one shot or multi-step fashion?
            additional_args (`dict`): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.initialize_system_prompt()
        system_prompt_step = SystemPromptStep(system_prompt=self.system_prompt)

        if reset:
            self.logs = []
            self.logs.append(system_prompt_step)
            self.monitor.reset()
        else:
            if len(self.logs) > 0:
                self.logs[0] = system_prompt_step
            else:
                self.logs.append(system_prompt_step)

        console.print(
            Panel(
                f"\n[bold]{self.task.strip()}\n",
                title="[bold]New run",
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                border_style=YELLOW_HEX,
                subtitle_align="left",
            )
        )

        self.logs.append(TaskStep(task=self.task))

        if single_step:
            step_start_time = time.time()
            step_log = ActionStep(start_time=step_start_time)
            step_log.end_time = time.time()
            step_log.duration = step_log.end_time - step_start_time

            # Run the agent's step
            result = self.step(step_log)
            return result

        if stream:
            return self.stream_run(self.task)
        else:
            return self.direct_run(self.task)

    def stream_run(self, task: str):
        """
        Runs the agent in streaming mode, yielding steps as they are executed: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            step_start_time = time.time()
            step_log = ActionStep(iteration=iteration, start_time=step_start_time)
            try:
                if (
                    self.planning_interval is not None
                    and iteration % self.planning_interval == 0
                ):
                    self.planning_step(
                        task, is_first_step=(iteration == 0), iteration=iteration
                    )
                console.print(
                    Rule(f"[bold]Step {iteration}", characters="━", style=YELLOW_HEX)
                )

                # Run one step!
                final_answer = self.step(step_log)
            except AgentError as e:
                step_log.error = e
            finally:
                step_log.end_time = time.time()
                step_log.duration = step_log.end_time - step_start_time
                self.logs.append(step_log)
                for callback in self.step_callbacks:
                    callback(step_log)
                iteration += 1
                yield step_log

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = ActionStep(error=AgentMaxIterationsError(error_message))
            self.logs.append(final_step_log)
            final_answer = self.provide_final_answer(task)
            console.print(Text(f"Final answer: {final_answer}"))
            final_step_log.action_output = final_answer
            final_step_log.end_time = time.time()
            final_step_log.duration = step_log.end_time - step_start_time
            for callback in self.step_callbacks:
                callback(final_step_log)
            yield final_step_log

        yield handle_agent_output_types(final_answer)

    def direct_run(self, task: str):
        """
        Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            step_start_time = time.time()
            step_log = ActionStep(iteration=iteration, start_time=step_start_time)
            try:
                if (
                    self.planning_interval is not None
                    and iteration % self.planning_interval == 0
                ):
                    self.planning_step(
                        task, is_first_step=(iteration == 0), iteration=iteration
                    )
                console.print(
                    Rule(f"[bold]Step {iteration}", characters="━", style=YELLOW_HEX)
                )

                # Run one step!
                final_answer = self.step(step_log)

            except AgentError as e:
                step_log.error = e
            finally:
                step_end_time = time.time()
                step_log.end_time = step_end_time
                step_log.duration = step_end_time - step_start_time
                self.logs.append(step_log)
                for callback in self.step_callbacks:
                    callback(step_log)
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = ActionStep(error=AgentMaxIterationsError(error_message))
            self.logs.append(final_step_log)
            final_answer = self.provide_final_answer(task)
            console.print(Text(f"Final answer: {final_answer}"))
            final_step_log.action_output = final_answer
            final_step_log.duration = 0
            for callback in self.step_callbacks:
                callback(final_step_log)

        return handle_agent_output_types(final_answer)

    def planning_step(self, task, is_first_step: bool, iteration: int):
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): The task to perform
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            iteration (`int`): The number of the current step, used as an indication for the LLM.
        """
        if is_first_step:
            message_prompt_facts = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS,
            }
            message_prompt_task = {
                "role": MessageRole.USER,
                "content": f"""Here is the task:
```
{task}
```
Now begin!""",
            }

            answer_facts = self.model([message_prompt_facts, message_prompt_task])

            message_system_prompt_plan = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_PLAN,
            }
            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN.format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(
                        self.tool_description_template
                    ),
                    managed_agents_descriptions=(
                        show_agents_descriptions(self.managed_agents)
                    ),
                    answer_facts=answer_facts,
                ),
            }
            answer_plan = self.model(
                [message_system_prompt_plan, message_user_prompt_plan],
                stop_sequences=["<end_plan>"],
            )

            final_plan_redaction = f"""Here is the plan of action that I will follow to solve the task:
```
{answer_plan}
```"""
            final_facts_redaction = f"""Here are the facts that I know so far:
```
{answer_facts}
```""".strip()
            self.logs.append(
                PlanningStep(plan=final_plan_redaction, facts=final_facts_redaction)
            )
            console.print(
                Rule("[bold]Initial plan", style="orange"), Text(final_plan_redaction)
            )
        else:  # update plan
            agent_memory = self.write_inner_memory_from_logs(
                summary_mode=False
            )  # This will not log the plan but will log facts

            # Redact updated facts
            facts_update_system_prompt = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS_UPDATE,
            }
            facts_update_message = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_FACTS_UPDATE,
            }
            facts_update = self.model(
                [facts_update_system_prompt] + agent_memory + [facts_update_message]
            )

            # Redact updated plan
            plan_update_message = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_PLAN_UPDATE.format(task=task),
            }
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN_UPDATE.format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(
                        self.tool_description_template
                    ),
                    managed_agents_descriptions=(
                        show_agents_descriptions(self.managed_agents)
                    ),
                    facts_update=facts_update,
                    remaining_steps=(self.max_iterations - iteration),
                ),
            }
            plan_update = self.model(
                [plan_update_message] + agent_memory + [plan_update_message_user],
                stop_sequences=["<end_plan>"],
            )

            # Log final facts and plan
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(
                task=task, plan_update=plan_update
            )
            final_facts_redaction = f"""Here is the updated list of the facts that I know:
```
{facts_update}
```"""
            self.logs.append(
                PlanningStep(plan=final_plan_redaction, facts=final_facts_redaction)
            )
            console.print(
                Rule("[bold]Updated plan", style="orange"), Text(final_plan_redaction)
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        system_prompt: Optional[str] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = TOOL_CALLING_SYSTEM_PROMPT
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            planning_interval=planning_interval,
            **kwargs,
        )

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.input_messages = agent_memory

        # Add new step in logs
        log_entry.agent_memory = agent_memory.copy()

        try:
            tool_name, tool_arguments, tool_call_id = self.model.get_tool_call(
                self.input_messages,
                available_tools=list(self.toolbox._tools.values()),
                stop_sequences=["Observation:"],
            )
        except Exception as e:
            raise AgentGenerationError(
                f"Error in generating tool call with model:\n{e}"
            )

        log_entry.tool_call = ToolCall(
            name=tool_name, arguments=tool_arguments, id=tool_call_id
        )

        # Execute
        console.print(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}"))
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
                console.print(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'."
                )
            else:
                final_answer = answer
                console.print(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}")
                )

            log_entry.action_output = final_answer
            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            observation_type = type(observation)
            if observation_type in [AgentImage, AgentAudio]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: observation naming could allow for different names of same type

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            console.print(f"Observations: {updated_information}")
            log_entry.observations = updated_information
            return None


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        use_e2b_executor: bool = False,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = CODE_SYSTEM_PROMPT
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )

        self.additional_authorized_imports = (
            additional_authorized_imports if additional_authorized_imports else []
        )
        if use_e2b_executor and len(self.managed_agents) > 0:
            raise Exception(
                f"You passed both {use_e2b_executor=} and some managed agents. Managed agents is not yet supported with remote code execution."
            )

        all_tools = {**self.toolbox.tools, **self.managed_agents}
        if use_e2b_executor:
            self.python_executor = E2BExecutor(
                self.additional_authorized_imports, list(all_tools.values())
            )
        else:
            self.python_executor = LocalPythonInterpreter(
                self.additional_authorized_imports, all_tools
            )
        self.authorized_imports = list(
            set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        if "{{authorized_imports}}" not in self.system_prompt:
            raise AgentError(
                "Tag '{{authorized_imports}}' should be provided in the prompt."
            )
        self.system_prompt = self.system_prompt.replace(
            "{{authorized_imports}}", str(self.authorized_imports)
        )

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.input_messages = agent_memory.copy()

        # Add new step in logs
        log_entry.agent_memory = agent_memory.copy()

        try:
            additional_args = (
                {"grammar": self.grammar} if self.grammar is not None else {}
            )
            llm_output = self.model(
                self.input_messages,
                stop_sequences=["<end_action>", "Observation:"],
                **additional_args,
            )
            log_entry.llm_output = llm_output
        except Exception as e:
            console.print_exception()
            raise AgentGenerationError(f"Error in generating model output:\n{e}")

        if self.verbose:
            console.print(
                Group(
                    Rule(
                        "[italic]Output message of the LLM:",
                        align="left",
                        style="orange",
                    ),
                    Syntax(
                        llm_output,
                        lexer="markdown",
                        theme="github-dark",
                        word_wrap=True,
                    ),
                )
            )

        # Parse
        try:
            code_action = parse_code_blob(llm_output)
        except Exception as e:
            console.print_exception()
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)

        log_entry.tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.logs)}",
        )

        # Execute
        console.print(
            Panel(
                Syntax(
                    code_action,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                    line_numbers=True,
                ),
                title="[bold]Executing this code:",
                title_align="left",
            )
        )
        observation = ""
        try:
            output, execution_logs = self.python_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation += "Execution logs:\n" + execution_logs
        except Exception as e:
            if isinstance(e, SyntaxError):
                error_msg = (
                    f"Code execution failed on line {e.lineno} due to: {type(e).__name__}\n"
                    f"{e.text}"
                    f"{' ' * (e.offset or 0)}^\n"
                    f"Error: {str(e)}"
                )
            else:
                error_msg = f"Code execution failed: {str(e)}"
            raise AgentExecutionError(error_msg)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        log_entry.observations = observation

        is_final_answer = False
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                is_final_answer = True
                break

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        console.print(Group(*execution_outputs_console))
        log_entry.action_output = output
        return output if is_final_answer else None


class ManagedAgent:
    def __init__(
        self,
        agent,
        name,
        description,
        additional_prompting: Optional[str] = None,
        provide_run_summary: bool = False,
        managed_agent_prompt: Optional[str] = None,
    ):
        self.agent = agent
        self.name = name
        self.description = description
        self.additional_prompting = additional_prompting
        self.provide_run_summary = provide_run_summary
        self.managed_agent_prompt = (
            managed_agent_prompt if managed_agent_prompt else MANAGED_AGENT_PROMPT
        )

    def write_full_task(self, task):
        """Adds additional prompting for the managed agent, like 'add more detail in your answer'."""
        full_task = self.managed_agent_prompt.format(name=self.name, task=task)
        if self.additional_prompting:
            full_task = full_task.replace(
                "\n{{additional_prompting}}", self.additional_prompting
            ).strip()
        else:
            full_task = full_task.replace("\n{{additional_prompting}}", "").strip()
        return full_task

    def __call__(self, request, **kwargs):
        full_task = self.write_full_task(request)
        output = self.agent.run(full_task, **kwargs)
        if self.provide_run_summary:
            answer = (
                f"Here is the final answer from your managed agent '{self.name}':\n"
            )
            answer += str(output)
            answer += f"\n\nFor more detail, find below a summary of this agent's work:\nSUMMARY OF WORK FROM AGENT '{self.name}':\n"
            for message in self.agent.write_inner_memory_from_logs(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += f"\nEND OF SUMMARY OF WORK FROM AGENT '{self.name}'."
            return answer
        else:
            return output


class ChainedPromptAgent(MultiStepAgent):
    """
    Agent that implements prompt chaining workflow by breaking tasks into sequential subtasks
    with validation gates between steps.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        subtask_definitions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        validation_functions: Optional[Dict[str, Callable]] = None,
        **kwargs
    ):
        """
        Initialize ChainedPromptAgent.

        Args:
            tools: List of tools available to the agent
            model: Language model callable
            subtask_definitions: List of dicts defining sequential subtasks, each containing:
                - name: Name of the subtask
                - prompt_template: Template for the subtask prompt
                - requires_validation: Whether this step needs validation
            validation_functions: Dict mapping subtask names to their validation functions
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(tools=tools, model=model, system_prompt=system_prompt, **kwargs)
        self.subtask_definitions = subtask_definitions
        self.validation_functions = validation_functions or {}
        self.current_subtask_index = 0
        self.subtask_outputs = {}
        
    def validate_subtask_output(self, subtask_name: str, output: Any) -> Tuple[bool, str]:
        """
        Validate the output of a subtask using its validation function.
        
        Args:
            subtask_name: Name of the subtask to validate
            output: Output to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if subtask_name not in self.validation_functions:
            return True, ""
            
        validation_fn = self.validation_functions[subtask_name]
        try:
            is_valid = validation_fn(output)
            return is_valid, "" if is_valid else f"Validation failed for {subtask_name}"
        except Exception as e:
            return False, f"Validation error for {subtask_name}: {str(e)}"

    def format_subtask_prompt(self, subtask_def: Dict[str, Any]) -> str:
        """
        Format the prompt template for a subtask using previous outputs.
        
        Args:
            subtask_def: Subtask definition dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt_template = subtask_def["prompt_template"]
        return prompt_template.format(
            **self.subtask_outputs,
            task=self.task
        )

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Execute one step in the prompt chain.
        
        Returns None if more steps are needed, or final output if chain is complete.
        """
        if self.current_subtask_index >= len(self.subtask_definitions):
            # Chain complete - return final output
            return self.subtask_outputs[self.subtask_definitions[-1]["name"]]
            
        current_subtask = self.subtask_definitions[self.current_subtask_index]
        subtask_name = current_subtask["name"]
        
        # Format prompt using previous outputs
        formatted_prompt = self.format_subtask_prompt(current_subtask)
        
        # Add to agent memory
        agent_memory = self.write_inner_memory_from_logs()
        agent_memory.append({
            "role": MessageRole.USER,
            "content": formatted_prompt
        })
        
        # Get model output for this subtask
        try:
            subtask_output = self.model(agent_memory)
            log_entry.llm_output = subtask_output
        except Exception as e:
            raise AgentGenerationError(f"Error in subtask {subtask_name}: {str(e)}")
            
        # Validate output if required
        if current_subtask.get("requires_validation", False):
            is_valid, error_msg = self.validate_subtask_output(subtask_name, subtask_output)
            if not is_valid:
                raise AgentExecutionError(error_msg)
                
        # Store output and advance to next subtask
        self.subtask_outputs[subtask_name] = subtask_output
        self.current_subtask_index += 1
        
        # Log the completion of this subtask
        log_entry.observations = f"Completed subtask: {subtask_name}"
        return None


class RouteDefinition:
    """Defines a route with its classifier and handler"""
    def __init__(
        self,
        name: str,
        description: str,
        handler: Union[MultiStepAgent, Callable],
        classifier_prompt: Optional[str] = None,
        classifier_function: Optional[Callable] = None,
    ):
        """
        Initialize a route definition.
        
        Args:
            name: Unique identifier for this route
            description: Description of what kind of tasks this route handles
            handler: Agent or function to handle tasks of this type
            classifier_prompt: Prompt template for LLM-based classification
            classifier_function: Optional function for rule-based classification
        """
        self.name = name
        self.description = description
        self.handler = handler
        self.classifier_prompt = classifier_prompt
        self.classifier_function = classifier_function


class RoutingAgent(MultiStepAgent):
    """
    Agent that implements routing workflow by classifying inputs and directing them 
    to specialized handlers.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        routes: List[RouteDefinition],
        system_prompt: Optional[str] = None,
        use_llm_classifier: bool = True,
        fallback_route: Optional[RouteDefinition] = None,
        **kwargs
    ):
        """
        Initialize RoutingAgent.

        Args:
            tools: List of tools available to the agent
            model: Language model callable
            routes: List of RouteDefinition objects defining possible routes
            system_prompt: Optional custom system prompt
            use_llm_classifier: Whether to use LLM for classification
            fallback_route: Route to use when classification fails
            **kwargs: Additional arguments passed to parent class
        """
        if system_prompt is None:
            route_descriptions = "\n".join(
                f"- {route.name}: {route.description}" for route in routes
            )
            system_prompt = f"""You are a routing agent that classifies tasks and directs them to the appropriate specialized handler.
Available routes:
{route_descriptions}

Your job is to analyze each task and determine which route would handle it best.
Respond with just the route name, or "unknown" if none of the routes seem appropriate."""

        super().__init__(tools=tools, model=model, system_prompt=system_prompt, **kwargs)
        self.routes = {route.name: route for route in routes}
        self.use_llm_classifier = use_llm_classifier
        self.fallback_route = fallback_route
        self.selected_route = None
        
    def classify_task(self, task: str) -> str:
        """
        Classify the task using either LLM or rule-based classification.
        
        Args:
            task: Task to classify
            
        Returns:
            Name of the selected route
        """
        # Try rule-based classification first
        for route in self.routes.values():
            if route.classifier_function is not None:
                try:
                    if route.classifier_function(task):
                        return route.name
                except Exception as e:
                    console.print(f"Error in classifier function for {route.name}: {e}")
                    continue
        
        # Fall back to LLM classification if enabled
        if self.use_llm_classifier:
            try:
                agent_memory = self.write_inner_memory_from_logs()
                classification = self.model(agent_memory).strip().lower()
                if classification in self.routes:
                    return classification
            except Exception as e:
                console.print(f"Error in LLM classification: {e}")
        
        # Use fallback route if available
        if self.fallback_route is not None:
            return self.fallback_route.name
            
        raise AgentExecutionError("Could not classify task and no fallback route defined")

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Execute one step in the routing workflow.
        First step classifies the task, second step handles it.
        """
        # First step: classify if not already classified
        if self.selected_route is None:
            route_name = self.classify_task(self.task)
            self.selected_route = self.routes[route_name]
            log_entry.observations = f"Classified task to route: {route_name}"
            return None
            
        # Second step: handle the task with the selected route
        try:
            if isinstance(self.selected_route.handler, MultiStepAgent):
                result = self.selected_route.handler.run(
                    self.task,
                    reset=True,
                    stream=False
                )
            else:
                result = self.selected_route.handler(self.task)
                
            log_entry.observations = f"Completed task using route: {self.selected_route.name}"
            log_entry.action_output = result
            return result
            
        except Exception as e:
            raise AgentExecutionError(
                f"Error in route handler {self.selected_route.name}: {str(e)}"
            )


class ParallelizationType(Enum):
    """Type of parallelization to use"""
    SECTIONING = "sectioning"
    VOTING = "voting"


@dataclass
class Section:
    """Definition of a section for parallel processing"""
    name: str
    description: str
    prompt_template: str
    weight: float = 1.0
    
    def format_prompt(self, task: str) -> str:
        """Format the section's prompt template with the task"""
        return self.prompt_template.format(task=task)


@dataclass
class VotingConfig:
    """Configuration for voting-based parallelization"""
    num_voters: int
    threshold: float
    aggregation_function: Callable[[List[Any]], Any]
    diversity_prompt: Optional[str] = None


class ParallelAgent(MultiStepAgent, Generic[T]):
    """
    Agent that implements parallel processing through either sectioning or voting.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        parallelization_type: ParallelizationType,
        sections: Optional[List[Section]] = None,
        voting_config: Optional[VotingConfig] = None,
        system_prompt: Optional[str] = None,
        max_workers: int = 4,
        **kwargs
    ):
        """
        Initialize ParallelAgent.

        Args:
            tools: List of tools available to the agent
            model: Language model callable
            parallelization_type: Type of parallelization to use
            sections: List of sections for sectioning parallelization
            voting_config: Configuration for voting parallelization
            system_prompt: Optional custom system prompt
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments passed to parent class
        """
        if system_prompt is None:
            if parallelization_type == ParallelizationType.SECTIONING:
                if not sections:
                    raise ValueError("Sections must be provided for sectioning parallelization")
                sections_desc = "\n".join(
                    f"- {section.name}: {section.description}" for section in sections
                )
                system_prompt = f"""You are a specialized agent focusing on one aspect of a larger task.
Available sections:
{sections_desc}

Your job is to analyze your specific aspect thoroughly and provide a detailed response."""
            else:  # VOTING
                if not voting_config:
                    raise ValueError("Voting config must be provided for voting parallelization")
                system_prompt = """You are a voting agent providing an independent perspective on a task.
Analyze the task thoroughly and provide your vote/opinion."""

        super().__init__(tools=tools, model=model, system_prompt=system_prompt, **kwargs)
        self.parallelization_type = parallelization_type
        self.sections = sections
        self.voting_config = voting_config
        self.max_workers = max_workers
        self.results: Dict[str, Any] = {}
        
    def process_section(self, section: Section, task: str) -> Tuple[str, Any]:
        """Process a single section"""
        try:
            agent_memory = self.write_inner_memory_from_logs()
            formatted_prompt = section.format_prompt(task)
            agent_memory.append({
                "role": MessageRole.USER,
                "content": formatted_prompt
            })
            result = self.model(agent_memory)
            return section.name, result
        except Exception as e:
            raise AgentExecutionError(f"Error processing section {section.name}: {str(e)}")

    def process_vote(self, task: str, voter_id: int) -> Any:
        """Process a single vote"""
        try:
            agent_memory = self.write_inner_memory_from_logs()
            
            # Add diversity to prompting if configured
            if self.voting_config.diversity_prompt:
                prompt = self.voting_config.diversity_prompt.format(
                    voter_id=voter_id,
                    task=task
                )
            else:
                prompt = task
                
            agent_memory.append({
                "role": MessageRole.USER,
                "content": prompt
            })
            
            return self.model(agent_memory)
        except Exception as e:
            raise AgentExecutionError(f"Error processing vote {voter_id}: {str(e)}")

    def aggregate_sections(self) -> T:
        """Aggregate results from different sections"""
        if not all(section.name in self.results for section in self.sections):
            raise AgentExecutionError("Not all sections have been processed")
            
        # Weight and combine section results
        weighted_results = []
        for section in self.sections:
            result = self.results[section.name]
            weighted_results.append((result, section.weight))
            
        # Create a formatted response combining all sections
        combined_response = "Combined analysis from all sections:\n\n"
        for section in self.sections:
            combined_response += f"[{section.name}]:\n{self.results[section.name]}\n\n"
            
        return combined_response

    def aggregate_votes(self, votes: List[Any]) -> T:

        if len(votes) < self.voting_config.num_voters:
            raise AgentExecutionError("Not enough votes collected")
            
        try:
            return self.voting_config.aggregation_function(votes)
        except Exception as e:
            raise AgentExecutionError(f"Error aggregating votes: {str(e)}")

    def step(self, log_entry: ActionStep) -> Union[None, T]:
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                if self.parallelization_type == ParallelizationType.SECTIONING:
                    # Process all sections in parallel
                    future_to_section = {
                        executor.submit(self.process_section, section, self.task): section
                        for section in self.sections
                    }
                    
                    for future in as_completed(future_to_section):
                        section = future_to_section[future]
                        try:
                            section_name, result = future.result()
                            self.results[section_name] = result
                        except Exception as e:
                            raise AgentExecutionError(
                                f"Error in section {section.name}: {str(e)}"
                            )
                    
                    final_result = self.aggregate_sections()
                    
                else:  # VOTING
                    # Collect votes in parallel
                    futures = [
                        executor.submit(self.process_vote, self.task, i)
                        for i in range(self.voting_config.num_voters)
                    ]
                    
                    votes = []
                    for future in as_completed(futures):
                        try:
                            vote = future.result()
                            votes.append(vote)
                        except Exception as e:
                            raise AgentExecutionError(f"Error collecting vote: {str(e)}")
                    
                    final_result = self.aggregate_votes(votes)
                
            log_entry.action_output = final_result
            return final_result
            
        except Exception as e:
            raise AgentExecutionError(f"Error in parallel processing: {str(e)}")


class WorkerDefinition:
    """Defines a worker with its capabilities and specialization"""
    def __init__(
        self,
        name: str,
        description: str,
        model: Callable,
        tools: List[Tool],
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt


class OrchestratorAgent(MultiStepAgent):
    """
    Agent that implements orchestrator-workers workflow by dynamically delegating tasks
    to specialized workers and synthesizing their results.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        workers: List[WorkerDefinition],
        system_prompt: Optional[str] = None,
        max_delegation_depth: int = 3,
        **kwargs
    ):
        if system_prompt is None:
            workers_desc = "\n".join(
                f"- {worker.name}: {worker.description}" for worker in workers
            )
            system_prompt = f"""You are an orchestrator agent that breaks down complex tasks and delegates them to specialized workers.
Available workers:
{workers_desc}

Your job is to:
1. Analyze the task and break it into subtasks
2. Assign each subtask to appropriate workers
3. Synthesize their results into a final solution

Respond with a JSON object containing:
{{"subtasks": [
    {{"name": "subtask_name", "worker": "worker_name", "description": "subtask_description"}}
]}}"""

        super().__init__(tools=tools, model=model, system_prompt=system_prompt, **kwargs)
        self.workers = {worker.name: worker for worker in workers}
        self.max_delegation_depth = max_delegation_depth
        self.current_depth = 0
        self.subtasks = []
        self.results = {}

    def delegate_subtask(self, subtask: Dict[str, str]) -> Any:
        """Delegate a subtask to a worker and get result"""
        worker = self.workers[subtask["worker"]]
        worker_agent = MultiStepAgent(
            tools=worker.tools,
            model=worker.model,
            system_prompt=worker.system_prompt
        )
        return worker_agent.run(subtask["description"])

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Execute one step in the orchestration workflow:
        1. First step: Break down task into subtasks
        2. Middle steps: Execute subtasks with workers
        3. Final step: Synthesize results
        """
        try:
            # First step: Break down task if not done
            if not self.subtasks:
                agent_memory = self.write_inner_memory_from_logs()
                breakdown = self.model(agent_memory)
                try:
                    parsed_breakdown = json.loads(breakdown)
                    self.subtasks = parsed_breakdown["subtasks"]
                except Exception as e:
                    raise AgentParsingError(f"Error parsing task breakdown: {str(e)}")
                
                log_entry.observations = f"Broke down task into {len(self.subtasks)} subtasks"
                return None

            # Middle steps: Execute remaining subtasks
            if len(self.results) < len(self.subtasks):
                current_subtask = self.subtasks[len(self.results)]
                result = self.delegate_subtask(current_subtask)
                self.results[current_subtask["name"]] = result
                
                log_entry.observations = f"Completed subtask: {current_subtask['name']}"
                return None

            # Final step: Synthesize results
            synthesis_prompt = f"""Task: {self.task}

Results from subtasks:
"""
            for subtask in self.subtasks:
                synthesis_prompt += f"\n[{subtask['name']}]:\n{self.results[subtask['name']]}"
            
            synthesis_prompt += "\n\nSynthesize these results into a final solution."
            
            agent_memory = self.write_inner_memory_from_logs()
            agent_memory.append({
                "role": MessageRole.USER,
                "content": synthesis_prompt
            })
            
            final_result = self.model(agent_memory)
            log_entry.action_output = final_result
            return final_result

        except Exception as e:
            raise AgentExecutionError(f"Error in orchestration: {str(e)}")


class EvaluatorOptimizerAgent(MultiStepAgent):
    """
    Agent that implements evaluator-optimizer workflow with one LLM generating responses
    and another providing evaluation and feedback in a loop.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        optimizer_model: Callable,
        evaluator_model: Callable,
        evaluation_criteria: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        evaluator_prompt: Optional[str] = None,
        max_iterations: int = 3,
        improvement_threshold: float = 0.1,
        **kwargs
    ):
        if system_prompt is None:
            system_prompt = """You are an optimizer agent that generates and improves responses based on evaluation feedback."""
            
        if evaluator_prompt is None:
            criteria_desc = "\n".join(
                f"- {criterion['name']}: {criterion['description']}"
                for criterion in evaluation_criteria
            )
            evaluator_prompt = f"""You are an evaluator agent that assesses responses based on specific criteria.
Evaluation criteria:
{criteria_desc}

Analyze the response and provide:
1. A score (0-1) for each criterion
2. Specific feedback for improvement
3. Overall score (0-1)
4. Whether significant improvement is likely with another iteration

Respond in JSON format:
{{
    "criteria_scores": {{"criterion_name": score}},
    "feedback": "detailed feedback",
    "overall_score": overall_score,
    "continue_iteration": boolean
}}"""

        super().__init__(
            tools=tools,
            model=optimizer_model,
            system_prompt=system_prompt,
            **kwargs
        )
        self.optimizer_model = optimizer_model
        self.evaluator_model = evaluator_model
        self.evaluation_criteria = evaluation_criteria
        self.evaluator_prompt = evaluator_prompt
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.current_iteration = 0
        self.best_response = None
        self.best_score = 0
        self.previous_score = 0
        
    def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Get evaluation and feedback for a response"""
        evaluation_prompt = f"""Task: {self.task}

Response to evaluate:
{response}

Provide evaluation and feedback."""

        agent_memory = [{
            "role": MessageRole.SYSTEM,
            "content": self.evaluator_prompt
        }, {
            "role": MessageRole.USER,
            "content": evaluation_prompt
        }]
        
        try:
            evaluation = self.evaluator_model(agent_memory)
            return json.loads(evaluation)
        except Exception as e:
            raise AgentExecutionError(f"Error in evaluation: {str(e)}")

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Execute one step in the evaluation-optimization loop:
        1. Generate/improve response
        2. Evaluate response
        3. Decide whether to continue iteration
        """
        try:
            # Check iteration limit
            if self.current_iteration >= self.max_iterations:
                return self.best_response

            # Generate/improve response
            agent_memory = self.write_inner_memory_from_logs()
            if self.current_iteration > 0:
                improvement_prompt = f"""Previous response:
{self.best_response}

Evaluation feedback:
{self.last_evaluation['feedback']}

Please improve the response based on this feedback."""
                
                agent_memory.append({
                    "role": MessageRole.USER,
                    "content": improvement_prompt
                })
            
            current_response = self.optimizer_model(agent_memory)
            
            # Evaluate response
            evaluation = self.evaluate_response(current_response)
            self.last_evaluation = evaluation
            current_score = evaluation["overall_score"]
            
            # Update best response if improved
            if current_score > self.best_score:
                self.best_response = current_response
                self.best_score = current_score
            
            # Log progress
            log_entry.observations = (
                f"Iteration {self.current_iteration + 1}: "
                f"Score = {current_score:.2f} "
                f"(Best = {self.best_score:.2f})"
            )
            
            # Decide whether to continue
            score_improvement = current_score - self.previous_score
            should_continue = (
                evaluation["continue_iteration"] and
                score_improvement >= self.improvement_threshold
            )
            
            self.previous_score = current_score
            self.current_iteration += 1
            
            if should_continue:
                return None
            else:
                log_entry.action_output = self.best_response
                return self.best_response

        except Exception as e:
            raise AgentExecutionError(f"Error in evaluation-optimization: {str(e)}")


__all__ = [
    "ManagedAgent",
    "MultiStepAgent",
    "ChainedPromptAgent",
    "RoutingAgent",
    "RouteDefinition",
    "OrchestratorAgent",
    "WorkerDefinition",
    "EvaluatorOptimizerAgent"
]
