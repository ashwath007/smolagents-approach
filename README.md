# AI Agents Guide

This guide explains diffrent types of AI agents approach from Antropic. Each agent is designed for specific use cases and can help you solve various problems.

## MultiStepAgent

This is the base agent that all other agents inherit from. It works by:
- Taking a task
- Breaking it down into steps
- Using tools to complete each step
- Keeping track of what its done

Its like having a worker who breaks down a big job into smaller tasks and completes them one by one.

## ManagedAgent

Think of this as a supervisor for other agents. It:
- Has a name and description
- Can handle specific types of tasks
- Provides additional instructions to other agents
- Can give summaries of what other agents have done

Its useful when you need to coordinate multiple agents working together.

## ChainedPromptAgent

This agent works like an assembly line. It:
- Has predefined steps that must be done in order
- Validates the output of each step
- Only moves forward when a step is completed successfully
- Uses the output of previous steps as input for next steps

Perfect for tasks that need to be done in a specific sequence!

## RoutingAgent and RouteDefinition

These work together like a traffic controller:

RouteDefinition:
- Defines where different types of tasks should go
- Includes rules for classifying tasks
- Specifies which agent should handle each type of task

RoutingAgent:
- Looks at incoming tasks
- Decides which route they should take
- Sends them to the right handler
- Can use either rules or AI to make decisions

## OrchestratorAgent and WorkerDefinition

Think of this as a project manager with specialized workers:

WorkerDefinition:
- Defines what each worker is good at
- Specifies what tools they can use
- Has instructions for how they should work

OrchestratorAgent:
- Breaks down big tasks into smaller pieces
- Assigns work to the right specialists
- Combines everyones work into final result
- Makes sure everything works together

## EvaluatorOptimizerAgent

This is like having a writer and an editor working together:
- One part generates solutions
- Another part evaluates and suggests improvements
- They work in loops to make the solution better
- Stops when the solution is good enough or after maximum tries

## When to Use Each Agent?

- Use **MultiStepAgent** for basic step-by-step tasks
- Use **ManagedAgent** when you need to supervise other agents
- Use **ChainedPromptAgent** for tasks that need strict order
- Use **RoutingAgent** when you need to sort tasks to different handlers
- Use **OrchestratorAgent** for complex projects needing multiple specialists
- Use **EvaluatorOptimizerAgent** when you need high-quality output with feedback loops

## Example Usage

```python
from agents import MultiStepAgent, RoutingAgent

# Create a basic agent
agent = MultiStepAgent(tools=my_tools, model=my_model)

# Run a task
result = agent.run("Calculate the square root of 16")
```

Remember, each agent is designed for specific use cases, so choose the one that best matches your needs!



## Credits

This project is inspired by and builds upon concepts from [SmoLAGents](https://github.com/huggingface/smolagents), a lightweight framework for building language agents. We extend our gratitude to the SmoLAGents team for their foundational work in this space.

