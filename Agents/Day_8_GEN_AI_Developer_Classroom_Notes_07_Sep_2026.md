# Gen-AI Developer Classroom Notes - 07/Sep/2026

## Introduction to LangGraph: Building Custom AI Agents

**Reference Blog**: [Direct AI Blog - Day 8: Introduction to LangGraph](https://directai.blog/2026/09/07/gen-ai-developer-classroom-notes-07-sep-2026/)

---

## Learning Objectives

By the end of this guide, you will understand:
- What LangGraph is and why it's important for AI agents
- The core concepts of graphs in LangGraph
- How to create your first LangGraph agent
- Step-by-step setup and implementation
- How to visualize your agent workflows

---

## What is an Agent in LangGraph?

### Understanding the Basics

An agent created with `create_agent` is essentially a **graph** at its core. This means:

- **Agent = Graph**: Every agent you build is a graph structure
- **LangGraph**: The framework that makes these graphs possible
- **Custom Workflows**: LangGraph allows you to create custom agents with your own workflows

### Why Use Graphs for Agents?

```
Traditional Agent Approach:
User Input → LLM → Response

Graph-Based Agent Approach:
User Input → Graph (Multiple Steps) → Response
```

Graphs allow for:
- **Complex workflows**: Multiple steps and decisions
- **Flexible routing**: Different paths based on conditions
- **State management**: Keep track of data across steps
- **Visual debugging**: See how your agent processes information

---

## LangGraph Core Concepts

### 1. Graph Terminology

LangGraph uses standard graph terminology:

```
┌─────────────────────────────────────────────────────────────┐
│                    LANGGRAPH TERMINOLOGY                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐    │
│  │   NODE   │────────▶│   NODE   │────────▶│   NODE   │    │
│  │ (Add)    │   EDGE  │ (Sub)    │   EDGE  │ (Mul)    │    │
│  └──────────┘         └──────────┘         └──────────┘    │
│                                                              │
│  NODES: Where processing happens                            │
│  EDGES: Define the direction of flow                       │
│  STATE: Data being processed by the graph                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Key Components Explained

#### **Nodes**
- **What they are**: Processing units where work happens
- **Example**: A function that adds two numbers
- **Purpose**: Each node performs a specific task

#### **Edges**
- **What they are**: Connections between nodes
- **Example**: Connection from "add" node to "subtract" node
- **Purpose**: Define the flow direction in your graph

#### **State**
- **What it is**: Data that flows through the graph
- **Example**: Numbers to be processed, intermediate results
- **Purpose**: Maintains information across nodes

---

## Setting Up Your LangGraph Project

### Step-by-Step Setup Guide

#### Step 1: Create a New Project Folder

```bash
mkdir hello-langgraph
cd hello-langgraph
```

#### Step 2: Initialize with UV

```bash
uv init .
```

**What this does**:
- Creates a new Python project
- Sets up project structure
- Creates `pyproject.toml` file
- Initializes a virtual environment

#### Step 3: Add Required Packages

```bash
uv add langgraph
uv add langchain-google-genai
uv add python-dotenv
```

**What each package does**:
- **langgraph**: Core framework for building graph-based agents
- **langchain-google-genai**: Integration with Google AI models
- **python-dotenv**: Environment variable management for API keys

#### Step 4: Add Visualization Support (Optional but Recommended)

```bash
uv add "langgraph-cli[inmem]"
uv pip install colorama
```

**What this does**:
- Enables visual representation of your graph
- Provides color-coded output for better debugging
- Helps you understand agent workflows

---

## Understanding State in LangGraph

### What is State?

State is the data that flows through your graph. It can be:

1. **TypedDict**: A dictionary with type hints
2. **Pydantic Model**: A data validation model
3. **Custom Class**: Your own class definition

### State Example

```python
from typing import TypedDict

class OperationsState(TypedDict):
    a: int              # First number
    b: int              # Second number
    sum: None|int       # Sum result (initially None)
    product: None|int   # Product result (initially None)
    diff: None|int      # Difference result (initially None)
```

**Key Points**:
- State holds all data needed for processing
- Each node can read and modify the state
- State flows from node to node through edges

---

## Building Your First LangGraph Agent

### Complete Implementation Example

Here's a complete, working example of a LangGraph agent that performs mathematical operations:

```python
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from typing import TypedDict
import time

# Define the state structure
class OperationsState(TypedDict):
    a: int
    b: int
    sum: None|int
    product: None|int
    diff: None|int

# Define node functions
def add(state: OperationsState) -> OperationsState:
    """Add two numbers"""
    time.sleep(2)  # Simulate processing time
    state['sum'] = state['a'] + state['b']
    return state

def sub(state: OperationsState) -> OperationsState:
    """Subtract two numbers"""
    time.sleep(2)  # Simulate processing time
    state['diff'] = state['a'] - state['b']
    return state

def mul(state: OperationsState) -> OperationsState:
    """Multiply two numbers"""
    time.sleep(2)  # Simulate processing time
    state['product'] = state['a'] * state['b']
    return state

# Create the graph
state_graph = StateGraph(OperationsState)

# Add nodes to the graph
state_graph.add_node("add", add)
state_graph.add_node("sub", sub)
state_graph.add_node("mul", mul)

# Define edges (connections between nodes)
state_graph.add_edge(START, "add")
state_graph.add_edge("add", "sub")
state_graph.add_edge("sub", "mul")
state_graph.add_edge("mul", END)

# Compile the graph
graph = state_graph.compile()

# Run the graph
if __name__ == "__main__":
    result = graph.invoke(OperationsState(a=5, b=4))
    print(result)
```

### Understanding the Code Flow

```
┌─────────────────────────────────────────────────────────────┐
│              GRAPH EXECUTION FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  START                                                       │
│   │                                                          │
│   ▼                                                          │
│  ┌──────────┐                                                │
│  │   ADD    │  Input: a=5, b=4                              │
│  │  Node    │  Output: sum=9                                │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │   SUB    │  Input: a=5, b=4                              │
│  │  Node    │  Output: diff=1                               │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │   MUL    │  Input: a=5, b=4                              │
│  │  Node    │  Output: product=20                           │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  END                                                         │
│                                                              │
│  Final Result: {a: 5, b: 4, sum: 9, diff: 1, product: 20}    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Visualizing Your Graph

### Setting Up Visualization

To make your graph executions visual, you need to:

#### 1. Create `langgraph.json` Configuration File

Create a file named `langgraph.json` in your project root:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent_1": "./hello_agent_1.py:graph"
  }
}
```

**What this does**:
- Tells LangGraph where to find your graph
- Specifies the Python file and graph variable name
- Enables the visualization server

#### 2. Run the Visualization Server

```bash
langgraph dev
```

**What this does**:
- Starts a local web server
- Provides a visual interface for your graph
- Shows real-time execution flow
- Helps with debugging and understanding

---

## Key Concepts Summary

### Graph Structure

```
Graph = Nodes + Edges + State

Nodes: Processing units (functions)
Edges: Connections (flow direction)
State: Data (information flow)
```

### Execution Flow

```
Input State → Node 1 → Node 2 → Node 3 → Output State
```

### Benefits of LangGraph

1. **Visual Debugging**: See how your agent processes information
2. **Complex Workflows**: Handle multi-step processes easily
3. **State Management**: Keep track of data across operations
4. **Custom Agents**: Build agents tailored to your needs
5. **Scalability**: Easy to add new nodes and edges

---

## Common Use Cases for LangGraph

### 1. Sequential Processing
- Data processing pipelines
- Multi-step analysis
- Workflow automation

### 2. Conditional Routing
- Decision-based workflows
- Error handling paths
- Dynamic task selection

### 3. Parallel Processing
- Multiple operations simultaneously
- Data aggregation
- Distributed computing

### 4. Human-in-the-Loop
- Approval workflows
- Interactive processes
- Collaborative decision making

---

## Best Practices

### 1. Start Simple
- Begin with basic graphs
- Add complexity gradually
- Test each node independently

### 2. Clear State Design
- Define state structure clearly
- Use type hints for validation
- Keep state minimal and focused

### 3. Modular Node Functions
- Each node should do one thing well
- Keep functions short and clear
- Add descriptive docstrings

### 4. Error Handling
- Handle exceptions in nodes
- Provide meaningful error messages
- Consider fallback paths

### 5. Visualization
- Use `langgraph dev` for debugging
- Understand your graph flow
- Document complex workflows

---

## Troubleshooting Common Issues

### Issue 1: Graph Not Compiling
**Solution**: Check that all nodes are properly defined and connected with edges.

### Issue 2: State Not Updating
**Solution**: Ensure nodes return the updated state and use correct state keys.

### Issue 3: Visualization Not Working
**Solution**: Verify `langgraph.json` configuration and ensure all dependencies are installed.

### Issue 4: Nodes Not Executing in Order
**Solution**: Check edge definitions and ensure proper START and END connections.

---

## Next Steps

### 1. Experiment with Different States
- Try different state structures
- Use Pydantic models for validation
- Add more complex data types

### 2. Build Complex Graphs
- Add conditional edges
- Implement parallel processing
- Create multi-agent systems

### 3. Integrate with LLMs
- Add LLM-powered nodes
- Implement tool calling
- Build conversational agents

### 4. Deploy Your Agents
- Set up production environments
- Add monitoring and logging
- Implement error handling

---

## Summary

LangGraph provides a powerful framework for building custom AI agents using graph-based workflows. Key takeaways:

1. **Agents are Graphs**: Every agent is fundamentally a graph structure
2. **Three Core Components**: Nodes (processing), Edges (flow), State (data)
3. **Visual Development**: Use `langgraph dev` to visualize and debug
4. **Flexible Architecture**: Build custom workflows for your specific needs
5. **Beginner-Friendly**: Start simple and gradually add complexity

The graph-based approach makes it easier to understand, debug, and scale your AI agents compared to traditional linear approaches.

---

**Document Version**: 1.0  
**Last Updated**: September 25, 2026  
**Author**: Generated from Direct AI Blog Classroom Notes  
**License**: Educational Use

---

*This study guide is based on the Gen-AI Developer Classroom Notes from Direct AI Blog, enhanced with detailed explanations, flowcharts, and step-by-step implementation examples for comprehensive learning.*