
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage


# Step 1: load the env
load_dotenv()

# Step 2: Get API KEY
api_key = os.getenv("GOOGLE_API_KEY")

# Step 3: Initialize the env
llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    api_key=api_key
)


@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        Returns sum of a and b.
    """

    return a + b

@tool
def sub(a: float, b: float) -> float:
    """Subtract two numbers.

    Args:
        a: The first number.
        b: The second numner.
    
    Returns:
        Returns difference of a and b.

    """

    return a - b


# Creating an agent from the tool
agent = create_agent(
    model=llm,
    tools=[add, sub] 
)

# Using messages with the agent
messages = [
    SystemMessage("You are a helpfull assistant"),
    HumanMessage("What is 5 + 3?"),
    HumanMessage("What is 8 - 5?")
]

# Invoke the agent
response = agent.invoke({
    "messages":messages
})

#Analyse the response.
for message in response['messages']:
    print(type(messages))
    message.pretty_print()

final_answer = response['messages'][-1].content[0]['text']
print(f"Final Answer: {final_answer}")
