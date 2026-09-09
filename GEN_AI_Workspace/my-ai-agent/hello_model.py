import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env variables
load_dotenv()

# Get API KEY
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=api_key
)

# Send a message to the model
response = llm.invoke("Hello! How are you today?")
response.pretty_print()

# Print the response
#print(f"Model response: {response.content}")

result = llm.invoke("What is the capital of India?")
result.pretty_print()


