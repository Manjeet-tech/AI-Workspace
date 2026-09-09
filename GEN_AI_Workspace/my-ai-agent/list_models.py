import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)
models = genai.list_models()
for model in models:
    print(f"Model: {model.name}, Supported methods: {model.supported_generation_methods}")