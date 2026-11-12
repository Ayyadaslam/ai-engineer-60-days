import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

try:
    print("Google ke available models ki list:\n")
    models = client.models.list()
    for model in models:
        print(model.id)
except Exception as e:
    print(f"Error: {e}")