import os
import aisuite as ai
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment. Make sure .env exists and contains OPENAI_API_KEY, or set the variable in your environment.")

client = ai.Client()

models = ["openai:gemini-2.5-flash-free", "openai:gpt-5-chat-free"]

messages = [
    {"role": "system", "content": "Response in Pirate English."},
    {"role": "user", "content": "Tell me a joke."}
]

for model in models:
    responce = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.
    )
    print(responce.choices[0].message.content)
    print("---------------------------------")