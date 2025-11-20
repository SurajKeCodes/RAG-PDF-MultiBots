import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Configure Gemini API with your key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
models = genai.list_models()

# Print model names
print("\nAvailable Gemini Models:\n")
for model in models:
    print(f" - {model.name}")
