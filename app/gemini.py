from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

google_api = os.getenv("API_KEY_GEMINI")
if google_api is None:
    raise EnvironmentError("API_KEY_GEMINI not set in environment variables.")

client = genai.Client(api_key = google_api)  

class GeminiLLM:
    def generate(self, prompt: str) -> str:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        if response.text is None:
            raise RuntimeError("Gemini returned no text...")
        return response.text

