from google import genai
import os

client = genai.Client(api_key="AIzaSyB7sETj_9G7k3y2UMYSW9oIpaAMYfTvn7I")  

class GeminiLLM:
    def generate(self, prompt: str) -> str:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        if response.text is None:
            raise RuntimeError("Gemini returned no text")
        return response.text

