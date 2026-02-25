# This class contains methods for querying local and remote LLMs
import requests

class LLMClient:
    def __init__(self, model, api_url, temperature, api_key):
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.api_key = api_key


    def generate(self, messages):
        """
        Use Ollama API format
        args - messages: list of message dicts with 'role' and 'content' keys
        """
        headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        response = requests.post(f"{self.api_url}/api/chat", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()

        # Extract assistant message content
        if "message" in content and "content" in content["message"]:
            return content["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected response: {content}")
        

if __name__ == "__main__":    # Example usage
    import os
    llm = LLMClient(model="gpt-oss:120b-cloud",
                    api_url="https://ollama.com",
                    temperature=0.1,  
                    api_key=os.getenv("OLLAMA_API_KEY"))
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)
    print(response)  # Should print "Paris" 
