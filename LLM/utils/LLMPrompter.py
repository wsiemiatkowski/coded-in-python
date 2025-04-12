# This code requires a local LLM running on ollama, please refer to README.

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMPrompter:
    def __init__(self, model=os.getenv("BASE_MODEL")):
        self.model = model
        self.client = OpenAI(
            base_url=os.getenv("URL"), api_key="ollama"  # This is required tho unused
        )

    def prompt_llm(self, message):
        llm_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        )
        return llm_response.choices[0].message.content


if __name__ == "__main__":
    llm_prompter = LLMPrompter(model="qwen2.5:14b")

    # A simple example
    prompt = "What is the capital of France?"
    response = llm_prompter.prompt_llm(prompt)

    print(response)
