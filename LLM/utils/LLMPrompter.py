# This code requires a local LLM running on ollama, please refer to README.

import os
from typing import Tuple, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMPrompter:
    def __init__(self, model=os.getenv("BASE_MODEL")):
        self.model = model
        self.client = OpenAI(
            base_url=os.getenv("URL"), api_key="ollama"  # This is required tho unused
        )

    def prompt_llm(self, message: str) -> str:
        llm_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        )

        return llm_response.choices[0].message.content

    def prompt_llm_few_shot(
        self, message: str, history: List[Tuple[str, str]], system_prompt: str
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        for chat in history:
            messages.append({"role": "user", "content": chat[0]})
            messages.append({"role": "system", "content": chat[1]})

        messages.append({"role": "user", "content": message})

        llm_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return llm_response.choices[0].message.content
