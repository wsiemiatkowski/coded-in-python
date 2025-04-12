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


class FewShotLLMPrompter(LLMPrompter):
    def __init__(self, model=os.getenv("BASE_MODEL")):
        """
        This is a few shot version of LLMPrompter.
        It allows for more context to be provided.
        """
        super().__init__(model)
        self.history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def add_few_shot_examples(self, examples):
        """
        Add few-shot examples to the history.
        format: List of tuples (like self.history)
        """
        for user_msg, assistant_reply in examples:
            self.history.append({"role": "user", "content": user_msg})
            self.history.append({"role": "assistant", "content": assistant_reply})

    def prompt_llm(self, message):
        prompt_messages = self.history + [{"role": "user", "content": message}]

        llm_response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt_messages,
        )

        reply = llm_response.choices[0].message.content

        return reply
