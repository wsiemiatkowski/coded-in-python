import random

import pandas as pd

from LLM.utils.LLMPrompter import LLMPrompter

MESSAGE_TOPICS = ["meeting", "entertainment", "business", "family", "party"]


def generate_query(prompter: LLMPrompter):
    topic = random.choice(MESSAGE_TOPICS)

    system_prompt = f"""
    You are a data specialist developing an NLP model designed to classify user messages into one of five categories:
    {MESSAGE_TOPICS}

    Since real customer data cannot be used, your task is to generate synthetic training messages for each topic.

    For a given topic, create a realistic and diverse message that follows this exact format:
    Message;topic;sender;location;time

    — Ensure the message is semicolon-separated with exactly five fields.
    — Use language appropriate to the topic, with varied vocabulary and phrasing.
    — Avoid repeating structures or content across messages.
    — Keep the messages natural and representative of real user communication.
    - Do not add any additional information.
    """

    history = [
        (
            "Target message topic: meeting",
            "Hi Robert, could we reschedule our coffee date to 2pm?;meeting;Anastasia;Berlin;15/05/2021",
        ),
        (
            "Target message topic: meeting",
            "I will meet ya near the Lucia's pizzeria.;meeting;Mario;Rome;30/09/2020",
        ),
        (
            "Target message topic: entertainment",
            "Hi, I have confirmed our escape room reservation, see ya.;entertainment;Linus;Amsterdam;01/02/2023",
        ),
        (
            "Target message topic: entertainment",
            "The Beer Museum opens at 10am, is 11am ok for you?;entertainment;Gniewko;Prague;01/02/2023",
        ),
        (
            "Target message topic: business",
            "I will be late to work today Sorry boss, bad tire.;business;Smith A.;Chicago;05/11/2020",
        ),
        (
            "Target message topic: business",
            "Miss Morgan, please contact me in regards to the contract that has been delivered to us today. "
            "Thank you, James S.;business;Morgan M.;Sydney;06/06/2000",
        ),
        (
            "Target message topic: family",
            "Will you be coming home for dinner darling?;family;mom;London;13/04/2025",
        ),
        (
            "Target message topic: family",
            "Remember to pick up Greg from preschool. xoxo;family;honey;New York;01/02/2023",
        ),
        (
            "Target message topic: party",
            "Don't forget to bring that pizza man;party;Mia;Tokyo;01/03/2025",
        ),
        (
            "Target message topic: party",
            "Yo, are you coming tonight? Boys are already here!;party;Julia;Warsaw;22/06/2024",
        ),
    ]

    message = f"Target system topic: {topic}"
    query = prompter.prompt_llm_few_shot(message, history, system_prompt)

    return topic, query


def generate_synthetic_dataset(number_of_samples):
    prompter = LLMPrompter()
    synthetic_dataset = []

    for _ in range(number_of_samples):
        topic, query = generate_query(prompter)
        semicolon_count = query.count(";")

        # If LLM does not generate a full query, skip it
        if semicolon_count == 4:
            synthetic_dataset.append(query)

    df = pd.DataFrame(
        [line.split(";") for line in synthetic_dataset],
        columns=["message", "topic", "sender", "location", "time"],
    )

    return df


if __name__ == "__main__":
    result_file = "qwen_generated_synthetic_dataset.tsv"
    synthetic_data = generate_synthetic_dataset(50)
    synthetic_data.to_csv(result_file, sep="\t", index=False)
