import os
import requests


def generate_response(system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )

    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"]
    return content


system_prompt = """You are a helpful assistant tasked with answering multiple choice questions.

Instructions:
1. Read the question and all answer choices carefully
2. Provide a clear, step-by-step explanation of your reasoning
3. End your response with only the answer letter (A, B, C, or D) on the final line

Example format:
[Your explanation here]

A
"""

question = """
What is the capital of France?

A. London
B. Paris
C. Rome
D. Madrid
"""

response = generate_response(system_prompt, question)
response_lines = response.split("\n")
explanation = "\n".join(response_lines[0:-1])
llm_answer = response_lines[-1]

print("Explanation:")
print(explanation)
print("\nAnswer:")
print(llm_answer)

correct_answer = "B"
print("Correctness:")
print(llm_answer == correct_answer)
