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


system_prompt = "You are a helpful assistant."

user_prompt = (
    "Write a short 3-4 sentence email to a friend about the weather in San Francisco."
)

response = generate_response(system_prompt, user_prompt)

print(response)

judge_system_prompt = """
You are a judge.
Your job is to judge how well the LLM followed the user's instructions.

Instructions:
1. Read the LLM's response carefully
2. Judge how well the LLM followed the user's instructions
3. Output a score between 1 and 5, where 1 is the worst and 5 is the best

Example format:
[Your explanation here]

5
"""

judge_user_prompt = f"""
LLM response:
{response}

User instructions:
{user_prompt}
"""

judge_result = generate_response(judge_system_prompt, judge_user_prompt)

judge_result_lines = judge_result.split("\n")
explanation = "\n".join(judge_result_lines[0:-1])
score = judge_result_lines[-1]

print("Explanation:")
print(explanation)
print("Score:")
print(score)
