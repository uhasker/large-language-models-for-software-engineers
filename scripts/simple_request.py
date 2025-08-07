import os, requests

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How are you?"},
]

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o",
        "messages": messages,
    },
)

response_json = response.json()
assistant_message = response_json["choices"][0]["message"]
print(assistant_message)

messages.append(assistant_message)
messages.append({"role": "user", "content": "What is the capital of France?"})

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o",
        "messages": messages,
    },
)

response_json = response.json()
assistant_message = response_json["choices"][0]["message"]
print(assistant_message)

messages.append(assistant_message)

print(messages)
