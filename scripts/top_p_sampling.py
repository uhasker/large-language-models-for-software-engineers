import os, requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "How are you?"}],
        "top_p": 0.9,
    },
)

response_json = response.json()
content = response_json["choices"][0]["message"]["content"]
print(content)
