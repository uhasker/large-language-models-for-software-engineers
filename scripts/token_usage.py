import os, requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello, world"}]},
)

response_json = response.json()
usage = response_json["usage"]
prompt_tokens = usage["prompt_tokens"]
completion_tokens = usage["completion_tokens"]
total_tokens = usage["total_tokens"]

print(f"Completion: {response_json['choices'][0]['message']['content']}")

print(f"Prompt tokens: {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")
print(f"Total tokens: {total_tokens}")
