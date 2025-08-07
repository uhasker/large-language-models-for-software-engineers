import os, requests

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}

API_KEY = os.environ["OPENAI_API_KEY"]
url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "Extract the person information."},
        {
            "role": "user",
            "content": "Alice is 25 years old and works as a software engineer.",
        },
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": schema,
        },
    },
}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
resp.raise_for_status()

print(resp.json()["choices"][0]["message"]["content"])
