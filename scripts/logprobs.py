import math
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
        "logprobs": True,
        "top_logprobs": 5,
    },
)

response_json = response.json()
logprobs = response_json["choices"][0]["logprobs"]
print(logprobs)
next_token_logprobs = logprobs["content"][0]["top_logprobs"]

for item in next_token_logprobs:
    token, logprob = item["token"], item["logprob"]
    prob = math.exp(logprob)
    print(token, prob)
