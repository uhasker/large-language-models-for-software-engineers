import math, os, requests

response = requests.post(
    "https://api.openai.com/v1/embeddings",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={"input": "Your text string goes here", "model": "text-embedding-3-small"},
)

response_json = response.json()
embedding = response_json["data"][0]["embedding"]
print(embedding[:5])
print(len(embedding))


def get_norm(embedding):
    return math.sqrt(sum(x**2 for x in embedding))


print(get_norm(embedding))
