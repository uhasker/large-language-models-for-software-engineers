import os, requests


def generate_embedding(text):
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={"input": text, "model": "text-embedding-3-small"},
    )

    response_json = response.json()
    embedding = response_json["data"][0]["embedding"]
    return embedding


documents = [
    "Example Corp was founded in 2020",
    "The capital of France is Paris",
    "Example Corp is a technology company that develops AI solutions",
    "The capital of Germany is Berlin",
    "Example Corp is headquartered in San Francisco",
    "The capital of Spain is Madrid",
    "The CEO of Example Corp is John Doe",
    "The capital of Italy is Rome",
]

user_query = "Who is the CEO of Example Corp?"

document_embeddings = [generate_embedding(doc) for doc in documents]
user_query_embedding = generate_embedding(user_query)


def get_dot_product(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def get_most_similar_documents(query_embedding, document_embeddings, top_k=5):
    similarities = [
        get_dot_product(query_embedding, doc_embedding)
        for doc_embedding in document_embeddings
    ]
    most_similar_indices = sorted(
        range(len(similarities)), key=lambda i: similarities[i], reverse=True
    )[:top_k]
    return [(documents[i], similarities[i]) for i in most_similar_indices]


most_similar_documents = get_most_similar_documents(
    user_query_embedding, document_embeddings
)
for doc, similarity in most_similar_documents:
    print(f"Document: {doc}, Similarity: {round(similarity, 2)}")


def generate_response(user_query, most_similar_documents):
    prompt = f"""
    Answer the user query based on the following documents:
    {"\n".join(most_similar_documents)}

    User query: {user_query}
    """

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]


response = generate_response(user_query, [doc[0] for doc in most_similar_documents])
print(response)
