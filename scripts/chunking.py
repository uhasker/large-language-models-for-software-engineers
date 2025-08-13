import math
import os
import requests


document = """
John Doe is the CEO of ExampleCorp.
He's a skilled software engineer with a focus on scalable systems.
In his spare time, he plays guitar and reads science fiction.

ExampleCorp was founded in 2020 and is based in San Francisco.
It builds AI solutions for various industries.
John still finds time for music and books, even with a busy schedule.

The company is a subsidiary of Example Inc, a tech conglomerate.
Example Inc started in 2015 and is headquartered in New York.
ExampleCorp keeps its startup energy despite the parent company.

San Francisco and New York serve as the main hubs.
This supports talent on both coasts.
John's mix of tech and creativity shapes a forward-thinking culture.
"""


def fixed_size_chunking(document, chunk_size):
    return [document[i : i + chunk_size] for i in range(0, len(document), chunk_size)]


print("Fixed size chunking:")
chunks = fixed_size_chunking(document, 100)
for chunk in chunks:
    print(repr(chunk))


def sliding_window_chunking(document, chunk_size, overlap):
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunks.append(document[i : i + chunk_size])
    return chunks


print("Sliding window chunking:")
chunks = sliding_window_chunking(document, 100, 20)
for chunk in chunks[:3]:
    print(repr(chunk))


def recursive_chunking(text, separators, max_len):
    if len(text) <= max_len or not separators:
        return [text]

    sep = separators[0]
    parts = text.split(sep)

    chunks = []
    for part in parts:
        if not part.strip():
            continue  # Skip empty parts

        # If still too long, recurse with other separators
        if len(part) > max_len and len(separators) > 1:
            chunks.extend(recursive_chunking(part, separators[1:], max_len))
        else:
            chunks.append(part)

    return chunks


print("Recursive chunking:")
chunks = recursive_chunking(document, ["\n\n", ".", ","], 100)
for chunk in chunks[:3]:
    print(repr(chunk))


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


def dot_product(embedding1, embedding2):
    return sum(x * y for x, y in zip(embedding1, embedding2))


def get_sentences(document):
    return [s.strip() for s in document.split(".") if s.strip()]


def semantic_chunking(document, threshold):
    sentences = get_sentences(document)
    embeddings = [generate_embedding(sentence) for sentence in sentences]

    chunks = []
    for i in range(len(sentences)):
        if i == 0:
            chunks.append(sentences[i])
        else:
            embedding_similarity = dot_product(embeddings[i - 1], embeddings[i])
            if embedding_similarity < threshold:
                chunks.append(sentences[i])
            else:
                chunks[-1] += ". " + sentences[i]
    return chunks


print("Semantic chunking:")
chunks = semantic_chunking(document, 0.3)
for chunk in chunks:
    print(repr(chunk))


def get_min_sim(chunk_embeddings):
    if len(chunk_embeddings) == 1:
        return 0.3

    min_sim = float("inf")
    for i in range(len(chunk_embeddings)):
        for j in range(i + 1, len(chunk_embeddings)):
            sim = dot_product(chunk_embeddings[i], chunk_embeddings[j])
            min_sim = min(min_sim, sim)
    return min_sim


def get_max_sim(chunk_embeddings, sentence_embedding):
    max_sim = 0
    for chunk_embedding in chunk_embeddings:
        sim = dot_product(sentence_embedding, chunk_embedding)
        max_sim = max(max_sim, sim)
    return max_sim


def max_min_semantic_chunking(document):
    sentences = get_sentences(document)
    embeddings = [generate_embedding(sentence) for sentence in sentences]

    chunks = []
    current_chunk = sentences[:1]
    current_chunk_embeddings = embeddings[:1]

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_embedding = embeddings[i]
        min_sim = get_min_sim(current_chunk_embeddings)
        max_sim = get_max_sim(current_chunk_embeddings, sentence_embedding)

        if max_sim > min_sim:
            current_chunk.append(sentence)
            current_chunk_embeddings.append(sentence_embedding)
        else:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_embeddings = [sentence_embedding]

    if current_chunk:
        chunks.append(". ".join(current_chunk))

    return chunks


print("\nMax-min semantic chunking:")
chunks = max_min_semantic_chunking(document)
for chunk in chunks:
    print(repr(chunk))
