import math


def get_idf(keyword, documents):
    N = len(documents)
    n_q = sum(1 for doc in documents if keyword in doc)

    idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
    return idf


def get_bm25(query_keywords, document, documents, k1=1.5, b=0.75):
    avgdl = sum(len(doc) for doc in documents) / len(documents)

    doc_len = len(document)

    score = 0
    for keyword in query_keywords:
        f_qi_d = document.count(keyword)

        if f_qi_d == 0:
            continue

        idf = get_idf(keyword, documents)

        keyword_score = (
            idf * (f_qi_d * (k1 + 1)) / (f_qi_d + k1 * (1 - b + b * (doc_len / avgdl)))
        )
        score += keyword_score

    return score


documents = [
    "TS-01 Can't access my account with my password",
    "TS-02 My password is not working and I don't know what it is so I need help",
    "TS-03 I need help with my account and I can't log in",
    "TS-04 I am having trouble with my setup and I don't know what it is",
    "TS-05 I can't access my account with my password",
    "TS-06 I need help",
]
documents = [doc.split() for doc in documents]

idf_i = get_idf("I", documents)
print(idf_i)

idf_password = get_idf("password", documents)
print(idf_password)

idf_ts01 = get_idf("TS-01", documents)
print(idf_ts01)


query = ["TS-01", "I", "password"]
for i, doc in enumerate(documents):
    score = get_bm25(query, doc, documents)
    print(f"Document {i + 1} BM25 score: {round(score, 2)}")
