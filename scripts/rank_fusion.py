def rrf(first_results, second_results, k=60):
    all_docs = set(doc_id for doc_id, _ in first_results) | set(doc_id for doc_id, _ in second_results)
    
    first_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(first_results)}
    second_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(second_results)}
    
    rrf_scores = []
    for doc_id in all_docs:
        score = 0
        
        if doc_id in first_ranks:
            score += 1 / (k + first_ranks[doc_id])
        
        if doc_id in second_ranks:
            score += 1 / (k + second_ranks[doc_id])
        
        rrf_scores.append((doc_id, score))
    
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    return rrf_scores



semantic_results = [
    ("doc1", 0.95),
    ("doc3", 0.87),
    ("doc5", 0.82),
    ("doc2", 0.78),
    ("doc4", 0.65)
]

bm25_results = [
    ("doc2", 2.53),
    ("doc1", 1.84),
    ("doc4", 1.12),
    ("doc6", 0.95),
    ("doc3", 0.71)
]

fused_results = rrf(semantic_results, bm25_results)

for rank, (doc_id, score) in enumerate(fused_results, 1):
    print(f"{rank}. {doc_id}: {round(score, 4)}")


