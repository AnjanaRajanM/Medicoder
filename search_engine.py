import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Load SAME MODEL used during indexing
# --------------------------------------------------

model = SentenceTransformer("pritamdeka/S-BioBERT-snli-multinli-stsb")

# --------------------------------------------------
# Load FAISS index
# --------------------------------------------------

index = faiss.read_index("data/icd_faiss.index")

# --------------------------------------------------
# Load records
# --------------------------------------------------

with open("data/icd_embeddings_ready.json") as f:
    records = json.load(f)

# --------------------------------------------------
# Search function
# --------------------------------------------------

# def search_icd(query, top_k=5):

#     print(f"\nQuery: {query}\n")

#     query_vector = model.encode([query], convert_to_numpy=True)

#     faiss.normalize_L2(query_vector)

#     distances, indices = index.search(query_vector, top_k)

#     results = []

#     for score, idx in zip(distances[0], indices[0]):
#         record = records[idx]
#         results.append({
#             "code": record["code"],
#             "description": record["description"],
#             "category": record["category"],
#             "score": float(score)
#         })

#     return results

def search_icd(query, top_k=5):

    print(f"\nQuery: {query}\n")

    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, 20)  # retrieve more first

    results = []

    for score, idx in zip(distances[0], indices[0]):
        record = records[idx]

        # Keyword boost
        keyword_bonus = 0
        if record["description"] in query.lower():
            keyword_bonus += 0.1

        if record["hierarchy_level"] == "Subcategory":
            keyword_bonus += 0.05

        final_score = float(score) + keyword_bonus

        results.append({
            "code": record["code"],
            "description": record["description"],
            "category": record["category"],
            "score": final_score
        })

    # Sort after boosting
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    results = search_icd("patient with typhoid meningitis")

    for r in results:
        print(r)
