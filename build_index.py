import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

os.makedirs("data", exist_ok=True)

# --------------------------------------------------
# STEP 1: Load Records
# --------------------------------------------------

with open("data/icd_embeddings_ready.json") as f:
    records = json.load(f)

texts = [r["embedding_text"] for r in records]

print("Total texts:", len(texts))

# --------------------------------------------------
# STEP 2: Load Medical Embedding Model
# --------------------------------------------------

print("Loading BioBERT embedding model...")

model = SentenceTransformer("pritamdeka/S-BioBERT-snli-multinli-stsb")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,   # smaller batch for large model
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)

# --------------------------------------------------
# STEP 3: Normalize for Cosine Similarity
# --------------------------------------------------

faiss.normalize_L2(embeddings)

# --------------------------------------------------
# STEP 4: Build FAISS Index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# --------------------------------------------------
# STEP 5: Save Files
# --------------------------------------------------

np.save("data/icd_embeddings.npy", embeddings)
faiss.write_index(index, "data/icd_faiss.index")

print("FAISS index built and saved.")
