import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="MediCoder", layout="centered")

st.title("🩺 MediCoder - ICD Code Suggestion System")
st.write("Enter a clinical description and get suggested ICD-10 codes.")

# ---------------------------------------------
# LOAD MODEL + INDEX (Load Once)
# ---------------------------------------------
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(
    "pritamdeka/S-BioBERT-snli-multinli-stsb",
    device="cpu"
)

    index = faiss.read_index("data/icd_faiss.index")

    with open("data/icd_embeddings_ready.json") as f:
        records = json.load(f)

    return model, index, records

model, index, records = load_model_and_index()

# ---------------------------------------------
# SEARCH FUNCTION
# ---------------------------------------------
def search_icd(query, top_k=3):

    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, 10)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        record = records[idx]

        # Boost subcategory (more specific codes)
        bonus = 0
        if record["hierarchy_level"] == "Subcategory":
            bonus += 0.05

        final_score = float(score) + bonus

        results.append({
            "code": record["code"],
            "description": record["description"],
            "category": record["category"],
            "score": final_score
        })

    # Sort after boost
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]

# ---------------------------------------------
# UI INPUT SECTION
# ---------------------------------------------

user_input = st.text_area(
    "Enter Clinical Description",
    placeholder="Example: Patient with typhoid meningitis and persistent fever...",
    height=150
)

# ---------------------------------------------
# SEARCH BUTTON
# ---------------------------------------------
if st.button("Suggest ICD Codes"):

    if user_input.strip() == "":
        st.warning("Please enter a clinical description.")
    else:
        with st.spinner("Analyzing and retrieving ICD codes..."):
            results = search_icd(user_input)

        st.success("Top Suggested ICD Codes")

        options = []

        for r in results:
            label = f"{r['code']} - {r['description']} ({r['category']})"
            options.append(label)

        selected_code = st.selectbox(
            "Select the most appropriate ICD code:",
            options
        )

        st.info(f"You selected: {selected_code}")
