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
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="MediCoder", layout="centered")
st.title("🩺 MediCoder - Multi-Disease ICD Engine")

# -------------------------------------------------
# LOAD MODELS + INDEX
# -------------------------------------------------
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer(
        "pritamdeka/S-BioBERT-snli-multinli-stsb",
        device="cpu"
    )

    index = faiss.read_index("data/icd_faiss.index")

    with open("data/icd_embeddings_ready.json") as f:
        records = json.load(f)

    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

    return embed_model, index, records, llm


embed_model, index, records, llm = load_resources()

# -------------------------------------------------
# LLM DIAGNOSIS EXTRACTION
# -------------------------------------------------
def extract_diagnoses(clinical_text):

    prompt = f"""
    Extract only distinct medical diagnoses from the following clinical description.
    Return as a comma-separated list.

    Clinical Description:
    {clinical_text}
    """

    response = llm(prompt, max_length=128, do_sample=False)[0]["generated_text"]

    diagnoses = [d.strip().lower() for d in response.split(",") if d.strip() != ""]

    return diagnoses

# -------------------------------------------------
# ICD SEARCH
# -------------------------------------------------
def search_icd(query, top_k=3):

    query_vector = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, 10)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        record = records[idx]

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

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]

# -------------------------------------------------
# PARALLEL RETRIEVAL
# -------------------------------------------------
def retrieve_for_disease(disease):
    return disease, search_icd(disease)

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "final_results" not in st.session_state:
    st.session_state.final_results = None

# -------------------------------------------------
# UI INPUT
# -------------------------------------------------
user_input = st.text_area(
    "Enter Full Clinical Description",
    height=200,
    placeholder="Example: Patient with type 2 diabetes and peripheral neuropathy and hypertension..."
)

# -------------------------------------------------
# PROCESS BUTTON
# -------------------------------------------------
if st.button("Analyze & Suggest ICD Codes"):

    if user_input.strip() == "":
        st.warning("Please enter clinical description.")
    else:
        with st.spinner("Extracting diagnoses using LLM..."):
            diagnoses = extract_diagnoses(user_input)

        with st.spinner("Retrieving ICD codes in parallel..."):
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(retrieve_for_disease, diagnoses))

        st.session_state.final_results = results

# -------------------------------------------------
# DISPLAY RESULTS (Persistent)
# -------------------------------------------------
if st.session_state.final_results:

    st.subheader("🔍 Extracted Diagnoses & ICD Suggestions")

    for disease, icd_results in st.session_state.final_results:

        st.markdown(f"### 🧾 Diagnosis: {disease}")

        options = [
            f"{r['code']} - {r['description']} ({r['category']})"
            for r in icd_results
        ]

        st.selectbox(
            f"Select ICD code for {disease}",
            options,
            key=disease
        )
