import pandas as pd
import json
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/ICD10codes_cleaned.csv")

print("Loaded dataset shape:", df.shape)

# --------------------------------------------------
# STEP 1: Create Enriched Embedding Text
# --------------------------------------------------

def create_embedding_text(row):
    return (
        f"ICD Code {row['Code']}. "
        f"Medical condition: {row['Description']}. "
        f"Category: {row['Category']}. "
        f"Hierarchy level: {row['Hierarchy_Level']}."
    )

df["embedding_text"] = df.apply(create_embedding_text, axis=1)

print("Embedding text column created.")

# --------------------------------------------------
# STEP 2: Create Final Structured Records
# --------------------------------------------------

records = []

for _, row in df.iterrows():
    record = {
        "code": row["Code"],
        "description": row["Description"],
        "category": row["Category"],
        "hierarchy_level": row["Hierarchy_Level"],
        "parent_code": row["Parent_Code"],
        "embedding_text": row["embedding_text"]
    }
    records.append(record)

# --------------------------------------------------
# STEP 3: Save JSON
# --------------------------------------------------

with open("data/icd_embeddings_ready.json", "w") as f:
    json.dump(records, f, indent=2)

print("Saved icd_embeddings_ready.json")
print("Total records:", len(records))
