import pandas as pd
import numpy as np
import re

# =====================================================
# PHASE 1 – LOAD DATASET
# =====================================================

df = pd.read_csv("ICD10codes.csv", header=None)

df.columns = [
    "Block_Code",
    "Sub_Block",
    "Code",
    "Short_Description",
    "Long_Description",
    "Category"
]

print("===== BASIC INFO =====")
print("Shape:", df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print("Exact duplicate rows:", df.duplicated().sum())

print("\n===== UNIQUE VALUE COUNT PER COLUMN =====")
print(df.nunique())

# =====================================================
# PHASE 2 – DESCRIPTION LENGTH STATS
# =====================================================

df["desc_length"] = df["Long_Description"].astype(str).apply(len)

print("\n===== LONG DESCRIPTION LENGTH STATS =====")
print(df["desc_length"].describe())

# =====================================================
# PHASE 3 – STRUCTURAL CLEANUP
# =====================================================

df_clean = df[[
    "Code",
    "Long_Description",
    "Category",
    "Short_Description"
]].copy()

df_clean = df_clean.rename(columns={
    "Long_Description": "Description"
})

print("\nAfter structural cleanup:", df_clean.shape)

# =====================================================
# PHASE 4 – REMOVE CRITICAL NULLS
# =====================================================

df_clean = df_clean.dropna(subset=["Code", "Description"])

df_clean = df_clean[
    (df_clean["Code"].astype(str).str.strip() != "") &
    (df_clean["Description"].astype(str).str.strip() != "")
]

print("After removing critical nulls:", df_clean.shape)

# =====================================================
# PHASE 5 – DEDUPLICATION
# =====================================================

df_clean = df_clean.drop_duplicates()
df_clean = df_clean.drop_duplicates(subset=["Code"])

print("After deduplication:", df_clean.shape)

# =====================================================
# PHASE 6 – TEXT NORMALIZATION
# =====================================================

def normalize_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\.\,\-\s\/]", "", text)
    return text.strip()

df_clean["Code"] = df_clean["Code"].astype(str).str.strip().str.upper()
df_clean["Description"] = df_clean["Description"].apply(normalize_text)
df_clean["Category"] = df_clean["Category"].apply(normalize_text)

print("\nNormalization completed.")
print(df_clean.head())

# =====================================================
# PHASE 7 – ADD METADATA (Hierarchy + Parent)
# =====================================================

def get_hierarchy_level(code):
    code = str(code)
    length = len(code)

    if length == 3:
        return "Block"
    elif length == 4:
        return "Category"
    elif length >= 5:
        return "Subcategory"
    else:
        return "Unknown"

def get_parent_code(code):
    code = str(code)

    if len(code) >= 5:
        return code[:4]
    elif len(code) == 4:
        return code[:3]
    else:
        return None

df_clean["Hierarchy_Level"] = df_clean["Code"].apply(get_hierarchy_level)
df_clean["Parent_Code"] = df_clean["Code"].apply(get_parent_code)

# =====================================================
# PHASE 8 – VALIDATION CHECKS
# =====================================================

print("\n===== HIERARCHY DISTRIBUTION =====")
print(df_clean["Hierarchy_Level"].value_counts())

print("\n===== CODE LENGTH DISTRIBUTION =====")
print(df_clean["Code"].apply(len).value_counts())

# Validate ICD format
invalid_codes = df_clean[
    ~df_clean["Code"].str.match(r"^[A-Z][0-9]+$")
]

print("\nInvalid code count:", invalid_codes.shape[0])

# Parent existence check
existing_codes = set(df_clean["Code"])

df_clean["Parent_Exists"] = df_clean["Parent_Code"].apply(
    lambda x: x in existing_codes if pd.notnull(x) else True
)

print("Parent missing count:",
      df_clean[df_clean["Parent_Exists"] == False].shape[0])

print("\n===== CLEANED DESCRIPTION LENGTH STATS =====")
print(df_clean["Description"].apply(len).describe())

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n===== FINAL CLEAN DATASET SHAPE =====")
print(df_clean.shape)

print("\n===== SAMPLE FINAL DATA =====")
print(df_clean.head(10))

# Optional: Save cleaned dataset
df_clean.to_csv("ICD10codes_cleaned.csv", index=False)

print("\nCleaned dataset saved as ICD10codes_cleaned.csv")
