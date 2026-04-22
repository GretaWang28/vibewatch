"""
preprocess.py
-------------
Cleans and prepares the Netflix dataset for embedding.

Steps:
  1. Drop rows missing a description (useless for semantic search)
  2. Remove exact duplicate titles
  3. Clean text fields (strip whitespace, remove special chars)
  4. Cap cast lists to top 5 names (long lists add noise)
  5. Normalize genres into a clean comma-separated string
  6. Build a single rich `_text` field per title for embedding
"""

import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CAST        = 5      # only keep top N cast members
MIN_DESC_LENGTH = 20     # drop descriptions shorter than this (junk rows)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip extra whitespace and remove non-printable characters."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F]", "", text)  # keep latin chars
    return text


def cap_cast(cast_str: str, max_names: int = MAX_CAST) -> str:
    """Keep only the first N cast members."""
    if not cast_str or cast_str == "nan":
        return ""
    names = [n.strip() for n in cast_str.split(",")]
    return ", ".join(names[:max_names])


def normalize_genres(genres_str: str) -> str:
    """Clean and deduplicate genre tags."""
    if not genres_str or genres_str == "nan":
        return ""
    genres = [g.strip() for g in genres_str.split(",")]
    seen = set()
    unique = []
    for g in genres:
        key = g.lower()
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return ", ".join(unique)


def build_embedding_text(row: pd.Series) -> str:
    """
    Combine cleaned fields into a single rich string for embedding.

    Format (pipe-separated sections):
      Title | Type | Genres | Rating | Description | Director | Cast
    """
    parts = [
        row.get("title", ""),
        row.get("type", ""),
        row.get("listed_in_clean", ""),
        row.get("rating", ""),
        row.get("description", ""),
        row.get("director", ""),
        row.get("cast_clean", ""),
    ]
    return " | ".join(p for p in parts if p and p != "nan")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the Netflix CSV.
    Returns a cleaned DataFrame with a `_text` column ready for embedding.
    """
    logger.info(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    original_count = len(df)
    logger.info(f"Loaded {original_count} rows.")

    # 1. Drop rows with missing or too-short descriptions
    df = df[df["description"].str.len() >= MIN_DESC_LENGTH].copy()
    logger.info(f"After dropping short/empty descriptions: {len(df)} rows "
                f"(removed {original_count - len(df)})")

    # 2. Remove exact duplicate titles (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["title"], keep="first")
    logger.info(f"After deduplication: {len(df)} rows (removed {before - len(df)} duplicates)")

    # 3. Clean individual text fields
    for col in ["title", "director", "description", "country", "rating", "type"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # 4. Cap cast lists
    df["cast_clean"] = df["cast"].apply(lambda x: cap_cast(x, MAX_CAST))

    # 5. Normalize genres
    df["listed_in_clean"] = df["listed_in"].apply(normalize_genres)

    # 6. Build the combined embedding text
    df["_text"] = df.apply(build_embedding_text, axis=1)

    # 7. Drop rows where _text is still too short after cleaning
    before = len(df)
    df = df[df["_text"].str.len() >= 30].copy()
    logger.info(f"After final text quality filter: {len(df)} rows "
                f"(removed {before - len(df)})")

    # 8. Reset index for clean integer indexing
    df = df.reset_index(drop=True)

    logger.info(f"Preprocessing complete. {len(df)} titles ready for embedding.")
    return df


# ── Quick test (run directly) ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = preprocess("netflix_data.csv")
    print("\nSample cleaned rows:")
    print(df[["title", "type", "listed_in_clean", "cast_clean"]].head(5).to_string())
    print(f"\nSample _text:\n{df['_text'].iloc[0]}")
    print(f"\nFinal shape: {df.shape}")
