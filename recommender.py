"""
recommender.py
--------------
Structured recommendation engine using OpenAI embeddings + GPT for query parsing.

Embedding model:  text-embedding-3-small  (~$0.00002 / 1K tokens)
LLM model:        gpt-4o-mini             (query parsing only, very cheap)

Total cost for 8,807 titles: ~$0.01
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import json
import re
from openai import OpenAI

logger = logging.getLogger(__name__)

CACHE_FILE      = "embeddings_cache.pkl"
EMBED_MODEL     = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
EMBED_DIM       = 1536  # text-embedding-3-small dimension

# ── Field weights (must sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "description": 0.35,
    "genres":      0.25,
    "director":    0.20,
    "cast":        0.20,
}

def get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ── Structured query ──────────────────────────────────────────────────────────
@dataclass
class StructuredQuery:
    description_intent: str
    genres:             str
    director:           str
    cast:               str
    raw:                str


# ── LLM query parser ──────────────────────────────────────────────────────────
PARSE_PROMPT = """
You are a movie search query parser.

A user typed a natural language search query. Your job is to decompose it
into structured fields that will each be matched independently against a
movie database.

═══════════════════════════════════════════
FIELDS TO EXTRACT
═══════════════════════════════════════════

description_intent:
  A rich 2–4 sentence description of the themes, mood, tone, setting,
  and narrative style the user is looking for.
  Even if the user only wrote 3 words, expand it into a full description.
  This is used for semantic similarity against movie plot descriptions.

genres:
  Comma-separated genre tags inferred from the query.
  Use standard Netflix genre labels where possible:
  e.g. "Dramas, Thrillers, International Movies, TV Comedies"
  If no genre is implied, return an empty string.

director:
  The director's full name if the user mentioned one, otherwise "".
  Do not invent names. Only include if explicitly stated.

cast:
  Comma-separated actor names if the user mentioned any, otherwise "".
  Do not invent names. Only include if explicitly stated.

═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
- Output ONLY valid JSON. No preamble, no markdown fences, no explanation.
- Never invent directors or cast members that weren't in the query.
- description_intent must always be filled — it is the semantic core.

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════
{
  "description_intent": "...",
  "genres": "...",
  "director": "...",
  "cast": "..."
}
"""

def parse_query(raw_query: str) -> StructuredQuery:
    """Use GPT-4o-mini to decompose the query into structured fields."""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PARSE_PROMPT},
                {"role": "user",   "content": raw_query},
            ],
            temperature=0,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        logger.info(f"Parsed query: {json.dumps(parsed, indent=2)}")
        return StructuredQuery(
            description_intent=parsed.get("description_intent", raw_query),
            genres=parsed.get("genres", ""),
            director=parsed.get("director", ""),
            cast=parsed.get("cast", ""),
            raw=raw_query,
        )
    except Exception as e:
        logger.warning(f"LLM parse failed, using raw query: {e}")
        return StructuredQuery(
            description_intent=raw_query,
            genres="", director="", cast="",
            raw=raw_query,
        )


# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_batch(texts: list[str], batch_size: int = 500) -> np.ndarray:
    """Embed texts using OpenAI in batches of 500. No rate limit issues."""
    client = get_client()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_safe = [t if t and t.strip() else "unknown" for t in batch]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch_safe,
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(embeddings, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed a single string."""
    if not text or not text.strip():
        return np.zeros(EMBED_DIM, dtype=np.float32)
    client = get_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(response.data[0].embedding, dtype=np.float32)


# ── Cache management ──────────────────────────────────────────────────────────
def build_embeddings(df: pd.DataFrame) -> dict[str, np.ndarray]:
    logger.info("Building per-field embeddings via OpenAI …")
    field_map = {
        "description": "description",
        "genres":      "listed_in_clean",
        "director":    "director",
        "cast":        "cast_clean",
    }
    cache = {}
    for field_name, col in field_map.items():
        texts = df[col].fillna("").tolist() if col in df.columns else [""] * len(df)
        logger.info(f"  Embedding field '{field_name}' …")
        cache[field_name] = embed_batch(texts)
    return cache


def load_or_build_embeddings(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if os.path.exists(CACHE_FILE):
        logger.info("Loading embeddings from cache …")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if all(k in cache for k in WEIGHTS):
            return cache
        logger.warning("Cache incomplete — rebuilding …")

    cache = build_embeddings(df)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Embeddings cached.")
    return cache


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_field(query_vec: np.ndarray, movie_matrix: np.ndarray) -> np.ndarray:
    if query_vec is None or np.all(query_vec == 0):
        return np.full(len(movie_matrix), 0.5, dtype=np.float32)
    return cosine_similarity(query_vec.reshape(1, -1), movie_matrix)[0].astype(np.float32)


def recommend(query: StructuredQuery, df: pd.DataFrame,
              embeddings: dict[str, np.ndarray], top_k: int = 10) -> pd.DataFrame:
    query_vecs = {
        "description": embed_single(query.description_intent),
        "genres":      embed_single(query.genres),
        "director":    embed_single(query.director),
        "cast":        embed_single(query.cast),
    }

    field_scores = {f: score_field(query_vecs[f], embeddings[f]) for f in WEIGHTS}
    final_scores = sum(WEIGHTS[f] * field_scores[f] for f in WEIGHTS)

    top_indices = final_scores.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["score"]             = final_scores[top_indices]
    results["score_description"] = field_scores["description"][top_indices]
    results["score_genres"]      = field_scores["genres"][top_indices]
    results["score_director"]    = field_scores["director"][top_indices]
    results["score_cast"]        = field_scores["cast"][top_indices]
    return results.reset_index(drop=True)
