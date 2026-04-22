import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging

from preprocess import preprocess
from recommender import parse_query, load_or_build_embeddings, recommend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CSV_FILE       = "netflix_data.csv"
TOP_K          = 10

app = FastAPI(title="Movie Recommender")

logger.info("Starting up …")
df = preprocess(CSV_FILE)
embeddings = load_or_build_embeddings(df)
logger.info(f"Ready. {len(df)} titles loaded.")


class QueryRequest(BaseModel):
    query: str

class Recommendation(BaseModel):
    title:             str
    type:              str
    genres:            str
    description:       str
    rating:            str
    director:          str
    cast:              str
    country:           str
    release_year:      int
    duration:          str
    score:             float
    score_description: float
    score_genres:      float
    score_director:    float
    score_cast:        float

class RecommendResponse(BaseModel):
    parsed_query: dict
    results:      list[Recommendation]


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    structured = parse_query(req.query)
    results_df = recommend(structured, df, embeddings, top_k=TOP_K)

    items = []
    for _, row in results_df.iterrows():
        items.append(Recommendation(
            title=str(row["title"]),
            type=str(row["type"]),
            genres=str(row.get("listed_in_clean", row.get("listed_in", ""))),
            description=str(row["description"]),
            rating=str(row["rating"]),
            director=str(row.get("director", "")),
            cast=str(row.get("cast_clean", row.get("cast", ""))),
            country=str(row.get("country", "")),
            release_year=int(row["release_year"]) if str(row.get("release_year","")).isdigit() else 0,
            duration=str(row.get("duration", "")),
            score=float(row["score"]),
            score_description=float(row["score_description"]),
            score_genres=float(row["score_genres"]),
            score_director=float(row["score_director"]),
            score_cast=float(row["score_cast"]),
        ))

    return RecommendResponse(
        parsed_query={
            "description_intent": structured.description_intent,
            "genres":             structured.genres,
            "director":           structured.director,
            "cast":               structured.cast,
        },
        results=items,
    )
