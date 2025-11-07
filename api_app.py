from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="SHL Assessment Recommendation API")

# Global cache
model, index, df = None, None, None


def load_resources():
    """Load model, dataset, and FAISS index (lazy loading)."""
    global model, index, df
    if model is None:
        print("🔹 Loading model and dataset for the first time...")

        # ✅ Auto-detect dataset path for Render
        dataset_path = "Gen_AI Dataset.xlsx"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        df = pd.read_excel(dataset_path)
        df["combined_text"] = df.astype(str).apply(" ".join, axis=1)
        df["combined_text"] = df["combined_text"].str.replace("\n", " ").str.strip()

        # ✅ Use a lightweight model for faster inference on Render
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # Create FAISS index
        embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings))

    return model, index, df


# ✅ Input Schema
class QueryInput(BaseModel):
    query: str
    top_k: int = 5


# ✅ Health Endpoint
@app.get("/health")
def health_check():
    """Simple endpoint to check if the API is running."""
    return {"status": "healthy"}


# ✅ Recommendation Endpoint
@app.post("/recommend")
def recommend(data: QueryInput):
    """Given a job description or query, recommend relevant SHL assessments."""
    model, index, df = load_resources()

    query = data.query.strip()
    k = min(max(data.top_k, 1), 10)  # between 1 and 10

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Compute embedding & similarity
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)

    recommendations = []
    for idx, score in zip(I[0], D[0]):
        row = df.iloc[idx]
        recommendations.append({
            "url": row.get("Assessment_url", ""),
            "name": row.get("Assessment_Name", "N/A"),
            "adaptive_support": row.get("Adaptive_Support", "No"),
            "description": row.get("Description", "N/A"),
            "duration": int(row.get("Duration", 0)),
            "remote_support": row.get("Remote_Support", "Yes"),
            "test_type": [row.get("Test_Type", "Knowledge & Skills")]
        })

    return {"recommended_assessments": recommendations}
