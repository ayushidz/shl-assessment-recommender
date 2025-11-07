from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI(title="SHL Assessment Recommendation API")

# Lazy global variables
model = None
index = None
df = None

def load_resources():
    global model, index, df
    if model is None:
        print("🔹 Loading model and data for the first time...")
        df = pd.read_excel("Gen_AI Dataset.xlsx")
        df["combined_text"] = df.astype(str).apply(" ".join, axis=1)
        df["combined_text"] = df["combined_text"].str.replace("\n", " ").str.strip()

        # ✅ Use smaller, faster model for Render
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

        embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings))

    return model, index, df

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(data: Query):
    model, index, df = load_resources()
    q_emb = model.encode([data.query], normalize_embeddings=True)
    D, I = index.search(q_emb, data.top_k)
    results = df.iloc[I[0]][["Query", "Assessment_url"]].copy()
    results["similarity_score"] = D[0]
    return results.to_dict(orient="records")
