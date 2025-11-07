from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ✅ Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# ✅ Load dataset
df = pd.read_excel("Gen_AI Dataset.xlsx")
df["combined_text"] = df.astype(str).apply(" ".join, axis=1)
df["combined_text"] = df["combined_text"].str.replace("\n", " ").str.strip()

# ✅ Load model + FAISS index
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings))

# ✅ Request schema
class Query(BaseModel):
    query: str
    top_k: int = 5

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

# ✅ Recommendation endpoint
@app.post("/recommend")
def recommend(data: Query):
    q_emb = model.encode([data.query], normalize_embeddings=True)
    D, I = index.search(q_emb, data.top_k)
    results = df.iloc[I[0]][["Query", "Assessment_url"]].copy()
    results["similarity_score"] = D[0]
    return results.to_dict(orient="records")

