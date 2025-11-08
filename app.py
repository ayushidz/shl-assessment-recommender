from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd, numpy as np, faiss, os
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
dataset_path = "Gen_AI Dataset.xlsx"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

df = pd.read_excel(dataset_path)
df["combined_text"] = df.astype(str).apply(" ".join, axis=1).str.replace("\n", " ").str.strip()

# Load lightweight model (fast on Hugging Face)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Build FAISS index
embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings))

# Request schema
class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {
        "message": "Welcome to SHL Assessment Recommendation API!",
        "endpoints": ["/health", "/recommend"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(data: Query):
    try:
        q_emb = model.encode([data.query], normalize_embeddings=True)
        D, I = index.search(q_emb, data.top_k)
        results = df.iloc[I[0]].copy()

        recommendations = []
        for _, row in results.iterrows():
            recommendations.append({
                "name": row.get("Assessment_Name", "N/A"),
                "url": row.get("Assessment_url", ""),
                "adaptive_support": row.get("Adaptive_Support", "No"),
                "description": row.get("Description", "N/A"),
                "duration": int(row.get("Duration", 0)),
                "remote_support": row.get("Remote_Support", "Yes"),
                "test_type": [row.get("Test_Type", "Knowledge & Skills")]
            })
        return {"recommended_assessments": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
