from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

app = FastAPI(title="SHL Assessment Recommendation API")

model = None
index = None
df = None

def load_resources():
    global model, index, df
    try:
        if model is None:
            print("🔹 Initializing model and data...")
            dataset_path = "Gen_AI Dataset.xlsx"

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            df = pd.read_excel(dataset_path)
            df["combined_text"] = df.astype(str).apply(" ".join, axis=1)
            df["combined_text"] = df["combined_text"].str.replace("\n", " ").str.strip()

            # lightweight model for Render
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2', cache_folder="/tmp")

            embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.array(embeddings))
            print("✅ Model and FAISS index loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        raise

    return model, index, df

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
        model, index, df = load_resources()
        q_emb = model.encode([data.query], normalize_embeddings=True)
        D, I = index.search(q_emb, data.top_k)
        results = df.iloc[I[0]].copy()

        recommendations = []
        for _, row in results.iterrows():
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

    except Exception as e:
        print(f"❌ Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host="0.0.0.0", port=10000)
