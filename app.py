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
    if model is None:
        print("🔹 Loading model and data...")
        dataset_path = "Gen_AI Dataset.xlsx"

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        df = pd.read_excel(dataset_path)
        df["combined_text"] = df.astype(str).apply(" ".join, axis=1)
        df["combined_text"] = df["combined_text"].str.replace("\n", " ").str.strip()

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
        embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings))
        print("✅ Model and FAISS index ready.")
    return model, index, df

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "Welcome to SHL Assessment Recommendation API!", "endpoints": ["/health", "/recommend"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(data: Query):
    try:
        print("🔍 Received query:", data.query)
        model, index, df = load_resources()
        print("✅ Model and data loaded successfully")

        q_emb = model.encode([data.query], normalize_embeddings=True)
        D, I = index.search(q_emb, data.top_k)
        results = df.iloc[I[0]].copy()
        print(f"✅ Retrieved top {data.top_k} results")

        recommendations = []
        for _, row in results.iterrows():
            # Safe handling for missing duration values
            duration_val = row.get("Duration", 0)
            if pd.isna(duration_val):
                duration_val = 0
            try:
                duration_val = int(float(duration_val))
            except:
                duration_val = 0

            recommendations.append({
                "url": str(row.get("Assessment_url", "")).strip(),
                "name": str(row.get("Assessment_Name", "N/A")).strip(),
                "adaptive_support": str(row.get("Adaptive_Support", "No")).strip(),
                "description": str(row.get("Description", "N/A")).strip(),
                "duration": duration_val,
                "remote_support": str(row.get("Remote_Support", "Yes")).strip(),
                "test_type": [str(row.get("Test_Type", "Knowledge & Skills")).strip()]
            })

        print("✅ Successfully generated recommendations")
        return {"recommended_assessments": recommendations}

    except Exception as e:
        print(f"❌ ERROR in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

