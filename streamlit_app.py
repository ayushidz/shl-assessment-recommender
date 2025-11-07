import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load Excel data
df = pd.read_excel("C:\\Users\\ayush\\Downloads\\Gen_AI Dataset.xlsx")
df["combined_text"] = df.astype(str).apply(" ".join, axis=1)
df["combined_text"] = df["combined_text"].str.replace("\n", " ").str.strip()

# Cache model + index to avoid reloading each time
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index

model, index = load_model_and_index()

# Streamlit UI
st.title("🧠 SHL Assessment Recommendation System")
st.markdown("Get top SHL assessments by entering a job description or hiring query.")

query = st.text_area("✍️ Enter a job description or query:")
top_k = st.slider("Number of recommendations", 5, 10, 5)

if st.button("🚀 Recommend"):
    if not query.strip():
        st.warning("Please enter a query before clicking Recommend.")
    else:
        with st.spinner("Finding best assessments... ⏳"):
            q_emb = model.encode([query], normalize_embeddings=True)
            D, I = index.search(q_emb, top_k)
            results = df.iloc[I[0]][["Query", "Assessment_url"]].copy()
            results["similarity_score"] = D[0]

        st.success("✅ Recommendations generated!")
        st.dataframe(results)

        # Option to download results
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv, "recommendations.csv", "text/csv")
