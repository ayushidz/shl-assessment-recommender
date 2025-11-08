import streamlit as st
import requests
import pandas as pd

# 🔗 Backend API (Hugging Face Space)
API_BASE = "https://ayushio-o-shl-assessment-recommender-api.hf.space"

st.set_page_config(page_title="🧠 SHL Assessment Recommender", page_icon="🧩", layout="wide")

# --- Title & Intro ---
st.title("🧠 SHL Assessment Recommender System")
st.markdown("""
This app recommends the most relevant **SHL Assessments** based on your job description or hiring needs.  
Built with `FastAPI + Sentence Transformers + FAISS + Streamlit`.
""")

# --- Input Mode ---
mode = st.radio("Select Input Type", ["Free text", "JD URL"], horizontal=True)

payload = {}
if mode == "Free text":
    payload["query"] = st.text_area("🧾 Paste a Job Description or Query", 
        placeholder="e.g., Hiring a data scientist proficient in Python and analytics")
else:
    payload["url"] = st.text_input("🌐 Paste JD URL")

top_k = st.slider("Number of Recommendations", 5, 10, 5)

# --- Action Button ---
if st.button("🚀 Get Recommendations"):
    if not payload.get("query") and not payload.get("url"):
        st.warning("⚠️ Please enter a query or URL first.")
    else:
        payload["top_k"] = top_k
        with st.spinner("🔍 Fetching recommendations... please wait"):
            try:
                response = requests.post(f"{API_BASE}/recommend", json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json().get("recommended_assessments", [])
                    if data:
                        df = pd.DataFrame(data)
                        st.success(f"✅ Found {len(df)} relevant assessments")
                        st.dataframe(df, use_container_width=True)

                        # Download button
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Results as CSV", csv, "recommendations.csv", "text/csv")
                    else:
                        st.info("🤔 No matching assessments found for your input.")
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                st.error("⏳ The request timed out. Please try again.")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")


