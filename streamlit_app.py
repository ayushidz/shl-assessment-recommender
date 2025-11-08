import streamlit as st
import requests
import pandas as pd

# Your deployed FastAPI backend URL
API_BASE = "https://Ayushio-o-shl-assessment-recommender-api.hf.space"

st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🧠", layout="wide")

st.title("🧠 SHL Assessment Recommender")
st.write("Enter a job description, query, or JD URL to get matching SHL assessments.")

# Choose input mode
mode = st.radio("Select Input Type", ["Free text", "JD URL"], horizontal=True)

payload = {}
if mode == "Free text":
    payload["query"] = st.text_area("Paste a job description or hiring requirement:")
else:
    payload["url"] = st.text_input("Paste JD URL:")

top_k = st.slider("Number of recommendations", 5, 10, 5)

if st.button("Get Recommendations"):
    if not payload.get("query") and not payload.get("url"):
        st.warning("Please enter a query or URL first.")
    else:
        payload["top_k"] = top_k
        with st.spinner("Fetching recommendations..."):
            try:
                r = requests.post(f"{API_BASE}/recommend", json=payload, timeout=60)
                if r.ok:
                    data = r.json().get("recommended_assessments", [])
                    if data:
                        df = pd.DataFrame(data)
                        st.success(f"✅ Found {len(df)} recommendations")
                        st.dataframe(df)
                    else:
                        st.info("No recommendations found for your query.")
                else:
                    st.error(f"❌ API Error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")
