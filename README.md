
# 🧠 SHL Assessment Recommender System

This project is an end-to-end **Recommendation Engine** designed to suggest the most relevant SHL Assessments based on a given job role, skill, or query text.  
It consists of two components — a **FastAPI backend** (deployed on Hugging Face Spaces) and a **Streamlit frontend** (for easy testing and visualization).

---

## 🚀 Project Structure

| Component | Description | Deployment |
|------------|--------------|-------------|
| **Backend (API)** | FastAPI application that encodes queries using Sentence Transformers, performs FAISS similarity search, and returns JSON recommendations. | Hugging Face Spaces |
| **Frontend (UI)** | Streamlit web app that sends user queries to the API and displays recommended assessments interactively. | Streamlit Cloud |

---

## ⚙️ Tech Stack

- **SentenceTransformer:** `paraphrase-MiniLM-L3-v2`
- **FAISS:** for efficient semantic similarity search
- **FastAPI:** for RESTful endpoints
- **Streamlit:** for interactive frontend
- **Pandas / NumPy:** data preprocessing
- **Torch / Transformers:** embedding computation

---

## 🔗 Deployment URLs

| Component | URL |
|------------|-----|
| 🧩 API Endpoint | [`https://ayushio-o-shl-assessment-recommender-api.hf.space/recommend`](https://ayushio-o-shl-assessment-recommender-api.hf.space/recommend) |
| 💻 Frontend Web App | [`https://shl-assessment-recommender.streamlit.app`](https://shl-assessment-recommender.streamlit.app) |
| 📁 GitHub Repository | [`https://github.com/ayushidz/shl-assessment-recommender`](https://github.com/ayushidz/shl-assessment-recommender) |

---

## 📜 API Usage

**Endpoint:**  
`POST /recommend`

**Input JSON:**
```json
{
  "query": "Software Engineer cognitive test",
  "top_k": 5
}

