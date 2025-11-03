# ===============================================================
# ADVISER RECOMMENDER API (TF-IDF + Experience)
# FastAPI + Supabase
# TF-IDF + Experience (Main)
# Wildcards based solely on Research Interest using TF-IDF
# Includes past theses in recommendations and wildcards
# Author: Prince Roniver A. Magsalos & Andrew R. Villalon
# ===============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib
import os
from dotenv import load_dotenv


# ===============================================================
# Load Environment / Supabase
# ===============================================================
print("Initializing Supabase connection...")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===============================================================
# Helper: Clean Text
# ===============================================================
def clean_text(text):
    if not text:
        return ""
    if isinstance(text, list):
        text = " ".join(map(str, text))
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.lower().strip()

# ===============================================================
# Load Pretrained Model (PKL)
# ===============================================================
MODEL_PATH = "adviser_recommender.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f" Model file not found: {MODEL_PATH}. Please train and export it first.")

model_data = joblib.load(MODEL_PATH)
vectorizer = model_data["vectorizer"]
tfidf_matrix = model_data["tfidf_matrix"]
df = model_data["df"]
experience_points = model_data["experience_points"]

print(f"Loaded pretrained model: {MODEL_PATH}")
print(f"Loaded {len(df)} theses and {len(experience_points)} adviser experience scores.")

# ===============================================================
# FastAPI Setup
# ===============================================================
app = FastAPI(title="Hybrid Adviser Recommender API v3.6", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================
# Pydantic Model
# ===============================================================
class Project(BaseModel):
    title: str
    abstract: str
    student_id: str

# ===============================================================
# Supabase Helper Functions
# ===============================================================
def map_adviser_names_to_ids(adviser_names: list[str]) -> dict[str, str]:
    mapping = {}
    for name in adviser_names:
        response = supabase.table("user_profiles").select("user_id").eq("full_name", name).execute()
        mapping[name] = response.data[0]["user_id"] if response.data else None
    return mapping

def get_sent_advisers(student_id: str) -> set[str]:
    res = supabase.table("student_requests").select("adviser_id").eq("student_id", student_id)\
        .in_("status", ["pending", "accepted"]).execute()
    return {r["adviser_id"] for r in res.data} if res.data else set()

def get_adviser_current_leaders(adviser_id: str):
    res = supabase.table("adviser_current_leaders").select("current_leaders").eq("adviser_id", adviser_id).execute()
    if res.data:
        cap = res.data[0]
        currentLeaders = cap.get("current_leaders", 0)
        availability = "Available" if currentLeaders > 0 else "Unavailable"
        return currentLeaders, availability
    return 0, "Available"

# ===============================================================
# Main Recommendation Endpoint
# ===============================================================
@app.post("/recommend")
def recommend(project: Project):
    try:
        title, abstract = project.title.strip(), project.abstract.strip()
        if not title or not abstract:
            raise HTTPException(status_code=400, detail="Title and abstract are required.")

        sent_advisers = get_sent_advisers(project.student_id)
        print("📬 Sent advisers:", sent_advisers)

        # Encode user text
        user_text = clean_text(title + " " + abstract)
        user_vec = vectorizer.transform([user_text])
        tfidf_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

        # Combine TF-IDF and experience
        exp_scaled = np.log1p(df["experience_score"] * 10) / np.log1p(10)
        w_tfidf, w_exp = 0.9, 0.1
        total_scores = (w_tfidf * tfidf_sim) + (w_exp * exp_scaled)

        df["similarity"] = tfidf_sim
        df["total_score"] = total_scores

        # Aggregate by adviser
        adviser_scores = df.groupby("adviser_name")["total_score"].mean().sort_values(ascending=False).head(5)
        adviser_names = adviser_scores.index.tolist()
        name_to_id = map_adviser_names_to_ids(adviser_names)

        results = []
        recommended_adviser_ids = []

        for adviser_name, score in adviser_scores.items():
            adviser_id = name_to_id.get(adviser_name)
            if not adviser_id:
                continue
            recommended_adviser_ids.append(adviser_id)

            currentLeaders, availability = get_adviser_current_leaders(adviser_id)

            profile_res = supabase.table("user_profiles").select(
                "prefix, full_name, suffix, profile_picture, email, position, research_interest, bio"
            ).eq("user_id", adviser_id).execute()
            profile = profile_res.data[0] if profile_res.data else {}

            full_name = (
                (profile.get("prefix") + " " if profile.get("prefix") else "") +
                profile.get("full_name", adviser_name) +
                (", " + profile.get("suffix") if profile.get("suffix") else "")
            )

            adv_theses = df[df["adviser_name"] == adviser_name].sort_values(by="similarity", ascending=False).head(5)

            results.append({
                "id": adviser_id,
                "full_name": full_name,
                "score": float(score),
                "availability": availability,
                "current_leaders": currentLeaders,
                "already_requested": adviser_id in sent_advisers,
                "profile_picture": profile.get("profile_picture"),
                "email": profile.get("email"),
                "position": profile.get("position"),
                "research_interest": profile.get("research_interest"),
                "bio": profile.get("bio"),
                "projects": [
                    {
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "similarity": float(row["similarity"]),
                        "status": row["status"],
                    }
                    for _, row in adv_theses.iterrows()
                ],
            })

        # ===============================================================
        # Wildcards Based Solely on Research Interest
        # ===============================================================
        user_profiles = supabase.table("user_profiles").select("full_name, research_interest").execute()
        df_users = pd.DataFrame(user_profiles.data)
        df_users["research_interest"] = df_users["research_interest"].fillna("").apply(clean_text)
        df_users = df_users[
            ~df_users["research_interest"].str.strip().str.lower().isin(
                ["", "to be provided", "tba", "n/a", "none", "not available", "pending"]
            )
        ].copy()

        # Exclude top 5 advisers
        df_users = df_users[~df_users["full_name"].isin(adviser_names)]

        if not df_users.empty:
            ri_vectorizer = TfidfVectorizer(stop_words="english")
            ri_matrix = ri_vectorizer.fit_transform(df_users["research_interest"].tolist())
            user_vec_ri = ri_vectorizer.transform([user_text])
            sims = cosine_similarity(user_vec_ri, ri_matrix).flatten()
            df_users["wildcard_score"] = sims
            top_wildcards = df_users.sort_values(by="wildcard_score", ascending=False).head(3)
        else:
            top_wildcards = pd.DataFrame()

        wildcards = []
        for _, row in top_wildcards.iterrows():
            adviser_name = row["full_name"]
            adviser_id = map_adviser_names_to_ids([adviser_name]).get(adviser_name)
            if not adviser_id:
                continue

            currentLeaders, availability = get_adviser_current_leaders(adviser_id)
            profile_res = supabase.table("user_profiles").select(
                "prefix, full_name, suffix, profile_picture, email, position, research_interest, bio"
            ).eq("user_id", adviser_id).execute()
            profile = profile_res.data[0] if profile_res.data else {}

            full_name = (
                (profile.get("prefix") + " " if profile.get("prefix") else "") +
                profile.get("full_name", adviser_name) +
                (", " + profile.get("suffix") if profile.get("suffix") else "")
            )

             # Fetch past theses for this wildcard adviser
            adv_theses = df[df["adviser_name"] == adviser_name].sort_values(by="similarity", ascending=False).head(5)
            projects = [
                {
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "similarity": float(row["similarity"]),
                    "status": row["status"],
                }
                for _, row in adv_theses.iterrows()
            ]


            wildcards.append({
                "id": adviser_id,
                "full_name": full_name,
                "availability": availability,
                "current_leaders": currentLeaders,
                "already_requested": adviser_id in sent_advisers,
                "profile_picture": profile.get("profile_picture"),
                "email": profile.get("email"),
                "position": profile.get("position"),
                "research_interest": profile.get("research_interest"),
                "bio": profile.get("bio"),
                "wildcard_score": float(row["wildcard_score"]),
                "projects": projects,
            })

        return {
            "recommendations": results,
            "recommended_adviser_ids": recommended_adviser_ids,
            "wildcard_advisers": wildcards,
        }

    except Exception as e:
        import traceback
        print("ERROR in /recommend endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
