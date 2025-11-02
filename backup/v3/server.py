# ===============================================================
# Hybrid Adviser Recommender API (FastAPI + Supabase)
# TF-IDF + SBERT + Logistic Regression + Wildcards + Adviser Helpers
# ===============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack
from supabase import create_client
from dotenv import load_dotenv
import re
import os

# ===============================================================
# Load Environment and Supabase
# ===============================================================
print("🌐 Initializing Supabase connection...")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    test = supabase.table("users").select("id, user_id").limit(1).execute()
    if test.data:
        print("✅ Supabase connected successfully. Sample user:", test.data[0])
    else:
        print("⚠️ Supabase connected but no users found.")
except Exception as e:
    print("❌ Supabase connection failed:", e)

# ===============================================================
# Load Models and Data
# ===============================================================
print("🔧 Loading model components...")

vectorizer = joblib.load("vectorizer.pkl")
log_reg = joblib.load("log_reg.pkl")
df = pd.read_pickle("adviser_df.pkl")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Models and embeddings loaded successfully.")

# ===============================================================
# FastAPI Setup
# ===============================================================
app = FastAPI(title="Hybrid Adviser Recommender API", version="3.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://archivia-official.vercel.app"],
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
    student_id: str  # for Supabase lookups

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
# Supabase Helper Functions
# ===============================================================
def map_adviser_names_to_ids(adviser_names: list[str]) -> dict[str, str]:
    mapping = {}
    for name in adviser_names:
        response = supabase.table("user_profiles").select("user_id").eq("full_name", name).execute()
        mapping[name] = response.data[0]["user_id"] if response.data else None
    return mapping

def get_full_name_with_title(adviser_name: str) -> str:
    adviser_id = map_adviser_names_to_ids([adviser_name]).get(adviser_name)
    if not adviser_id:
        return adviser_name
    profile_res = supabase.table("user_profiles") \
        .select("prefix, full_name, suffix") \
        .eq("user_id", adviser_id).execute()
    profile = profile_res.data[0] if profile_res.data else {}
    return (
        (profile.get("prefix") + " " if profile.get("prefix") else "") +
        profile.get("full_name", adviser_name) +
        (", " + profile.get("suffix") if profile.get("suffix") else "")
    )

def get_sent_advisers(student_id: str) -> set[str]:
    response = (
        supabase.table("student_requests")
        .select("adviser_id")
        .eq("student_id", student_id)
        .in_("status", ["pending", "accepted"])
        .execute()
    )
    return {r["adviser_id"] for r in response.data} if response.data else set()


def get_adviser_current_leaders(adviser_id: str) -> tuple[int, str]:
    response = (
        supabase.table("adviser_current_leaders")
        .select("current_leaders")
        .eq("adviser_id", adviser_id)
        .execute()
    )
    if response.data:
        cap = response.data[0]
        currentLeaders = cap.get("current_leaders", 0)
        availability = "Available" if currentLeaders > 0 else "Unavailable"
        return currentLeaders, availability
    return 0, "Available"

# ===============================================================
# Main Endpoint
# ===============================================================
@app.post("/recommend")
def recommend(project: Project):
    try:
        title, abstract = project.title.strip(), project.abstract.strip()
        if not title or not abstract:
            raise HTTPException(status_code=400, detail="Title and abstract are required.")

        print(f"🔹 Received project from {project.student_id}: {title[:40]}...")

        sent_advisers = get_sent_advisers(project.student_id)
        print("📬 Sent advisers:", sent_advisers)

        user_text = clean_text(title + " " + abstract)

        # Encode user input
        user_vec_tfidf = vectorizer.transform([user_text])
        user_vec_sbert = sbert_model.encode([user_text])
        user_exp_feature = np.array([[0.5]])  # neutral experience level
        user_vec_lr = hstack([user_vec_tfidf, user_exp_feature])

        # Similarities
        tfidf_sim = cosine_similarity(user_vec_tfidf, vectorizer.transform(df["combined_text"])).flatten()
        sbert_sim = cosine_similarity(user_vec_sbert, sbert_model.encode(df["combined_text"].tolist())).flatten()

        lr_probs = log_reg.predict_proba(user_vec_lr)[0]
        lr_classes = log_reg.classes_
        lr_score_dict = dict(zip(lr_classes, lr_probs))

        # Compute composite scores
        scores, contrib = {}, {}
        for idx in range(len(df)):
            adviser = df.iloc[idx]["adviser_name"]
            if adviser:
                tfidf_score = 0.5 * tfidf_sim[idx]
                sbert_score = 0.2 * sbert_sim[idx]
                research_interest = df.iloc[idx]["research_interest"]
                research_text = clean_text(research_interest if pd.notna(research_interest) else "")
                research_score = 0.1 * cosine_similarity(
                    sbert_model.encode([research_text]), user_vec_sbert
                )[0][0]
                lr_score = 0.15 * lr_score_dict.get(adviser, 0)
                total = tfidf_score + sbert_score + research_score + lr_score

                scores[adviser] = scores.get(adviser, 0) + total
                contrib[adviser] = contrib.get(adviser, [0, 0, 0, 0])
                contrib[adviser][0] += tfidf_score
                contrib[adviser][1] += sbert_score
                contrib[adviser][2] += research_score
                contrib[adviser][3] += lr_score

        # Top 5 advisers
        top_advisers = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        adviser_names = [a for a, _ in top_advisers]
        name_to_id = map_adviser_names_to_ids(adviser_names)

        recommendations = []
        recommended_adviser_ids = []

        for adv, score in top_advisers:
            adviser_id = name_to_id.get(adv)
            if adviser_id is None:
                continue

            recommended_adviser_ids.append(adviser_id)

            currentLeaders, availability = get_adviser_current_leaders(adviser_id)

            # Fetch profile details from Supabase
            profile_res = supabase.table("user_profiles") \
                .select("prefix, full_name, suffix, profile_picture, email, position, research_interest, bio") \
                .eq("user_id", adviser_id).execute()
            profile = profile_res.data[0] if profile_res.data else {}

            # Construct full name with prefix and suffix
            full_name_with_title = (
                (profile.get("prefix") + " " if profile.get("prefix") else "") +
                profile.get("full_name", adv) +
                (", " + profile.get("suffix") if profile.get("suffix") else "")
            )

            adv_theses = df[df["adviser_name"] == adv].copy()
            if adv_theses.empty:
                continue

            theses_embs = sbert_model.encode(adv_theses["combined_text"].tolist(), show_progress_bar=False)
            sim_scores = cosine_similarity(user_vec_sbert, theses_embs).flatten()
            adv_theses["similarity"] = sim_scores
            top_theses = adv_theses.sort_values(by="similarity", ascending=False).head(5)

            recommendations.append({
                "id": adviser_id,
                "full_name": full_name_with_title,
                "score": float(score),
                "availability": availability,
                "current_leaders": currentLeaders,
                "tfidf": float(contrib[adv][0]),
                "sbert": float(contrib[adv][1]),
                "research": float(contrib[adv][2]),
                "lr": float(contrib[adv][3]),
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
                    for _, row in top_theses.iterrows()
                ],
            })

        # Wildcard advisers (ignore 'To be provided' research interests)
        valid_interest_df = df[
            ~df["research_interest"].str.strip().str.lower().isin(
                ["to be provided", "tba", "n/a", "none", "not available", "pending"]
            )
        ].copy()

        adviser_interest_texts = valid_interest_df["research_interest"].fillna("").apply(clean_text).tolist()
        adviser_texts = valid_interest_df["combined_text"].fillna("").apply(clean_text).tolist()

        tfidf_sims = cosine_similarity(user_vec_tfidf, vectorizer.transform(adviser_texts)).flatten()
        sbert_sims = cosine_similarity(user_vec_sbert, sbert_model.encode(adviser_texts)).flatten()
        research_sims = cosine_similarity(
            sbert_model.encode([user_text]), sbert_model.encode(adviser_interest_texts)
        ).flatten()
        exp_scores = valid_interest_df["experience_score"].fillna(0).to_numpy()

        valid_interest_df["wildcard_score"] = (
            (0.10 * tfidf_sims) +
            (0.15 * sbert_sims) +
            (0.45 * research_sims) +
            (0.30 * exp_scores)
        )

        top_adviser_names = {a for a, _ in top_advisers}
        wildcard_df = valid_interest_df[~valid_interest_df["adviser_name"].isin(top_adviser_names)]

        top_wildcards = (
            wildcard_df.groupby("adviser_name")["wildcard_score"]
            .max()
            .sort_values(ascending=False)
            .head(3)
        )

        wildcards = []
        for adv, score in top_wildcards.items():
            adviser_id = map_adviser_names_to_ids([adv]).get(adv)
            if adviser_id is None:
                continue
            
            adv_theses = wildcard_df[wildcard_df["adviser_name"] == adv].copy()
            if adv_theses.empty:
                continue

            currentLeaders, availability = get_adviser_current_leaders(adviser_id)

             # Fetch profile details from Supabase
            profile_res = supabase.table("user_profiles") \
                .select("prefix, full_name, suffix, profile_picture, email, position, research_interest, bio") \
                .eq("user_id", adviser_id).execute()
            profile = profile_res.data[0] if profile_res.data else {}

            full_name_with_title = (
                (profile.get("prefix") + " " if profile.get("prefix") else "") +
                profile.get("full_name", adv) +
                (", " + profile.get("suffix") if profile.get("suffix") else "")
            )

            theses_embs = sbert_model.encode(adv_theses["combined_text"].tolist(), show_progress_bar=False)
            sim_scores = cosine_similarity(user_vec_sbert, theses_embs).flatten()
            adv_theses["similarity"] = sim_scores
            top_theses = adv_theses.sort_values(by="similarity", ascending=False).head(5)

            wildcards.append({
                "id": adviser_id,
                "full_name": full_name_with_title,
                "availability": availability,
                "current_leaders": currentLeaders,
                "already_requested": adviser_id in sent_advisers,
                "profile_picture": profile.get("profile_picture"),
                "email": profile.get("email"),
                "position": profile.get("position"),
                "research_interest": profile.get("research_interest"),
                "bio": profile.get("bio"),
                "wildcard_score": float(score),
                "projects": [
                    {
                        "title": row["title"],
                        "similarity": float(row["similarity"]),
                        "abstract": row["abstract"],
                        "status": row.get("status", "active"),
                    }
                    for _, row in top_theses.iterrows()
                ],
        })

        # Return JSON response
        return {
            "recommendations": recommendations,
            "recommended_adviser_ids": recommended_adviser_ids,
            "wildcard_advisers": wildcards
        }

    except Exception as e:
        import traceback
        print("❌ ERROR in /recommend endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
