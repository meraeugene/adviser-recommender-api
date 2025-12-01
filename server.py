from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
import pandas as pd
import numpy as np
import re
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

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
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
    return text.lower().strip()

# ===============================================================
# Load Thesis Data
# ===============================================================
response = supabase.table("ml_thesis_view").select("*").execute()
df = pd.DataFrame(response.data)
df = df[df["status"]=="active"].copy()
panel_columns = ["panel_member1","panel_member2","panel_member3"]

# ===============================================================
# SBERT Setup
# ===============================================================
print("Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to {device}")

# Precompute embeddings at API startup
print("Precomputing SBERT embeddings for all theses...")
df["combined_text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).apply(clean_text)
df["embedding"] = list(sbert_model.encode(df["combined_text"].tolist(), convert_to_tensor=True))
print(f"{len(df)} thesis embeddings precomputed.")

# ===============================================================
# Precompute Wildcard Adviser Embeddings
# ===============================================================
print("Precomputing wildcard adviser embeddings...")
df_users = pd.DataFrame(supabase.table("user_profiles").select("user_id, full_name, research_interest").execute().data)
df_users["research_interest_clean"] = df_users["research_interest"].fillna("").apply(clean_text)

df_users_nonempty = df_users[df_users["research_interest_clean"].str.strip() != ""].copy()
if not df_users_nonempty.empty:
    df_users_nonempty["embedding"] = list(
        sbert_model.encode(df_users_nonempty["research_interest_clean"].tolist(), convert_to_tensor=True)
    )
print(f"{len(df_users_nonempty)} wildcard adviser embeddings precomputed.")

# ===============================================================
# Precompute mappings for fast lookup
# ===============================================================
# Adviser -> theses
adviser_to_theses = {adv: df[df["adviser_name"] == adv].copy() for adv in df["adviser_name"].unique()}

# Adviser -> panel membership counts
panel_membership = {}
for adv in df["adviser_name"].unique():
    panel_count = len(df[df[panel_columns].apply(lambda r: adv in r.values, axis=1)])
    panel_membership[adv] = panel_count

# Adviser name -> user_id mapping
all_user_profiles = pd.DataFrame(supabase.table("user_profiles")
                                 .select("user_id, full_name, prefix, suffix, profile_picture, email, position, research_interest, bio")
                                 .execute().data)
name_to_id_global = {r["full_name"]: r["user_id"] for r in all_user_profiles.to_dict(orient="records")}

# ===============================================================
# FastAPI Setup
# ===============================================================
app = FastAPI(title="SBERT Adviser Recommender API", version="1.0")
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
# Supabase helper functions
# ===============================================================
def get_sent_advisers(student_id: str) -> set[str]:
    res = supabase.table("student_requests").select("adviser_id").eq("student_id", student_id)\
        .in_("status", ["pending", "accepted"]).execute()
    return {r["adviser_id"] for r in res.data} if res.data else set()

def get_adviser_current_leaders(adviser_id: str):
    res = supabase.table("adviser_current_leaders").select("current_leaders, max_limit").eq("adviser_id", adviser_id).execute()
    if res.data:
        cap = res.data[0]
        currentLeaders = cap.get("current_leaders", 0)
        limit = cap.get("max_limit", 0)
        is_full = limit > 0 and currentLeaders >= limit
        availability = "Unavailable" if is_full else "Available"
        return currentLeaders, limit, is_full, availability
    return 0, 0, False, "Available"


# ===============================================================
# Optimized Recommendation Endpoint (Precomputed Embeddings)
# ===============================================================
@app.post("/recommend")
def recommend(project: Project):
    try:
        title, abstract = project.title.strip(), project.abstract.strip()
        if not title or not abstract:
            raise HTTPException(status_code=400, detail="Title and abstract are required.")

        user_text = clean_text(title + " " + abstract)
        user_embedding = sbert_model.encode([user_text], convert_to_tensor=True)
        sent_advisers = get_sent_advisers(project.student_id)

        advisers = list(adviser_to_theses.keys())
        adviser_scores, adviser_top_thesis, topic_experience = {}, {}, {}

        # ================== Vectorized Similarity ==================
        all_embeddings = torch.cat([torch.stack(adviser_to_theses[adv]["embedding"].tolist()) for adv in advisers])
        all_adviser_index = []
        for adv in advisers:
            all_adviser_index.extend([adv]*len(adviser_to_theses[adv]))
        cos_scores_all = torch.nn.functional.cosine_similarity(
            user_embedding.repeat(all_embeddings.shape[0],1), all_embeddings
        ).cpu().numpy()

        df_scores = pd.DataFrame({"adviser": all_adviser_index, "score": cos_scores_all})

        for adv in advisers:
            adv_scores = df_scores[df_scores["adviser"] == adv]["score"].values
            adv_thesis_df = adviser_to_theses[adv]

            if len(adv_scores) == 0:
                adviser_scores[adv] = 0
                adviser_top_thesis[adv] = {"title": "", "similarity": 0}
                topic_experience[adv] = 0
                continue

            # Top 5 similar theses for this adviser
            top_idx = np.argsort(adv_scores)[::-1][:5]
            top_cos = adv_scores[top_idx]

            # Adviser score combining mean and max similarity
            adviser_scores[adv] = 0.7 * np.mean(top_cos) + 0.3 * np.max(top_cos)

            # Record top thesis
            top_thesis_idx = np.argmax(adv_scores)
            adviser_top_thesis[adv] = {
                "title": adv_thesis_df.iloc[top_thesis_idx]["title"],
                "similarity": float(adv_scores[top_thesis_idx])
            }

            # ===== New Experience Calculation =====
            points = 0

            for idx in top_idx:
                thesis_row = adv_thesis_df.iloc[idx]
                # 1 point if adviser supervised it
                if thesis_row["adviser_name"] == adv:
                    points += 1
                # 0.3 points for each panel membership
                points += sum([0.3 for col in panel_columns if adv == thesis_row[col]])
                # # Optional: similarity contribution
                points += 0.5 * adv_scores[idx]
    
            topic_experience[adv] = points

        # Normalize experience
        max_exp = max(topic_experience.values()) if topic_experience else 1
        topic_exp_norm = {k: v / max_exp for k, v in topic_experience.items()}

        total_scores = {adv: 0.9*adviser_scores[adv]+0.1*topic_exp_norm[adv] for adv in advisers}

        top_advisers = sorted(total_scores.items(), key=lambda x:x[1], reverse=True)[:5]
        top_adviser_names = [a for a,_ in top_advisers]

        # ================== Results ==================
        top_profiles = all_user_profiles[all_user_profiles["full_name"].isin(top_adviser_names)].set_index("full_name").to_dict(orient="index")
        results = []

        for adv in top_adviser_names:
            profile = top_profiles.get(adv, {})
            adviser_id = profile.get("user_id", name_to_id_global.get(adv))
            if not adviser_id:
                continue
            currentLeaders, limit, is_full, availability = get_adviser_current_leaders(adviser_id)

            adv_theses = adviser_to_theses[adv].copy()
            adv_theses = adv_theses.assign(
                similarity=lambda x: torch.nn.functional.cosine_similarity(
                    user_embedding.repeat(len(x),1), torch.stack(x["embedding"].tolist())
                ).cpu().numpy()
            ).sort_values("similarity", ascending=False).head(5)

            full_name = (profile.get("prefix","") + " " if profile.get("prefix") else "") + \
                        profile.get("full_name", adv) + \
                        (", " + profile.get("suffix") if profile.get("suffix") else "")

            results.append({
                "id": adviser_id,
                "full_name": full_name,
                "score": float(total_scores[adv]),
                "current_leaders": currentLeaders,
                "limit": limit,
                "is_full": is_full,
                "availability": availability,
                "already_requested": adviser_id in sent_advisers,
                "profile_picture": profile.get("profile_picture"),
                "email": profile.get("email"),
                "position": profile.get("position"),
                "research_interest": profile.get("research_interest"),
                "bio": profile.get("bio"),
                "projects": [
                {"title": row["title"], "abstract": row["abstract"], "similarity": float(row["similarity"])}
                for _, row in adv_theses.iterrows()
                ],
            })

       # ================== Wildcards ==================
        wildcards = []

        df_wildcards = df_users_nonempty[
        ~df_users_nonempty["full_name"].isin(top_adviser_names)
        ].copy()


        if not df_wildcards.empty:
            ri_embeddings = torch.stack(df_wildcards["embedding"].tolist())
            user_emb_norm = user_embedding / user_embedding.norm(dim=1, keepdim=True)
            ri_emb_norm = ri_embeddings / ri_embeddings.norm(dim=1, keepdim=True)

            sims = torch.matmul(user_emb_norm, ri_emb_norm.T).squeeze(0).cpu().numpy()

            df_wildcards.loc[:, "wildcard_score"] = sims

            top_wildcards = df_wildcards.sort_values(
                "wildcard_score", ascending=False
            ).head(3)

            for _, row in top_wildcards.iterrows():   
                adv_id = row["user_id"]
                currentLeaders, limit, is_full, availability = get_adviser_current_leaders(adv_id)

                profile_rows = all_user_profiles[all_user_profiles["user_id"] == adv_id]
                if profile_rows.empty:
                    continue
                profile = profile_rows.to_dict(orient="records")[0]


                full_name = (
                    (profile.get("prefix","") + " " if profile.get("prefix") else "")
                    + profile.get("full_name", row["full_name"])
                    + (", " + profile.get("suffix") if profile.get("suffix") else "")
                )

                wildcards.append({   
                    "id": adv_id,
                    "full_name": full_name,
                    "current_leaders": currentLeaders,
                    "limit": limit,
                    "is_full": is_full,
                    "availability": availability,
                    "already_requested": adv_id in sent_advisers,
                    "profile_picture": profile.get("profile_picture"),
                    "email": profile.get("email"),
                    "position": profile.get("position"),
                    "research_interest": profile.get("research_interest"),
                    "bio": profile.get("bio"),
                    "wildcard_score": float(row["wildcard_score"])
                })


        # ================== Radar Chart Data (Frontend Ready) ==================
        radar_data = []

        for adv in top_adviser_names:
            top_thesis = adviser_top_thesis[adv] 
            radar_data.append({
                "name": adv,
                "similarity": float(adviser_scores[adv]),
                "experience": float(topic_exp_norm[adv]),
                "overall": float(total_scores[adv]),
                 "top_project": {
                "title": top_thesis["title"],
                "similarity": float(top_thesis["similarity"])
                }
            })


        # ================== Explanations ==================
        top_similarity_adv = max({k:adviser_scores[k] for k in top_adviser_names}, key=lambda k: adviser_scores[k])
        top_experience_adv = max({k:topic_exp_norm[k] for k in top_adviser_names}, key=lambda k: topic_exp_norm[k])
        top_overall_adv = max({k:total_scores[k] for k in top_adviser_names}, key=lambda k: total_scores[k])
        top1_adv = top_adviser_names[0]

        paragraph_1 = (
            f"We recommend these advisers because their past research is closely related to your thesis topic. "
            f"{top_similarity_adv} has the most similar past projects to your idea."
        )
        paragraph_2 = (
            f"\n\nRegarding experience, {top_experience_adv} has the highest topic-based experience "
            "(supervised or paneled on similar theses), which means they can provide valuable guidance based on prior experience."
        )
        paragraph_3 = (
            f"\n\nLooking at the most closely matching past thesis, {adviser_top_thesis[top1_adv]['title']} "
            f"was supervised by {top1_adv} with a similarity score of {adviser_top_thesis[top1_adv]['similarity']:.3f}."
        )
        paragraph_4 = (
            f"\n\nOverall, {top_overall_adv} is the best match considering both similarity and experience. "
            "All advisers above are strong candidates to guide your thesis."
        )

        overall_explanation = paragraph_1 + paragraph_2 + paragraph_3 + paragraph_4

        
        # ---------------------- Top 1 Adviser Detailed Paragraph ----------------------
        adv_theses_count = len(adviser_to_theses[top1_adv])
        panel_count = panel_membership.get(top1_adv, 0)
        top1_thesis = adviser_top_thesis[top1_adv]

        top1_paragraph = (
            f"{top1_adv} is recommended as your top adviser because their research experience closely matches your thesis topic. "
            f"They have supervised {adv_theses_count} theses similar to your research area and served as panel member for {panel_count} related theses. "
            f"Notably, one of their past theses, '{top1_thesis['title']}', has a semantic similarity score of {top1_thesis['similarity']:.3f}. "
            "This combination of experience and topic relevance makes them highly suitable to guide you."
        )

        return {
            "recommendations": results,
            "recommended_adviser_ids": [name_to_id_global[a] for a in top_adviser_names],
            "wildcard_advisers": wildcards,
            "radar_data": radar_data,
            # "pie_chart_base64": pie_base64,
            "overall_explanation": overall_explanation,
            "top1_adviser_explanation": top1_paragraph
        }

    except Exception as e:
        import traceback
        print("ERROR in /recommend endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
