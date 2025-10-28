from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# ---------------- Supabase setup ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Supabase connection test ----------------
try:
    test = supabase.table("users").select("id, user_id").limit(1).execute()
    if test.data:
        print("Supabase connected successfully. Sample user:", test.data[0])
    else:
        print("Supabase connected but no users found.")
except Exception as e:
    print("Supabase connection failed:", e)


# ---------------- FastAPI setup ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://archivia-official.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load trained models ----------------
vectorizer = joblib.load("vectorizer.pkl")
lr_model = joblib.load("lr_model.pkl")
clf_df = pd.read_pickle("clf_df.pkl")  # Contains columns: TITLE, ABSTRACT, ADVISER, COMBINED_TEXT

# ---------------- Pydantic model ----------------
class Project(BaseModel):
    title: str
    abstract: str
    student_id: str 

# ---------------- Helper: map adviser names to Supabase IDs ----------------
def map_adviser_names_to_ids(adviser_names: list[str]) -> dict[str, str]:
    mapping = {}
    for name in adviser_names:
  
        response = supabase.table("user_profiles").select("user_id").eq("full_name", name).execute()
        if response.data:
            mapping[name] = response.data[0]["user_id"]
        else:
            mapping[name] = None
    return mapping

# ---------------- Helper: get sent advisers for a student ----------------
def get_sent_advisers(student_id: str) -> set[str]:
    response = supabase.table("student_requests") \
        .select("adviser_id") \
        .eq("student_id", student_id) \
        .in_("status", ["pending", "accepted"]) \
        .execute()

    return {r["adviser_id"] for r in response.data} if response.data else set()


# ---------------- Helper: get adviser current leaders ----------------
def get_adviser_current_leaders(adviser_id: str) -> tuple[int, str]:
    response = supabase.table("adviser_current_leaders") \
        .select("current_leaders") \
        .eq("adviser_id", adviser_id) \
        .execute()

    if response.data:
        cap = response.data[0]
        currentLeaders = cap.get("current_leaders", 0)
        availability = "Available" if currentLeaders > 0 else "Unavailable"
        return currentLeaders, availability

    return 0, "Available"  

@app.post("/recommend")
def recommend(project: Project):
    try:
        print("Received project:", project.dict())

        if not project.title.strip() or not project.abstract.strip():
            raise HTTPException(status_code=400, detail="Title and abstract are required.")

        sent_advisers = get_sent_advisers(project.student_id)
        print("🔹 Sent advisers:", sent_advisers)

        user_text = f"{project.title} {project.abstract}"
        user_vec = vectorizer.transform([user_text])
        print("Vectorized text successfully")

        similarities = cosine_similarity(user_vec, vectorizer.transform(clf_df["COMBINED_TEXT"])).flatten()
        print("Calculated similarities")

        clf_df["similarity"] = similarities
        adviser_scores = clf_df.groupby("ADVISER")["similarity"].max().sort_values(ascending=False)
        adviser_names = adviser_scores.head(5).index.tolist()
        print("Adviser scores:", adviser_scores.head(5).to_dict())

        name_to_id = map_adviser_names_to_ids(adviser_names)
        print("Adviser name → ID mapping:", name_to_id)

        results = []

        for adviser_name, score in adviser_scores.head(5).items():
            print(f"Processing adviser: {adviser_name} ({score:.4f})")
            adviser_id = name_to_id.get(adviser_name)

            if adviser_id is None:
                print(f"Skipping adviser '{adviser_name}' — no matching Supabase ID")
                continue

            currentLeaders, availability = get_adviser_current_leaders(adviser_id)
            print(f"Capacity for {adviser_name}: {currentLeaders} ({availability})")

            supabase_user = supabase.table("user_profiles") \
                .select("prefix, full_name, suffix, profile_picture, email, position, research_interest, bio") \
                .eq("user_id", adviser_id).execute()

            user = supabase_user.data[0] if supabase_user.data else {}
            full_name_with_title = f"{user.get('prefix', '') + ' ' if user.get('prefix') else ''}{user.get('full_name', adviser_name)}{', ' + user.get('suffix') if user.get('suffix') else ''}"

            adviser_projects = (
                clf_df[clf_df["ADVISER"] == adviser_name]
                .sort_values(by="similarity", ascending=False)[["TITLE", "ABSTRACT", "similarity"]]
                .to_dict(orient="records")
            )

            already_requested = adviser_id in sent_advisers
            print(f"Adviser '{adviser_name}' already requested:", already_requested)

            results.append({
                "full_name": full_name_with_title,
                "id": adviser_id,
                "score": float(score),
                "availability": availability,
                "current_leaders": currentLeaders,
                "projects": [
                    {"title": p["TITLE"], "abstract": p["ABSTRACT"], "similarity": float(p["similarity"])}
                    for p in adviser_projects
                ],
                "profile_picture": user.get("profile_picture"),
                "email": user.get("email"),
                "position": user.get("position"),
                "research_interest": user.get("research_interest"),
                "bio": user.get("bio"),
                "already_requested": already_requested,
            })

        if not results:
            print("No advisers found in results.")
            raise HTTPException(status_code=404, detail="No advisers found in the database.")

        print("Successfully generated recommendations.")
        return {"recommendations": results}

    except Exception as e:
        import traceback
        print("ERROR OCCURRED IN /recommend ENDPOINT")
        print(traceback.format_exc())  # shows exact stack trace
        raise HTTPException(status_code=500, detail=str(e))
