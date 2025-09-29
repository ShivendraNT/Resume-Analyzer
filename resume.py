from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from pathlib import Path
from scipy.spatial.distance import cosine
from keybert import KeyBERT

# Loading model
model=SentenceTransformer("all-MiniLM-L6-v2")

# Base directory: where this script sits
BASE_DIR = Path(__file__).resolve().parent
JD_DIR = BASE_DIR / "data" / "jds"
RESUME_DIR = BASE_DIR / "data" 

def load_jds(filename: str):
    file_path = JD_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"JD file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Loading Resumes

file_path=RESUME_DIR / "Resume.csv"
df=pd.read_csv(file_path)
resumes = df["Resume"].dropna().tolist()
category=df['Category'].tolist()
print("Loading Successfully Done")

resume_embedded=model.encode(resumes)

# JDs are in json format but bert expect a continous text

jd_text=""
for i in range(1,16):
    jd = load_jds(f"jd{i}.json")
    if isinstance(jd,dict):
        parts=[]
        if 'title' in jd : parts.append(jd['title'])
        if "location" in jd : parts.append(f"Location: {jd['location']}")
        if 'role' in jd : parts.append(f"Role: {jd['role']}")
        if 'responsibilities' in jd : parts.append("Responsibility: " + " ".join(jd['responsibilities']))
        if 'requirements' in jd : parts.append("Requirements: "+" ".join(jd['requirements']))
        jd_text=jd_text+" ".join(parts)
    else:
        jd_text=str(jd)

# Embedding JDs

jd_embedded=model.encode(jd_text)

# Computing cosine similarities for the first 10 resumes

scores=[]

for i,emb in enumerate(resume_embedded[:]):
    sim=1-cosine(jd_embedded,emb)
    scores.append((i,sim,category[i]))

# Creating Jd