from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from pathlib import Path

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

# just filename, no path prefix
file_path=RESUME_DIR / "Resume.csv"
df=pd.read_csv(file_path)
resumes = df["Resume"].dropna().tolist()
category=df['Category'].tolist()
print("Loading Successfully Done")

resume_embedded=model.encode(resumes)

jd_text=""
for i in range(1,6):
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


jd_embedded=model.encode(jd_text)
print(resume_embedded)
print(jd_embedded)


