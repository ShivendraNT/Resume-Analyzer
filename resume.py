TECH_SKILLS = {
    # Programming Languages
    "python", "java", "c", "c++", "c#", "go", "rust", "scala", "javascript", "typescript", "r", "matlab", "bash",
    # Data / ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost", "catboost", "lightgbm", "mlflow",
    "huggingface", "transformers", "spacy", "nltk", "gensim",
    # Data Engineering / Big Data
    "spark", "hadoop", "kafka", "airflow", "dbt",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "snowflake", "bigquery",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "jenkins", "terraform", "ansible",
    # Tools / Visualization
    "powerbi", "tableau", "superset", "excel", "matplotlib", "seaborn", "plotly",
    # Concepts
    "nlp", "cnn", "rnn", "lstm", "gan", "transformer", "deep learning", "machine learning", "ai", "ml",
    # MLOps / CI/CD
    "fastapi", "flask", "django", "rest api", "grpc",
    # Misc
    "git", "github", "linux", "unix"
}



from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from pathlib import Path
from scipy.spatial.distance import cosine
from keybert import KeyBERT

# Loading model
model=SentenceTransformer("models/resume_model_finetuned")
kw_model=KeyBERT(model="all-MiniLM-L6-v2")

# Base directory: where this script sits
BASE_DIR = Path(__file__).resolve().parent
JD_DIR = BASE_DIR / "data" / "jds"
RESUME_DIR = BASE_DIR / "data" 

# Loading all the JDs
def load_all_jds():
    jds = []
    for jd_file in JD_DIR.glob("*.json"):
        with open(jd_file, "r", encoding="utf-8") as f:
            jd = json.load(f)
        parts = []
        if "title" in jd: parts.append(jd["title"])
        if "role" in jd: parts.append(jd["role"])
        if "responsibilities" in jd: parts.append(" ".join(jd["responsibilities"]))
        if "requirements" in jd: parts.append(" ".join(jd["requirements"]))
        jd_text = " ".join(parts)
        jds.append((jd_file.stem, jd_text))
    return jds

# Analysis Function

def analyze_resume(resume_text:str):
    resume_emb=model.encode(resume_text)
    jds=load_all_jds()

    jd_scores=[]
    for jd_name,jd_text in jds:
        jd_emb=model.encode(jd_text)
        sim=1-cosine(jd_emb,resume_emb)
        jd_scores.append([jd_name,jd_text,sim])
    
    #Picking best JD match
    best_jd,best_jd_text,best_score=max(jd_scores,key=lambda x:x[2])

    # Extract Keywords and find missing ones
    raw_keywords = [kw for kw, _ in kw_model.extract_keywords(best_jd_text, top_n=20)]
    jd_keywords = [kw for kw in raw_keywords if kw.lower() in TECH_SKILLS]
    if not jd_keywords:
        jd_keywords = raw_keywords

    missing = [kw for kw in jd_keywords if kw.lower() not in resume_text.lower()]
    present_keywords = [kw for kw in jd_keywords if kw.lower() not in [m.lower() for m in missing]]

# Simple coverage score: ratio of present keywords to total JD keywords
    if jd_keywords:
        coverage_score = len(present_keywords) / len(jd_keywords)
    else:
        coverage_score = 0.0

    rating = round(coverage_score * 10, 1)  # out of 10

    if rating >= 8:
        rating_label = "Excellent ✅"
    elif rating >= 5:
        rating_label = "Decent ⚠️"
    else:
        rating_label = "Needs Improvement ❌"

    return {
        "best_jd": best_jd,
        "similarity": best_score,
        "jd_keywords": jd_keywords,
        "missing_keywords": missing,
        "coverage_score": coverage_score,
        "rating": rating,
        "rating_label": rating_label
    }

# Extract text from file

def extract_text(file):
    import os
    ext=os.path.splitext(file.name)[1].lower()
    text=""

    if ext==".txt":
        text = file.read().decode("utf-8")
    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        pdf = PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return text


if __name__=="__main__":
    import pandas as pd
    df = pd.read_csv(BASE_DIR / "data" / "Resume.csv")
    sample_resume = df["Resume"].iloc[0]
    result = analyze_resume(sample_resume)
    print("\n=== Resume Analysis Result ===")
    print(f"Best Matched JD: {result['best_jd']}")
    print(f"Similarity Score: {result['similarity']:.4f}")
    print(f"JD Keywords: {', '.join(result['jd_keywords'])}")
    print(f"Missing Keywords: {', '.join(result['missing_keywords']) if result['missing_keywords'] else 'None ✅'}")