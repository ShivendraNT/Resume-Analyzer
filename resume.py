from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

# Loading a pretrained model
model=SentenceTransformer("all-MiniLM-L6-v2")

# Loading JDs

BASE_DIR = Path(__file__).resolve().parent  # folder where resume.py is
JD_DIR = BASE_DIR / "data" / "jds"

def load_jds(path):
    with open (path,'r')as f:
        return json.load(f)
    
jd=load_jds("jd1.json")
print(jd['title'])
print(jd['requirements'])