from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import json
from pathlib import Path
import random

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
JD_DIR = BASE_DIR / "data" / "jds"
RESUME_FILE = BASE_DIR / "data" / "Resume.csv"
MODEL_SAVE_PATH = BASE_DIR / "models" / "resume_model_finetuned"

# --- Load Resume Data ---
df = pd.read_csv(RESUME_FILE)
resumes = df["Resume"].dropna().tolist()
categories = df["Category"].tolist()
unique_categories = sorted(set(categories))

# --- Helper: Load JD text for a given category ---
def get_jd_text_for_category(category):
    jd_file = JD_DIR / f"jd_{category.lower().replace(' ', '_')}.json"
    if not jd_file.exists():
        # fallback: just use category name as text
        return f"Job Role: {category}"
    with open(jd_file, "r", encoding="utf-8") as f:
        jd = json.load(f)
    parts = []
    if "title" in jd: parts.append(jd["title"])
    if "role" in jd: parts.append(jd["role"])
    if "responsibilities" in jd: parts.append(" ".join(jd["responsibilities"]))
    if "requirements" in jd: parts.append(" ".join(jd["requirements"]))
    return " ".join(parts)

# --- Build Training Examples ---
examples = []
for resume_text, cat in zip(resumes, categories):
    # Positive pair
    jd_text = get_jd_text_for_category(cat)
    examples.append(InputExample(texts=[resume_text, jd_text], label=1.0))

    # Negative pairs (sample 2 random other categories)
    negative_categories = random.sample([c for c in unique_categories if c != cat], 2)
    for neg_cat in negative_categories:
        neg_jd_text = get_jd_text_for_category(neg_cat)
        examples.append(InputExample(texts=[resume_text, neg_jd_text], label=0.0))

print(f"Built {len(examples)} training pairs ({len(resumes)} resumes).")

# --- Train Model ---
model = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

print("Starting fine-tuning...")
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save(str(MODEL_SAVE_PATH))
print(f"Model saved to {MODEL_SAVE_PATH}")
