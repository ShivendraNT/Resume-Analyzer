import json
from pathlib import Path

# Loading model


# Base directory: where this script sits
BASE_DIR = Path(__file__).resolve().parent
JD_DIR = BASE_DIR / "data" / "jds"

def load_jds(filename: str):
    file_path = JD_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"JD file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    try:
        jd = load_jds("jd1.json")  # just filename, no path prefix
        print("Title:", jd.get("title"))
        print("Requirements:", jd.get("requirements"))
    except Exception as e:
        print("Error loading JD:", e)