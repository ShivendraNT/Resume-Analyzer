import streamlit as st
from resume import analyze_resume,extract_text
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Resume Analyzer",page_icon="🤖",layout="wide")
st.title("📄 AI Resume Analyzer")
st.write("Upload your resume to see which JD it matches the best, what skills you're missing, and get a resume rating!")
uploaded_file=st.file_uploader("Upload Your Resume",type=['pdf','txt'])

if uploaded_file:
    with st.spinner("Analysing your resume..."):
        resume_text=extract_text(uploaded_file)
        result=analyze_resume(resume_text)
    
    # Display of result
    st.subheader(f"🔍 Best Matched JD: **{result['best_jd']}**")
    st.metric("Similarity Score",f"{result['similarity']:.4f}")

    st.subheader("📊 Resume Rating")
    st.metric("Coverage", f"{result['rating']}/10", delta=None, help="Based on how many JD keywords are present")
    st.write(f"**Assessment:** {result['rating_label']}")

    st.subheader("🧠 JD Keywords")
    st.write(", ".join(result["jd_keywords"]))

    st.subheader("⚠️ Missing Keywords")
    if result["missing_keywords"]:
        st.error(", ".join(result["missing_keywords"]))
    else:
        st.success("No major missing keywords detected ✅")

    # --- Bar Chart Visualization ---
    present = [kw for kw in result["jd_keywords"] if kw not in result["missing_keywords"]]
    missing = result["missing_keywords"]

    if present or missing:
        df = pd.DataFrame({
            "Keyword": present + missing,
            "Status": ["Present"] * len(present) + ["Missing"] * len(missing)
        })
        df["Count"] = 1  # Dummy value for bar length

        fig, ax = plt.subplots()
        df.groupby(["Keyword", "Status"])["Count"].count().unstack().plot(
            kind="barh", stacked=True, ax=ax, color={"Present": "green", "Missing": "red"}
        )
        ax.set_xlabel("Keyword Presence")
        ax.set_ylabel("Keywords")
        ax.set_title("Resume Keyword Coverage")
        st.pyplot(fig)
        
