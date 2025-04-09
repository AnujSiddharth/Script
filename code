import streamlit as st
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text  # for stopwords

# ---------- Preprocessing ----------
def preprocess_script(script_text):
    script_text = script_text.lower()
    script_text = re.sub(r"[^a-z\s]", "", script_text)
    tokens = script_text.split()
    stop_words = text.ENGLISH_STOP_WORDS
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------- Vectorization ----------
def vectorize_scripts(scripts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(scripts)

# ---------- Similarity Calculation ----------
def calculate_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

# ---------- Visualization ----------
def display_heatmap(similarity_matrix, filenames):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=filenames, yticklabels=filenames,
                cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

# ---------- Main App ----------
st.title("🎬 Script Plagiarism Detector")
st.write("Upload multiple movie scripts in `.txt` format to compare their similarity.")

uploaded_files = st.file_uploader("Choose script files", type="txt", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Please upload **at least two** script files to check for plagiarism.")
    else:
        raw_scripts = []
        filenames = []

        for file in uploaded_files:
            text_data = file.read().decode("utf-8")
            raw_scripts.append(preprocess_script(text_data))
            filenames.append(file.name)

        st.success("Scripts uploaded and preprocessed successfully.")

        with st.spinner("Calculating similarity..."):
            tfidf_matrix = vectorize_scripts(raw_scripts)
            similarity_matrix = calculate_similarity(tfidf_matrix)

        st.subheader("🔍 Similarity Heatmap")
        display_heatmap(similarity_matrix, filenames)

        st.subheader("⚠️ Potential Plagiarism Cases")
        threshold = 0.55
        flagged = []
        n = len(filenames)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > threshold:
                    flagged.append((filenames[i], filenames[j], similarity_matrix[i][j]))

        if flagged:
            for file1, file2, score in flagged:
                percentage = score * 100
                st.write(f"📝 **'{file1}' and '{file2}' are {percentage:.0f}% similar.** This may indicate potential plagiarism.")
            
            st.markdown("---")
            st.subheader("📢 What to do if plagiarism is detected?")
            st.markdown("""
If you suspect plagiarism:
- 📁 **Save the similarity report** or screenshot the results.
- 🧑‍🏫 **Inform your course instructor** or TA with the report and files involved.
- 🛑 **Avoid making accusations without evidence** — let the academic team investigate.
- 📧 Many institutions have a plagiarism reporting system or email — check your university's academic integrity policies.
""")
        else:
            st.info("✅ No suspicious similarity scores found above threshold.")
