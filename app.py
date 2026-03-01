import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Universal RAG Assistant", page_icon="🤖")
st.title("🤖 RAG Based  Chatbot")

# ---------------------------
# Initialize session state
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# File uploader (ALL FILES)
# ---------------------------
uploaded_file = st.file_uploader("Upload any file (joblib, audio, video, etc.)")

if uploaded_file is not None:

    file_name = uploaded_file.name
    file_extension = file_name.split(".")[-1].lower()

    # Save file
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    save_path = os.path.join("uploads", file_name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ File '{file_name}' uploaded successfully!")

    # If joblib → load embeddings
    if file_extension == "joblib":
        try:
            st.session_state.df = joblib.load(save_path)
            st.success("✅ Embeddings loaded successfully!")
        except Exception as e:
            st.error(f"Error loading joblib: {e}")

    else:
        st.info("File saved. If you want chatbot to work, upload embeddings.joblib.")

# Stop if embeddings not loaded
if st.session_state.df is None:
    st.warning("⚠ Upload embeddings.joblib to start chatting.")
    st.stop()

df = st.session_state.df

# ---------------------------
# Embedding function
# ---------------------------
def create_embedding(text_list):
    r = requests.post(
        "http://127.0.0.1:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return r.json()["embeddings"]

# ---------------------------
# LLM function
# ---------------------------
def inference(prompt):
    r = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]

# ---------------------------
# Show chat history
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Chat input
# ---------------------------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    question_embedding = create_embedding([user_input])[0]

    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()

    top_results = 5
    max_indx = similarities.argsort()[::-1][:top_results]
    new_df = df.loc[max_indx]

    prompt = f"""
You are a C++ course assistant.

Here are relevant video subtitle chunks:
{new_df[['title','number','start','end','text']].to_json(orient="records")}

User Question:
{user_input}

Answer clearly.
Mention video title, number and timestamp.
If unrelated, say you only answer course questions.
"""

    answer = inference(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)