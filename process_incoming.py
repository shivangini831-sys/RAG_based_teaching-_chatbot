import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from read_chunks import create_embedding
import numpy as np
import joblib
import requests

def create_embedding(text_list):
    r = requests.post(
        "http://127.0.0.1:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    embeddings = r.json()['embeddings']
    return embeddings

def inference(prompt):
    r = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.2",   # ✅ removed duplicate model
            "prompt": prompt,     # ✅ added missing comma
            "stream": False
        }
    )
    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question:")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [question_embedding]
).flatten()

print(similarities)

top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]

new_df = df.loc[max_indx]

# ✅ Made it f-string so variables work
prompt = f'''  I am teaching c++. Here are video subtitle chunks containing video title,video number ,start time in second,end time in second, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
........................

"{incoming_query}"
user asked this question related to the video chunks, you have to answer in a human way (dont mention the above format,its just for you where and how much content is taught in which video where (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course

'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = (inference(prompt)["response"])
print(response)

with open("response.txt", "w") as f:
    f.write(response)