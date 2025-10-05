# app.py - Content-Based Song Recommendation
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Page setup
st.set_page_config(page_title="🎧 Song Recommender", layout="centered")
st.title("🎵 Content-Based Song Recommendation System")
st.write("Enter a song name and get top 5 similar songs based on audio features.")


# Step 2: Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/spotify.csv").head(50000)  # first 50k rows
    feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    return df, scaled_features

df, scaled_features = load_data()

# Step 3: Recommendation function
def recommend(song_name, n=5):
    lower_songs = df['track_name'].str.lower()
    song_name_lower = song_name.lower()
    
    if song_name_lower not in lower_songs.values:
        st.error(f"Song '{song_name}' not found in the dataset. Try another song from the list.")
        return None
    
    idx = lower_songs[lower_songs == song_name_lower].index[0]
    song_vector = scaled_features[idx].reshape(1, -1)
    sim_scores = cosine_similarity(song_vector, scaled_features).flatten()
    sim_scores[idx] = -1
    top_idx = np.argsort(sim_scores)[-n:][::-1]
    recs = df.iloc[top_idx][['track_name', 'artist_name']]
    return recs

# Step 4: User Input
song_input = st.text_input("Enter a song name (exact title from dataset):")

if st.button("Recommend"):
    results = recommend(song_input)
    if results is not None:
        st.subheader(f"Top 5 songs similar to '{song_input}':")
        st.dataframe(results)

# Step 5: Optional dataset preview
if st.checkbox("Show dataset preview"):
    st.write(df.head())
