# =========================
# IMPORTS
# =========================
import os
import subprocess
import streamlit as st
from src.content_based_filtering import recommend
from src.collaborative_based_inference import collaborative_recommendation
from src.hybrid_recommendation import HybridRecommenderSystem
from scipy.sparse import load_npz
from numpy import load
import pandas as pd


# =========================
# ENSURE DATA (DVC PULL)
# =========================
def ensure_data():
    check_path = "data/processed/interaction_matrix.npz"

    # if file exists → local run
    if os.path.exists(check_path):
        return

    # else → pull from DVC remote
    st.info("Downloading data from DVC remote...")

    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)

    if result.returncode != 0:
        st.error("DVC pull failed")
        st.text(result.stderr)
        st.stop()


# CALL BEFORE LOADING DATA
ensure_data()


# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_all_data():
    return {
        "songs_data": pd.read_csv("data/processed/cleaned_music_info.csv"),
        "transformed_data": load_npz("data/transformed/transformed_music_info.npz"),
        "track_ids": load("data/processed/tracks_ids.npy", allow_pickle=True),
        "filtered_data": pd.read_csv("data/processed/collab_filtered_data.csv"),
        "interaction_matrix": load_npz("data/processed/interaction_matrix.npz"),
        "transformed_hybrid_data": load_npz("data/transformed/transformed_hybrid_data.npz"),
    }


data = load_all_data()

# STORE IN SESSION STATE
st.session_state.songs_data = data["songs_data"]
st.session_state.transformed_data = data["transformed_data"]
st.session_state.track_ids = data["track_ids"]
st.session_state.filtered_data = data["filtered_data"]
st.session_state.interaction_matrix = data["interaction_matrix"]
st.session_state.transformed_hybrid_data = data["transformed_hybrid_data"]


# =========================
# UI
# =========================
st.title("Music Recommendation System")
st.write("### Enter a song name to get Personalized and Diverse song recommendations!")

# INPUTS
song_name = st.text_input("Enter a song name:")
artist_name = st.text_input("Enter the Artist name:")

# LOWERCASE
song_name = song_name.lower()
artist_name = artist_name.lower()

# NUMBER OF RECOMMENDATIONS
k = st.selectbox("How many recommendations do you want?", options=[5, 10, 15, 20], index=1)


# =========================
# FILTER OPTIONS
# =========================
if ((st.session_state.filtered_data['name'] == song_name) &
    (st.session_state.filtered_data['artist'] == artist_name)).any():

    filtering_type = st.selectbox(
        'Select the type of filtering:',
        ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'],
        index=2
    )

    diversity = st.slider("Diversity in Recommendations", 1, 10, 5, 1)
    content_based_weight = 1 - (diversity / 10)

else:
    filtering_type = st.selectbox(
        'Select the type of filtering:',
        ['Content-Based Filtering']
    )


# =========================
# CONTENT-BASED
# =========================
if filtering_type == 'Content-Based Filtering':
    if st.button("Get Recommendations"):

        if ((st.session_state.songs_data['name'] == song_name) &
            (st.session_state.songs_data['artist'] == artist_name)).any():

            st.write(f"Recommendations for **{song_name}** by **{artist_name}**")

            recommendations = recommend(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=st.session_state.songs_data,
                transformed_data=st.session_state.transformed_data,
                k=k
            )

            for ind, row in recommendations.iterrows():
                name = row['name'].title()
                artist = row['artist'].title()

                if ind == 0:
                    st.markdown("## Current Playing Song:")
                elif ind == 1:
                    st.markdown("## Next Up:")

                st.markdown(f"#### {ind}. **{name}** by **{artist}**")
                st.audio(row['spotify_preview_url'], format='audio/mp3')
                st.write('---')

        else:
            st.error("Song not found. Try another one.")


# =========================
# COLLABORATIVE
# =========================
elif filtering_type == 'Collaborative Filtering':
    if st.button("Get Recommendations"):

        if ((st.session_state.filtered_data["name"] == song_name) &
            (st.session_state.filtered_data["artist"] == artist_name)).any():

            st.write(f"Recommendations for **{song_name}** by **{artist_name}**")

            recommendations = collaborative_recommendation(
                song_name=song_name,
                artist_name=artist_name,
                track_ids=st.session_state.track_ids,
                songs_data=st.session_state.filtered_data,
                interaction_matrix=st.session_state.interaction_matrix,
                k=k
            )

            for ind, row in recommendations.iterrows():
                name = row['name'].title()
                artist = row['artist'].title()

                if ind == 0:
                    st.markdown("## Current Playing Song:")
                elif ind == 1:
                    st.markdown("## Next Up:")

                st.markdown(f"#### {ind}. **{name}** by **{artist}**")
                st.audio(row['spotify_preview_url'], format='audio/mp3')
                st.write('---')

        else:
            st.error("Song not found. Try another one.")


# =========================
# HYBRID
# =========================
elif filtering_type == 'Hybrid Recommender System':
    if st.button("Get Recommendations"):

        if ((st.session_state.filtered_data["name"] == song_name) &
            (st.session_state.filtered_data["artist"] == artist_name)).any():

            st.write(f"Recommendations for **{song_name}** by **{artist_name}**")

            recommender = HybridRecommenderSystem(
                number_of_recommendations=k,
                weight_content_based=content_based_weight
            )

            recommendations = recommender.give_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=st.session_state.filtered_data,
                transformed_matrix=st.session_state.transformed_hybrid_data,
                track_ids=st.session_state.track_ids,
                interaction_matrix=st.session_state.interaction_matrix
            )

            for ind, row in recommendations.iterrows():
                name = row['name'].title()
                artist = row['artist'].title()

                if ind == 0:
                    st.markdown("## Current Playing Song:")
                elif ind == 1:
                    st.markdown("## Next Up:")

                st.markdown(f"#### {ind}. **{name}** by **{artist}**")
                st.audio(row['spotify_preview_url'], format='audio/mp3')
                st.write('---')

        else:
            st.error("Song not found. Try another one.")