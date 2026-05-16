# =========================
# IMPORTS
# =========================
import streamlit as st
from src.content_based_filtering import recommend
from src.collaborative_based_inference import collaborative_recommendation
from src.hybrid_recommendation import HybridRecommenderSystem
from scipy.sparse import load_npz
from numpy import load
import pandas as pd

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
        "unique_songs_artists": pd.read_csv("data/processed/unique_songs&artist_combo.csv")
    }


data = load_all_data()

# STORE IN SESSION STATE
st.session_state.songs_data = data["songs_data"]
st.session_state.transformed_data = data["transformed_data"]
st.session_state.track_ids = data["track_ids"]
st.session_state.filtered_data = data["filtered_data"]
st.session_state.interaction_matrix = data["interaction_matrix"]
st.session_state.transformed_hybrid_data = data["transformed_hybrid_data"]
st.session_state.unique_songs_artists = data["unique_songs_artists"]

# =========================
# UI
# =========================
st.title("Music Recommendation System")
st.info("💡 **Get Started:** Enter a song name below to get personalized and diverse song recommendations!")

## INPUTS
song_list = st.session_state.unique_songs_artists['name'].dropna().unique().tolist()

song_name = st.selectbox("Enter a song name:", 
                          options=song_list,
                          index=None,
                          placeholder="Type a song name..."
)

# Handle Artist Selectbox dynamically only if song is selected
artist_name = None
if song_name:
    artist_list = st.session_state.unique_songs_artists[
        st.session_state.unique_songs_artists['name'] == song_name
    ]['artist'].dropna().unique().tolist()
    
    artist_name = st.selectbox("Select an artist:", options=artist_list)


# 4. Display and backend processing
if song_name and artist_name:
    
    # Create beautifully capitalized versions JUST for the UI display
    display_song = song_name.title()
    display_artist = artist_name.title()
    
    # Show the clean title-cased text to the user
    st.success(f"You selected: **{display_song}** by {display_artist}")
    st.subheader(f"🎵 Recommendations for {display_song} by {display_artist}")
    
    # NOW convert the original variables to lowercase for your backend/model logic
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

                st.subheader(
                f"🎵 Recommendations for {song_name.title()} by {artist_name.title()}"
                )
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