## STREAMLIT APP FOR THE CONTENT BASED FILTERING RECOMMENDATION SYSTEM
import streamlit as st
from src.content_based_filtering import recommend
from src.collaborative_based_filtering import collaborative_recommendation
from scipy.sparse import load_npz
from numpy import load
import pandas as pd


## LOAD THE TRANSFORM DATA 
transformed_data_path = "data/transformed/transformed_music_info.npz"
transformed_data = load_npz(transformed_data_path)

## LOAD THE CLEANED DATA 
cleaned_data_path = "data/processed/cleaned_music_info.csv"
songs_data = pd.read_csv(cleaned_data_path)

## LOAD THE TRACK IDS
track_ids_path = "data/processed/tracks_ids.npy"
track_ids = load(track_ids_path,allow_pickle=True)

## LOAD THE FILTERED DATA
filtered_data_path = "data/processed/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_data_path)

## LOAD THE INTERACTION MATRIX
interaction_matrix_path = "data/processed/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

## TITLE
st.title("Music Recommendation System")

# SUBHEADER
st.write("### Enter a song name to get similar or diverse song recommendations!")

## INPUT:- SONG NAME
song_name = st.text_input("Enter a song name:")
st.write("You entered:", song_name)
## INPUT:- ARTIST NAME
artist_name = st.text_input("Enter the Artist name:")
st.write("You entered:", artist_name)

## LOWERCASING
song_name = song_name.lower()
artist_name = artist_name.lower()

## k-recommendations
k = st.selectbox("How many recommendations do you want?", options=[5, 10, 15, 20], index=1)

## TYPE OF FILTERING
filtering_type = st.selectbox('What type of Recommendation you want:',['Content-Based Filtering', 'Collaborative Filtering'])

## BUTTON
if filtering_type == 'Content-Based Filtering':
    if st.button("Get Recommendations"):
        if ((songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")          
            recommendations = recommend(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=songs_data,
                                                     transformed_data=transformed_data,
                                                     k=k)
            ## DISPLAY RECOMMENDATIONS
            for ind, recommendations in recommendations.iterrows():
                song_name = recommendations['name'].title()
                artist_name = recommendations['artist'].title()

                if ind == 0:
                    st.markdown("## Current Playing Song:")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendations['spotify_preview_url'], format='audio/mp3')
                    st.write('---')
                elif ind == 1:
                    st.markdown("## Next Up:")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendations['spotify_preview_url'], format='audio/mp3')
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendations['spotify_preview_url'], format='audio/mp3')
                    st.write('---')
                
        else:
            st.markdown(f"Sorry, the song **{song_name}** was not found in the database. Please try another song.")

elif filtering_type == 'Collaborative Filtering':
    if st.button('Get Recommendations'):
        if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = collaborative_recommendation(song_name=song_name,
                                                           artist_name=artist_name,
                                                           track_ids=track_ids,
                                                           songs_data=filtered_data,
                                                           interaction_matrix=interaction_matrix,
                                                           k=k)
            ## DISPLAY RECOMMENDATIONS
            for ind, recommendations in recommendations.iterrows():
                song_name = recommendations['name'].title()
                artist_name = recommendations['artist'].title()

                if ind == 0:
                    st.markdown("## Current Playing Song:")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendations['spotify_preview_url'], format='audio/mp3')
                    st.write('---')
                elif ind == 1:
                    st.markdown("## Next Up:")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendations['spotify_preview_url'], format='audio/mp3')
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendations['spotify_preview_url'], format='audio/mp3')
                    st.write('---')
                
        else:
            st.markdown(f"Sorry, the song **{song_name}** was not found in the database. Please try another song.")
