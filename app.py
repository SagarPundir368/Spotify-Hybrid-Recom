## STREAMLIT APP FOR THE CONTENT BASED FILTERING RECOMMENDATION SYSTEM
import streamlit as st
from src.content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd


## TRANSFORM DATA PATH
transformed_data_path = "data/transformed/transformed_music_info.npz"

## CLEANED DATA PATH
cleaned_data_path = "data/processed/cleaned_music_info.csv"

## LOAD THE CLEANED DATA AND TRANSFORMED DATA
data = pd.read_csv(cleaned_data_path)
transformed_data = load_npz(transformed_data_path)

## TITLE
st.title("Content-Based Music Recommendation System")

# SUBHEADER
st.write("### Enter a song name to get similar song recommendations!")

## TEXT INPUT
song_name = st.text_input("Enter a song name:")
st.write("You entered:", song_name)

## LOWERCASING
song_name = song_name.lower()

## k-recommendations
k = st.selectbox("How many recommendations do you want?", options=[5, 10, 15, 20], index=1)

## BUTTON
if st.button("Get Recommendations"):
    if (data['name'] == song_name).any():
        st.write("Song found in the dataset. Generating recommendations...")
        recommendations = recommend(song_name, data, transformed_data, k)

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
        st.write("Sorry, the song you entered was not found in the dataset. Please try another song.")