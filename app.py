## STREAMLIT APP FOR THE CONTENT BASED FILTERING RECOMMENDATION SYSTEM
import streamlit as st
from src.content_based_filtering import recommend
from src.collaborative_based_filtering import collaborative_recommendation
from src.hybrid_recommendation import HybridRecommenderSystem 
from scipy.sparse import load_npz
from numpy import load
import pandas as pd

## LOAD THE CLEANED DATA 
cleaned_data_path = "data/processed/cleaned_music_info.csv"
st.session_state.songs_data = pd.read_csv(cleaned_data_path)

## LOAD THE TRANSFORM DATA 
transformed_data_path = "data/transformed/transformed_music_info.npz"
st.session_state.transformed_data = load_npz(transformed_data_path)

## LOAD THE TRACK IDS
track_ids_path = "data/processed/tracks_ids.npy"
st.session_state.track_ids = load(track_ids_path,allow_pickle=True)

## LOAD THE FILTERED DATA
filtered_data_path = "data/processed/collab_filtered_data.csv"
st.session_state.filtered_data = pd.read_csv(filtered_data_path)

## LOAD THE INTERACTION MATRIX
interaction_matrix_path = "data/processed/interaction_matrix.npz"
st.session_state.interaction_matrix = load_npz(interaction_matrix_path)

## LOAD THE TRANSFORMED HYBRID DATA
transformed_hybrid_data_path = "data/transformed/transformed_hybrid_data.npz"
st.session_state.transformed_hybrid_data = load_npz(transformed_hybrid_data_path)

## TITLE
st.title("Music Recommendation System")

# SUBHEADER
st.write("### Enter a song name to get Personalized and Diverse song recommendations!")

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

if ((st.session_state.filtered_data['name']== song_name) &(st.session_state.filtered_data['artist']== artist_name)).any():
    ## TYPE OF FILTERING
    filtering_type = st.selectbox(label='Select the type of filtering:',
                                options=['Content-Based Filtering',
                                            'Collaborative Filtering',
                                            'Hybrid Recommender System'],
                                index=2)
    ## DIVERSITY SLIDER
    diversity = st.slider(label="Diversity in Recommendations",
                         min_value=1, max_value=10,
                         value=5,step=1)
    content_based_weight = 1 - (diversity / 10)
else:
     ## TYPE OF FILTERING
    filtering_type = st.selectbox(label='Select the type of filtering:',
                                options=['Content-Based Filtering'])
                                            
## BUTTON
if filtering_type == 'Content-Based Filtering':
    if st.button("Get Recommendations"):
        if ((st.session_state.songs_data['name'] == song_name) & (st.session_state.songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")          
            recommendations = recommend(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=st.session_state.songs_data,
                                                     transformed_data=st.session_state.transformed_data,
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
        if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = collaborative_recommendation(song_name=song_name,
                                                           artist_name=artist_name,
                                                           track_ids=st.session_state.track_ids,
                                                           songs_data=st.session_state.filtered_data,
                                                           interaction_matrix=st.session_state.interaction_matrix,
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

elif filtering_type == 'Hybrid Recommender System':
    if st.button('Get Recommendations'):
        if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommender = HybridRecommenderSystem(
                                number_of_recommendations=k,
                                weight_content_based = content_based_weight)
            
            ## GET THE RECOMMENDATIONS
            recommendations = recommender.give_recommendations(song_name=song_name,artist_name=artist_name,
                                                                songs_data=st.session_state.filtered_data,
                                                                transformed_matrix = st.session_state.transformed_hybrid_data,
                                                                track_ids=st.session_state.track_ids,
                                                                interaction_matrix=st.session_state.interaction_matrix)
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
