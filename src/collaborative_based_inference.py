import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ONLY INFERENCE LOGIC
def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    ## LOWERCASING
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    ## FETCH THE ROW FROM THE SONGS DATA
    song_row = songs_data.loc[
        (songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)
    ]

    if song_row.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' not found.")

    ## TRACK ID OF INPUT SONG
    input_track_id = song_row['track_id'].values.item()

    ## INDEX VALUE OF TRACK ID
    ind = np.where(track_ids == input_track_id)[0].item()

    ## FETCH THE INPUT VECTOR
    input_array = interaction_matrix[ind]

    # GET THE SIMILARITY SCORES
    similarity_scores = cosine_similarity(input_array, interaction_matrix)

    # INDEX VALUE OF RECOMMENDATION
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

    # GET TOP K RECOMMENDATIONS
    recommendation_track_ids = track_ids[recommendation_indices]

    # GET TOP SCORES
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]

    # CREATE SCORE DATAFRAME
    scores_df = pd.DataFrame({
        "track_id": recommendation_track_ids.tolist(),
        "score": top_scores
    })

    # MERGE WITH SONG DATA
    top_k_songs = (
        songs_data
        .loc[songs_data["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )

    return top_k_songs