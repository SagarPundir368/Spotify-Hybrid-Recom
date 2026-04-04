import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

## SET PATHS
### output paths
track_ids_save_path = "data/processed/tracks_ids.npy"
filtered_data_save_path = "data/processed/collab_filtered_data.csv"
interaction_matrix_save_path = "data/processed/interaction_matrix.npz"

### input paths
songs_data_path = "data/processed/cleaned_music_info.csv"
user_listening_history_data_path = "data/raw/User Listening History.csv"


def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path:str) -> pd.DataFrame:
    """
        Filter the songs data for the given track ids
    """
    ## FILTER DATA BASED ON THE TRACK IDS
    filtered_data = songs_data[songs_data['track_id'].isin(track_ids)]
    ## SORT THE DATA BY TRACK ID
    filtered_data.sort_values(by='track_id',inplace=True)
    ## RESET INDEX
    filtered_data.reset_index(drop=True,inplace=True)
    ## SAVE THE DATA
    filtered_data.to_csv(save_df_path, index=False)

    return filtered_data


def create_interaction_matrix(history_data:dd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
    ## MAKE A COPY OF DATA
    df = history_data.copy()
    
    ## CONVERT THE PLAYCOUNT COLUMN TO FLOAT
    df['playcount'] = df['playcount'].astype(np.float64)
    
    ## CONVERT STRING COLUMN TO CATEGORICAL
    df = df.categorize(columns=['user_id', 'track_id'])
    
    ## CONVERT USER_ID AND TRACK_ID TO NUMERICAL INDICES
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes
    
    ## GET THE LIST OF TRACK_IDS
    track_ids = df['track_id'].cat.categories.values
    
    # SAVE THE CATEGORIES
    np.save(track_ids_save_path, track_ids, allow_pickle=True)
    
    # ADD THE INDEX COLUMNS TO THE DF
    df = df.assign(
        user_idx=user_mapping,
        track_idx=track_mapping
    )
    
    # CREATE THE INTERACTION MATRIX
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    
    # cOMPUTE THE MATRIX
    interaction_matrix = interaction_matrix.compute()
    
    # GET THE INDICES TO FORM SPARSE MATRIX
    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']
    
    # GET THE SHAPE OF SPARSE MATRIX
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()
    
    # CREATE THE SPARSE MATRIX
    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    
    # SAVE THE SPARSE MATRIX
    save_npz(save_matrix_path, interaction_matrix)



def collaborative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k=5):
    ## LOWERCASING
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    ## FETCH THE ROW FROM THE SONGS DATA
    song_row = songs_data.loc[(songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)]
    ## TRACK ID OF INPUT SONG
    input_track_id = song_row['track_id'].values.item()
    ## INDEX VALUE OF TRACK ID
    ind = np.where(track_ids==input_track_id)[0].item()
    ## FETCH THE INPUT VECTOR
    input_array = interaction_matrix[ind]
    # GET THE SIMILARITY SCORES
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    # INDEX VALUE OF RECOMMENDATION
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # GET TOP K RECOM
    recommendation_track_ids = track_ids[recommendation_indices]
    # GET TOP SCORES
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    # GET THE SONGS FROM DATA AND PRINT
    scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
                            "score":top_scores})
    
    top_k_songs = (
                    songs_data
                    .loc[songs_data["track_id"].isin(recommendation_track_ids)]
                    .merge(scores_df,on="track_id")
                    .sort_values(by="score",ascending=False)
                    .drop(columns=["track_id","score"])
                    .reset_index(drop=True)
                    )
    
    return top_k_songs


def main():
    # LOAD THE USER DATA
    user_data = dd.read_csv(user_listening_history_data_path)
    
    # GET THE UNIQUE TRACK IDS
    unique_track_ids = user_data.loc[:,"track_id"].unique().compute()
    unique_track_ids = unique_track_ids.tolist()
    # FILTER THE SONGS DATA
    songs_data = pd.read_csv(songs_data_path)
    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    
    # CREATING THE INTERACTION MATRIX
    create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)


if __name__ == "__main__":
    main()