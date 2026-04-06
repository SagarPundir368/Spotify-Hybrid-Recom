import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

## CREATING A CLASS

class HybridRecommenderSystem:

    def __init__(self,
                number_of_recommendations: int, 
                weight_content_based):
        
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based
        
    def __calculate_content_based_similarities(self,song_name,artist_name,songs_data,transformed_matrix):
        ## FILTER OUT THE SONG FROM THE DATA
        song_row = songs_data.loc[(songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)]
        ## GET THE INDEX OF THE SONG
        song_idx = song_row.index[0]
        ## GENERATE THE INPUT VECTOR
        input_vector = transformed_matrix[song_idx].reshape(1,-1)
        ## CALCULATE SIMILARITY SCORES
        content_similarity_scores = cosine_similarity(input_vector,transformed_matrix)

        return content_similarity_scores
    
    def __calculate_collaborative_filtering_similarities(self,song_name,artist_name,track_ids,songs_data,interaction_matrix):
        ## FILTER OUT THE SONG FROM THE DATA
        song_row = songs_data[(songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)]
        ## TRACK ID OF INPUT SONG
        input_track_id = song_row['track_id'].values.item()
        ## GET THE INDEX OF THE INPUT TRACK ID
        track_id_idx = np.where(track_ids == input_track_id)[0].item()
        ## GENERATE THE INPUT VECTOR
        input_array = interaction_matrix[track_id_idx]
        ## CALCULATE SIMILARITY SCORES
        collaborative_similarity_scores = cosine_similarity(input_array,interaction_matrix)

        return collaborative_similarity_scores
    
    def __normalize_similarities(self,similarity_scores):
        min = np.min(similarity_scores)
        max = np.max(similarity_scores)
        normalized_scores = (similarity_scores - min)/(max- min)
        return normalized_scores
    
    def __weighted_combination(self,content_based_scores,collaborative_filtering_scores):
        weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)
        return weighted_scores
    
    def give_recommendations(self,song_name,
                                artist_name,songs_data, track_ids,                                 
                                transformed_matrix,interaction_matrix):
        ## CONTENT BASED SIMILARITIES
        content_based_similarities = self.__calculate_content_based_similarities(song_name= song_name, 
                                                                               artist_name= artist_name, 
                                                                               songs_data= songs_data, 
                                                                               transformed_matrix= transformed_matrix)
        
        ## COLLABORAIVE FILTERING SIMILARITIES
        collaborative_filtering_similarities = self.__calculate_collaborative_filtering_similarities(song_name= song_name, 
                                                                                                   artist_name= artist_name, 
                                                                                                   track_ids= track_ids, 
                                                                                                   songs_data= songs_data,
                                                                                                   interaction_matrix= interaction_matrix)

        ## NORMALIZING SCORES
        normalize_content_based_similarities = self.__normalize_similarities(content_based_similarities)
        normalize_collaborative_filtering_similarities = self.__normalize_similarities(collaborative_filtering_similarities)

        ## WEIGHTED COMBINATION OF SIMILARITIES
        weighted_scores = self.__weighted_combination(normalize_content_based_similarities,normalize_collaborative_filtering_similarities)

        ## INDEX VALUES OF RECOMMENDATION
        recommendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]

        ## GET TOP K RECOMMENDATIONS
        recommendation_track_ids = track_ids[recommendation_indices]

        ## GET TOP SCORES
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]
        
        # get the songs from data and print
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
    
                                 