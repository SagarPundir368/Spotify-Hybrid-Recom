import pandas as pd

## INPUT PATH
songs_data_path = "data/processed/cleaned_music_info.csv"

## OUTPUT PATH
output_path = "data/processed/unique_songs&artist_combo.csv"

def create_unique_songs_artist_combo(songs_data_path, output_path):
    ## LOAD THE SONGS DATA
    songs_data = pd.read_csv(songs_data_path)

    ## CREATE A NEW DATAFRAME WITH UNIQUE SONG-ARTIST COMBOS
    unique_songs_artists = songs_data[['name', 'artist']].drop_duplicates().reset_index(drop=True)

    ## SAVE THE UNIQUE SONG-ARTIST COMBO DATAFRAME
    unique_songs_artists.to_csv(output_path, index=False)

    return unique_songs_artists

if __name__ == "__main__":
    create_unique_songs_artist_combo(songs_data_path, output_path)