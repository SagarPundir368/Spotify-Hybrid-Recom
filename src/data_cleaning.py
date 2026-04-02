import pandas as pd
import os

DATA_PATH = 'data/raw/Music Info.csv' ## Not use .. becuase we will run this file in the pipeline by 
                                    # dvc repro and it works from the root directory

def clean_data(data):
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    return (
        data
        .drop_duplicates(subset='track_id')
        .drop(columns=['genre', 'spotify_id'])
        .fillna({'tags': 'no_tags'})
        .assign(
            name=lambda x: x['name'].str.lower(),
            artist=lambda x: x['artist'].str.lower(),
            tags=lambda x: x['tags'].str.lower()
        )
        .reset_index(drop=True)
    )

def data_for_content_filtering(data):
    """
    Cleans the input DataFrame by dropping specific columns.

    This function takes a DataFrame and removes the columns "track_id", "name",
    and "spotify_preview_url". It is intended to prepare thedata for content based
    filtering by removing unnecessary features.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing songs information.

    Returns:
    pandas.DataFrame: A DataFrame with the specified columns removed.
    """
    return (
        data
        .drop(columns=["track_id", "name", "spotify_preview_url"])
    ) 

def main(data_path):
    """
    Main function to load, clean, and save data.
    Parameters:
    data_path (str): The file path to the raw data CSV file.
    Returns:
    None
    """
    ## Load the dataset
    data = pd.read_csv(data_path)

    ## perform data cleaning
    cleaned_data = clean_data(data)

    ## Create the directory if it doesn't exist
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    ## save cleaned data
    output_path = os.path.join(output_dir, 'cleaned_music_info.csv')
    ## save cleaned data
    cleaned_data.to_csv('data/processed/cleaned_music_info.csv', index=False)


if __name__ == "__main__":
    main(DATA_PATH)

