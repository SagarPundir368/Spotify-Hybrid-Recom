import pandas as pd
from src.data_cleaning import data_for_content_filtering
from src.content_based_filtering import transform_data, save_transformed_data

## PATH OF FILTERED DATA
filtered_data_path = 'data/processed/collab_filtered_data.csv'

## SAVE PATH
save_path = 'data/transformed/transformed_hybrid_data.npz'


def main(data_path,save_path):
    ## LOAD THE DATA
    filtered_data = pd.read_csv(data_path)

    ## CLEAN THE DATA
    filtered_data_cleaned = data_for_content_filtering(filtered_data)

    ## TRANSFORMED THE DATA INTO MATRIX
    transformed_data = transform_data(filtered_data_cleaned)

    ## SAVE THE TRANSFORMED DATA
    save_transformed_data(transformed_data, save_path)


if __name__ == '__main__':
    main(filtered_data_path,save_path)