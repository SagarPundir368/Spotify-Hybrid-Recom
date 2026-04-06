# 🎵 Hybrid Music Recommendation System

**Personalized & Diverse Song Recommendations**  
*Based on the current song and artist*

A smart **hybrid recommender system** that suggests the next song you’ll love by intelligently combining **Content-Based Filtering** and **Collaborative Filtering**. Built using real Spotify music data from Kaggle (provided by Vijay).

---

## 📊 Dataset

The system is powered by two clean CSV files:

- **Song Information** (`songs.csv` / `track_metadata.csv`):  
  Contains rich metadata for each track — song name, artist(s), album, genre, audio features (danceability, energy, tempo, valence, etc.), release year, and more.

- **User Listening History** (`user_history.csv` / `listening_history.csv`):  
  Contains user-song interactions (user IDs, song IDs, play counts, timestamps, etc.).

Both files are used together to train and evaluate the recommendation models.

---

## 🛠️ System Architecture & Approach

We follow a **three-stage development** process to build a robust hybrid recommender:

### 1. Content-Based Recommendation System
- Uses **only song metadata and audio features** (no user data required).
- For any **currently playing song + artist**, the system computes similarity (cosine similarity / Euclidean distance) with all other songs.
- Recommends the **most similar tracks** based on musical characteristics, genre, artist style, mood, and audio profile.
- **Strength**: Highly personalized to the current listening context and works instantly even for new songs/users (cold-start friendly).

### 2. Collaborative Filtering Recommendation System
- Leverages the **user-song interaction matrix** to capture listening patterns.
- Implemented in two variants:

  **a. User-Based Collaborative Filtering**  
  - Finds users with the **most similar music taste** (using Pearson correlation or cosine similarity on listening history).  
  - Recommends songs that similar users have loved but the current user hasn’t heard yet.

  **b. Item-Based Collaborative Filtering**  
  - Computes **song-to-song similarity** based on how users interact with them.  
  - Recommends songs that are most similar to the ones the user has already played.

- **Strength**: Captures real user behavior and serendipitous discoveries.

### 3. Hybrid Recommendation System (Final Model)
- **Combines both Content-Based and Collaborative Filtering** outputs intelligently.
- Uses a **weighted fusion / ranking aggregation** technique:
  - Takes top-N recommendations from both models.
  - Applies a novel scoring mechanism that balances **musical similarity** (content) with **community preference** (collaborative).
  - Final output: **Personalized + Diverse** song suggestions that feel fresh yet relevant.
- Result: Overcomes limitations of individual approaches (e.g., content-based lacks diversity; collaborative suffers from cold-start).

---

## ✨ Key Features

- ✅ Real-time recommendations based on **currently playing song & artist**
- ✅ Personalized to individual taste
- ✅ Promotes **diversity** to avoid repetitive suggestions
- ✅ Handles both known and new songs (hybrid advantage)
- ✅ Scalable and modular design (easy to extend with deep learning later)

---

## 📁 Project Structure (Planned)

```
HYBRID-REC-PROJECT/
├── .dvc/                          # DVC configuration for data versioning
├── .git/                          # Git repository
├── data/
│   ├── MP3-Example/               # Sample audio files
│   ├── processed/
│   │   ├── gitignore
│   │   ├── cleaned_music_info.csv
│   │   ├── collab_filtered_data.csv
│   │   ├── interaction_matrix.npz
│   │   └── tracks_ids.npy
│   ├── raw/
│   │   ├── gitignore
│   │   ├── Music_Info.csv
│   │   ├── Music_Info.csv.dvc
│   │   ├── User_Listening_History.csv
│   │   └── User_Listening_History.csv.dvc
│   └── transformed/
│       ├── gitignore
│       ├── transformed_hybrid_data.npz
│       └── transformed_music_info.npz
├── hrvenv/
├── models/
│   ├── gitignore
|   └── cb_transformer.joblib
├── notebooks/
│   ├── EDA_Spotify_Dataset.ipynb
│   ├── Spotify_Collaborative_Filtering.ipynb
│   ├── Spotify_Content_Based_Filtering.ipynb
│   ├── dask-expr.svg
│   └── mydask.png
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── collaborative_based_filtering.py
│   ├── content_based_filtering.py
│   ├── data_cleaning.py
│   ├── hybrid_recommendation.py
│   └── transformed_filtered_data.py
├── .dvcignore
├── .gitignore
├── app.py                         # Main application (Streamlit/FastAPI/Gradio)
├── dvc.lock
├── dvc.yaml
├── README.md
├── requirements.txt
└── requirements_deploy.txt
 
```