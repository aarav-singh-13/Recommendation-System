import streamlit as st
import pandas as pd
import os
import pickle
from src.data_processing import load_movie_titles, load_ratings
from src.recommend import generate_user_recommendations, map_recommendations_to_titles
from PIL import Image

# Streamlit App Configuration
st.set_page_config(page_title="Netflix Recommender", layout="wide", page_icon="🎬")

@st.cache_data
def load_data(data_dir):
    movies_df = load_movie_titles(data_dir)
    try:
        ratings_df = load_ratings(data_dir, sample_frac=1.0)
    except FileNotFoundError:
        ratings_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
    return movies_df, ratings_df

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_path = os.path.join(base_dir, "models", "svd_model.pkl")
figures_dir = os.path.join(base_dir, "deliverables", "figures")

st.title("🎬 Netflix Prize Recommendation System")
st.markdown("""
This dashboard demonstrates a personalized recommendation system trained on the Netflix Prize Dataset using Matrix Factorization (SVD).
""")

# Load Data and Model
movies, ratings = load_data(data_dir)
if os.path.exists(model_path):
    svd_model = load_model(model_path)
else:
    svd_model = None
    st.warning("Model not found. Please run `src/train.py` first to generate the model.")

tab1, tab2 = st.tabs(["Personalized Recommendations", "Exploratory Data Analysis (EDA)"])

with tab1:
    st.header("Get Recommendations")
    
    if not ratings.empty and svd_model:
        # Let user select from top 100 most active users to ensure good recommendations
        top_users = ratings['user_id'].value_counts().head(100).index.tolist()
        
        selected_user = st.selectbox("Select a User ID:", top_users)
        
        if selected_user:
            # Show past highly rated movies
            user_ratings = ratings[ratings['user_id'] == selected_user]
            user_favorites = user_ratings[user_ratings['rating'] >= 4].merge(movies, on='movie_id')
            
            st.subheader(f"User {selected_user}'s Favorite Movies")
            if not user_favorites.empty:
                st.dataframe(user_favorites[['title', 'rating']].head(5))
            else:
                st.write("No ratings >= 4 found for this user.")
                
            # Generate Recommendations
            st.subheader("Top-10 Recommended Movies")
            with st.spinner("Generating recommendations..."):
                all_movie_ids = movies['movie_id'].unique()
                user_rated_movies = user_ratings['movie_id'].unique()
                
                raw_recs = generate_user_recommendations(svd_model, selected_user, all_movie_ids, user_rated_movies, n=10)
                enriched_recs = map_recommendations_to_titles(raw_recs, movies)
                
                recs_df = pd.DataFrame(enriched_recs)
                # Display table with formatting
                st.dataframe(recs_df.style.highlight_max(subset=['estimated_rating'], color='lightgreen'))
                
    else:
        st.info("Waiting for data and model to be available...")

with tab2:
    st.header("Exploratory Data Analysis")
    st.write("Insights derived from the dataset.")
    
    col1, col2 = st.columns(2)
    
    if os.path.exists(os.path.join(figures_dir, 'rating_distribution.png')):
        with col1:
            st.image(Image.open(os.path.join(figures_dir, 'rating_distribution.png')), caption="Distribution of Ratings")
    if os.path.exists(os.path.join(figures_dir, 'user_activity.png')):
        with col2:
            st.image(Image.open(os.path.join(figures_dir, 'user_activity.png')), caption="User Activity (Log Scale)")
            
    if os.path.exists(os.path.join(figures_dir, 'content_popularity.png')):
        st.image(Image.open(os.path.join(figures_dir, 'content_popularity.png')), caption="Content Popularity (Long Tail)")
