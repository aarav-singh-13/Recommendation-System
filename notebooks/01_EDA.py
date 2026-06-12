import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data_processing import load_movie_titles, load_ratings

def perform_eda(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data for EDA...")
    try:
        ratings_df = load_ratings(data_dir, sample_frac=1.0)
        movies_df = load_movie_titles(data_dir)
    except FileNotFoundError:
        print("Data files not found. Please ensure data is present.")
        return
        
    print(f"Total Ratings: {len(ratings_df)}")
    print(f"Total Users: {ratings_df['user_id'].nunique()}")
    print(f"Total Movies: {ratings_df['movie_id'].nunique()}")
    
    # 1. Rating Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=ratings_df, palette='viridis')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating (1-5 Stars)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
    plt.close()
    
    # 2. User Activity (Ratings per User)
    user_activity = ratings_df.groupby('user_id').size()
    plt.figure(figsize=(8, 5))
    sns.histplot(user_activity, bins=50, kde=True, color='blue')
    plt.title('User Activity: Number of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Frequency')
    # Use log scale since it's usually highly skewed
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'user_activity.png'))
    plt.close()
    
    # 3. Content Popularity (Ratings per Movie)
    movie_popularity = ratings_df.groupby('movie_id').size().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.plot(movie_popularity.values)
    plt.title('Content Popularity: Ratings per Movie (Long Tail)')
    plt.xlabel('Movies (ordered by popularity)')
    plt.ylabel('Number of Ratings')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'content_popularity.png'))
    plt.close()
    
    # 4. Sparsity Calculation
    total_possible_ratings = ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique()
    actual_ratings = len(ratings_df)
    sparsity = 1.0 - (actual_ratings / total_possible_ratings)
    
    with open(os.path.join(output_dir, 'eda_summary.txt'), 'w') as f:
        f.write(f"Total Ratings: {actual_ratings}\n")
        f.write(f"Total Users: {ratings_df['user_id'].nunique()}\n")
        f.write(f"Total Movies: {ratings_df['movie_id'].nunique()}\n")
        f.write(f"Dataset Sparsity: {sparsity * 100:.4f}%\n")
        
    print("EDA Complete. Plots saved to", output_dir)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'deliverables', 'figures')
    perform_eda(data_dir, output_dir)
