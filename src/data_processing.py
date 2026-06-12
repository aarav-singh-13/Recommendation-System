import pandas as pd
import numpy as np
import os
import glob

def load_movie_titles(data_dir):
    """
    Load movie titles from movie_titles.csv
    Returns a DataFrame with columns: ['movie_id', 'year', 'title']
    """
    file_path = os.path.join(data_dir, 'movie_titles.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected to find movie_titles.csv in {data_dir}")
        
    df = pd.read_csv(
        file_path, 
        encoding='iso-8859-1', 
        header=None, 
        names=['movie_id', 'year', 'title'],
        on_bad_lines='skip'
    )
    return df

def load_ratings(data_dir, max_files=None, sample_frac=1.0):
    """
    Load ratings from combined_data_*.txt files.
    Because the file format is custom (MovieID: followed by UserID,Rating,Date), 
    we need to parse it carefully.
    
    Returns a DataFrame with columns: ['movie_id', 'user_id', 'rating', 'date']
    """
    files = glob.glob(os.path.join(data_dir, 'combined_data_*.txt'))
    if not files:
        raise FileNotFoundError(f"Expected to find combined_data_*.txt in {data_dir}")
        
    if max_files:
        files = files[:max_files]
        
    data = []
    
    for file in files:
        with open(file, 'r') as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    # Random sampling to save memory during parsing if requested
                    if sample_frac < 1.0 and np.random.rand() > sample_frac:
                        continue
                        
                    parts = line.split(',')
                    if len(parts) == 3:
                        user_id, rating, date = parts
                        data.append([movie_id, int(user_id), int(rating), date])
                        
    df = pd.DataFrame(data, columns=['movie_id', 'user_id', 'rating', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    return df

def filter_sparse_data(df, min_user_ratings=5, min_movie_ratings=5):
    """
    Filter out users and movies with too few ratings to reduce sparsity and memory usage.
    """
    user_counts = df['user_id'].value_counts()
    movie_counts = df['movie_id'].value_counts()
    
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    filtered_df = df[df['user_id'].isin(valid_users) & df['movie_id'].isin(valid_movies)]
    return filtered_df

if __name__ == "__main__":
    # Test data processing on synthetic data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    print("Loading movies...")
    movies = load_movie_titles(data_dir)
    print(f"Loaded {len(movies)} movies.")
    
    print("Loading ratings...")
    ratings = load_ratings(data_dir)
    print(f"Loaded {len(ratings)} ratings.")
    
    print("Filtering data...")
    filtered_ratings = filter_sparse_data(ratings, min_user_ratings=2, min_movie_ratings=2)
    print(f"Filtered down to {len(filtered_ratings)} ratings.")
