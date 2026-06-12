import os
import random
import csv
from datetime import datetime, timedelta

def generate_synthetic_data(data_dir="data", num_movies=500, num_users=2000, num_ratings=20000):
    """
    Generates synthetic data mimicking the Netflix Prize dataset format.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    movie_titles_path = os.path.join(data_dir, "movie_titles.csv")
    combined_data_path = os.path.join(data_dir, "combined_data_1.txt")
    
    print(f"Generating synthetic movie titles at {movie_titles_path}...")
    with open(movie_titles_path, "w", newline="", encoding="iso-8859-1") as f:
        writer = csv.writer(f)
        for movie_id in range(1, num_movies + 1):
            year = random.randint(1980, 2005)
            title = f"Synthetic Movie Title {movie_id}"
            writer.writerow([movie_id, year, title])
            
    print(f"Generating synthetic ratings at {combined_data_path}...")
    
    # Assign users
    user_ids = [random.randint(1000000, 2999999) for _ in range(num_users)]
    
    # Pre-generate ratings per movie to ensure exact formatting
    movie_ratings = {m: [] for m in range(1, num_movies + 1)}
    
    start_date = datetime(2000, 1, 1)
    
    for _ in range(num_ratings):
        movie_id = random.randint(1, num_movies)
        user_id = random.choice(user_ids)
        rating = random.randint(1, 5)
        # Give some bias to make data slightly non-uniform
        if random.random() < 0.3: 
            rating = random.choices([4, 5], weights=[0.4, 0.6])[0]
            
        days_offset = random.randint(0, 2000)
        date_str = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        
        movie_ratings[movie_id].append((user_id, rating, date_str))
        
    with open(combined_data_path, "w") as f:
        for movie_id in range(1, num_movies + 1):
            f.write(f"{movie_id}:\n")
            for user_id, rating, date_str in movie_ratings[movie_id]:
                f.write(f"{user_id},{rating},{date_str}\n")
                
    print("Synthetic data generation complete!")

if __name__ == "__main__":
    generate_synthetic_data()
