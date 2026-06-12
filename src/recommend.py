import pandas as pd
from collections import defaultdict

def get_top_n_recommendations(predictions, n=10):
    """
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions: The list of predictions, as returned by the test method of an algorithm.
        n: The number of recommendation to output for each user. Default is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def generate_user_recommendations(model, user_id, all_movie_ids, user_rated_movies, n=10):
    """
    Generate Top-N recommendations for a specific user.
    """
    # Find movies the user hasn't rated yet
    unrated_movies = [m for m in all_movie_ids if m not in user_rated_movies]
    
    # Predict ratings for all unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movies]
    
    # Sort by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Return top N
    return [(pred.iid, pred.est) for pred in predictions[:n]]

def map_recommendations_to_titles(recommendations, movie_df):
    """
    Maps movie IDs to their titles for display.
    """
    movie_dict = pd.Series(movie_df.title.values, index=movie_df.movie_id).to_dict()
    
    enriched_recs = []
    for iid, est in recommendations:
        title = movie_dict.get(iid, f"Unknown Movie {iid}")
        enriched_recs.append({"movie_id": iid, "title": title, "estimated_rating": round(est, 2)})
        
    return enriched_recs

if __name__ == "__main__":
    pass
