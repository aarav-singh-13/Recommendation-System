import numpy as np
from collections import defaultdict
from surprise import accuracy

def compute_rmse(predictions):
    """
    Computes Root Mean Squared Error.
    """
    return accuracy.rmse(predictions, verbose=False)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Return precision and recall at k metrics for each user.
    Relevance threshold is specified by the problem statement (>= 3.5).
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) 
                              for (est, true_r) in user_ratings[:k])
        
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
    return precisions, recalls

def average_precision_at_k(user_ratings, k=10, threshold=3.5):
    """
    Calculates Average Precision at K for a single user.
    """
    # user_ratings is a list of tuples: (est, true_r)
    # Sort by estimated rating descending
    user_ratings.sort(key=lambda x: x[0], reverse=True)
    
    # Take top k
    top_k = user_ratings[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, (est, true_r) in enumerate(top_k):
        if true_r >= threshold: # If item is relevant
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    # Normalize by number of relevant items in the whole user set or max k
    total_relevant = min(len([r for (_, r) in user_ratings if r >= threshold]), k)
    if total_relevant == 0:
        return 0.0
        
    return score / total_relevant

def compute_map_at_k(predictions, k=10, threshold=3.5):
    """
    Computes Mean Average Precision at K (MAP@K) across all users.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
        
    ap_scores = []
    for uid, user_ratings in user_est_true.items():
        ap = average_precision_at_k(user_ratings, k, threshold)
        ap_scores.append(ap)
        
    return np.mean(ap_scores)

if __name__ == "__main__":
    from surprise import SVD, Dataset, Reader
    import pandas as pd
    
    # Mock data to test MAP@10 logic
    ratings_dict = {
        'user_id': [1, 1, 1, 1, 2, 2, 2, 2],
        'item_id': [10, 20, 30, 40, 10, 20, 30, 40],
        'rating': [5.0, 4.0, 3.0, 2.0, 5.0, 1.0, 4.0, 2.0]
    }
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    rmse = compute_rmse(predictions)
    map10 = compute_map_at_k(predictions, k=10, threshold=3.5)
    
    print(f"Test RMSE: {rmse}")
    print(f"Test MAP@10: {map10}")
