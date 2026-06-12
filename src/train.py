import os
import argparse
from data_processing import load_movie_titles, load_ratings, filter_sparse_data
from models import RecommendationModels
from evaluation import compute_rmse, compute_map_at_k
from surprise.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Train Recommendation Models")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Directory containing Netflix data")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of data to sample (0.0 to 1.0)")
    parser.add_argument("--model_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "models"), help="Directory to save trained models")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    print("1. Loading Data...")
    try:
        ratings_df = load_ratings(data_dir, sample_frac=args.sample_frac)
    except FileNotFoundError:
        print(f"Error: Ratings files not found in {data_dir}. Please run data_generator.py or provide actual Netflix data.")
        return
        
    print(f"Loaded {len(ratings_df)} ratings. Filtering sparse data...")
    ratings_df = filter_sparse_data(ratings_df, min_user_ratings=5, min_movie_ratings=5)
    print(f"Filtered down to {len(ratings_df)} ratings.")

    print("2. Preparing Data for Modeling...")
    models = RecommendationModels()
    data = models.prepare_data(ratings_df)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("3. Training SVD Model...")
    svd_model = models.train_svd(trainset)
    svd_predictions = svd_model.test(testset)
    
    svd_rmse = compute_rmse(svd_predictions)
    svd_map10 = compute_map_at_k(svd_predictions, k=10)
    
    print(f"--- SVD Results ---")
    print(f"RMSE: {svd_rmse:.4f}")
    print(f"MAP@10: {svd_map10:.4f}")
    
    print("\n4. Training KNN Model...")
    # NOTE: KNN can be extremely memory intensive on large datasets. 
    # If it crashes, comment this section out or use a much smaller sample_frac.
    knn_model = models.train_knn(trainset)
    knn_predictions = knn_model.test(testset)
    
    knn_rmse = compute_rmse(knn_predictions)
    knn_map10 = compute_map_at_k(knn_predictions, k=10)
    
    print(f"--- KNN Results ---")
    print(f"RMSE: {knn_rmse:.4f}")
    print(f"MAP@10: {knn_map10:.4f}")

    print("\n5. Saving Models...")
    models.save_model(svd_model, os.path.join(model_dir, "svd_model.pkl"))
    # We may choose not to save KNN if it's too large, but doing it here for completeness
    models.save_model(knn_model, os.path.join(model_dir, "knn_model.pkl"))
    
    print("Training Pipeline Complete!")

if __name__ == "__main__":
    main()
