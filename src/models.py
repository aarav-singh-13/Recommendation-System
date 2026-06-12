import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pickle
import os

class RecommendationModels:
    def __init__(self):
        self.svd_model = SVD(random_state=42)
        self.knn_model = KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': True})
        
    def prepare_data(self, df):
        """
        Convert Pandas DataFrame to Surprise Dataset format.
        Surprise expects: user_id, item_id, rating
        """
        reader = Reader(rating_scale=(1, 5))
        # Ensure correct column order for Surprise
        data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
        return data
        
    def train_svd(self, trainset):
        """Train SVD (Matrix Factorization) model"""
        print("Training SVD Matrix Factorization model...")
        self.svd_model.fit(trainset)
        print("SVD training complete.")
        return self.svd_model
        
    def train_knn(self, trainset):
        """Train KNN (User-Based Collaborative Filtering) model"""
        print("Training User-Based KNN model...")
        self.knn_model.fit(trainset)
        print("KNN training complete.")
        return self.knn_model
        
    def save_model(self, model, filepath):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
    def load_model(self, filepath):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    from data_processing import load_ratings
    # Quick test
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = load_ratings(data_dir, sample_frac=0.1)
    
    models = RecommendationModels()
    data = models.prepare_data(df)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Train SVD
    svd = models.train_svd(trainset)
    predictions_svd = svd.test(testset)
    print("SVD RMSE:", accuracy.rmse(predictions_svd))
    
    # Train KNN
    knn = models.train_knn(trainset)
    predictions_knn = knn.test(testset)
    print("KNN RMSE:", accuracy.rmse(predictions_knn))
