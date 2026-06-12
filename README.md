# 🎬 Netflix Prize Recommendation System

A complete, end-to-end recommendation system built for the Open Projects 2026 challenge using the legendary **[Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)**.

This project implements Collaborative Filtering models capable of predicting user preferences across a massive, highly sparse matrix of over 100 million movie ratings. It evaluates these models using strict RMSE and MAP@10 metrics and wraps the final predictive engine in a beautiful, interactive Streamlit web dashboard.

---

## ✨ Features

- **Matrix Factorization (SVD):** A highly scalable implementation of Singular Value Decomposition that handles 98%+ sparsity with ease.
- **Advanced Evaluation:** Includes custom logic to compute `MAP@10` (Mean Average Precision) alongside traditional `RMSE`.
- **Interactive Dashboard:** A local web app that allows you to select users, view their historical favorites, and instantly generate Top-10 personalized movie recommendations.
- **Automated EDA:** Scripts to visualize rating distributions, the "long tail" of content popularity, and user activity patterns.

---

## 📁 Repository Structure

```
netflix_recommendation/
│
├── data/                       # Contains dataset files (or synthetic data)
├── src/                        # Core logic (data processing, SVD/KNN models, evaluation)
├── notebooks/                  # EDA Scripts & Visualizations
├── deliverables/               # Final deliverables (Technical Report & Presentation)
│   └── figures/                # Output graphs and dashboard screenshots
├── models/                     # Saved model artifacts (.pkl files)
├── app.py                      # Streamlit Interactive Dashboard
├── requirements.txt            # Python dependencies
└── .gitignore                  # Prevents huge data/model files from being uploaded
```

---

## 🚀 How to Train on the Real Dataset (Kaggle)

Because the full dataset is over 2GB, training it locally can cause out-of-memory errors on standard laptops. We highly recommend doing the "heavy lifting" on Kaggle.

1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. Click **Add Input** and attach the **[Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)**.
3. Ensure **Internet Access** is toggled ON in your Notebook Settings.
4. Run this in your **first cell**:
   ```python
   !pip install "numpy<2.0.0" scikit-surprise
   ```
5. Paste the entire training pipeline from `src/train.py` (or the monolithic snippet provided in the documentation) into your **second cell** and run it.
6. Once training finishes, Kaggle will generate `svd_model.pkl` in the `/kaggle/working/` directory. **Download this file** along with the `movie_titles.csv` file!

---

## 💻 How to Run the Dashboard Locally

Once you have your trained `svd_model.pkl` and `movie_titles.csv` from Kaggle, you can run the dashboard on your own machine!

1. Place `svd_model.pkl` into the `models/` folder.
2. Place `movie_titles.csv` into the `data/` folder.
3. Open your terminal in this project directory and activate the virtual environment:
   ```bash
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Mac/Linux:
   source venv/bin/activate
   ```
4. Install dependencies (if you haven't already):
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the dashboard!
   ```bash
   streamlit run app.py
   ```

A browser window will automatically open showing your interactive recommendation system.

---

## 📈 Experimental Results

Our Matrix Factorization (SVD) model achieved the following metrics when trained on a sample of the dataset:
- **RMSE:** 0.9736
- **MAP@10:** 0.7102

*(User-Based KNN failed to scale to this massive dataset due to OOM constraints, proving SVD's architectural superiority).*
