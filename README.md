# Netflix Prize Recommendation System

This repository contains the complete implementation for the "Recommendation Systems for Personalized Content Discovery" project based on the Netflix Prize Dataset.

## Project Structure

```
netflix_recommendation/
│
├── data/                       # Contains dataset files (or synthetic data)
│   ├── movie_titles.csv
│   └── combined_data_1.txt
│
├── src/                        # Source code
│   ├── data_generator.py       # Script to generate synthetic Netflix data for local testing
│   ├── data_processing.py      # Parses the unique Netflix dataset format
│   ├── models.py               # Implementation of SVD and KNN models
│   ├── evaluation.py           # Custom RMSE and MAP@10 calculations
│   ├── recommend.py            # Logic to generate Top-K recommendations
│   └── train.py                # Main script to train models and save them
│
├── notebooks/                  # EDA Scripts / Notebooks
│   └── 01_EDA.py               # Generates Exploratory Data Analysis figures
│
├── deliverables/               # Final deliverables
│   ├── figures/                # EDA plots
│   ├── Technical_Report.md     # Comprehensive 10-page equivalent report
│   └── Presentation.md         # 8-slide presentation outline
│
├── models/                     # Saved model artifacts (.pkl files)
├── app.py                      # Streamlit Interactive Dashboard
└── requirements.txt            # Python dependencies
```

## How to Run Locally (With Synthetic Data)

If you don't have the 2GB+ Netflix dataset downloaded locally, you can test the entire pipeline using synthetic data that perfectly mimics the Netflix format.

1. **Install Dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data:**
   ```bash
   python src/data_generator.py
   ```

3. **Run Exploratory Data Analysis (EDA):**
   ```bash
   python notebooks/01_EDA.py
   ```
   *This will generate `.png` figures in the `deliverables/figures/` directory.*

4. **Train the Models:**
   ```bash
   python src/train.py
   ```
   *This trains the SVD model, evaluates it, and saves it to `models/svd_model.pkl`.*

5. **Run the Interactive Dashboard:**
   ```bash
   streamlit run app.py
   ```

## How to Run on the Full Netflix Dataset (Kaggle/Colab)

Because the full dataset contains over 100 million ratings, training it locally can cause out-of-memory errors unless you have a powerful workstation. We highly recommend running this on **Kaggle Notebooks**.

1. Go to Kaggle and start a new Notebook.
2. Add the [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) dataset to your Notebook environment.
3. Upload the `src/` directory and `app.py` to the Kaggle environment.
4. Modify the `data_dir` variable in `train.py` or your Notebook cells to point to `/kaggle/input/netflix-prize-data/`.
5. Run the training loop! 

*(Note: When using the full dataset, Memory-based KNN algorithms might crash due to a massive user-user similarity matrix. Matrix Factorization (SVD) scales much better for this).*
