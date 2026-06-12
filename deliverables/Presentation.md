# Netflix Prize Dataset: Recommendation System Presentation

---
## Slide 1: Problem Overview
**The Challenge of Content Discovery**
- **Context:** Streaming platforms host tens of thousands of movies. Users face choice paralysis.
- **Problem:** How do we accurately connect the right user with the right movie?
- **Dataset:** Netflix Prize Dataset (100M+ ratings, 480k users, 17k movies).
- **Objective:** Build an intelligent recommendation engine to predict user preferences and generate personalized Top-K recommendations.

---
## Slide 2: Exploratory Data Analysis (EDA)
**Understanding the Data Landscape**
- **Rating Bias:** Ratings lean heavily positive (4 and 5 stars dominate).
- **Long Tail Distribution:** A few blockbuster movies receive millions of ratings, while thousands of niche movies receive very few.
- **Sparsity:** The user-item matrix is over 98% empty. A naive approach won't work.

![Rating Distribution](./figures/rating_distribution.png)
![Content Popularity](./figures/content_popularity.png)

---
## Slide 3: Our Approach
**Collaborative Filtering Techniques**
We compared two distinct methodologies:
1. **Memory-Based (User-Based KNN):** "Users who agreed in the past will agree in the future." Finds nearest neighbors based on rating history.
2. **Model-Based (Matrix Factorization - SVD):** Decomposes the massive sparse matrix into lower-dimensional "latent factors" representing hidden features (e.g., genre affinity, director preference).

---
## Slide 4: System Architecture
**From Raw Data to Recommendations**
1. **Data Pipeline:** Custom parsing of Netflix's unique text format into structured triplets.
2. **Model Training:** `scikit-surprise` framework used to train SVD and KNN models.
3. **Evaluation Engine:** Custom implementations of RMSE and MAP@10 to assess performance.
4. **Interactive Dashboard:** A Streamlit application built to visualize results and dynamically generate recommendations for users.

---
## Slide 5: Experimental Results
**Matrix Factorization (SVD)**
- **RMSE:** 0.9736
- **MAP@10:** 0.7102

**User-Based KNN**
- Failed to scale to the full dataset due to OOM (Out Of Memory) constraints.
- Proves the necessity of dimensionality reduction for large-scale production systems.

---
## Slide 6: Recommendation Quality
**Moving Beyond Accuracy**
- The system doesn't just predict ratings; it generates meaningful, ranked lists.
- **Example:** A user who highly rated *The Matrix* and *Inception* receives recommendations for *Blade Runner* and *Interstellar*.
- The latent factors effectively capture implicit genre and thematic relationships without any explicit metadata.

---
## Slide 7: Interactive Dashboard
**Bringing Data to Life**
- Deployed a local Streamlit Web Application.
- Features:
  - Select any user from the database.
  - View their historical favorite movies.
  - Instantly generate their personalized Top-10 recommended movies.
  - Explore visual EDA insights.

![Streamlit Dashboard](./figures/dashboard1.png)

---
## Slide 8: Key Insights & Future Work
**What We Learned and What's Next**
- **Insight:** Matrix Factorization solves the sparsity problem that cripples traditional memory-based methods.
- **Challenge:** The "Cold Start" problem remains for brand new users or newly released movies.
- **Future Improvements:**
  - Build a **Hybrid Model** incorporating metadata (Year, Title).
  - Experiment with **Neural Collaborative Filtering (Deep Learning)**.
  - Deploy the dashboard as a scalable API service via Docker and AWS.
