# CSCI 441/541 – Project 1: Predicting Minnesota Housing Prices

## 1. Project Overview
This project replicates the *California Housing Prices* case study, applying it to **Minnesota housing data**.  
The goal is to predict **median home values** using demographic, geographic, and housing-related features.  
This project explores the full **machine learning pipeline** — including data analysis, preparation, model training, evaluation, and deployment — while also considering the **ethical implications** of predictive modeling in real estate.

---

## 2. Summary of Steps Taken

Following the project guidelines, the work was divided into **five major steps**, each addressing key parts of the ML workflow.

### **Step 1 – Exploratory Data Analysis (EDA)**
Dataset source: [Kaggle – United States Redfin Housing Market](https://www.kaggle.com/datasets/soulaimanebenayad/united-states-redfin-housing-market-csv)

- Combined Minnesota housing data from **MN-1.csv** and **MN-2.csv** into a single dataset (`housing_data.csv`).
- Removed irrelevant attributes such as sale dates, property addresses, and URLs.
- Standardized column names to lowercase (e.g., `PRICE → median_house_value`, `YEAR BUILT → housing_median_age`).
- Converted text-based numeric columns (e.g., those containing `$` or commas) into numeric form.
- Handled missing values by dropping incomplete records and imputing medians.

**Visual analysis included:**
- **Histograms** of numeric features to understand distributions and detect outliers.
- **Scatter plots** between `median_house_value` and predictors such as `square_feet`, `beds`, and `baths`, showing strong positive relationships.
- **Correlation heatmap** highlighting that house size and bedroom count had the strongest correlation with price.
- **Geographic scatter plot** showing higher-priced homes clustered near the **Minneapolis–Saint Paul** metro area.

---

### **Step 2 – Data Preparation**
Focused on cleaning, transforming, and structuring the dataset for model training.

- Applied **stratified sampling** (if `median_income` was available) to maintain balanced distributions between training and test sets; otherwise, used an **80/20 random split**.
- Imputed missing values:
  - Numeric columns → filled with **median values**.
  - Categorical columns → filled with **most frequent values**.
- Engineered new features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- Scaled numerical data with **StandardScaler** and encoded categorical data with **OneHotEncoder** using a **ColumnTransformer**.
- Combined all preprocessing steps into a **single Scikit-Learn pipeline** for consistency and reusability.

---

### **Step 3 – Model Training**
Trained and compared three regression models to predict median housing values:
1. **Linear Regression** – baseline model; interpretable but underfit the data.  
2. **Decision Tree Regressor** – captured non-linear patterns but overfit the training data.  
3. **Random Forest Regressor** – ensemble model with the best balance between bias and variance.

**Cross-validation:**  
Used **5-fold cross-validation** and evaluated models using **Root Mean Squared Error (RMSE)**.

**Hyperparameter tuning:**
- **GridSearchCV** – explored structured parameter grids.  
- **RandomizedSearchCV** – searched broader parameter ranges efficiently.  

The **tuned Random Forest model** achieved the lowest RMSE and was selected as the final model for testing.

---

### **Step 4 – Model Evaluation**
The tuned Random Forest pipeline was evaluated on the unseen test set.

- **RMSE** was used to measure performance, quantifying the average prediction error.  
- The Random Forest model achieved the lowest test RMSE, outperforming the baseline models.  
- **Feature importances** revealed that:
  - `square_feet`, `beds`, and `baths` were the strongest predictors.
  - Geographic features (`latitude`, `longitude`) captured regional price variations.
  - Engineered metrics like `price_per_sqft` added additional explanatory power.

These results confirmed that **size and location** are the dominant determinants of housing value in Minnesota.

---

### **Step 5 – Final Pipeline & Prediction**
- The final pipeline (including preprocessing and the tuned Random Forest model) was saved as  
  **`mn_housing_final_pipeline.joblib`**.
- Example predictions on synthetic and new data confirmed the model ran smoothly.
- The pipeline is **fully reusable** — it can be applied to other regions or deployed in an application for automated price prediction.

