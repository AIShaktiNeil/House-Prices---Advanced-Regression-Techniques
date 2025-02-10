
# House Prices Prediction 

This repository contains an end-to-end solution for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) competition on Kaggle. The project leverages multiple regression models and extensive hyperparameter tuning to predict house sale prices accurately.

## Project Overview

- **Objective:**  
  Develop a predictive model for house sale prices using various regression techniques, and select the best performing model based on test scores.

- **Approach:**  
  Evaluated multiple models including:
  - `LogisticRegression`
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - `SVR`
  - `XGBRegressor`
  
  After thorough experimentation and hyperparameter tuning, the **GradientBoostingRegressor** emerged as the best model with a test score of **0.9719814215737473**.

## Data

- **Dataset:**  
  The analysis utilizes the House Prices dataset from Kaggle, which includes diverse features such as:
  - **Property Characteristics:** MSSubClass, LotArea, YearBuilt, etc.
  - **Neighborhood Details:** Neighborhood, Condition1, Condition2, etc.
  - **House Features:** OverallQual, OverallCond, GrLivArea, etc.
  - And many other columns that provide insights into property attributes.
  
- **Source:**  
  [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

## Models & Methodology

1. **Preprocessing:**  
   - **Cleaned** the dataset by handling missing values and outliers.
   - **Engineered** relevant features and performed normalization/encoding for optimal model performance.

2. **Modeling:**  
   - **Implemented** and compared various regression models: `LogisticRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`, and `XGBRegressor`.
   - **Conducted** hyperparameter tuning to optimize model performance for each approach.

3. **Outcome:**  
   - **Selected** the `GradientBoostingRegressor` as the final model after achieving a **test score of 0.9719814215737473**.
   - **Validated** the model’s performance using cross-validation and out-of-sample testing.

## How to Run the Project

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/AIShaktiNeil/House-Prices---Advanced-Regression-Techniques/blob/main/house_price_train.ipynb
   cd house-prices-prediction
   ```

2. **Install Dependencies:**

   Ensure you have Python installed, then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Jupyter Notebook:**

   Start Jupyter Notebook and open the project file:

   ```bash
   http://localhost:8888/lab/tree/Downloads/house_price_train.ipynb
   ```

4. **Run the Analysis:**

   Execute the notebook cells sequentially to reproduce the data preprocessing, model training, and evaluation results.

## Technologies & Libraries

- **Python**
- **Pandas** & **NumPy** – Data manipulation and numerical operations.
- **Scikit-learn** – Implementing regression models and hyperparameter tuning.
- **XGBoost** – For the XGBRegressor model.
- **Matplotlib** & **Seaborn** – Data visualization.
- **Jupyter Notebook** – Interactive coding environment.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to the Kaggle community for providing the House Prices dataset and valuable insights.
- Inspired by the methodologies and kernels shared by the data science community on Kaggle.
