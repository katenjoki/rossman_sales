# <h1>Rossman Pharmaceutical Sales prediction across multiple stores</h1>

<h3>Objective</h3>

* The goal is to forecast sales in all Rossman Pharmaceutical store across several cities six weeks ahead of time. 
* Factors such as promotions, competition, school and state holidays, seasonality, and locality are necessary for predicting the sales across the various stores.
* This project aims to build and serve an end-to-end product that delivers this prediction, making it easy to derive insights.

<h3>Exploratory Data Analysis</h3>

Check out this [streamlit dashboard](https://rossman-sales-prediction.streamlit.app/) that enables users to visualise different features and detect any trends that may be influencing Sales values.

<h3>Machine Learning</h3>
To predict Sales(target variable), these ml models were used:

* Decision Tree Regressor
* Random Forest Regressor
* Stochastic Gradient Regressor

<h3>Evaluation metrics </h3>

* Random Forest Regression Model

| metric | validation score | test score |
|----|----|---|
| RMSE | 911.90 | 944.31|
| MAE | 595.25 | 594.3 |
| Coefficient of Determination (R<sup>2</sup>) | 0.9061 | 0.8980 | 

* Decision Tree Regression Model

| metric | validation score | test score |
|----|----|---|
| RMSE | 1,235.82 | 1,254.37 |
| MAE | 799.03 | 808.20 |
| Coefficient of Determination (R<sup>2</sup>) | 0.8246 | 0.8226 |

* Both models are able to generalise well when exposed to unseen data.
* The **Random Forest Regression** model however has the best performace, as a RMSE test score of ~944 means that on average the predictions made by the Random Forest Regression model differ from the actual Sales values by about ~944 sales. This is better than the Decision Tree Regression model, whose predictions differ from the actual Sales by ~ +/-1254.
* A R<sup>2</sup> validation score of 0.9061 means that ~91% of the variability in Sales is explained by the features in the Random Forest Regressor model.

  <h3>MlOps</h3>
  
