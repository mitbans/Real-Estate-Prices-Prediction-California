# Real Estate Price Prediction - California
Predicting Real Estate Price in California using Linear Regression model for non-linear features.

### Objective
- Build "Best Model" for predicting real estate prices in California.
- Interpret the model using built in methods from scikitlearn, permutation_importance.
Take a look at the user guide for permutation_importance <a href="https://scikit-learn.org/stable/modules/permutation_importance.html">here</a>.

### Building "Best Model" Strategy

- Train / Test split the data and assign y = median_house_value
- Calculate baseline predictions
- Examine the Correlations
- Measure the multicollinearity using builtin function VIF and drop features with high multicolinearity
- Create the Transformers: 
  - SimpleImputer: Imputing missing values as median for numerical and most_frequent for categorical columns. 
  - PolynomialFeatures of varying degree in range 1 to 5
  - One-hot encoder for categorical feature "ocean_proximity"
  - KMeans clustering for longitude and latitude (10 clusters)
  - Pass through any remaining columns (remainder='passthrough')
- Create the Pipeline with above mentioned Transformer and LinearRegression 
- Fit / Predict / calculate mses

#### Pipeline
<img width="1082" alt="image" src="https://github.com/mitbans/real-estate-price-prediction-california/assets/166747739/3920868a-0efc-4b75-9197-abd5c6c8b93a">

#### Training and Testing MSEs vs Polynomial Degree
<img width="578" alt="image" src="https://github.com/mitbans/real-estate-price-prediction-california/assets/166747739/9870feae-2a76-4cfd-b243-f66dab6dcd3b">

#### KMeans Clusters
<img width="578" alt="image" src="https://github.com/mitbans/real-estate-price-prediction-california/assets/166747739/83ebfcaa-fbd4-433f-a974-836e46409cd8">

### Optimal Model Complexity using Simple Cross Validation
- Result: The best degree polynomial Model is 3 with smallest mse = 3.9M
- Plotting Model output / predictions vs actual values with degree = 3
<img width="629" alt="image" src="https://github.com/mitbans/real-estate-price-prediction-california/assets/166747739/8167cb13-3d3d-4318-81bd-30dc698c5f26">

### Interpreting the Model using Permutation Importance 
- Conclusion:
  - Geographic location (longitude and latitude) of the housing units has a significant influence on the target variable
  - Population and total bedrooms also have relatively high permutation importance compared to the rest of the features
  - Median income has a moderate permutation importance
  - Housing Median Age, Households, Ocean Proximity, Total Rooms have relatively low permutation importance

## Repository Structure
- <code>data/</code>: Contains dataset used in the analysis.
- <code>notebooks/real-estate-price-prediction.ipynb</code>: Jupyter notebook with code for data analysis.
- <code>README.md</code>: Summary of findings and link to notebook

## Notebook
The detailed analysis and code can be found in the Jupyter notebook <a href="https://github.com/mitbans/real-estate-price-prediction-california/blob/main/notebooks/real-estate-price-prediction.ipynb">here</a>.
