# CO2-Emissions-Regression

# CO2 Emissions Prediction from Vehicle Data

This project builds machine learning models to predict CO2 emissions of vehicles using the EPA Vehicle Fuel Economy dataset. The pipeline includes data cleaning, feature engineering, model training with Random Forest and XGBoost, evaluation, visualization, and saving the best model.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Results](#results)
- [Saving & Loading Models](#saving--loading-models)
- [License](https://github.com/VenuOnTech/CO2-Emissions-Regression/blob/main/LICENSE)

---

## Project Overview

Accurate estimation of CO2 emissions is critical for environmental policies and automotive industry innovation. This project uses historical vehicle attributes to predict CO2 emissions with machine learning models, enabling better understanding and forecasting of environmental impact.

---

## Dataset

The dataset is based on the EPA Vehicle Fuel Economy data, containing multiple vehicle specifications along with CO2 emission values. The dataset includes columns such as:

- Make, Model, Year
- Cylinders, Engine Displacement
- Fuel Type, Drive Type
- Transmission, Vehicle Class
- CO2 Emissions (target variable)

**Dataset Download Link:**  [Vehicles](https://www.fueleconomy.gov/feg/epadata/vehicles.csv)

(The dataset is also upload in the files with the name 'vehicles.csv' if you wanna download go throught the above link.)

---

## Features Used

| Feature Name | Description                      |
|--------------|--------------------------------|
| make         | Manufacturer of the vehicle    |
| model        | Vehicle model                  |
| year         | Manufacturing year             |
| cylinders    | Number of cylinders in engine  |
| displ        | Engine displacement (liters)   |
| fuelType     | Fuel type (e.g., Gasoline)     |
| drive        | Drive type (e.g., FWD, RWD)    |
| trany        | Transmission type              |
| VClass       | Vehicle class                  |
| co2          | CO2 emissions (g/mile) [Target]|

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/co2-emissions-prediction.git
    cd co2-emissions-prediction
    ```

2. Install required dependencies (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have the dataset (`vehicles.csv`) in the project directory or update the file path in the notebook/script.

---

## Usage

Run the notebook or Python script to perform the following steps:

- Load and clean the dataset
- Drop leakage columns
- Select relevant features and drop missing values
- Split data into training and testing sets
- Preprocess categorical and numerical data
- Train Random Forest and XGBoost regression models
- Evaluate model performance using R² and RMSE metrics
- Visualize feature importance and actual vs predicted CO2 emissions
- Save the best model (`xgb_model_co2.pkl`)

---

## Modeling

### Preprocessing

- One-hot encoding is applied to categorical features.
- Numeric features are passed through without transformation.

### Models

- **Random Forest Regressor**
- **XGBoost Regressor** with hyperparameter tuning using GridSearchCV.

---

## Results

| Model         |       R² Score       |         RMSE          |
|---------------|----------------------|-----------------------|
| Random Forest | 0.9974464778086294   | 10.06522460701715     |
| XGBoost       | 0.991543710231781    | 18.31654578814944     |

*Note: Results may vary depending on dataset version and parameter tuning.*

### Visualizations

- Feature importance from Random Forest model (top 20 features)
- Scatter plots of Actual vs Predicted CO2 emissions for both models

---

## Saving & Loading Models

The best trained model (XGBoost) is saved using `joblib`:

```python
import joblib

# Save model
joblib.dump(grid.best_estimator_, "xgb_model_co2.pkl")

# Load model
model = joblib.load("xgb_model_co2.pkl")
```

### License

