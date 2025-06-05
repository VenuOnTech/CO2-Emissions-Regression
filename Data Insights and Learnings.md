## 📊 Data Insights and Learnings

During the exploratory data analysis and model building for CO2 emissions prediction, we uncovered several important insights about the EPA Vehicle Fuel Economy dataset:

### 1. **Data Leakage Awareness**

- The `CO2` target variable is **calculated directly from MPG (miles per gallon) values** such as `comb08`, `city08`, and `highway08`.
- Including these MPG-related features causes leakage — the model essentially “sees the answer” and memorizes the formula instead of learning meaningful relationships.
- Removing these leakage features led to more realistic model performance.

### 2. **Feature Importance**

- Categorical variables like **vehicle make, model, fuel type, drive type, and transmission** have a significant impact on CO2 emission levels.
- Numeric features like **engine displacement, cylinder count, and vehicle year** also influence emissions, reflecting how newer or more powerful engines affect fuel consumption and emissions.

### 3. **Model Performance**

- After cleaning the data and preventing leakage, both Random Forest and XGBoost models achieved high but realistic accuracy (R² around 0.99), demonstrating that the selected features effectively predict CO2 emissions.
- The Root Mean Squared Error (RMSE) values indicate the models’ predictions are close to actual values with a small margin of error, suitable for practical applications.

### 4. **Relationship Between Variables**

- Higher engine displacement and cylinder count generally lead to increased CO2 emissions, confirming that bigger engines tend to consume more fuel and pollute more.
- Vehicle year showed a trend where newer vehicles tend to have lower emissions, likely due to improved fuel efficiency and emission standards.
- Different fuel types and drive configurations correspond to varying emission levels, highlighting the impact of vehicle design on environmental footprint.

### 5. **Data Quality and Completeness**

- Dropping rows with missing values in key features ensured clean, reliable data but reduced dataset size. Future work could explore advanced imputation techniques to leverage more data.

---

## 🔍 What We Learned

- **Understanding the dataset and domain is crucial** before model building to avoid pitfalls like data leakage.
- **Feature engineering and selection** can greatly impact model reliability and interpretability.
- Predicting CO2 emissions using vehicle characteristics is feasible with high accuracy using ensemble models.
- The project highlights the **importance of transparency and rigor in environmental data science**, aiding efforts to monitor and reduce vehicle emissions.
