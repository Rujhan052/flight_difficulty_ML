Flight Delay Prediction (Regression Analysis)

Project Overview:

This project attempts to build a machine learning model to predict flight departure delays. The goal was to determine if pre-flight operational data (such as baggage counts, passenger load, and scheduled ground time) could be used to accurately forecast departure_delay_minutes.

** The data was sourced from a dataset provided during the United Airlines Hackathon. **

Analysis Workflow & Model Evaluation

The project followed an iterative process of building a baseline model, diagnosing its weaknesses, and attempting to improve it through feature engineering and selection.

1. Baseline Model (Simple Train-Test Split)
A baseline Random Forest model was first trained on a standard 80/20 train-test split, using numerical, time-based, and one-hot encoded station features.

    Result: This model produced a promising R2 score of ~0.40.

    Problem: A single split can be misleading or "lucky." A more robust validation method was needed to check for stability.

2. Model Validation (5-Fold Cross-Validation)
To test the model's stability, a 5-fold cross-validation was performed on the same dataset.

    Result: The average R2 score dropped to ~-0.204, with a very high standard deviation.

    Diagnosis: A negative R2 score indicates the model is performing worse than just guessing the average delay. The high standard deviation confirmed the model was severely overfitting and highly unstable.

3. Feature Selection (Hypothesis)
The hypothesis was that the model was overfitting due to too many noisy, irrelevant features (especially the one-hot encoded airport codes).

    Action: A Random Forest model was used to rank all features by their feature_importances_.

    Finding: Features like scheduled_ground_time_minutes and TOTAL_PASSENGERS were identified as most important.

4. Final Model (Re-Validation)
A new, simpler model was trained using only the top 10 most important features. This model was then re-evaluated using 5-fold cross-validation to see if removing the noise had fixed the stability problem.

    Result: The average R2 score was still negative, at ~-0.237.

    Std Deviation: The standard deviation remained high at ~0.59.

Key Findings & Conclusion

* This project successfully demonstrates the critical importance of using cross-validation to diagnose model instability.

* The key finding is that the available features (passenger counts, baggage, scheduled time, etc.) do not have sufficient predictive power to reliably forecast flight delays.

* The model's poor performance, even after feature selection, proves that the primary drivers of delays are external factors not present in this dataset. These likely include:

       1) Weather conditions

       2) Air Traffic Control status

       3) Crew or maintenance issues

       4) Cascading delays from a plane's previous flight

Conclusion: The analysis shows that simply tuning the model would be ineffective. The correct business recommendation is that a reliable delay prediction model would require acquiring new, more relevant data sources (like weather, flight status, and crew data) before proceeding.

Technologies Used:

* Python

* Pandas

* NumPy

* Scikit-learn (RandomForestRegressor, LinearRegression, KNeighborsRegressor,  train_test_split, cross_val_score)

* XGBoost (XGBRegressor)

