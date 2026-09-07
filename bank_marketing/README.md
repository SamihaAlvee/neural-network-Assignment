 Bank Marketing Term Deposit Prediction

## Problem Statement
Build a Logistic Regression model to predict whether a bank customer will subscribe to a term deposit (`yes`) or not (`no`) using demographic, account, and campaign information.

## Dataset
- Records: 45,211
- Attributes: 17
- Target: `y`
- `yes` = subscribed
- `no` = did not subscribe
  
### Approach
The dataset was preprocessed by encoding categorical features and scaling numerical features. The data was split into training and testing sets, and a *Logistic Regression* model was trained to predict whether a customer would subscribe to a term deposit. The model was evaluated using standard classification metrics.

## Methodology
1. Loaded `bank-full.csv` using Pandas with `;` as the separator.
2. Inspected shape, data types, and missing values.
3. Separated `y` as the target variable.
4. Encoded the target as `no -> 0` and `yes -> 1`.
5. Applied `StandardScaler` to numerical features.
6. Applied `OneHotEncoder` to categorical features.
7. Used an 80/20 stratified train-test split.
8. Trained a Logistic Regression classifier with `class_weight="balanced"` because the target classes are imbalanced.
9. Evaluated the model using Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix.



## Findings
The accuracy of the model was roughly 84.57%. Strong discrimination between subscribers and non-subscribers is indicated by the ROC-AUC of roughly 90.79%.

With a recall of almost 81.47%, the model was able to identify a significant percentage of real subscribers. Due to the decreased precision (41.82%), some clients who were anticipated to subscribe would not do so.

Accuracy alone is insufficient because to the imbalanced dataset; recall, F1-score, and ROC-AUC are all crucial.

## Limitations and Future Improvements
- Tune the classification threshold based on the bank's business objective.
- Try hyperparameter tuning and cross-validation.
- Compare with tree-based classification models.
- Consider excluding `duration` for a true pre-contact prediction scenario because call duration is only known after a customer contact.
- Explore feature importance and model interpretability.

