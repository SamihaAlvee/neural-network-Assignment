# Bank Marketing Term Deposit Prediction

## Problem Statement
Build a Logistic Regression model to predict whether a bank customer will subscribe to a term deposit (`yes`) or not (`no`) using demographic, account, and campaign information.

## Dataset
- Records: 45,211
- Attributes: 17
- Target: `y`
- `yes` = subscribed
- `no` = did not subscribe

## Approach
1. Loaded `bank-full.csv` using Pandas with `;` as the separator.
2. Inspected shape, data types, and missing values.
3. Separated `y` as the target variable.
4. Encoded the target as `no -> 0` and `yes -> 1`.
5. Applied `StandardScaler` to numerical features.
6. Applied `OneHotEncoder` to categorical features.
7. Used an 80/20 stratified train-test split.
8. Trained a Logistic Regression classifier with `class_weight="balanced"` because the target classes are imbalanced.
9. Evaluated the model using Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix.

## Results

| Metric | Score |
|---|---:|
| Accuracy | 84.57% |
| Precision | 41.82% |
| Recall | 81.47% |
| F1 Score | 55.27% |
| ROC-AUC | 90.79% |

### Confusion Matrix
```
[[6786 1199]
 [ 196  862]]
```

## Findings
The model achieved an accuracy of approximately 84.57%. The ROC-AUC of approximately 90.79% indicates strong discrimination between subscribers and non-subscribers.

Recall was approximately 81.47%, meaning the model identified a large proportion of actual subscribers. Precision was lower (41.82%), so some customers predicted as subscribers would not actually subscribe.

Because the dataset is imbalanced, accuracy alone is not sufficient; recall, F1-score, and ROC-AUC are also important.

## Limitations and Future Improvements
- Tune the classification threshold based on the bank's business objective.
- Try hyperparameter tuning and cross-validation.
- Compare with tree-based classification models.
- Consider excluding `duration` for a true pre-contact prediction scenario because call duration is only known after a customer contact.
- Explore feature importance and model interpretability.

## How to Run
```bash
pip install -r requirements.txt
python logistic_regression.py
```
