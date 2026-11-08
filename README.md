# Titanic Survival Prediction

A Random Forest model for the Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The project is split into two notebooks: one for feature engineering and one for training and prediction.

## Results

| Metric | Score |
| --- | --- |
| 5-fold CV accuracy (train set) | 0.8373 |
| **Kaggle public leaderboard score** | **0.75837** |

The gap between CV accuracy and the leaderboard score is common on this dataset (small train set of 891 rows, and the test set's distribution differs slightly), so the CV number should be treated as an optimistic estimate.

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── train.csv              # Original Kaggle training data
│   │   └── test.csv               # Original Kaggle test data
│   └── processed/
│       ├── train.csv              # Engineered training features
│       ├── test.csv               # Engineered test features
│       └── submission.csv         # Predictions to submit to Kaggle
├── data_processing.ipynb          # Cleaning + feature engineering
├── model_train.ipynb                    # Model tuning + prediction
└── README.md
```

## Pipeline

### 1. `data_processing.ipynb` – Feature Engineering

Train and test sets are concatenated (with an `is_train` flag) so that both receive identical transformations. All imputation statistics are computed **from the training rows only**, to avoid leaking test information.

| Step | Description |
| --- | --- |
| **Title extraction** | Pulled from `Name` via regex. `Mlle`/`Ms` → `Miss`, `Mme` → `Mrs`, and uncommon titles (Lady, Countess, Capt, Col, Don, Dr, Major, Rev, Sir, Jonkheer, Dona) → `Rare`. |
| **Age imputation** | Missing ages filled with the median age of the passenger's title group, falling back to the global median. |
| **Fare imputation** | Missing fares filled with the median fare of the passenger's `Pclass`. |
| **Group features** | `ticket_size` (passengers sharing a ticket) and `family_size` (`SibSp + Parch + 1`) are combined into `group_size = max(ticket_size, family_size)`. From it: `is_alone` and `fare_per_person = Fare / group_size`. |
| **Deck** | First letter of `Cabin`; missing cabins → `M`. The single `T` deck is merged into `M`. |
| **Ship side** | Derived from cabin number parity: even → `Port`, odd → `Starboard`, missing → `M`. |
| **Embarked** | Missing values filled with the training-set mode. |
| **Encoding** | One-hot encoding (`drop_first=True`) of `Sex`, `title`, `deck`, `ship_side`, `Embarked`. |
| **Dropped columns** | `Cabin`, `Cabin_Num`, `Name`, `Ticket`, `ticket_size`, `family_size`, `SibSp`, `Parch`. |

The processed data is saved to `data/processed/train.csv` and `data/processed/test.csv`.

### 2. `model_train.ipynb` – Modeling

- **Model:** `RandomForestClassifier(random_state=1)`
- **Tuning:** `GridSearchCV` with 5-fold CV, scored on accuracy, over:

| Hyperparameter | Values |
| --- | --- |
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 5, 8, 10, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

- **Best parameters:** `max_depth=None`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=200`
- **Output:** the best estimator predicts on the test set and writes `data/processed/submission.csv` (`PassengerId`, `Survived`).



## Possible Improvements

- Try gradient boosting models (XGBoost, LightGBM, CatBoost) or an ensemble.
- Use repeated or stratified CV for a more stable estimate, since the leaderboard score is noticeably lower than CV accuracy.
- Reduce the tuning search space or use `RandomizedSearchCV` to limit overfitting to the CV folds.
