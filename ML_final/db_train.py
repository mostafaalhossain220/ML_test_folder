import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

from sklearn.metrics import accuracy_score, classification_report


def main():
    df = pd.read_csv("diabetes.csv")
    print("Dataset Loaded Successfully!")
    print("Shape:", df.shape)

    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X.columns

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numeric_features)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_dist = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": randint(2, 10),
        "model__min_samples_leaf": randint(1, 5)
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print("\nTraining + Hyperparameter Tuning Started...")
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    print("\nBest CV Accuracy:", random_search.best_score_)
    print("Best Parameters:", random_search.best_params_)

    y_pred = best_model.predict(X_test)

    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(best_model, "model.pkl")
    print("\nBest model saved as model.pkl")


if __name__ == "__main__":
    main()