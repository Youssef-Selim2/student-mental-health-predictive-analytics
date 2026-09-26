"""Student depression prediction with classical machine-learning models.

This script reproduces the final analysis pipeline used in the project:
- label-encode categorical features
- median-impute missing values
- standardize predictors
- rank features with a Random Forest
- compare KNN, Random Forest, and Gradient Boosting with stratified 5-fold CV
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(data_path: Path):
    """Load the dataset and reproduce the project's preprocessing steps."""
    df = pd.read_csv(data_path)
    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    for col in X.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y


def rank_features(X, X_scaled, y, figure_path: Path):
    """Rank predictors using Random Forest feature importance and save a plot."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    ranking = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\n=== Feature Importance Ranking (Random Forest) ===")
    print(ranking.to_string(index=False))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ranking["Feature"], ranking["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title("Feature Importance Ranking (Random Forest)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_models(X_scaled, y):
    """Evaluate three classifiers with stratified 5-fold cross-validation."""
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=42
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_scaled,
            y,
            cv=cv,
            scoring=["roc_auc", "precision", "recall", "f1"],
            return_train_score=False,
        )
        results[name] = {
            metric: np.mean(scores[f"test_{metric}"])
            for metric in ["roc_auc", "precision", "recall", "f1"]
        }

    print("\n=== Model Performance ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Student depression ML analysis")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/student_depression_dataset.csv"),
        help="Path to the Student Depression Dataset CSV.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/feature_importance.png"),
        help="Where to save the feature-importance plot.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data}. Download it from the source linked "
            "in README.md and place it there, or pass --data PATH_TO_CSV."
        )

    X, X_scaled, y = load_and_preprocess(args.data)
    rank_features(X, X_scaled, y, args.figure)
    evaluate_models(X_scaled, y)


if __name__ == "__main__":
    main()
