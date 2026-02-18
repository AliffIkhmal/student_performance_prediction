import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


class StudentPerformanceModel:
    """ML model that auto-selects the best classifier for student grade prediction."""

    FEATURES = [
        "Age", "Gender", "ParentalEducation", "StudyTimeWeekly",
        "Absences", "ParentalSupport", "Extracurricular",
        "Sports", "Music", "Volunteering",
    ]

    GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}

    # All candidate models to compare
    CANDIDATE_MODELS = {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": lambda: SVC(kernel="rbf", random_state=42),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
        "Neural Network": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.best_model_name = None
        self.feature_importance = None
        self.is_trained = False
        self.comparison_results = None

    # ── Imbalance check ──────────────────────────────────────────────────
    @staticmethod
    def check_imbalance(y):
        """Analyze class distribution and return imbalance info."""
        counts = y.value_counts().sort_index()
        total = len(y)
        ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")

        distribution = {}
        grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
        for cls, count in counts.items():
            label = grade_map.get(cls, str(cls))
            distribution[label] = {"count": int(count), "percent": round(count / total * 100, 1)}

        return {
            "distribution": distribution,
            "total": total,
            "ratio": round(ratio, 1),
            "is_imbalanced": bool(ratio > 3),
        }

    # ── Training ─────────────────────────────────────────────────────────
    def train(self, X, y):
        """
        Compare all candidate models, select the best by F1 score,
        and return its metrics plus comparison results.

        SMOTE is applied AFTER the train/test split to avoid data leakage.
        """
        # 1. Split first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. SMOTE on training data only
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 3. Scale
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)

        # 4. Compare all models
        comparison = {}
        best_f1 = -1

        for name, model_fn in self.CANDIDATE_MODELS.items():
            m = model_fn()
            m.fit(X_train_scaled, y_train_res)
            y_pred = m.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            comparison[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            }

            if f1 > best_f1:
                best_f1 = f1
                self.model = m
                self.best_model_name = name

        self.is_trained = True
        self.comparison_results = comparison

        # 5. Feature importance (if available)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = pd.DataFrame({
                "feature": X.columns,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)

        # 6. Return metrics of the best model
        best_metrics = comparison[self.best_model_name]
        best_metrics["best_model"] = self.best_model_name
        return best_metrics

    # ── Prediction ───────────────────────────────────────────────────────
    def predict(self, X):
        """Predict grade class for new data. Returns numeric class."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_grade_label(self, X):
        """Predict and return the grade letter (A-F)."""
        prediction = self.predict(X)[0]
        return self.GRADE_MAP.get(prediction, "Unknown")

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path="trained_model.pkl"):
        """Save the trained model, scaler, and metadata to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model")

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "best_model_name": self.best_model_name,
            "feature_importance": self.feature_importance,
            "comparison_results": self.comparison_results,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path="trained_model.pkl"):
        """Load a previously saved model from disk."""
        if not os.path.exists(path):
            return None

        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.best_model_name = data["best_model_name"]
        instance.feature_importance = data["feature_importance"]
        instance.comparison_results = data["comparison_results"]
        instance.is_trained = True
        return instance
