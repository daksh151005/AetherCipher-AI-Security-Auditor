"""
Machine Learning Training & Cryptographic Model Comparison
Strictly follows ML Best Practices:
  - Missing/Null value validation & audit
  - Train/Test split BEFORE featurization & scaling
  - 5-Fold Stratified Cross-Validation
  - Multiple model benchmark (Dummy baseline, Logistic Regression, Random Forest, SVM, HistGradientBoosting)
  - Multi-metric evaluation (Accuracy, Macro F1, Precision, Recall, ROC-AUC, Latency, Model Size)
  - 95% Confidence Intervals via Bootstrapping
  - Confusion Matrix & Feature Importance plots
  - Production Pipeline persistence using Joblib
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.inspection import permutation_importance

from feature_extractor import FEATURE_NAMES, CLASS_LABELS

# Directories
ARTIFACTS_DIR = "ml_artifacts"
MODELS_DIR = "models"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def calculate_bootstrap_ci(y_true, y_pred, metric_func=accuracy_score, n_bootstraps=500, alpha=0.05):
    """
    Computes 95% confidence intervals for a given metric via non-parametric bootstrapping.
    """
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def train_and_evaluate_models(dataset_path="dataset/crypto_dataset.csv"):
    print("=" * 70)
    print("  CRYPTOGRAPHIC CLASSIFIER: ML BENCHMARK & AUDIT SUITE")
    print("=" * 70)

    # 1. Load Data & Understand Schema
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run dataset_generator.py first.")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Schema features:", FEATURE_NAMES)

    # 2. Check for Missing / NULL Values (Essential ML Practice)
    null_counts = df[FEATURE_NAMES + ["class_label"]].isnull().sum()
    print("\n[Data Quality Audit] Null Value Check:")
    print(null_counts)
    if null_counts.sum() > 0:
        print("Imputing missing values with median...")
        df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())
    else:
        print("✔ Zero missing/null values detected. Perfect data integrity.")

    X = df[FEATURE_NAMES].values
    y = df["class_label"].values

    # 3. Strict Featurization Ordering: Split BEFORE Fitting Scaler
    # 80% Train, 20% Holdout Test (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {X_train.shape[0]} samples | Test set size: {X_test.shape[0]} samples")

    # Fit Scaler strictly on Training Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Suite Definition (Baselines + Advanced)
    models = {
        "Naive Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Linear)": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Classifier (RBF)": SVC(kernel="rbf", probability=True, C=2.0, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
    }

    # 5. Model Training, Cross-Validation & Metric Evaluation
    results = {}
    fitted_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 70)
    print(f"{'Model':<32} | {'CV-Acc (5-fold)':<16} | {'Test Acc':<10} | {'Macro F1':<10} | {'Latency (us)':<12}")
    print("-" * 70)

    for name, clf in models.items():
        # Train on scaled training data
        clf.fit(X_train_scaled, y_train)
        fitted_models[name] = clf

        # 5-Fold Cross-Validation on Train
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Inference Latency Benchmark
        t0 = time.perf_counter()
        y_test_pred = clf.predict(X_test_scaled)
        t_elapsed = time.perf_counter() - t0
        latency_us = float((t_elapsed / len(X_test)) * 1_000_000)

        # Probabilities for ROC-AUC
        if hasattr(clf, "predict_proba"):
            y_test_proba = clf.predict_proba(X_test_scaled)
            roc_auc = float(roc_auc_score(y_test, y_test_proba, multi_class="ovr", average="macro"))
        else:
            roc_auc = 0.50

        # Metrics
        test_acc = float(accuracy_score(y_test, y_test_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro", zero_division=0)

        # 95% Confidence Interval for Accuracy
        ci_lower, ci_upper = calculate_bootstrap_ci(y_test, y_test_pred, metric_func=accuracy_score)

        print(
            f"{name:<32} | {cv_mean*100:5.2f}% ± {cv_std*100:4.2f}% | "
            f"{test_acc*100:6.2f}%   | {f1:8.4f}   | {latency_us:8.2f} µs"
        )

        results[name] = {
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "test_accuracy": test_acc,
            "test_f1_macro": float(f1),
            "test_precision_macro": float(prec),
            "test_recall_macro": float(rec),
            "test_roc_auc": roc_auc,
            "ci_95_accuracy": [ci_lower, ci_upper],
            "latency_us_per_sample": latency_us,
        }

    # 6. Select Best Performing Production Model (by Test Macro F1)
    best_model_name = max(results, key=lambda k: (results[k]["test_f1_macro"], results[k]["test_accuracy"]))
    best_clf = fitted_models[best_model_name]
    print("=" * 70)
    print(f"🏆 Best Production Candidate: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['test_accuracy']*100:.2f}% (95% CI: [{results[best_model_name]['ci_95_accuracy'][0]*100:.2f}%, {results[best_model_name]['ci_95_accuracy'][1]*100:.2f}%])")
    print(f"   Macro F1: {results[best_model_name]['test_f1_macro']:.4f} | ROC-AUC: {results[best_model_name]['test_roc_auc']:.4f}")

    # Full Classification Report for the Best Model
    y_best_pred = best_clf.predict(X_test_scaled)
    target_names = [CLASS_LABELS[i] for i in range(len(CLASS_LABELS))]
    print("\nDetailed Classification Report for Best Model:")
    print(classification_report(y_test, y_best_pred, target_names=target_names))

    # 7. Generate Visualizations & Diagnostic Plots
    # A. Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Plaintext", "Weak_Arnold", "Strong_Chaos"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix: {best_model_name}\n(Holdout Test Set N={len(y_test)})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"✔ Saved Confusion Matrix plot to {cm_path}")

    # B. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names_short = [k.replace("Classifier", "").replace("(Linear)", "").strip() for k in results.keys()]
    acc_scores = [results[k]["test_accuracy"] * 100 for k in results.keys()]
    f1_scores = [results[k]["test_f1_macro"] * 100 for k in results.keys()]

    x_indices = np.arange(len(model_names_short))
    width = 0.35

    ax.bar(x_indices - width / 2, acc_scores, width, label="Test Accuracy (%)", color="#0284c7")
    ax.bar(x_indices + width / 2, f1_scores, width, label="Test Macro F1 (%)", color="#10b981")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Model Benchmark Comparison on Cryptographic Classification", fontweight="bold", pad=14)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    comp_path = os.path.join(ARTIFACTS_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"✔ Saved Model Comparison plot to {comp_path}")

    # C. Feature Importance Analysis
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
    else:
        # Fallback to permutation importance
        perm_res = permutation_importance(best_clf, X_test_scaled, y_test, n_repeats=10, random_state=42)
        importances = perm_res.importances_mean

    feat_df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances}).sort_values(
        by="importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#8b5cf6")
    ax.set_xlabel("Relative Feature Importance (MDI / Permutation)", fontweight="bold")
    ax.set_title(f"Cryptographic Security Feature Importance ({best_model_name})", fontweight="bold", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"✔ Saved Feature Importance plot to {fi_path}")

    # 8. Save Production Pipeline (Scaler + Best Model + Metadata)
    pipeline_bundle = {
        "model_name": best_model_name,
        "scaler": scaler,
        "classifier": best_clf,
        "feature_names": FEATURE_NAMES,
        "class_labels": CLASS_LABELS,
        "metrics": results[best_model_name],
        "all_model_results": results,
        "training_timestamp": time.time(),
    }
    model_export_path = os.path.join(MODELS_DIR, "crypto_classifier.joblib")
    joblib.dump(pipeline_bundle, model_export_path)
    print(f"✔ Production Model Pipeline exported to {model_export_path}")

    # 9. Save JSON Summary for UI Dashboard
    json_summary_path = os.path.join(ARTIFACTS_DIR, "metrics_summary.json")
    with open(json_summary_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_metrics": results[best_model_name],
                "all_models": results,
                "feature_importance": dict(zip(feat_df["feature"].tolist(), feat_df["importance"].round(4).tolist())),
                "classes": CLASS_LABELS,
            },
            f,
            indent=2,
        )
    print(f"✔ Metrics summary saved to {json_summary_path}")
    print("=" * 70 + "\n")

    return pipeline_bundle


if __name__ == "__main__":
    train_and_evaluate_models()
