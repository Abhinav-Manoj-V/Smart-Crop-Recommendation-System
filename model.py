"""
model.py — Crop Recommendation model training pipeline
=========================================================
Converted from crop_recommendation.ipynb into a reproducible training script.

Trains a Stacking Ensemble (Random Forest + SVM(RBF) + XGBoost -> Logistic
Regression meta-learner) for crop recommendation, and saves the fitted model
+ label encoder as pickle files for the Streamlit app to consume.

IMPORTANT — feature order
--------------------------
The model is trained on features in this exact order:

    N, P, K, temperature, humidity, ph, rainfall

Any code that builds a raw input array for `model.predict(...)` /
`model.predict_proba(...)` MUST supply the features in this order, or
predictions will be silently wrong (this was a real bug found in the
original app.py — pH and temperature/humidity were swapped).

What changed vs. the notebook (and why it improves confidence)
----------------------------------------------------------------
1. Bug fix: feature order is now defined once (FEATURE_ORDER) and reused
   everywhere, instead of being implicitly assumed.
2. RF / XGBoost are trained on RAW (unscaled) features, exactly like the
   notebook, so SHAP TreeExplainer on the RF sub-model still works directly
   on raw user inputs in the Streamlit app.
3. The stacking meta-learner (Logistic Regression) is wrapped in its own
   Pipeline([StandardScaler, LogisticRegression]) so the concatenated
   meta-feature matrix (base-model probabilities + optional passthrough
   raw features) is on a comparable scale. Without this, LogisticRegression
   fails to converge and produces mushy, low-confidence probabilities.
4. `passthrough=True` gives the meta-learner access to the raw features in
   addition to base-model probabilities, which sharpens its decisions.
5. `cv=10` (was 5) for the internal stacking cross-validation, so base
   learners generate less noisy out-of-fold probability estimates for the
   meta-learner to train on.
6. Hyperparameters tuned for this dataset (which is small, clean, and
   near-perfectly separable): more trees, shallower regularization on RF's
   leaf size, less regularization on the meta-learner (C=5.0) so it doesn't
   over-flatten already-confident base-model probabilities.
7. Everything is wrapped into reusable functions with a CLI, instead of a
   linear notebook, so it can be re-run / re-trained deterministically.

Usage
-----
    python model.py --data Crop_recommendation.csv --output-dir models

This produces, inside --output-dir:
    stack_model_combined.pkl     label_encoder_combined.pkl
    stack_model_agri.pkl         label_encoder_agri.pkl
    stack_model_horti.pkl        label_encoder_horti.pkl

The Streamlit app (app.py) only needs the "combined" pair. The agri/horti
category-specific models are trained too (mirroring the notebook's
methodology) in case you want to switch the app to category-specific
prediction later — see the notebook's discussion in section 6.2.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Canonical feature order — used for training AND must be used at
#    inference time by app.py. Do not change without retraining. ─────────────
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

AGRICULTURAL_CROPS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "cotton", "jute",
]

HORTICULTURAL_CROPS = [
    "pomegranate", "banana", "mango", "grapes", "watermelon",
    "muskmelon", "apple", "orange", "papaya", "coconut", "coffee",
]

RANDOM_STATE = 42


# ── Base learners + stacking ensemble ─────────────────────────────────────────
def build_stack_model() -> StackingClassifier:
    """
    Builds the stacking ensemble: RF + SVM(RBF) + XGBoost -> Logistic
    Regression meta-learner. Hyperparameters are tuned (vs. the notebook's
    defaults) to sharpen predicted-class confidence without sacrificing
    generalization, since the underlying dataset is small and clean.
    """
    from xgboost import XGBClassifier

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    estimators = [("rf", rf), ("svm", svm_pipeline), ("xgb", xgb)]

    # Meta-learner gets its own scaler so the concatenated meta-feature
    # matrix (base-model probabilities in [0,1] + passthrough raw features
    # in very different ranges, e.g. rainfall up to ~300) is well conditioned.
    # This is what makes the ensemble's final probabilities confident and
    # well-separated instead of mushy / non-converged.
    final_estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=5.0, max_iter=5000)),
    ])

    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=10,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )
    return stack_model


# ── Training + evaluation for one dataset slice ───────────────────────────────
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, name: str):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"\n{'=' * 60}\n{name} model — training on {X_train.shape[0]} samples "
          f"({X.shape[1]} features, {len(encoder.classes_)} classes)\n{'=' * 60}")

    model = build_stack_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    top_conf = y_proba.max(axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Fit time:            {fit_seconds:.1f}s")
    print(f"Accuracy:            {acc * 100:.2f}%")
    print(f"Precision (macro):   {precision * 100:.2f}%")
    print(f"Recall (macro):      {recall * 100:.2f}%")
    print(f"F1-score (macro):    {f1 * 100:.2f}%")
    print(f"Mean top-class confidence:   {top_conf.mean() * 100:.2f}%")
    print(f"Median top-class confidence: {np.median(top_conf) * 100:.2f}%")
    print(f"Lowest top-class confidence: {top_conf.min() * 100:.2f}%  "
          f"(hardest test sample — some overlap between crops is expected)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

    return model, encoder, {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "mean_confidence": float(top_conf.mean()),
    }


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Train the crop recommendation stacking ensemble.")
    parser.add_argument("--data", default="Crop_recommendation.csv", help="Path to Crop_recommendation.csv")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained model/encoder pickles")
    parser.add_argument("--skip-category-models", action="store_true",
                         help="Only train the combined (22-class) model used by app.py; skip agri/horti models")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Could not find dataset at '{args.data}'. Pass --data <path to Crop_recommendation.csv>.")

    try:
        import xgboost  # noqa: F401
    except ImportError:
        sys.exit(
            "xgboost is not installed. Install dependencies first:\n"
            "    pip install scikit-learn xgboost shap pandas numpy"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    missing = set(FEATURE_ORDER + ["label"]) - set(df.columns)
    if missing:
        sys.exit(f"Dataset is missing expected columns: {missing}")

    print(f"Loaded dataset: {df.shape[0]} rows, {df['label'].nunique()} crop classes")

    # ── Combined model (all 22 crops) — this is the one app.py loads ──────────
    X_all = df[FEATURE_ORDER]
    y_all = df["label"]
    combined_model, combined_encoder, combined_metrics = train_and_evaluate(X_all, y_all, "Combined (22-class)")
    save_pickle(combined_model, os.path.join(args.output_dir, "stack_model_combined.pkl"))
    save_pickle(combined_encoder, os.path.join(args.output_dir, "label_encoder_combined.pkl"))

    if not args.skip_category_models:
        # ── Agricultural-crops model ───────────────────────────────────────────
        df_agri = df[df["label"].isin(AGRICULTURAL_CROPS)]
        X_agri, y_agri = df_agri[FEATURE_ORDER], df_agri["label"]
        agri_model, agri_encoder, agri_metrics = train_and_evaluate(X_agri, y_agri, "Agricultural crops")
        save_pickle(agri_model, os.path.join(args.output_dir, "stack_model_agri.pkl"))
        save_pickle(agri_encoder, os.path.join(args.output_dir, "label_encoder_agri.pkl"))

        # ── Horticultural-crops model ──────────────────────────────────────────
        df_horti = df[df["label"].isin(HORTICULTURAL_CROPS)]
        X_horti, y_horti = df_horti[FEATURE_ORDER], df_horti["label"]
        horti_model, horti_encoder, horti_metrics = train_and_evaluate(X_horti, y_horti, "Horticultural crops")
        save_pickle(horti_model, os.path.join(args.output_dir, "stack_model_horti.pkl"))
        save_pickle(horti_encoder, os.path.join(args.output_dir, "label_encoder_horti.pkl"))

        print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
        summary = pd.DataFrame({
            "Model": ["Combined", "Agricultural", "Horticultural"],
            "Accuracy (%)": [combined_metrics["accuracy"] * 100, agri_metrics["accuracy"] * 100, horti_metrics["accuracy"] * 100],
            "F1 macro (%)": [combined_metrics["f1"] * 100, agri_metrics["f1"] * 100, horti_metrics["f1"] * 100],
            "Mean confidence (%)": [combined_metrics["mean_confidence"] * 100, agri_metrics["mean_confidence"] * 100, horti_metrics["mean_confidence"] * 100],
        })
        print(summary.to_string(index=False))

    print(f"\nDone. Model files written to: {os.path.abspath(args.output_dir)}")
    print("app.py expects to find stack_model_combined.pkl and label_encoder_combined.pkl "
          "in a 'models/' folder next to (or one level above) the app's own directory.")


if __name__ == "__main__":
    main()
