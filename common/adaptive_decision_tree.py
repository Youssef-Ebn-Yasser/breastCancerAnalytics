"""
Learned Decision Tree for Adaptive Clinical Decision-Making.

This module builds a multi-criteria Decision Tree that combines:
1. Low-level Image Quality Assessment (IQA) metrics (Sharpness, Contrast, SNR, Entropy)
2. High-level CNN model confidence and prediction ambiguity

To make adaptive clinical routing decisions:
- Accept_Direct_Prediction: High image quality + High CNN confidence (Autonomous classification)
- Route_Image_Enhancement: Degraded contrast/noise, but salvageable (CLAHE / Denoising filter pipeline)
- Expert_Human_Review: Borderline quality or high classification uncertainty (Radiologist/Pathologist escalation)
- Reject_Reacquire: Severe blur/artifacts or blank slide (Request image re-scan)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


ROUTING_CLASSES = [
    "Accept_Direct_Prediction",
    "Route_Image_Enhancement",
    "Expert_Human_Review",
    "Reject_Reacquire",
]


def generate_ground_truth_routing_policy(df: pd.DataFrame) -> pd.Series:
    """
    Generate ground-truth routing labels based on physical quality bounds
    and classification confidence. This simulates the consensus clinical protocol
    used to train the learned decision tree.
    """
    decisions = []
    for _, row in df.iterrows():
        q_score = row.get("composite_quality_score", 0.5)
        conf = row.get("cnn_confidence", 0.85)
        lap_var = row.get("laplacian_variance", 100.0)
        contrast = row.get("rms_contrast", 0.20)
        snr = row.get("snr_db", 20.0)

        # 1. Severe defects: Reject & Reacquire
        if q_score < 0.20 or lap_var < 15.0:
            decisions.append("Reject_Reacquire")
        # 2. Low contrast or high noise with reasonable sharpness: Image Enhancement
        elif (contrast < 0.12 or snr < 10.0) and lap_var >= 25.0:
            decisions.append("Route_Image_Enhancement")
        # 3. High quality and high model confidence: Accept Direct
        elif q_score >= 0.60 and conf >= 0.80:
            decisions.append("Accept_Direct_Prediction")
        # 4. Borderline quality or model uncertainty: Escalate to Expert
        else:
            decisions.append("Expert_Human_Review")

    return pd.Series(decisions, index=df.index, name="adaptive_decision")


class AdaptiveClinicalDecisionTree:
    """
    Trained Decision Tree model for routing medical images adaptively.
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion="gini",
            random_state=42,
        )
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract relevant numeric features for the decision tree.
        """
        candidate_cols = [
            "composite_quality_score",
            "laplacian_variance",
            "tenengrad_gradient",
            "rms_contrast",
            "michelson_contrast",
            "shannon_entropy",
            "snr_db",
            "cnr",
            "cnn_confidence",
        ]
        available_cols = [col for col in candidate_cols if col in df.columns]
        if not available_cols:
            raise ValueError(f"None of candidate features {candidate_cols} found in dataframe.")
        return df[available_cols].values, available_cols

    def fit(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        """
        Train the adaptive decision tree.
        """
        X, feature_names = self.prepare_features(df)
        self.feature_columns = feature_names

        if target_col is None or target_col not in df.columns:
            y = generate_ground_truth_routing_policy(df)
        else:
            y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rules_str = export_text(self.model, feature_names=feature_names)

        feature_importances = dict(zip(feature_names, self.model.feature_importances_))

        return {
            "test_accuracy": float(np.mean(y_pred == y_test)),
            "classification_report": report,
            "feature_importances": feature_importances,
            "rules_text": rules_str,
            "classes": list(self.model.classes_),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict routing decisions for input samples.
        """
        if not self.is_fitted:
            raise RuntimeError("Decision Tree model must be fitted before predict.")
        X = df[self.feature_columns].values
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict routing decision probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Decision Tree model must be fitted before predict_proba.")
        X = df[self.feature_columns].values
        return self.model.predict_proba(X)
