from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


from dashboard.output_formatter import format_alert
from inference.alert_generator import generate_alert
from inference.risk_scoring import generate_risk_score
from models.adaboost_model import AdaBoostInsiderModel
from models.train_model import DEFAULT_DATA_PATH, train_and_evaluate
from preprocessing.data_cleaning import clean_data
from preprocessing.feature_extraction import extract_features
from security.access_control import check_alert_access
from security.confidential_processing import confidential_execution
from utils.config import (
    DEVICE_PATH,
    EMAIL_PATH,
    FEATURE_COLUMNS,
    FILE_PATH,
    LOGON_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from utils.logger import LOGGER, log_generated_alert, log_model_prediction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end insider-threat detection pipeline:\n"
            "1) Load raw log reports from data/raw\n"
            "2) Clean data\n"
            "3) Extract features\n"
            "4) Train or load model\n"
            "5) Run inference\n"
            "6) Generate risk scores\n"
            "7) Generate alerts\n"
            "8) Print formatted output"
        )
    )
    parser.add_argument(
        "--role",
        type=str,
        default="SOC",
        help="Caller role for access control (e.g. SOC, ADMIN).",
    )
    parser.add_argument(
        "--viewer-id",
        type=str,
        default="viewer-001",
        help=(
            "Anonymized identifier for the operator viewing alerts. "
            "Used only for audit logging, not printed directly."
        ),
    )
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help=(
            "If set, automatically train a model using the processed training "
            "dataset when no saved model is found."
        ),
    )
    return parser.parse_args()


def _load_or_train_model(train_if_missing: bool) -> AdaBoostInsiderModel:
    """
    Try to load an existing AdaBoost model; optionally train if not found.
    """
    try:
        model = AdaBoostInsiderModel.load_model()
        if model.model is None:
            raise RuntimeError("Loaded AdaBoostInsiderModel has no underlying estimator.")
        return model
    except FileNotFoundError:
        if not train_if_missing:
            raise FileNotFoundError(
                "No trained model found. Run the training pipeline first "
                "or re-run with --train-if-missing."
            )

        print(
            "No saved model detected. Training a new model using the "
            f"processed dataset at: {DEFAULT_DATA_PATH}"
        )
        model_path = train_and_evaluate(DEFAULT_DATA_PATH)
        print(f"Training complete. Model saved to: {model_path}")
        model = AdaBoostInsiderModel.load_model()
        if model.model is None:
            raise RuntimeError("Model training reported success but no estimator was loaded.")
        return model


def _load_raw_event_data_from_config_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw security event datasets from paths defined in utils.config.

    Each CSV is expected to be pre-anonymized and to avoid raw content such as
    email bodies or file contents. This function only handles file I/O and
    logging; feature logic remains in the preprocessing module.
    """
    def _load_single(label: str, path: "pd._typing.FilePath | None") -> pd.DataFrame:
        if path and Path(path).exists():
            LOGGER.info("Loading %s events from CSV: %s", label, path)
            return pd.read_csv(path)
        LOGGER.info("CSV for %s events not found at %s; skipping.", label, path)
        return pd.DataFrame()

    logon_df = _load_single("logon", LOGON_PATH)
    email_df = _load_single("email", EMAIL_PATH)
    file_df = _load_single("file", FILE_PATH)
    device_df = _load_single("device", DEVICE_PATH)

    if (
        logon_df.empty
        and email_df.empty
        and file_df.empty
        and device_df.empty
    ):
        LOGGER.info("No raw event CSV files were loaded; proceeding with empty datasets.")

    return logon_df, email_df, file_df, device_df


def main() -> None:
    args = _parse_args()

    print("=== Insider Threat Detection Pipeline (CLI) ===")
    print(f"Viewer role    : {args.role}")
    print(f"Viewer ID      : {args.viewer_id} (anonymized)\n")

    # 1. Load raw event data from configured CSV paths
    print("1) Loading raw log reports from data/raw...")
    logon_df, email_df, file_df, device_df = _load_raw_event_data_from_config_paths()

    # 2. Clean data
    print("2) Cleaning data...")
    logon_df_clean = clean_data(logon_df)
    email_df_clean = clean_data(email_df)
    file_df_clean = clean_data(file_df)
    device_df_clean = clean_data(device_df)

    # 3. Extract features
    print("3) Extracting per-user features...")
    features_df = extract_features(
        logon_df_clean,
        email_df_clean,
        file_df_clean,
        device_df_clean,
    )

    # Derive heuristic labels for training based on simple behavioral rules.
    features_df["label"] = (
        (features_df["off_hours_login_ratio"] > 0.4)
        | (features_df["file_access_count"] > 50)
        | (features_df["email_send_count"] > 30)
    ).astype(int)

    # Log label distribution for visibility (without exposing raw data).
    label_counts = features_df["label"].value_counts().to_dict()
    labeled_1 = int(label_counts.get(1, 0))
    labeled_0 = int(label_counts.get(0, 0))
    LOGGER.info("Label distribution - 1s: %d, 0s: %d", labeled_1, labeled_0)

    # Persist processed features for training / auditability.
    processed_path = PROCESSED_DATA_DIR / "processed_dataset.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(processed_path, index=False)

    if features_df.empty:
        print("No features extracted from raw event data; nothing to score.")
        return

    user_ids: List[str] = features_df["user_id"].astype(str).tolist()
    # Construct feature matrix strictly from configured numeric feature columns.
    X = features_df[FEATURE_COLUMNS]

    # 4. Train or load model
    print("4) Loading or training AdaBoost model...")
    model = _load_or_train_model(train_if_missing=args.train_if_missing)

    # 5. Run inference (wrapped in simulated confidential execution context)
    print("5) Running model inference inside simulated confidential environment...")
    with confidential_execution():
        proba = model.predict_proba(X)

    # Determine the probability associated with the "positive" insider-threat class.
    classes = getattr(model.model, "classes_", None)
    if classes is not None and len(classes) > 1:
        try:
            positive_index = list(classes).index(1)
        except ValueError:
            positive_index = int(np.argmax(classes))
    else:
        positive_index = 0

    threat_probabilities = proba[:, positive_index]

    # 6. Generate risk scores
    print("6) Generating risk scores...")
    risk_objects = [generate_risk_score(float(p)) for p in threat_probabilities]

    # 7. Generate alerts
    print("7) Generating alerts (non-PII)...")
    alerts = []
    for idx, (user_id, risk) in enumerate(zip(user_ids, risk_objects), start=1):
        risk_level = risk["level"]
        risk_score = risk["score"]

        # Centralized model prediction logging (non-PII identifiers only).
        log_model_prediction(
            employee_id=user_id,
            risk_score=risk_score,
            risk_level=risk_level,
            metadata={"source": "main_pipeline"},
        )

        alert = generate_alert(user_id, risk_level)
        alert_id = f"ALERT-{idx:06d}"
        alert["alert_id"] = alert_id

        log_generated_alert(
            alert_id=alert_id,
            employee_id=user_id,
            risk_level=risk_level,
            reason=f"model_probability={risk_score:.3f}",
            metadata={"source": "main_pipeline"},
        )

        alerts.append(alert)

    # 8. Print formatted output (respecting access control)
    print("8) Applying access control and printing formatted alerts...\n")

    if not check_alert_access(
        user_id=args.viewer_id,
        role=args.role,
        action="read",
        source_ip=None,
    ):
        print(
            f"[ACCESS DENIED] Role '{args.role}' is not authorized to view alerts."
        )
        return

    for alert in alerts:
        print(format_alert(alert))
        print()

    print("Pipeline run complete.")


if __name__ == "__main__":
    main()

