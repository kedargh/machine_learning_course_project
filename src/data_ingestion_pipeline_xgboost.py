from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None 


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create bank-campaign features inside each saved sklearn pipeline.

    Keeping this transformer inside the fitted pipeline means every saved model
    accepts the original raw bank predictor columns during inference.
    """

    RAW_PREDICTOR_COLUMNS = (
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    )

    NUMERIC_COLUMNS = (
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    )

    MONTH_TO_QUARTER = {
        "jan": "Q1",
        "feb": "Q1",
        "mar": "Q1",
        "apr": "Q2",
        "may": "Q2",
        "jun": "Q2",
        "jul": "Q3",
        "aug": "Q3",
        "sep": "Q3",
        "oct": "Q4",
        "nov": "Q4",
        "dec": "Q4",
    }

    ENGINEERED_COLUMNS = (
        "was_previously_contacted",
        "previous_success_flag",
        "total_contact_count",
        "contact_pressure",
        "debt_product_count",
        "recency_band",
        "contact_quarter",
        "balance_segment",
    )

    def __init__(self, include_duration: bool = False) -> None:
        self.include_duration = include_duration

    @staticmethod
    def _as_dataframe(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("BankFeatureEngineer expects a pandas DataFrame.")

        frame = features.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if "y" in frame.columns:
            frame = frame.drop(columns=["y"])

        missing = sorted(set(BankFeatureEngineer.RAW_PREDICTOR_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Prediction input is missing required bank columns: {missing}")

        frame = frame.loc[:, list(BankFeatureEngineer.RAW_PREDICTOR_COLUMNS)]
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].astype("string").str.strip().str.lower()
        for column in BankFeatureEngineer.NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "BankFeatureEngineer":
        del target
        frame = self._as_dataframe(features)
        low_cutoff, high_cutoff = frame["balance"].quantile([0.33, 0.66]).tolist()
        if pd.isna(low_cutoff) or pd.isna(high_cutoff):
            raise ValueError("The balance column contains no usable numeric values.")
        if low_cutoff >= high_cutoff:
            high_cutoff = low_cutoff + 1e-9

        self.low_balance_cutoff_ = float(low_cutoff)
        self.high_balance_cutoff_ = float(high_cutoff)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "low_balance_cutoff_"):
            raise RuntimeError("BankFeatureEngineer must be fitted before transform().")

        engineered = self._as_dataframe(features)
        engineered["was_previously_contacted"] = (engineered["pdays"] != -1).astype(int)
        engineered["previous_success_flag"] = (engineered["poutcome"] == "success").astype(int)
        engineered["total_contact_count"] = engineered["campaign"] + engineered["previous"]
        engineered["contact_pressure"] = engineered["campaign"] / (engineered["previous"] + 1.0)
        engineered["debt_product_count"] = (
            engineered[["default", "housing", "loan"]].eq("yes").sum(axis=1)
        )

        recency_conditions = [
            engineered["pdays"].eq(-1),
            engineered["pdays"].between(0, 30, inclusive="both"),
            engineered["pdays"].between(31, 90, inclusive="both"),
            engineered["pdays"].gt(90),
        ]
        engineered["recency_band"] = pd.Series(
            np.select(
                recency_conditions,
                ["not_previously_contacted", "0_30_days", "31_90_days", "over_90_days"],
                default="unknown",
            ),
            index=engineered.index,
            dtype="string",
        )
        engineered["contact_quarter"] = (
            engineered["month"].map(self.MONTH_TO_QUARTER).fillna("unknown")
        )
        engineered["balance_segment"] = pd.cut(
            engineered["balance"],
            bins=[-np.inf, self.low_balance_cutoff_, self.high_balance_cutoff_, np.inf],
            labels=["low", "medium", "high"],
            include_lowest=True,
        ).astype("string")

        if not self.include_duration:
            engineered = engineered.drop(columns=["duration"])
        return engineered

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("BankFeatureEngineer must be fitted before get_feature_names_out().")
        names = self.feature_names_in_.tolist() + list(self.ENGINEERED_COLUMNS)
        if not self.include_duration:
            names.remove("duration")
        return np.asarray(names, dtype=object)


# Make joblib files portable when this file is executed directly as a script.
sys.modules.setdefault("data_ingestion_pipeline", sys.modules[__name__])
BankFeatureEngineer.__module__ = "data_ingestion_pipeline"


class DataIngestionPipeline:
    """Load multiple Bank Marketing files, train four classifiers and save them.

    By default, ``duration`` is excluded because the current call duration is
    unavailable when the bank is deciding whom to call. Pass
    ``include_duration=True`` only when deliberately building a post-call
    analysis model.
    """

    TARGET_COLUMN = "y"

    EXPECTED_COLUMNS = (
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    )

    NUMERIC_COLUMNS = (
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    )

    MONTH_TO_QUARTER = {
        "jan": "Q1",
        "feb": "Q1",
        "mar": "Q1",
        "apr": "Q2",
        "may": "Q2",
        "jun": "Q2",
        "jul": "Q3",
        "aug": "Q3",
        "sep": "Q3",
        "oct": "Q4",
        "nov": "Q4",
        "dec": "Q4",
    }

    def __init__(
        self,
        *filepaths: str | Path | Iterable[str | Path],
        models_dir: str | Path | None = None,
        test_size: float = 0.20,
        random_state: int = 42,
        split_strategy: str = "stratified",
        include_duration: bool = False,
        drop_duplicates: bool = True,
        decision_threshold: float = 0.50,
    ) -> None:
        flattened_paths: list[Path] = []
        for item in filepaths:
            if isinstance(item, (list, tuple, set)):
                flattened_paths.extend(Path(path).expanduser() for path in item)
            else:
                flattened_paths.append(Path(item).expanduser())

        if not flattened_paths:
            raise ValueError("Provide at least one input CSV filepath.")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if split_strategy not in {"stratified", "ordered"}:
            raise ValueError("split_strategy must be 'stratified' or 'ordered'.")
        if not 0.0 < decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")

        self.filepaths = flattened_paths
        self.project_dir = Path(__file__).resolve().parent
        self.models_dir = (
            Path(models_dir).expanduser().resolve()
            if models_dir is not None
            else self.project_dir / "models"
        )
        self.test_size = test_size
        self.random_state = random_state
        self.split_strategy = split_strategy
        self.include_duration = include_duration
        self.drop_duplicates = drop_duplicates
        self.decision_threshold = decision_threshold

        self.data: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.trained_models: dict[str, Pipeline] = {}
        self.source_summaries: list[dict[str, Any]] = []
        self.ingestion_metadata: dict[str, Any] = {}

    @staticmethod
    def _detect_delimiter(filepath: Path) -> str:
        with filepath.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(8192)

        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ";" if sample.count(";") > sample.count(",") else ","

    @staticmethod
    def _normalise_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].astype("string").str.strip().str.lower()
        return frame

    def _read_file(self, filepath: Path) -> pd.DataFrame:
        filepath = filepath.resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Input file does not exist: {filepath}")
        if not filepath.is_file():
            raise ValueError(f"Input path is not a file: {filepath}")

        delimiter = self._detect_delimiter(filepath)
        frame = pd.read_csv(filepath, sep=delimiter, encoding="utf-8-sig")
        frame.columns = [str(column).strip().lower() for column in frame.columns]

        if frame.columns.duplicated().any():
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names in {filepath}: {duplicates}")

        expected = set(self.EXPECTED_COLUMNS)
        observed = set(frame.columns)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                f"Schema mismatch in {filepath}. Missing={missing}; extra={extra}. "
                f"Expected columns={list(self.EXPECTED_COLUMNS)}"
            )

        frame = frame.loc[:, list(self.EXPECTED_COLUMNS)]
        frame = self._normalise_text_columns(frame)

        for column in self.NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        self.source_summaries.append(
            {
                "filepath": str(filepath),
                "delimiter": delimiter,
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
            }
        )
        return frame

    def load_data(self) -> pd.DataFrame:
        """Read, validate and merge every supplied file."""
        self.source_summaries = []
        frames = [self._read_file(filepath) for filepath in self.filepaths]
        combined = pd.concat(frames, ignore_index=True)

        rows_before_deduplication = len(combined)
        if self.drop_duplicates:
            combined = combined.drop_duplicates(ignore_index=True)
        duplicates_removed = rows_before_deduplication - len(combined)

        invalid_targets = sorted(
            value for value in combined[self.TARGET_COLUMN].dropna().unique() if value not in {"yes", "no"}
        )
        if invalid_targets:
            raise ValueError(
                f"Target column '{self.TARGET_COLUMN}' contains invalid values: {invalid_targets}. "
                "Only 'yes' and 'no' are supported."
            )
        if combined[self.TARGET_COLUMN].isna().any():
            raise ValueError("Target column contains missing values.")

        if combined.empty:
            raise ValueError("No records remain after ingestion.")

        target_counts = combined[self.TARGET_COLUMN].value_counts().to_dict()
        self.ingestion_metadata = {
            "rows_before_deduplication": int(rows_before_deduplication),
            "rows_after_deduplication": int(len(combined)),
            "duplicates_removed": int(duplicates_removed),
            "target_counts": {str(key): int(value) for key, value in target_counts.items()},
            "sources": self.source_summaries,
        }
        self.data = combined
        return combined

    def prepare_train_test_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create the requested split before fitting any preprocessing step."""
        if self.data is None:
            self.load_data()
        assert self.data is not None

        target = self.data[self.TARGET_COLUMN].map({"no": 0, "yes": 1}).astype(int)
        features = self.data.drop(columns=[self.TARGET_COLUMN])

        if self.split_strategy == "stratified":
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=target,
            )
        else:
            split_index = int(len(features) * (1.0 - self.test_size))
            X_train = features.iloc[:split_index].copy()
            X_test = features.iloc[split_index:].copy()
            y_train = target.iloc[:split_index].copy()
            y_test = target.iloc[split_index:].copy()
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                raise ValueError(
                    "The ordered split does not contain both classes in train and test. "
                    "Use split_strategy='stratified' for these files."
                )

        self.X_train = X_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        return self.X_train, self.X_test, self.y_train, self.y_test

    @staticmethod
    def _make_one_hot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", self._make_one_hot_encoder()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
                ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
            ],
            remainder="drop",
        )

    def _build_classifiers(self, y_train: pd.Series) -> dict[str, Any]:
        if XGBClassifier is None:
            raise RuntimeError(
                "XGBoost is required for the fourth model. Install it using: pip install xgboost"
            )

        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())
        scale_pos_weight = negative_count / max(positive_count, 1)

        return {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.random_state,
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            "xgboost": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                n_jobs=-1,
                random_state=self.random_state,
            ),
        }

    def _evaluate_predictions(
        self,
        y_true: pd.Series,
        probabilities: np.ndarray,
    ) -> tuple[dict[str, float | int], np.ndarray]:
        predictions = (probabilities >= self.decision_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

        metrics: dict[str, float | int] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
        return metrics, predictions

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def train_evaluate_and_save(self) -> pd.DataFrame:
        """Train all four classifiers, evaluate them and save every pipeline."""
        if self.X_train is None or self.X_test is None:
            self.prepare_train_test_data()

        assert self.X_train is not None
        assert self.X_test is not None
        assert self.y_train is not None
        assert self.y_test is not None

        self.models_dir.mkdir(parents=True, exist_ok=True)
        classifiers = self._build_classifiers(self.y_train)
        score_rows: list[dict[str, Any]] = []
        classification_reports: dict[str, Any] = {}

        for model_name, classifier in classifiers.items():
            model_pipeline = Pipeline(
                steps=[
                    ("feature_engineering", BankFeatureEngineer(self.include_duration)),
                    ("preprocessor", self._build_preprocessor()),
                    ("classifier", classifier),
                ]
            )

            start_time = time.perf_counter()
            model_pipeline.fit(self.X_train, self.y_train)
            training_seconds = time.perf_counter() - start_time

            probabilities = model_pipeline.predict_proba(self.X_test)[:, 1]
            metrics, predictions = self._evaluate_predictions(self.y_test, probabilities)

            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model_pipeline, model_path, compress=3)
            self.trained_models[model_name] = model_pipeline

            score_rows.append(
                {
                    "model": model_name,
                    **metrics,
                    "threshold": self.decision_threshold,
                    "training_seconds": float(training_seconds),
                    "model_file": model_path.name,
                }
            )
            classification_reports[model_name] = classification_report(
                self.y_test,
                predictions,
                target_names=["no_subscription", "subscription"],
                output_dict=True,
                zero_division=0,
            )

        scores = pd.DataFrame(score_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
        best_model_name = str(scores.iloc[0]["model"])
        best_feature_engineer = self.trained_models[best_model_name].named_steps[
            "feature_engineering"
        ]
        joblib.dump(
            self.trained_models[best_model_name],
            self.models_dir / "best_model.joblib",
            compress=3,
        )

        scores.to_csv(self.models_dir / "model_scores.csv", index=False)
        with (self.models_dir / "model_scores.json").open("w", encoding="utf-8") as file:
            json.dump(score_rows, file, indent=2, default=self._json_default)
        with (self.models_dir / "classification_reports.json").open("w", encoding="utf-8") as file:
            json.dump(classification_reports, file, indent=2, default=self._json_default)

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_directory": str(self.project_dir),
            "models_directory": str(self.models_dir),
            "ingestion": self.ingestion_metadata,
            "split_strategy": self.split_strategy,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "duration_included": self.include_duration,
            "train_rows": int(len(self.X_train)),
            "test_rows": int(len(self.X_test)),
            "training_positive_rate": float(self.y_train.mean()),
            "test_positive_rate": float(self.y_test.mean()),
            "raw_model_input_columns": self.X_train.columns.tolist(),
            "balance_segment_train_cutoffs": {
                "low_to_medium": best_feature_engineer.low_balance_cutoff_,
                "medium_to_high": best_feature_engineer.high_balance_cutoff_,
            },
            "best_model_by_pr_auc": best_model_name,
            "saved_models": [f"{name}.joblib" for name in classifiers] + ["best_model.joblib"],
        }
        with (self.models_dir / "training_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, default=self._json_default)

        feature_schema = {
            "training_file_columns": list(self.EXPECTED_COLUMNS),
            "prediction_input_columns": list(BankFeatureEngineer.RAW_PREDICTOR_COLUMNS),
            "target_column": self.TARGET_COLUMN,
            "target_mapping": {"no": 0, "yes": 1},
            "duration_default": "excluded_from_pre_call_model",
            "saved_models_accept_raw_bank_predictors": True,
            "engineered_features": list(BankFeatureEngineer.ENGINEERED_COLUMNS),
        }
        with (self.models_dir / "feature_schema.json").open("w", encoding="utf-8") as file:
            json.dump(feature_schema, file, indent=2)

        return scores

    def run(self) -> pd.DataFrame:
        """Execute ingestion, preparation, training, evaluation and saving."""
        self.load_data()
        self.prepare_train_test_data()
        scores = self.train_evaluate_and_save()

        print("\nIngestion summary")
        print("-" * 80)
        print(f"Input files:                 {len(self.filepaths):,}")
        print(f"Rows before deduplication:   {self.ingestion_metadata['rows_before_deduplication']:,}")
        print(f"Exact duplicate rows removed:{self.ingestion_metadata['duplicates_removed']:>8,}")
        print(f"Rows used:                   {self.ingestion_metadata['rows_after_deduplication']:,}")
        print(f"Duration included:           {self.include_duration}")
        print(f"Split strategy:              {self.split_strategy}")

        display_columns = [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "training_seconds",
        ]
        print("\nModel scores, sorted by PR-AUC")
        print("-" * 80)
        print(scores[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"\nSaved models and reports to: {self.models_dir}")
        return scores


def load_parameters(parameters_path: str | Path) -> dict[str, Any]:
    """Load pipeline settings; resolve relative paths beside the JSON file."""
    parameters_path = Path(parameters_path).expanduser().resolve()
    with parameters_path.open("r", encoding="utf-8") as file:
        parameters = json.load(file)

    if not isinstance(parameters, dict):
        raise ValueError("parameters.json must contain a JSON object.")

    allowed_keys = {
        "filepaths", "models_dir", "test_size", "random_state",
        "split_strategy", "include_duration", "drop_duplicates",
        "decision_threshold",
    }
    unknown_keys = sorted(set(parameters) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown pipeline parameters: {unknown_keys}")

    filepaths = parameters.get("filepaths")
    if not isinstance(filepaths, list) or not filepaths or any(
        not isinstance(path, str) or not path.strip() for path in filepaths
    ):
        raise ValueError("'filepaths' must be a non-empty list of CSV path strings.")

    def resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (parameters_path.parent / path).resolve()

    parameters["filepaths"] = [resolve_path(path) for path in filepaths]
    if parameters.get("models_dir") is not None:
        models_dir = parameters["models_dir"]
        if not isinstance(models_dir, str) or not models_dir.strip():
            raise ValueError("'models_dir' must be a non-empty path string or null.")
        parameters["models_dir"] = resolve_path(models_dir)
    for key in ("include_duration", "drop_duplicates"):
        if key in parameters and not isinstance(parameters[key], bool):
            raise ValueError(f"'{key}' must be a JSON boolean (true or false).")
    return parameters


def main(parameters_path: str | Path | None = None) -> None:
    if parameters_path is None:
        parameters_path = Path(__file__).resolve().parent.parent / "parameters.json"
    parameters = load_parameters(parameters_path)
    filepaths = parameters.pop("filepaths")
    pipeline = DataIngestionPipeline(*filepaths, **parameters)
    pipeline.run()


if __name__ == "__main__":
    main()
