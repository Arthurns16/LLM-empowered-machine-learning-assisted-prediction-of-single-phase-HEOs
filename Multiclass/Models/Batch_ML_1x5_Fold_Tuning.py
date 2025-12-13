#!/usr/bin/env python3
# Batch ML with 1x5 Fold Tuning 

import re
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    average_precision_score, roc_auc_score, log_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import joblib

# Optional: XGBoost / LightGBM / CatBoost
_HAVE_XGB = _HAVE_LGBM = _HAVE_CAT = True
try:
    from xgboost import XGBClassifier
except Exception:
    _HAVE_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    _HAVE_LGBM = False
try:
    from catboost import CatBoostClassifier
except Exception:
    _HAVE_CAT = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# CONFIG
# -----------------------
TARGET_COL = "phase_disambiguated"
INPUT_DIR = Path("datasets_for_ML_FineTune")
OUT_ROOT  = Path("Tunned_ML_evaluation")

RANDOM_STATE = 42
TUNE_SEED = 1337
N_SPLITS_TRAIN_CV = 5
TEST_SIZE = 0.20
VAL_SIZE  = 0.20

N_ITER_TUNE = 60
N_JOBS_TUNE = -1


# -----------------------
# Data load helpers
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._\-]+", "_", stem).strip("_")
    return stem or "dataset"

def read_any_table(path: Path) -> dict[str, pd.DataFrame]:
    suf = path.suffix.lower()
    if suf == ".csv":
        return {safe_stem(path.name): pd.read_csv(path)}
    if suf in [".tsv", ".txt"]:
        return {safe_stem(path.name): pd.read_csv(path, sep="\t")}
    if suf in [".xlsx", ".xls"]:
        xls = pd.read_excel(path, sheet_name=None)
        out = {}
        for sheet_name, df in xls.items():
            tag = f"{safe_stem(path.name)}__{re.sub(r'[^A-Za-z0-9._\\-]+','_',str(sheet_name))}"
            out[tag] = df
        return out
    return {}

def infer_feature_types(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f'Target column "{target}" not found.')
    X = df.drop(columns=[target], errors="ignore")
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    y = df[target]
    return X, y, num_cols, cat_cols

def _ohe_kwargs():
    # sklearn rename: sparse -> sparse_output
    try:
        OneHotEncoder(sparse_output=False)
        return dict(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return dict(handle_unknown="ignore", sparse=False)

def needs_scaling(est_name: str) -> bool:
    return est_name in {"logreg", "svc_rbf"}

def make_preprocessor(num_cols, cat_cols, scale_numeric=False):
    transformers = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(**_ohe_kwargs()))
            ]),
            cat_cols
        ))
    return ColumnTransformer(transformers, remainder="drop")


# -----------------------
# Probability-based CV scorers (multiclass-safe)
# -----------------------
def _brier_multiclass_from_proba(y_true, y_proba, class_order):
    y_true = np.asarray(y_true)
    y_bin = (y_true[:, None] == class_order[None, :]).astype(float)
    return float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))

def logloss_scorer_cv(estimator, X, y):
    if not hasattr(estimator, "predict_proba"):
        return np.nan
    proba = estimator.predict_proba(X)
    clf = getattr(estimator, "named_steps", {}).get("clf", None) if hasattr(estimator, "named_steps") else estimator
    labels = getattr(clf, "classes_", None)
    try:
        return -log_loss(y, proba, labels=labels)
    except Exception:
        return -log_loss(y, proba)

def brier_scorer_cv(estimator, X, y):
    if not hasattr(estimator, "predict_proba"):
        return np.nan
    proba = estimator.predict_proba(X)
    clf = getattr(estimator, "named_steps", {}).get("clf", None) if hasattr(estimator, "named_steps") else estimator
    classes_ = getattr(clf, "classes_", None)
    if classes_ is None:
        classes_ = np.unique(y)
    classes_ = np.asarray(classes_)
    return -_brier_multiclass_from_proba(y, proba, classes_)


# -----------------------
# Metrics on a split (TRAIN / TEST)
# -----------------------
def compute_all_metrics(pipe: Pipeline, X, y_true, classes_k: int):
    y_pred = pipe.predict(X)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_macro": np.nan,
        "pr_auc_macro": np.nan,
        "log_loss": np.nan,
        "brier": np.nan,
    }

    clf = pipe.named_steps.get("clf", None)
    if clf is not None and hasattr(clf, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            # PR AUC macro (one-vs-rest)
            y_true_np = np.asarray(y_true)
            y_bin = (y_true_np[:, None] == np.arange(proba.shape[1])[None, :]).astype(int)
            out["pr_auc_macro"] = float(average_precision_score(y_bin, proba, average="macro"))
        except Exception:
            pass

        try:
            if classes_k == 2:
                out["roc_auc_macro"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                out["roc_auc_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except Exception:
            pass

        try:
            out["log_loss"] = float(log_loss(y_true, proba, labels=np.arange(classes_k)))
        except Exception:
            pass

        try:
            y_true_np = np.asarray(y_true)
            y_bin = (y_true_np[:, None] == np.arange(proba.shape[1])[None, :]).astype(float)
            out["brier"] = float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))
        except Exception:
            pass

    return out


# -----------------------
# Wrapper for contiguous labels 0..k-1
# -----------------------
class LabelContiguityWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y = np.asarray(y)
        self._classes_seen_ = np.unique(y)
        self._to_local_ = {lab: i for i, lab in enumerate(self._classes_seen_)}
        self._to_global_ = {i: lab for lab, i in self._to_local_.items()}
        y_local = np.vectorize(self._to_local_.get)(y)
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y_local)
        self.classes_ = self._classes_seen_
        return self

    def predict(self, X):
        y_local = self.est_.predict(X)
        return np.vectorize(self._to_global_.get)(y_local)

    def predict_proba(self, X):
        if not hasattr(self.est_, "predict_proba"):
            raise AttributeError("Wrapped estimator lacks predict_proba")
        return self.est_.predict_proba(X)

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator}

    def set_params(self, **params):
        if "base_estimator" in params:
            self.base_estimator = params["base_estimator"]
        return self


# -----------------------
# Models and tuning spaces
# -----------------------
def base_estimators(classes_k: int):
    models = {
        "logreg": LogisticRegression(max_iter=4000, solver="saga", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "svc_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    if _HAVE_XGB:
        obj = "binary:logistic" if classes_k == 2 else "multi:softprob"
        xgb = XGBClassifier(
            objective=obj,
            n_estimators=600, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE, reg_lambda=1.0, reg_alpha=0.0,
            eval_metric="mlogloss" if classes_k > 2 else "logloss",
            use_label_encoder=False,
        )
        models["xgb"] = LabelContiguityWrapper(xgb)

    if _HAVE_LGBM:
        objective = "binary" if classes_k == 2 else "multiclass"
        models["lgbm"] = LGBMClassifier(
            objective=objective,
            n_estimators=800, learning_rate=0.1,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1,
        )

    if _HAVE_CAT:
        loss = "Logloss" if classes_k == 2 else "MultiClass"
        models["cat"] = CatBoostClassifier(
            loss_function=loss,
            iterations=800, learning_rate=0.1,
            depth=6, rsm=0.9, l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=False,
        )

    return models

def param_distributions(model_name: str):
    if model_name == "logreg":
        return {
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__C": np.logspace(-3, 2, 30),
            "clf__l1_ratio": np.linspace(0.0, 1.0, 11),
            "clf__class_weight": [None, "balanced"],
        }
    if model_name == "svc_rbf":
        return {
            "clf__C": np.logspace(-3, 2, 30),
            "clf__gamma": np.logspace(-4, 0, 30),
            "clf__class_weight": [None, "balanced"],
        }
    if model_name == "rf":
        return {
            "clf__n_estimators": [300, 500, 800, 1000],
            "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__max_features": ["sqrt", "log2", 0.5, 0.75],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        }
    if model_name == "xgb":
        return {
            "clf__learning_rate": np.logspace(np.log10(0.005), np.log10(0.2), 30),
            "clf__max_depth": [3, 4, 5, 6, 8, 10],
            "clf__min_child_weight": [1, 3, 5, 7],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__reg_alpha": np.logspace(-8, -1, 8),
            "clf__reg_lambda": np.logspace(-3, 1, 9),
        }
    if model_name == "lgbm":
        return {
            "clf__learning_rate": np.logspace(np.log10(0.005), np.log10(0.2), 30),
            "clf__num_leaves": np.unique(np.linspace(15, 255, 25, dtype=int)).tolist(),
            "clf__max_depth": [-1, 3, 4, 5, 6, 8, 10, 12],
            "clf__min_child_samples": [5, 10, 20, 50],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__reg_alpha": np.logspace(-8, -1, 8),
            "clf__reg_lambda": np.logspace(-3, 1, 9),
        }
    if model_name == "cat":
        return {
            "clf__learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 20),
            "clf__depth": [3, 4, 5, 6, 8, 10],
            "clf__l2_leaf_reg": np.logspace(0, np.log10(30), 10),
            "clf__bagging_temperature": [0, 1, 5, 10],
            "clf__rsm": [0.6, 0.8, 1.0],
        }
    return {}


# -----------------------
# Tuning with lexicographic selection
# -----------------------
def tune_model(model_name, estimator, preprocessor, X_train, y_train, cv, n_iter=N_ITER_TUNE):
    """
    RandomizedSearchCV (refit=False), then manual selection by:
      maximize f1_macro -> maximize balanced_accuracy -> minimize log_loss -> minimize brier
    """
    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
    dist = param_distributions(model_name)
    if not dist:
        return clone(estimator), None, None, {"note": "no_param_space_for_model"}, {}

    scoring = {
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "log_loss": logloss_scorer_cv,  # negative
        "brier": brier_scorer_cv,       # negative
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=N_JOBS_TUNE,
        refit=False,
        random_state=TUNE_SEED,
        verbose=0,
        error_score=np.nan,
    )
    search.fit(X_train, y_train)

    res = pd.DataFrame(search.cv_results_)
    score_cols = ["mean_test_f1_macro", "mean_test_balanced_accuracy", "mean_test_log_loss", "mean_test_brier"]
    valid = res[score_cols].dropna()
    if valid.empty:
        return clone(estimator), res, None, {"note": "no_valid_cv_rows"}, {}

    f1 = np.nan_to_num(res["mean_test_f1_macro"].values, nan=-1e9)
    bacc = np.nan_to_num(res["mean_test_balanced_accuracy"].values, nan=-1e9)
    ll = np.nan_to_num(-res["mean_test_log_loss"].values, nan=1e9)       # minimize
    brier = np.nan_to_num(-res["mean_test_brier"].values, nan=1e9)      # minimize

    order = np.lexsort((brier, ll, -bacc, -f1))
    best_idx = int(order[0])

    best_params = search.cv_results_["params"][best_idx]
    clf_params = {k.replace("clf__", "", 1): v for k, v in best_params.items() if k.startswith("clf__")}
    tuned_est = clone(estimator).set_params(**clf_params)

    best_cv = {
        "f1_macro_mean": float(res["mean_test_f1_macro"].iloc[best_idx]),
        "balanced_accuracy_mean": float(res["mean_test_balanced_accuracy"].iloc[best_idx]),
        "log_loss_mean": float(ll[best_idx]),
        "brier_mean": float(brier[best_idx]),
        "tune_seed": int(TUNE_SEED),
        "n_iter": int(n_iter),
    }
    return tuned_est, res, best_idx, best_cv, best_params


# -----------------------
# Split saving
# -----------------------
def save_splits(out_splits_dir: Path,
               X_train, y_train, X_val, y_val, X_test, y_test,
               target_col: str, label_encoder: LabelEncoder):
    """
    Saves:
      - X/y in separate files (CSV)
      - combined files with target (CSV)
      - label mapping (JSON)
    """
    ensure_dir(out_splits_dir)

    def _save_xy(prefix: str, X, y):
        X.to_csv(out_splits_dir / f"{prefix}__X.csv", index=False)
        pd.Series(y, name=target_col).to_csv(out_splits_dir / f"{prefix}__y.csv", index=False)

        df_xy = X.copy()
        df_xy[target_col] = y
        df_xy.to_csv(out_splits_dir / f"{prefix}__Xy.csv", index=False)

    _save_xy("train", X_train, y_train)
    _save_xy("val",   X_val,   y_val)
    _save_xy("test",  X_test,  y_test)

    mapping = {int(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    save_json({"target": target_col, "label_mapping": mapping}, out_splits_dir / "label_mapping.json")


# -----------------------
# Per-dataset processing
# -----------------------
def evaluate_dataset(df: pd.DataFrame, ds_id: str, out_dir: Path):
    ensure_dir(out_dir)
    ds_tag = safe_stem(ds_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    splits_dir   = out_dir / f"{ds_tag}_splits"
    summaries_dir= out_dir / f"{ds_tag}_summaries"
    models_dir   = out_dir / f"{ds_tag}_best_models"
    ensure_dir(splits_dir)
    ensure_dir(summaries_dir)
    ensure_dir(models_dir)

    # Clean + target
    df = df.dropna(axis=1, how="all").copy()
    if TARGET_COL not in df.columns:
        raise ValueError(f'[{ds_tag}] Target "{TARGET_COL}" not found.')
    df = df[~df[TARGET_COL].isna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str)

    # Feature split + label encoding
    X_all, y_raw, num_cols, cat_cols = infer_feature_types(df, TARGET_COL)
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw.values)
    classes_k = len(le.classes_)

    # Save schema
    schema = {
        "dataset_id": ds_tag,
        "generated_at": timestamp,
        "n_rows": int(df.shape[0]),
        "n_features": int(X_all.shape[1]),
        "n_classes": int(classes_k),
        "classes": [str(c) for c in le.classes_],
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": TARGET_COL,
        "split": {"test_size": TEST_SIZE, "val_size": VAL_SIZE, "random_state": RANDOM_STATE},
        "tuning": {"n_iter": N_ITER_TUNE, "tune_seed": TUNE_SEED, "cv_splits_max": N_SPLITS_TRAIN_CV},
    }
    save_json(schema, out_dir / f"{ds_tag}__schema.json")

    # Train / Val / Test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    val_frac_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_of_temp, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Save splits
    save_splits(splits_dir, X_train, y_train, X_val, y_val, X_test, y_test, TARGET_COL, le)

    # CV folds on TRAIN
    min_class_count_train = int(pd.Series(y_train).value_counts().min()) if len(y_train) else 0
    n_splits_eff = max(3, min(N_SPLITS_TRAIN_CV, min_class_count_train)) if min_class_count_train >= 3 else 3
    cv_train = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_STATE)

    def preproc_for(model_name: str):
        return make_preprocessor(num_cols, cat_cols, scale_numeric=needs_scaling(model_name))

    models = base_estimators(classes_k)

    rows_train = []   # metrics on TRAIN (after final refit on TRAIN+VAL)
    rows_test  = []   # metrics on TEST  (after final refit on TRAIN+VAL)
    rows_cv    = []   # best CV means from tuning
    best_params_dump = {}  # per model

    # Refit set (TRAIN+VAL)
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = np.concatenate([y_train, y_val], axis=0)

    for name, base_est in models.items():
        preproc = preproc_for(name)

        tuned_est, cv_results_df, best_idx, best_cv, best_params_full = tune_model(
            name, base_est, preproc, X_train, y_train, cv_train, n_iter=N_ITER_TUNE
        )

        # Save tuning results (csv + json)
        model_tag = f"{ds_tag}__{name}"
        if cv_results_df is not None:
            cv_results_df.to_csv(summaries_dir / f"{model_tag}__tuning_cv_results.csv", index=False)
        save_json({"best_idx": best_idx, "best_cv": best_cv, "best_params": best_params_full},
                  summaries_dir / f"{model_tag}__best_params.json")
        best_params_dump[name] = best_params_full

        rows_cv.append({
            "model": name,
            "cv_splits": int(n_splits_eff),
            "f1_macro_mean": best_cv.get("f1_macro_mean", np.nan),
            "balanced_accuracy_mean": best_cv.get("balanced_accuracy_mean", np.nan),
            "log_loss_mean": best_cv.get("log_loss_mean", np.nan),
            "brier_mean": best_cv.get("brier_mean", np.nan),
            "n_iter": best_cv.get("n_iter", N_ITER_TUNE),
        })

        # Refit on TRAIN+VAL (best params), then evaluate TRAIN and TEST
        pipe = Pipeline([("prep", preproc), ("clf", clone(tuned_est))])
        pipe.fit(X_trval, y_trval)

        # Save best pipeline
        pkl_path = models_dir / f"{ds_tag}__{name}__best_pipeline.pkl"
        joblib.dump(pipe, pkl_path)

        # Metrics
        m_train = compute_all_metrics(pipe, X_trval, y_trval, classes_k)
        m_test  = compute_all_metrics(pipe, X_test,  y_test,  classes_k)

        rows_train.append({"model": name, **m_train})
        rows_test.append({"model": name, **m_test})

    # Save performance tables
    df_cv = pd.DataFrame(rows_cv).sort_values(
        ["f1_macro_mean", "balanced_accuracy_mean", "log_loss_mean", "brier_mean"],
        ascending=[False, False, True, True],
    )
    df_train = pd.DataFrame(rows_train).sort_values(
        ["f1_macro", "balanced_accuracy", "log_loss", "brier"],
        ascending=[False, False, True, True],
    )
    df_test = pd.DataFrame(rows_test).sort_values(
        ["f1_macro", "balanced_accuracy", "log_loss", "brier"],
        ascending=[False, False, True, True],
    )

    df_cv.to_csv(summaries_dir / f"{ds_tag}__leaderboard_trainCV_tuned.csv", index=False)
    df_train.to_csv(summaries_dir / f"{ds_tag}__final_train_metrics_table.csv", index=False)
    df_test.to_csv(summaries_dir / f"{ds_tag}__final_test_metrics_table.csv", index=False)

    # Compact manifest
    manifest = {
        "dataset_id": ds_tag,
        "generated_at": timestamp,
        "target": TARGET_COL,
        "splits": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "cv_splits_train": int(n_splits_eff),
        "folders": {
            "splits_dir": str(splits_dir.relative_to(out_dir)),
            "summaries_dir": str(summaries_dir.relative_to(out_dir)),
            "models_dir": str(models_dir.relative_to(out_dir)),
        },
        "saved_files": {
            "schema": f"{ds_tag}__schema.json",
            "cv_leaderboard": str((summaries_dir / f"{ds_tag}__leaderboard_trainCV_tuned.csv").relative_to(out_dir)),
            "train_metrics": str((summaries_dir / f"{ds_tag}__final_train_metrics_table.csv").relative_to(out_dir)),
            "test_metrics": str((summaries_dir / f"{ds_tag}__final_test_metrics_table.csv").relative_to(out_dir)),
            "pipelines_pkl": [str(p.relative_to(out_dir)) for p in sorted(models_dir.glob("*.pkl"))],
            "splits_csv": [str(p.relative_to(out_dir)) for p in sorted(splits_dir.glob("*.csv"))],
        },
    }
    save_json(manifest, out_dir / f"{ds_tag}__MANIFEST.json")


# -----------------------
# Main (batch)
# -----------------------
def main():
    ensure_dir(OUT_ROOT)
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f'Input folder "{INPUT_DIR}" not found.')

    files = [p for p in INPUT_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}]
    if not files:
        print(f'No compatible files found in "{INPUT_DIR}".')
        return

    for path in sorted(files):
        datasets = read_any_table(path)
        if not datasets:
            continue

        for ds_id, df in datasets.items():
            out_dir = OUT_ROOT / safe_stem(ds_id)
            print(f"\n=== Processing: {ds_id} -> {out_dir} ===")
            try:
                evaluate_dataset(df, ds_id, out_dir)
                print("Done.")
            except Exception as e:
                ensure_dir(out_dir)
                (out_dir / f"{safe_stem(ds_id)}__ERROR.txt").write_text(str(e), encoding="utf-8")
                print(f"ERROR on {ds_id}: {e}")

if __name__ == "__main__":
    main()
