# Feature selection pipeline (Filtering / PCA / Combined)

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA


# -----------------------------
# Data load
# -----------------------------
def load_xy(csv_path: str | Path, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    return X, y


# -----------------------------
# Processing pipeline
# -----------------------------
def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def get_feature_names(preprocessor, num_cols, cat_cols) -> np.ndarray:
    try:
        names = preprocessor.get_feature_names_out()
        names = [n.replace("num__", "").replace("cat__", "") for n in names]
        return np.array(names, dtype=object)
    except Exception:
        # fallback (less informative, but safe)
        names = num_cols.copy() + [f"{c}__ohe" for c in cat_cols]
        return np.array(names, dtype=object)


def safe_corr_prune(X_dense: np.ndarray, feat_names: np.ndarray, corr_threshold: float):
    """Drop one of each highly-correlated pair (|r| > threshold), using upper triangle pruning."""
    n = X_dense.shape[1]
    if n <= 1:
        keep_mask = np.ones(n, dtype=bool)
        return keep_mask, feat_names

    Xs = X_dense - X_dense.mean(axis=0, keepdims=True)
    std = Xs.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xs = Xs / std

    corr = np.corrcoef(Xs, rowvar=False)
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

    to_drop = set()
    for i, j in zip(*np.where((np.abs(corr) > corr_threshold) & upper)):
        to_drop.add(j)

    keep_mask = np.ones(n, dtype=bool)
    if to_drop:
        keep_mask[list(to_drop)] = False

    return keep_mask, feat_names[keep_mask]


def choose_mi_k(n_features: int, mi_top_frac: float, mi_top_cap: int) -> int:
    k = max(10, int(mi_top_frac * n_features))
    k = min(k, mi_top_cap, n_features)
    return max(1, k)


def pca_components(n_features: int, n_samples: int, user_k: int | None):
    hard_cap = max(1, min(n_features, max(1, n_samples - 1)))
    if user_k is None:
        return min(20, hard_cap)
    return min(user_k, hard_cap)


def to_dataframe(X_trans: np.ndarray, cols: list[str], y: pd.Series, target_col: str) -> pd.DataFrame:
    df_out = pd.DataFrame(X_trans, columns=cols, index=y.index)
    df_out[target_col] = y.values
    return df_out[[c for c in df_out.columns if c != target_col] + [target_col]]


def run_filtering(X: pd.DataFrame, y: pd.Series, target_col: str,
                  var_threshold: float, corr_threshold: float, mi_top_frac: float, mi_top_cap: int):
    pre, num_cols, cat_cols = build_preprocessor(X)
    X_pre = pre.fit_transform(X, y)
    feat_names = get_feature_names(pre, num_cols, cat_cols)

    # 1) low variance
    vt = VarianceThreshold(threshold=var_threshold)
    X_vt = vt.fit_transform(X_pre)
    names_vt = feat_names[vt.get_support()]

    # 2) correlation pruning
    keep_mask_corr, names_corr = safe_corr_prune(X_vt, names_vt, corr_threshold=corr_threshold)
    X_corr = X_vt[:, keep_mask_corr]

    # 3) mutual information (supervised)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    k = choose_mi_k(X_corr.shape[1], mi_top_frac, mi_top_cap)

    sel = SelectKBest(score_func=mutual_info_classif, k=k)
    X_sel = sel.fit_transform(X_corr, y_enc)
    names_sel = names_corr[sel.get_support()]

    df_out = to_dataframe(X_sel, list(names_sel), y, target_col)
    meta = {
        "method": "Filtering",
        "var_threshold": var_threshold,
        "corr_threshold": corr_threshold,
        "mi_top_frac": mi_top_frac,
        "mi_top_cap": mi_top_cap,
        "selected_feature_count": int(X_sel.shape[1]),
        "selected_features": names_sel.tolist(),
        "classes_": le.classes_.tolist(),
    }
    return df_out, meta


def run_pca(X: pd.DataFrame, y: pd.Series, target_col: str, pca_k: int | None):
    pre, num_cols, cat_cols = build_preprocessor(X)
    X_pre = pre.fit_transform(X, y)

    k_eff = pca_components(n_features=X_pre.shape[1], n_samples=X_pre.shape[0], user_k=pca_k)
    pca = PCA(n_components=k_eff, svd_solver="auto")
    X_pca = pca.fit_transform(X_pre)

    cols = [f"pca_comp_{i+1}" for i in range(X_pca.shape[1])]
    df_out = to_dataframe(X_pca, cols, y, target_col)

    meta = {
        "method": "DimensionalReduction",
        "algorithm": "PCA",
        "requested_components": pca_k,
        "effective_components": int(X_pca.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return df_out, meta


def run_combined(X: pd.DataFrame, y: pd.Series, target_col: str,
                 var_threshold: float, corr_threshold: float, mi_top_frac: float, mi_top_cap: int,
                 pca_k: int | None):
    df_filt, meta_filt = run_filtering(
        X, y, target_col,
        var_threshold=var_threshold,
        corr_threshold=corr_threshold,
        mi_top_frac=mi_top_frac,
        mi_top_cap=mi_top_cap,
    )

    Xf = df_filt.drop(columns=[target_col]).values
    k_eff = pca_components(n_features=Xf.shape[1], n_samples=Xf.shape[0], user_k=pca_k)

    pca = PCA(n_components=k_eff, svd_solver="auto")
    X_pca = pca.fit_transform(Xf)

    cols = [f"pca_comp_{i+1}" for i in range(X_pca.shape[1])]
    df_out = to_dataframe(X_pca, cols, df_filt[target_col], target_col)

    meta = {
        "method": "Combined",
        "algorithm": "Filtering+PCA",
        "filtering_meta": meta_filt,
        "requested_components": pca_k,
        "effective_components": int(X_pca.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return df_out, meta


# -----------------------------
# Save generated new dataframe(s)
# -----------------------------
def run_and_save(
    csv_path: str | Path,
    output_dir: str | Path,
    target_col: str = "target",
    var_threshold: float = 1e-5,
    corr_threshold: float = 0.95,
    mi_top_frac: float = 0.30,
    mi_top_cap: int = 300,
    pca_components_user: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(csv_path, target_col=target_col)
    stem = Path(csv_path).stem

    # Filtering
    df_filt, meta_filt = run_filtering(X, y, target_col, var_threshold, corr_threshold, mi_top_frac, mi_top_cap)
    df_filt.to_csv(output_dir / f"{stem}_Filter_features.csv", index=False)
    (output_dir / f"{stem}_Filtering_selected_features.meta.json").write_text(json.dumps(meta_filt, indent=2))

    # PCA
    df_pca, meta_pca = run_pca(X, y, target_col, pca_components_user)
    df_pca.to_csv(output_dir / f"{stem}_DimenReduct_features.csv", index=False)
    (output_dir / f"{stem}_DimensionalReduction_selected_features.meta.json").write_text(json.dumps(meta_pca, indent=2))

    # Combined
    df_comb, meta_comb = run_combined(
        X, y, target_col, var_threshold, corr_threshold, mi_top_frac, mi_top_cap, pca_components_user
    )
    df_comb.to_csv(output_dir / f"{stem}_Combined_features.csv", index=False)
    (output_dir / f"{stem}_Combined_selected_features.meta.json").write_text(json.dumps(meta_comb, indent=2))

