from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

#---------------------------
# Feature sets from schema_df
# ---------------------------

def plot_distribution(df: pd.DataFrame, feature: str, semantic_type: str, target: str, split: bool):
    s = df[feature]

    # --- Numeric: histogram ---
    if semantic_type == "number":
        fig, ax = plt.subplots(figsize=(10, 4.8))

        # Drop NaNs for plotting
        if not split:
            vals = s.dropna().astype(float)
            ax.hist(vals, bins=20, edgecolor="black")
            ax.set_title(f"Distribution: {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.grid(False)
            st.pyplot(fig, clear_figure=True)
            return

        # Split by target
        # Ensure target exists and has (at least) two groups
        if target not in df.columns:
            st.warning(f"Target-column '{target}' not found.")
            return

        # Sort labels for stable legend ordering (e.g., 0 then 1)
        groups = sorted(df[target].dropna().unique().tolist())

        # Use same bins for comparability
        all_vals = s.dropna().astype(float)
        bins = 20
        hist_range = (float(all_vals.min()), float(all_vals.max())) if len(all_vals) else None

        for g in groups:
            vals_g = df.loc[df[target] == g, feature].dropna().astype(float)
            ax.hist(vals_g, bins=bins, range=hist_range, alpha=0.45, label=f"{target} {g}", edgecolor="black")

        ax.set_title(f"Distribution: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(False)
        st.pyplot(fig, clear_figure=True)
        return

    # --- Categorical/Bool: bar chart ---
    if semantic_type in {"bool", "category", "category_str"}:
        # treat as string labels for stable plotting
        s_cat = s.astype("object").where(~s.isna(), other="(missing)").astype(str)

        if not split:
            counts = s_cat.value_counts(dropna=False).sort_index()
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(counts.index, counts.values, edgecolor="black")
            ax.set_title(f"Distribution: {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            st.pyplot(fig, clear_figure=True)
            return

        if target not in df.columns:
            st.warning(f"Target-column '{target}' not found.")
            return

        # Grouped counts by category and target
        tmp = pd.DataFrame({feature: s_cat, target: df[target].astype("object")})
        ct = pd.crosstab(tmp[feature], tmp[target])

        # Optional: limit too many categories
        MAX_CATS = 25
        if len(ct.index) > MAX_CATS:
            top_idx = s_cat.value_counts().head(MAX_CATS).index
            ct = ct.loc[ct.index.isin(top_idx)]
            ct = ct.sort_index()

        fig, ax = plt.subplots(figsize=(10, 4.8))

        x = np.arange(len(ct.index))
        cols = sorted(ct.columns.tolist())
        width = 0.8 / max(1, len(cols))

        for i, col in enumerate(cols):
            ax.bar(x + i * width, ct[col].values, width=width, label=f"{target} {col}", edgecolor="black")

        ax.set_xticks(x + (len(cols)-1) * width / 2)
        ax.set_xticklabels(ct.index, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Distribution: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        return

    # --- Fallback ---
    st.info(f"no visualization for semantic_type='{semantic_type}' implemented.")


def get_feature_sets(schema_df: pd.DataFrame, target_col: str) -> dict:
    """Builds robust feature lists based on your schema_df."""
    constants = schema_df.loc[schema_df["constant"] == True, "columnname"].tolist()

    bool_cols = schema_df.loc[schema_df["semantic_type"] == "bool", "columnname"].tolist()
    num_cols = schema_df.loc[schema_df["semantic_type"] == "number", "columnname"].tolist()
    cat_cols = schema_df.loc[schema_df["semantic_type"].isin({"category", "category_str"}), "columnname"].tolist()

    for lst in (bool_cols, num_cols, cat_cols):
        if target_col in lst:
            lst.remove(target_col)

    bool_cols = [c for c in bool_cols if c not in constants]
    num_cols  = [c for c in num_cols  if c not in constants]
    cat_cols  = [c for c in cat_cols  if c not in constants]

    feature_cols = [
        c for c in schema_df["columnname"].tolist()
        if c != target_col and c not in constants
    ]

    return {
        "feature_cols": feature_cols,
        "bool_cols": bool_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "constants": constants,
    }


# ---------------------------
# Transformers (sklearn-compatible feature names)
# ---------------------------

class BoolToIntTransformer(BaseEstimator, TransformerMixin):
    """Convert boolean columns to 0/1 and expose feature names for sklearn."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X can be DataFrame or ndarray depending on sklearn version
        if isinstance(X, pd.DataFrame):
            out = X.copy()
            out = out.fillna(False).astype(bool).astype(np.int64)
            return out.to_numpy()
        else:
            X = np.asarray(X)
            # NaN -> 0, everything else -> bool->int
            X = np.nan_to_num(X, nan=0.0)
            return X.astype(bool).astype(np.int64)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.array(input_features, dtype=object)


def _make_onehot_encoder():
    """
    sklearn compatibility:
    - newer: OneHotEncoder(..., sparse_output=False)
    - older: OneHotEncoder(..., sparse=False)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    *,
    bool_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:

    bool_tf = BoolToIntTransformer()

    if scale_numeric and len(num_cols) > 0:
        num_tf = Pipeline(steps=[("scaler", StandardScaler())])
    else:
        num_tf = "passthrough"

    cat_tf = _make_onehot_encoder()

    return ColumnTransformer(
        transformers=[
            ("bool", bool_tf, bool_cols),
            ("num",  num_tf,  num_cols),
            ("cat",  cat_tf,  cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ---------------------------
# Model registry
# ---------------------------

def build_model_pipelines(
    *,
    bool_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    random_state: int = 42,
) -> Dict[str, Pipeline]:

    specs = {
        "Nearest Centroid": (NearestCentroid(), True),
        "KNN (k=5)":        (KNeighborsClassifier(n_neighbors=5), True),
        "Decision Tree":    (DecisionTreeClassifier(max_depth=5, random_state=random_state), False),
        "Logistic Reg.":    (LogisticRegression(max_iter=3000, random_state=random_state), True),
    }

    pipelines: Dict[str, Pipeline] = {}
    for name, (clf, scale_numeric) in specs.items():
        pre = make_preprocessor(
            bool_cols=bool_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            scale_numeric=scale_numeric,
        )
        pipelines[name] = Pipeline(steps=[
            ("preprocess", pre),
            ("model", clf),
        ])
    return pipelines


# ---------------------------
# Training / Evaluation
# ---------------------------

@dataclass
class TrainResult:
    results_df: pd.DataFrame
    fitted_models: Dict[str, Pipeline]
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_sets: dict


def train_and_evaluate(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    target_col: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:

    fs = get_feature_sets(schema_df, target_col)

    X = df[fs["feature_cols"]].copy()

    y_raw = df[target_col]
    if y_raw.dtype == bool:
        y = y_raw.astype(np.int64)
    else:
        y = y_raw.astype(str).str.lower().map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
        if y.isna().any():
            y = pd.to_numeric(y_raw, errors="coerce")
        y = y.fillna(0).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model_pipes = build_model_pipelines(
        bool_cols=fs["bool_cols"],
        num_cols=fs["num_cols"],
        cat_cols=fs["cat_cols"],
        random_state=random_state,
    )

    rows = []
    fitted = {}

    for name, pipe in model_pipes.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        y_pred = pipe.predict(X_test)

        roc = np.nan
        model_step = pipe.named_steps["model"]
        if hasattr(model_step, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) == 2:
                roc = roc_auc_score(y_test, proba)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, pos_label=1),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "ROC_AUC": roc,
        })

    results_df = (
        pd.DataFrame(rows)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    return TrainResult(
        results_df=results_df,
        fitted_models=fitted,
        X_test=X_test,
        y_test=y_test,
        feature_sets=fs,
    )


def evaluate_single_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    out = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": cm,
    }

    model_step = model.named_steps["model"]
    if hasattr(model_step, "predict_proba") and len(np.unique(y_test)) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        out["roc_auc"] = roc_auc_score(y_test, proba)

    return out


# ---------------------------
# Feature names (safe for older sklearn)
# ---------------------------

def _cols_to_names(ct: ColumnTransformer, cols):
    """Convert ColumnTransformer column selector to list of column names."""
    all_names = list(getattr(ct, "feature_names_in_", []))
    if isinstance(cols, slice):
        return all_names[cols]
    if isinstance(cols, (list, tuple)):
        return list(cols)
    if isinstance(cols, np.ndarray) and cols.dtype == bool:
        return list(np.array(all_names)[cols])
    return list(cols) if cols is not None else []


def _get_names_for_transformer(trans, input_features: List[str]):
    input_features = np.array(input_features, dtype=object)

    if trans == "passthrough":
        return input_features
    if trans == "drop":
        return np.array([], dtype=object)

    if isinstance(trans, Pipeline):
        # take the LAST step that can output feature names
        for _, step in reversed(trans.steps):
            if hasattr(step, "get_feature_names_out"):
                return step.get_feature_names_out(input_features)
        return input_features

    if hasattr(trans, "get_feature_names_out"):
        try:
            return trans.get_feature_names_out(input_features)
        except TypeError:
            return trans.get_feature_names_out()

    return input_features


def safe_get_feature_names_out(ct: ColumnTransformer) -> np.ndarray:
    """Robust feature names for ColumnTransformer across sklearn versions."""
    names = []
    verbose = bool(getattr(ct, "verbose_feature_names_out", False))

    for name, trans, cols in ct.transformers_:
        if trans == "drop":
            continue
        in_names = _cols_to_names(ct, cols)
        out_names = _get_names_for_transformer(trans, in_names)
        out_names = np.array(out_names, dtype=object)

        if verbose and len(out_names) > 0:
            out_names = np.array([f"{name}__{n}" for n in out_names], dtype=object)

        names.append(out_names)

    if not names:
        return np.array([], dtype=object)
    return np.concatenate(names)


def get_feature_names(model: Pipeline) -> np.ndarray:
    pre = model.named_steps["preprocess"]
    try:
        return pre.get_feature_names_out()
    except Exception:
        # fallback for older / partially incompatible sklearn setups
        return safe_get_feature_names_out(pre)


# ---------------------------
# Feature relevance helpers
# ---------------------------

def base_feature_name(encoded_feature: str) -> str:
    """
    Encoded names look like:
      - cat__PaymentMethod_Electronic check
      - num__MonthlyCharges
      - bool__Partner
    Map back to base column: PaymentMethod, MonthlyCharges, Partner
    """
    if "__" in encoded_feature:
        _, rest = encoded_feature.split("__", 1)
    else:
        rest = encoded_feature

    if "_" in rest and not rest.startswith(("num", "bool")):
        return rest.split("_", 1)[0]
    return rest


def get_logreg_feature_table(model: Pipeline, top_n: int = 25) -> pd.DataFrame:
    feats = get_feature_names(model)
    coef = model.named_steps["model"].coef_.ravel()
    dfc = pd.DataFrame({"feature": feats, "coef": coef})
    dfc["abs"] = dfc["coef"].abs()
    return dfc.sort_values("abs", ascending=False).head(top_n).reset_index(drop=True)


def get_tree_feature_table(model: Pipeline, top_n: int = 25) -> pd.DataFrame:
    feats = get_feature_names(model)
    imp = model.named_steps["model"].feature_importances_
    dfi = pd.DataFrame({"feature": feats, "importance": imp})
    dfi["abs"] = dfi["importance"].abs()
    return dfi.sort_values("abs", ascending=False).head(top_n).reset_index(drop=True)


def aggregate_to_base_feature(df_feat: pd.DataFrame, score_col: str) -> pd.DataFrame:
    tmp = df_feat.copy()
    tmp["base_feature"] = tmp["feature"].map(base_feature_name)
    return (
        tmp.groupby("base_feature", as_index=False)[score_col]
        .sum()
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------
# Prediction UX helpers
# ---------------------------

def make_defaults(df: pd.DataFrame, feature_cols: List[str]) -> dict:
    defaults = {}
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            defaults[c] = float(df[c].median())
        else:
            mode = df[c].mode(dropna=True)
            defaults[c] = mode.iloc[0] if len(mode) else None
    return defaults


def pick_top_input_columns_for_model(
    model: Pipeline,
    feature_sets: dict,
    *,
    top_k: int = 10
) -> List[str]:
    """
    Returns base/original feature columns (not onehot columns) most relevant,
    if the model supports native importance. Otherwise fallback.
    """
    fs = feature_sets
    feature_cols = fs["feature_cols"]

    model_step = model.named_steps["model"]
    name = model_step.__class__.__name__

    if name == "LogisticRegression":
        dfc = get_logreg_feature_table(model, top_n=60)
        ranked = dfc["feature"].tolist()
    elif name == "DecisionTreeClassifier":
        dfi = get_tree_feature_table(model, top_n=60)
        ranked = dfi["feature"].tolist()
    else:
        return feature_cols[:top_k]

    seen = set()
    out = []
    for enc in ranked:
        base = base_feature_name(str(enc))
        if base in feature_cols and base not in seen:
            out.append(base)
            seen.add(base)
        if len(out) >= top_k:
            break

    return out if out else feature_cols[:top_k]


# ----------------------------------
# Helpers for Visualization:
#----------------------------------

DISCRETE_SEMS = {"bool", "num_bool", "category", "category_str"}


def get_semantic_type(
    schema_df: pd.DataFrame,
    col: str,
    col_field: str = "columnname",
    sem_field: str = "semantic_type",
) -> str:
    """Return semantic type for a column from your schema dataframe."""
    return schema_df.loc[schema_df[col_field] == col, sem_field].iloc[0]


def as_cat_str(s: pd.Series) -> pd.Series:
    """
    Stringify + normalize bool-ish values + fill missing.
    Produces labels like True/False and keeps other categories as-is.
    """
    out = s.astype("string").fillna("Missing").str.strip()
    low = out.str.lower()

    out = np.where(low.isin(["true", "1", "yes", "y"]), "True",
          np.where(low.isin(["false", "0", "no", "n"]), "False", out))

    return pd.Series(out, index=s.index, name=s.name)


def target_to_int(t: pd.Series) -> pd.Series:
    """
    Best-effort convert target to {0,1} for churn-rate computations.
    Returns a Series with NaNs where mapping fails.
    """
    if pd.api.types.is_bool_dtype(t):
        return t.astype(int)

    if pd.api.types.is_numeric_dtype(t):
        tn = pd.to_numeric(t, errors="coerce")
        vals = set(tn.dropna().unique())
        if vals.issubset({0, 1}):
            return tn.astype(int)
        if len(vals) <= 3 and len(vals) > 0 and min(vals) >= 0:
            return (tn > 0).astype(int)

    ts = t.astype("string").fillna("").str.strip().str.lower()
    mapping = {
        "true": 1, "1": 1, "yes": 1, "y": 1,
        "false": 0, "0": 0, "no": 0, "n": 0
    }
    return ts.map(mapping).astype("float")


def _ordered_categories(s_str: pd.Series) -> list[str]:
    """Stable-ish ordering: bool => False/True, else keep categorical order if available, else sorted."""
    # If original was categorical and passed through as strings, one can't recover category order.
    uniq = list(pd.unique(s_str))
    uniq_set = set(uniq)

    if uniq_set.issubset({"False", "True"}) and len(uniq_set) > 0:
        return [c for c in ["False", "True"] if c in uniq_set]

    # put Missing last, rest sorted
    return sorted(uniq, key=lambda x: (x == "Missing", str(x)))


def _deterministic_noise(index: pd.Index, jitter: float, salt: int = 0) -> np.ndarray:
    """
    Deterministic "random" noise in [-jitter, +jitter] based on the index.
    Keeps jitter stable across Streamlit reruns.
    """
    idx = pd.Index(index)
    h = pd.util.hash_pandas_object(idx, index=False).astype("uint64").values
    # mix salt
    h = h ^ np.uint64(0x9E3779B97F4A7C15 + salt)  # golden ratio constant-ish
    # to [0,1)
    u = (h / np.float64(2**64))
    return (u * 2.0 - 1.0) * jitter


def jitter_codes(s_str: pd.Series, jitter: float = 0.28, salt: int = 0):
    """
    Map categories to integer positions and add deterministic jitter.
    Returns (jittered_codes, categories_in_order).
    """
    cats = _ordered_categories(s_str)
    cat_to_code = {c: i for i, c in enumerate(cats)}

    base = s_str.map(cat_to_code).astype(float)
    noise = _deterministic_noise(s_str.index, jitter=jitter, salt=salt)

    return (base.values + noise), cats

#------------------- for Management Summary


def tenure_category_slider(df, *, col="tenure_category", key="tenure_cat"):
    if col not in df.columns:
        return None, df

    # stabile Sortierung (wenn ihr schon eine order-helper habt -> nutzen!)
    cats = sorted(df[col].dropna().unique().tolist())

    selected = st.select_slider(
        "tenure_category auswählen",
        options=cats,
        value=cats[0] if cats else None,
        key=key
    )
    if selected is None:
        return None, df

    return selected, df[df[col] == selected]


def churn_share_monthlycharges_bins(df, *, bins=None, churn_col="Churn"):
    if bins is None:
        bins = np.arange(0, 141, 10)

    d = df.copy()

    # Einheitliche Labels + gewünschte Reihenfolge
    # (Churn unten, No Churn oben)
    ORDER = ["Churn", "No Churn"]
    COLORS = {"Churn": "#ff7f0e", "No Churn": "#1f77b4"}  # orange / blau

    if churn_col in d.columns:
        # bool -> Label
        if d[churn_col].dtype == bool:
            d[churn_col] = d[churn_col].map({True: "Churn", False: "No Churn"})
        else:
            # falls schon Strings/Zahlen drin sind, robust normalisieren
            d[churn_col] = d[churn_col].astype(str).str.strip()
            d[churn_col] = d[churn_col].replace({
                "True": "Churn", "1": "Churn", "Yes": "Churn",
                "False": "No Churn", "0": "No Churn", "No": "No Churn"
            })

    d["mc_bin"] = pd.cut(d["MonthlyCharges"], bins=bins, include_lowest=True)

    g = (
        d.groupby(["mc_bin", churn_col], dropna=False)
         .size()
         .reset_index(name="count")
    )

    # normalize within each bin
    g["share"] = g["count"] / g.groupby("mc_bin")["count"].transform("sum")
    g["mc_bin_label"] = g["mc_bin"].astype(str)

    fig = px.bar(
        g,
        x="mc_bin_label",
        y="share",
        color=churn_col,
        barmode="stack",
        title="Relativer Churn-Anteil pro MonthlyCharges-Bin",
        category_orders={churn_col: ORDER},          # <-- Trace-Reihenfolge fixieren
        color_discrete_map=COLORS,                   # <-- Farben fixieren
    )

    fig.update_yaxes(range=[0, 1], title="Relativer Anteil im Bin")
    fig.update_xaxes(title="MonthlyCharges-Bins", tickangle=-35)
    fig.update_layout(legend_title_text="Churn", legend=dict(traceorder="normal"))

    return fig
