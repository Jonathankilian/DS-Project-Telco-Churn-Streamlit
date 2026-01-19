from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import numbers

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PAGES_DIR = PROJECT_ROOT / "pages"

# Data
CSV_FILE = DATA_DIR / "Telco_Churn_cleaned_data.csv"
TARGET_COLUMN = "Churn"

"""
# NICHT MEHR IN VERWENDUNG
@st.cache_data
def get_config():
    
    '''Nutzt das bereits geladene (gecachedte) DataFrame für die Config.'''
    # Ruft load_data() auf. Dank Cache wird die CSV NICHT erneut gelesen.
    df = load_data()

    config = {
        'bool_cols': df.select_dtypes(include=['bool']).columns.tolist(),
        'cat_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'num_cols': df.select_dtypes(include=['number']).columns.tolist(),
        'all_cols': df.columns.tolist()
    }
    return config
    """
@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)

@st.cache_data
def get_schema_df():
    """
    Possible Development:
    -> Statistical Values like mean, std, quantile
    """
    df = load_data()
    n_rows = len(df)

    # tuneables (small + memory friendly)
    PREVIEW_SAMPLE_N = 500     # rows to look at for preview/type inference
    PREVIEW_UNIQUE_K = 20      # values shown in schema_df
    INFER_SAMPLE_N = 300       # strings to test for datetime/number/text/id heuristics

    def infer_object_semantic(*, n_unique, cardinality_ratio, unique_preview, s_str, n_rows) -> str:
        """
        Uses only already-computed / bounded information:
        - unique_preview: <= PREVIEW_UNIQUE_K values
        - n_unique, cardinality_ratio: scalars
        - s_str: string-only view (non-strings -> NaN) already computed for missing counts
        Avoids full-column unique materialization.
        """
        if not unique_preview:
            return "object"  # all-null handled separately via all_null flag

        # types from preview (cheap)
        py_types = {type(x) for x in unique_preview if x is not None}

        if len(py_types) > 1:
            if py_types.issubset({str, int, float, np.int64, np.float64}):
                return "mixed_num_str"
            return "mixed"

        t = next(iter(py_types)) if py_types else object

        # object columns with non-string python types
        if t in {pd.Timestamp, np.datetime64}:
            return "datetime"
        if t is bool:
            return "bool"
        if issubclass(t, numbers.Number):
            # if t in {int, float, np.int64, np.float64}:
            return "number"
        if t is bytes:
            return "bytes"

        # string-ish object columns
        if t is str:
            s_str_nn = s_str.dropna()
            if s_str_nn.empty:
                return "string"

            # bounded sample for expensive inference
            sample_str = s_str_nn.head(INFER_SAMPLE_N).astype(str)

            # datetime-as-string?
            dt_ratio = float(pd.to_datetime(sample_str, errors="coerce").notna().mean())
            if dt_ratio >= 0.9:
                return "datetime_str"

            # numeric-as-string?
            num_ratio = float(pd.to_numeric(sample_str, errors="coerce").notna().mean())
            if num_ratio >= 0.9:
                return "string_number"

            # cheap string features
            avg_len = float(sample_str.str.len().mean()) if len(sample_str) else 0.0
            space_ratio = float(sample_str.str.contains(r"\s", regex=True).mean()) if len(sample_str) else 0.0

            # category vs id vs text (re-using n_unique/cardinality_ratio)
            if (n_unique <= 30) or (not np.isnan(cardinality_ratio) and cardinality_ratio <= 0.05):
                return "category_str"
            if (not np.isnan(cardinality_ratio) and cardinality_ratio >= 0.9) and avg_len <= 32 and space_ratio <= 0.05:
                return "id_str"
            if avg_len >= 40 or space_ratio >= 0.4:
                return "text"
            return "string"

        return "object_other"

    rows = []

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)

        n_isna = int(s.isna().sum())

        s_obj = s.astype("object")
        is_str = s_obj.apply(lambda x: isinstance(x, str))
        s_str = s_obj.where(is_str)  # non-strings become NaN (safe for .str ops)

        empty_string_count = int((s_str == "").sum())
        stripped = s_str.str.strip()
        whitespace_only_count = int(((stripped == "") & (s_str != "")).sum())

        semantic_missing_count = n_isna + empty_string_count + whitespace_only_count
        missing_pct = (semantic_missing_count / n_rows) * 100 if n_rows else 0.0

        # --- Scalars (compute once; used for rows + semantic inference) ---
        n_unique = int(s.nunique(dropna=True))
        cardinality_ratio = (n_unique / n_rows) if n_rows else np.nan
        all_null = bool(s.isna().all())
        constant = (n_unique == 1)

        # --- Preview uniques: bounded sample to avoid full unique materialization ---
        # deterministically take first PREVIEW_SAMPLE_N non-null values, then unique them
        sample_non_null = s.dropna().head(PREVIEW_SAMPLE_N)
        unique_values_preview = pd.unique(sample_non_null)[:PREVIEW_UNIQUE_K].tolist()

        # --- semantic_type
        if pd.api.types.is_bool_dtype(s):
            semantic = "bool"

        elif pd.api.types.is_numeric_dtype(s):
            # detect 0/1 numeric booleans without materializing full unique set
            vals = set(pd.unique(sample_non_null).tolist()) if len(sample_non_null) else set()
            semantic = "num_bool" if (len(vals) > 0 and vals.issubset({0, 1})) else "number"

        elif isinstance(s.dtype, pd.CategoricalDtype):
            semantic = "category"

        else:
            semantic = infer_object_semantic(
                n_unique=n_unique,
                cardinality_ratio=cardinality_ratio,
                unique_preview=unique_values_preview,
                s_str=s_str,
                n_rows=n_rows,
            )

        rows.append({
            "columnname": col,
            "dtype": dtype_str,
            "semantic_type": semantic,
            "n_unique": n_unique,
            "cardinality_ratio": cardinality_ratio,
            "unique_values": unique_values_preview,
            "n_isna": n_isna,
            "empty_string_count": empty_string_count,
            "whitespace_only_count": whitespace_only_count,
            "missing_pct": missing_pct,
            "all_null": all_null,
            "constant": constant,
            # ... statistics?
        })

    schema_df = pd.DataFrame(rows)

    # --- Arrow-safe: Listen/Mixed Types in Strings umwandeln ---
    def _stringify_list_cell(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return ", ".join(map(str, x))
        return "" if x is None else str(x)

    schema_df["unique_values"] = schema_df["unique_values"].apply(_stringify_list_cell)

    return schema_df

