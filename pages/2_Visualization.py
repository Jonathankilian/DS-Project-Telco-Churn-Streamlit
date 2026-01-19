import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from config import CSV_FILE, TARGET_COLUMN, load_data, get_schema_df
from utils.helpers import (
    DISCRETE_SEMS,
    as_cat_str,
    target_to_int,
    jitter_codes,
    get_semantic_type,
    tenure_category_slider,
    churn_share_monthlycharges_bins,
    plot_distribution
)


sys.path.insert(0, str(Path(__file__).parent.parent))


st.set_page_config(page_title="Visualization", page_icon="📊", layout="wide")

st.title("📊 Visualization")
st.subheader("Explore the features using interactive visualizations!")

df = load_data()
schema_df = get_schema_df()

BOOLEAN_COLUMNS = schema_df.loc[schema_df["semantic_type"] == "bool", "columnname"].tolist()

CATEGORICAL_TYPES = {"category", "category_str"}
CATEGORICAL_COLUMNS = schema_df.loc[schema_df["semantic_type"].isin(CATEGORICAL_TYPES), "columnname"].tolist()

NUMERIC_COLUMNS = schema_df.loc[schema_df["semantic_type"] == "number", "columnname"].tolist()

# --- colors for target classes (extend)
TARGET_COLORS = {
    "False": "#1f77b4",
    "True": "#ff7f0e",
    "0": "#1f77b4",
    "1": "#ff7f0e",
    "No": "#1f77b4",
    "Yes": "#ff7f0e",
}

CHURN_RATE_COLORSCALE = [
    [0.0, "green"],
    [0.5, "yellow"],
    [1.0, "red"],
]

# as_cat_str -> liefert i.d.R. "True"/"False"
df["_Churn_str"] = as_cat_str(df["Churn"])

# Optional: Legend order stabilisieren
CHURN_ORDER = {"_Churn_str": ["False", "True"], "Churn": ["False", "True"]}

JITTER = 0.28
MARKER_SIZE = 6
OPACITY = 0.65


tab0, tab1, tab2, tab3= st.tabs(["⚡ Management Summary", "📊 Distributions", "🔗 Correlations", "↔️ Comparisons"])

with tab0:
    st.header("Management Summary")

    # -----------------------------
    # Expander: vs Monthly Charges
    # -----------------------------
    with st.expander("vs Monthly Charges", expanded=True):

        # 1) PaymentMethod (x cat) vs MonthlyCharges
        s_x = as_cat_str(df["PaymentMethod"])
        xj, xcats = jitter_codes(s_x, jitter=JITTER, salt=11)

        d1 = df.copy()
        d1["_x"] = xj
        fig1 = px.scatter(
            d1,
            x="_x",
            y="MonthlyCharges",
            color="_Churn_str",
            color_discrete_map=TARGET_COLORS,
            category_orders=CHURN_ORDER,
            title="MonthlyCharges vs PaymentMethod (jittered)",
            hover_data={"PaymentMethod": True, "MonthlyCharges": True, "_x": False},
        )
        fig1.update_traces(marker=dict(size=MARKER_SIZE, opacity=OPACITY))
        fig1.update_layout(template="plotly_dark", height=450, margin=dict(l=40, r=20, t=60, b=40))
        fig1.update_xaxes(
            title="PaymentMethod",
            tickmode="array",
            tickvals=list(range(len(xcats))),
            ticktext=xcats,
        )
        fig1.update_yaxes(title="MonthlyCharges")
        st.plotly_chart(fig1, use_container_width=True)

        # 2) tenure_category (x cat) vs MonthlyCharges
        s_x = as_cat_str(df["tenure_category"])
        xj, xcats = jitter_codes(s_x, jitter=JITTER, salt=12)

        d2 = df.copy()
        d2["_x"] = xj
        fig2 = px.scatter(
            d2,
            x="_x",
            y="MonthlyCharges",
            color="_Churn_str",
            color_discrete_map=TARGET_COLORS,
            category_orders=CHURN_ORDER,
            title="MonthlyCharges vs tenure_category (jittered)",
            hover_data={"tenure_category": True, "MonthlyCharges": True, "_x": False},
        )
        fig2.update_traces(marker=dict(size=MARKER_SIZE, opacity=OPACITY))
        fig2.update_layout(template="plotly_dark", height=450, margin=dict(l=40, r=20, t=60, b=40))
        fig2.update_xaxes(
            title="tenure_category",
            tickmode="array",
            tickvals=list(range(len(xcats))),
            ticktext=xcats,
        )
        fig2.update_yaxes(title="MonthlyCharges")
        st.plotly_chart(fig2, use_container_width=True)

        # 3) Contract (x cat) vs MonthlyCharges
        s_x = as_cat_str(df["Contract"])
        xj, xcats = jitter_codes(s_x, jitter=JITTER, salt=13)

        d3 = df.copy()
        d3["_x"] = xj
        fig3 = px.scatter(
            d3,
            x="_x",
            y="MonthlyCharges",
            color="_Churn_str",
            color_discrete_map=TARGET_COLORS,
            category_orders=CHURN_ORDER,
            title="MonthlyCharges vs Contract (jittered)",
            hover_data={"Contract": True, "MonthlyCharges": True, "_x": False},
        )
        fig3.update_traces(marker=dict(size=MARKER_SIZE, opacity=OPACITY))
        fig3.update_layout(template="plotly_dark", height=450, margin=dict(l=40, r=20, t=60, b=40))
        fig3.update_xaxes(
            title="Contract",
            tickmode="array",
            tickvals=list(range(len(xcats))),
            ticktext=xcats,
        )
        fig3.update_yaxes(title="MonthlyCharges")
        st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # Expander: vs tenure
    # -----------------------------
    with st.expander("vs tenure", expanded=True):

        # 4) tenure (x num) vs PaymentMethod (y cat)
        s_y = as_cat_str(df["PaymentMethod"])
        yj, ycats = jitter_codes(s_y, jitter=JITTER, salt=21)

        d4 = df.copy()
        d4["_y"] = yj
        fig4 = px.scatter(
            d4,
            x="tenure",
            y="_y",
            color="_Churn_str",
            color_discrete_map=TARGET_COLORS,
            category_orders=CHURN_ORDER,
            title="PaymentMethod vs tenure (jittered)",
            hover_data={"PaymentMethod": True, "tenure": True, "_y": False},
        )
        fig4.update_traces(marker=dict(size=MARKER_SIZE, opacity=OPACITY))
        fig4.update_layout(template="plotly_dark", height=450, margin=dict(l=40, r=20, t=60, b=40))
        fig4.update_yaxes(
            title="PaymentMethod",
            tickmode="array",
            tickvals=list(range(len(ycats))),
            ticktext=ycats,
        )
        fig4.update_xaxes(title="tenure")
        st.plotly_chart(fig4, use_container_width=True)

        # 5) tenure (x num) vs Contract (y cat)
        s_y = as_cat_str(df["Contract"])
        yj, ycats = jitter_codes(s_y, jitter=JITTER, salt=22)

        d5 = df.copy()
        d5["_y"] = yj
        fig5 = px.scatter(
            d5,
            x="tenure",
            y="_y",
            color="_Churn_str",
            color_discrete_map=TARGET_COLORS,
            category_orders=CHURN_ORDER,
            title="Contract vs tenure (jittered)",
            hover_data={"Contract": True, "tenure": True, "_y": False},
        )
        fig5.update_traces(marker=dict(size=MARKER_SIZE, opacity=OPACITY))
        fig5.update_layout(template="plotly_dark", height=450, margin=dict(l=40, r=20, t=60, b=40))
        fig5.update_yaxes(
            title="Contract",
            tickmode="array",
            tickvals=list(range(len(ycats))),
            ticktext=ycats,
        )
        fig5.update_xaxes(title="tenure")
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # -----------------------------
    # Slider tenure_category + MonthlyCharges-Bins (ohne Gesamtview)
    # -----------------------------
    selected_cat, df_sub = tenure_category_slider(df, key="ms_tenure_cat")

    if selected_cat is not None and len(df_sub) > 0:
        st.caption(f"Filter: tenure_category = {selected_cat} | N = {len(df_sub):,}")

        bins = np.arange(0, 141, 10)

        # Helper returns fig; we update it to enforce color mapping + order
        fig_bins = churn_share_monthlycharges_bins(df_sub, bins=bins, churn_col="Churn")
        fig_bins.update_traces(marker_line_width=0)
        #fig_bins.update_layout(template="plotly_dark")
        fig_bins.update_layout(legend_title_text="Churn")

        # Force colors for bars (plotly express stores mapping in layout)
        fig_bins.update_layout(coloraxis=None)
        fig_bins.update_layout(
            legend=dict(traceorder="normal")
        )
        # Best: rebuild with px.bar inside helper using color_discrete_map.
        # If one keep the helper as-is, one can still set:
        fig_bins.update_layout(
            **{"colorway": [TARGET_COLORS["False"], TARGET_COLORS["True"]]}
        )

        st.plotly_chart(fig_bins, use_container_width=True)

    elif selected_cat is not None and len(df_sub) == 0:
        st.info("Keine Daten für diese tenure_category.")
    else:
        st.info("tenure_category ist nicht vorhanden oder leer.")

with tab1:
    st.header("Distribution Analysis")
    left, right = st.columns([1, 3])

    with left:
        feature_selected = st.selectbox(
            "Feature Selection",
            schema_df["columnname"].tolist(),
            index=0
        )
        split_by_target = st.checkbox("Split by target variable")

        # get semantic type for selected feature
        semantic_type = schema_df.loc[schema_df["columnname"] == feature_selected, "semantic_type"].iloc[0]

        # TASK: Slider ímplementation pending 😶‍🌫️
        #if semantic_type == "number":
        #    st.caption("Optional")
            # Sider for part of the Plot
            # slider for bin size


    with right:
        plot_distribution(df, feature_selected, semantic_type, TARGET_COLUMN, split_by_target)


with tab2:
    st.header("Correlation Analysis")

    # Build numeric dataframe: include numeric + bool converted to int
    df_corr_base = df.copy()

    # bool -> int (0/1) so it is included in correlations
    for c in BOOLEAN_COLUMNS:
        if c in df_corr_base.columns:
            # handle pandas nullable booleans if present
            df_corr_base[c] = df_corr_base[c].astype("Int64").astype(float)

    # keep only numeric columns (now includes converted bools)
    num_df = df_corr_base.select_dtypes(include=["number"]).copy()

    if num_df.shape[1] < 2:
        st.warning("Not enough numeric features for correlation analysis.")
    else:
        # Drop constant columns (avoid NaN correlations)
        nunique = num_df.nunique(dropna=True)
        num_df = num_df.loc[:, nunique > 1]

        if num_df.shape[1] < 2:
            st.warning("Not enough non-constant numeric features for correlation analysis.")
        else:
            method = st.selectbox("Method", ["pearson", "spearman"], index=0)

            corr = num_df.corr(method=method)

            # ---------------------------
            # 1) Correlation Matrix
            # ---------------------------
            st.subheader("Correlation Matrix")

            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)

            ax.set_title("Correlation Matrix")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index, fontsize=8)

            # annotate values (like your screenshot)
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    val = corr.iat[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation Coefficient")

            st.pyplot(fig, clear_figure=True)

            # ---------------------------
            # 3) Correlations with target variable
            # ---------------------------
            st.subheader(f"Correlations with Target Variable: {TARGET_COLUMN}")

            if TARGET_COLUMN not in corr.columns:
                st.warning(
                    f"Target column '{TARGET_COLUMN}' is not part of the numeric correlation matrix "
                    f"(it may be non-numeric or missing)."
                )
            else:
                target_corr = corr[TARGET_COLUMN].drop(labels=[TARGET_COLUMN]).dropna()

                if target_corr.empty:
                    st.info("No valid correlations with the target variable were found.")
                else:
                    n_max = min(30, len(target_corr))
                    n_show = st.slider("Number of features in plot", 5, n_max, min(15, n_max))

                    # take top-N by absolute correlation, then sort by signed value for a nicer plot
                    series_plot = target_corr.reindex(
                        target_corr.abs().sort_values(ascending=False).head(n_show).index
                    ).sort_values()

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    bars = ax2.barh(series_plot.index, series_plot.values, edgecolor="black")
                    ax2.axvline(0, linewidth=1)

                    ax2.set_title(f"Feature Correlations with {TARGET_COLUMN}")
                    ax2.set_xlabel(f"Correlation ({method}) with {TARGET_COLUMN}")
                    ax2.set_ylabel("Feature")

                    # value labels: inside (white, bold) if bar is wide enough, otherwise outside (black)
                    x_min, x_max = ax2.get_xlim()
                    span = x_max - x_min
                    inside_threshold = 0.08 * span  # threshold for "small" bars

                    for bar, val in zip(bars, series_plot.values):
                        y = bar.get_y() + bar.get_height() / 2
                        width = bar.get_width()

                        label = f"{val:.3f}"

                        # decide inside vs outside
                        if abs(width) >= inside_threshold:
                            # inside bar
                            x_text = width * 0.5
                            ax2.text(
                                x_text,
                                y,
                                label,
                                va="center",
                                ha="center",
                                color="white",
                                fontsize=10,
                                fontweight="bold"
                            )
                        else:
                            # outside bar (black)
                            offset = 0.01 * span
                            if val >= 0:
                                x_text = width + offset
                                ha = "left"
                            else:
                                x_text = width - offset
                                ha = "right"

                            ax2.text(
                                x_text,
                                y,
                                label,
                                va="center",
                                ha=ha,
                                color="black",
                                fontsize=9,
                                fontweight="bold"
                            )

                    st.pyplot(fig2, clear_figure=True)

with tab3:
    st.header("Comparison Analysis")

    # --- choices (exclude target, but allow same feature twice) ---
    feature_choices = [c for c in schema_df["columnname"].tolist() if c != TARGET_COLUMN]
    if not feature_choices:
        st.warning("No features available (all columns are excluded).")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X-Axis", feature_choices, index=0)
    with col2:
        default_y_idx = 1 if len(feature_choices) > 1 else 0
        y = st.selectbox("Y-Axis", feature_choices, index=default_y_idx)

    x_sem = get_semantic_type(schema_df, x)
    y_sem = get_semantic_type(schema_df, y)



    # --- build alias columns to avoid duplicate-name bugs (x==y etc.) ---
    plot_df = pd.DataFrame({
        "__x__": df[x],
        "__y__": df[y],
        "__t_raw__": df[TARGET_COLUMN],
    })
    plot_df["__t__"] = as_cat_str(plot_df["__t_raw__"])      # for discrete coloring
    plot_df["__t_int__"] = target_to_int(plot_df["__t_raw__"])  # for churn rate

    # common labels so legend title is TARGET_COLUMN (not "__t__")
    LABELS_COMMON = {"__t__": TARGET_COLUMN}

    fig = None

    # -------------------------
    # same feature twice -> 1D view
    # -------------------------
    if x == y:
        if x_sem == "number":
            fig = px.histogram(
                plot_df,
                x="__x__",
                color="__t__",
                color_discrete_map=TARGET_COLORS,
                barmode="overlay",
                template="plotly_dark",
                title=f"Distribution of {x} by {TARGET_COLUMN}",
                labels={**LABELS_COMMON, "__x__": x},
            )
            fig.update_traces(opacity=0.55)
            fig.update_xaxes(title=x)
            fig.update_yaxes(title="count")
            fig.update_layout(legend_title_text=TARGET_COLUMN)
        else:
            plot_df["__x_str__"] = as_cat_str(plot_df["__x__"])
            fig = px.histogram(
                plot_df,
                x="__x_str__",
                color="__t__",
                color_discrete_map=TARGET_COLORS,
                barmode="group",
                template="plotly_dark",
                title=f"{x} by {TARGET_COLUMN} (counts)",
                labels={**LABELS_COMMON, "__x_str__": x},
            )
            fig.update_xaxes(title=x)
            fig.update_yaxes(title="count")
            fig.update_layout(legend_title_text=TARGET_COLUMN)

    # -------------------------
    # numeric vs numeric -> scatter
    # -------------------------
    elif x_sem == "number" and y_sem == "number":
        fig = px.scatter(
            plot_df,
            x="__x__",
            y="__y__",
            color="__t__",
            color_discrete_map=TARGET_COLORS,
            template="plotly_dark",
            title=f"{y} vs {x}",
            labels={**LABELS_COMMON, "__x__": x, "__y__": y},
        )
        fig.update_traces(marker=dict(opacity=0.55, size=6))
        fig.update_layout(legend_title_text=TARGET_COLUMN)

    # -------------------------
    # discrete vs numeric -> jittered scatter cloud
    # -------------------------
    elif x_sem in DISCRETE_SEMS and y_sem == "number":
        plot_df["__x_str__"] = as_cat_str(plot_df["__x__"])
        xj, cats = jitter_codes(plot_df["__x_str__"], jitter=0.28, salt=1)
        plot_df["__x_j__"] = xj

        fig = px.scatter(
            plot_df,
            x="__x_j__",
            y="__y__",
            color="__t__",
            color_discrete_map=TARGET_COLORS,
            template="plotly_dark",
            title=f"{y} vs {x} (jittered)",
            labels={**LABELS_COMMON, "__x_j__": x, "__y__": y},
        )
        fig.update_traces(marker=dict(opacity=0.45, size=6))
        fig.update_xaxes(
            title=x,
            tickmode="array",
            tickvals=list(range(len(cats))),
            ticktext=cats,
        )
        fig.update_yaxes(title=y)
        fig.update_layout(legend_title_text=TARGET_COLUMN)

    # -------------------------
    # numeric vs discrete -> jittered scatter cloud
    # -------------------------
    elif x_sem == "number" and y_sem in DISCRETE_SEMS:
        plot_df["__y_str__"] = as_cat_str(plot_df["__y__"])
        yj, cats = jitter_codes(plot_df["__y_str__"], jitter=0.28, salt=2)
        plot_df["__y_j__"] = yj

        fig = px.scatter(
            plot_df,
            x="__x__",
            y="__y_j__",
            color="__t__",
            color_discrete_map=TARGET_COLORS,
            template="plotly_dark",
            title=f"{y} vs {x} (jittered)",
            labels={**LABELS_COMMON, "__x__": x, "__y_j__": y},
        )
        fig.update_traces(marker=dict(opacity=0.45, size=6))
        fig.update_yaxes(
            title=y,
            tickmode="array",
            tickvals=list(range(len(cats))),
            ticktext=cats,
        )
        fig.update_xaxes(title=x)
        fig.update_layout(legend_title_text=TARGET_COLUMN)

    # -------------------------
    # discrete vs discrete -> churn-rate heatmap + counts annotated
    # -------------------------
    elif x_sem in DISCRETE_SEMS and y_sem in DISCRETE_SEMS:
        plot_df["__x_str__"] = as_cat_str(plot_df["__x__"])
        plot_df["__y_str__"] = as_cat_str(plot_df["__y__"])

        valid_rate = plot_df["__t_int__"].notna().mean() >= 0.95

        if valid_rate:
            grp = (
                plot_df
                .groupby(["__y_str__", "__x_str__"], dropna=False)
                .agg(n=("__t_int__", "size"), rate=("__t_int__", "mean"))
                .reset_index()
            )

            rate_mat = grp.pivot(index="__y_str__", columns="__x_str__", values="rate")
            n_mat = grp.pivot(index="__y_str__", columns="__x_str__", values="n").fillna(0).astype(int)

            fig = go.Figure(data=go.Heatmap(
                z=rate_mat.values,
                x=rate_mat.columns.astype(str),
                y=rate_mat.index.astype(str),
                text=n_mat.values,
                texttemplate="%{text}",
                colorscale=CHURN_RATE_COLORSCALE,
                zmin=0,
                zmax=1,
                xgap=3,
                ygap=3,
                colorbar=dict(title=f"{TARGET_COLUMN} rate"),
                hovertemplate=(
                    f"{x}: %{{x}}<br>"
                    f"{y}: %{{y}}<br>"
                    "count: %{text}<br>"
                    f"{TARGET_COLUMN} rate: %{{z:.1%}}"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="black",
                paper_bgcolor="black",
                title=f"{TARGET_COLUMN} rate by {y} vs {x} (counts annotated)",
            )
            fig.update_xaxes(title=x)
            fig.update_yaxes(title=y)

        else:
            # fallback: raw counts heatmap
            ct = pd.crosstab(plot_df["__y_str__"], plot_df["__x_str__"])
            fig = go.Figure(data=go.Heatmap(
                z=ct.values,
                x=ct.columns.astype(str),
                y=ct.index.astype(str),
                colorscale="Blues",
                xgap=3,
                ygap=3,
                colorbar=dict(title="count"),
                hovertemplate=(
                    f"{x}: %{{x}}<br>"
                    f"{y}: %{{y}}<br>"
                    "count: %{z}"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="black",
                paper_bgcolor="black",
                title=f"{y} vs {x} (counts)",
            )
            fig.update_xaxes(title=x)
            fig.update_yaxes(title=y)

    else:
        st.warning(f"Unsupported combination: {x_sem} vs {y_sem}")

    if fig is not None:
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)