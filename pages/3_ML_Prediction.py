import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from config import load_data, get_schema_df, TARGET_COLUMN
from utils.helpers import (
    train_and_evaluate,
    evaluate_single_model,
    make_defaults,
    pick_top_input_columns_for_model,
    get_logreg_feature_table,
    get_tree_feature_table,
    aggregate_to_base_feature,
)

st.set_page_config(page_title="ML Prediction", page_icon="🤖", layout="wide")

st.title("🤖 ML Prediction")
st.subheader("Customer churn prediction")

df = load_data()
schema_df = get_schema_df()

tab1, tab2, tab3 = st.tabs(["Model Training & Evaluation", "Prediction", "Model Info"])


# ---------------------------
# TAB 1: Training & Evaluation
# ---------------------------
with tab1:
    st.markdown("## 🎯 Model Training & Evaluation")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with c1:
        test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    with c2:
        random_state = st.number_input("Random state", value=42, step=1)
    with c3:
        run_train = st.button("Train & evaluate", use_container_width=True)
    with c4:
        clear = st.button("Clear results", use_container_width=True)

    if clear:
        for k in ["train_result", "chosen_model_name", "chosen_model"]:
            st.session_state.pop(k, None)
        st.success("Cleared training results.")

    if run_train:
        with st.spinner("Training models..."):
            tr = train_and_evaluate(
                df=df,
                schema_df=schema_df,
                target_col=TARGET_COLUMN,
                test_size=float(test_size),
                random_state=int(random_state),
            )
            st.session_state["train_result"] = tr

            best_name = tr.results_df.loc[0, "Model"]
            st.session_state["chosen_model_name"] = best_name
            st.session_state["chosen_model"] = tr.fitted_models[best_name]

        st.success("Training completed.")

    if "train_result" not in st.session_state:
        st.info("Click **Train & evaluate** to run model comparison.")
    else:
        tr = st.session_state["train_result"]

        st.markdown("### 📊 Model comparison")
        st.dataframe(tr.results_df, use_container_width=True)

        # Accuracy bar plot with values inside bars (white, bold)
        fig_acc, ax = plt.subplots()
        models = tr.results_df["Model"].tolist()
        vals = tr.results_df["Accuracy"].to_numpy()

        bars = ax.bar(models, vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model comparison (Accuracy)")
        ax.tick_params(axis="x", rotation=25)

        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" for v in vals],
            label_type="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

        fig_acc.tight_layout()
        st.pyplot(fig_acc)
        plt.close(fig_acc)

        st.markdown("### ✅ Choose model")
        model_names = tr.results_df["Model"].tolist()

        current_choice = st.session_state.get("chosen_model_name", model_names[0])
        try:
            default_ix = model_names.index(current_choice)
        except ValueError:
            default_ix = 0

        chosen = st.selectbox("Choose model", model_names, index=default_ix)
        st.session_state["chosen_model_name"] = chosen
        st.session_state["chosen_model"] = tr.fitted_models[chosen]

        chosen_model = st.session_state["chosen_model"]
        metrics = evaluate_single_model(chosen_model, tr.X_test, tr.y_test)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("F1", f"{metrics['f1']:.3f}")
        m3.metric("Precision", f"{metrics['precision']:.3f}")
        m4.metric("Recall", f"{metrics['recall']:.3f}")
        m5.metric("ROC_AUC", f"{metrics.get('roc_auc', np.nan):.3f}" if "roc_auc" in metrics else "—")

        st.markdown("### 🧩 Confusion Matrix")

        cm = metrics["confusion_matrix"]
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax_cm, values_format="d", colorbar=False)
        ax_cm.set_title(f"Confusion Matrix – {chosen}")
        fig_cm.tight_layout()

        st.pyplot(fig_cm)
        plt.close(fig_cm)


# ---------------------------
# TAB 2: Prediction
# ---------------------------
with tab2:
    st.markdown("## 🔮 Prediction")

    if "train_result" not in st.session_state or "chosen_model" not in st.session_state:
        st.info("Train models first in **Model Training & Evaluation** and choose a model.")
    else:
        tr = st.session_state["train_result"]
        model = st.session_state["chosen_model"]
        chosen_name = st.session_state.get("chosen_model_name", tr.results_df.loc[0, "Model"])

        fs = tr.feature_sets
        feature_cols = fs["feature_cols"]

        st.caption(f"Using model: **{chosen_name}**")
        st.caption("Only a small number of important inputs necessary. The remaining features are auto-filled with median/mode defaults.")

        defaults = make_defaults(df, feature_cols)

        top_cols = pick_top_input_columns_for_model(model, fs, top_k=10)

        left, right = st.columns(2)
        user_vals = {}

        def render_input(colname: str, container):
            if colname in fs["num_cols"]:
                v = float(defaults[colname]) if defaults[colname] is not None else 0.0
                return container.number_input(colname, value=v)
            if colname in fs["bool_cols"]:
                v = bool(defaults[colname]) if defaults[colname] is not None else False
                return container.checkbox(colname, value=v)
            if colname in fs["cat_cols"]:
                options = sorted(df[colname].dropna().unique().tolist())
                if not options:
                    return None
                try:
                    idx = options.index(defaults[colname])
                except Exception:
                    idx = 0
                return container.selectbox(colname, options, index=idx)

            options = sorted(df[colname].dropna().unique().tolist())
            if options:
                return container.selectbox(colname, options, index=0)
            return defaults.get(colname, None)

        for i, c in enumerate(top_cols):
            container = left if i % 2 == 0 else right
            user_vals[c] = render_input(c, container)

        row = defaults.copy()
        row.update(user_vals)
        X_one = pd.DataFrame([row], columns=feature_cols)

        model_step = model.named_steps["model"]
        has_proba = hasattr(model_step, "predict_proba")

        if has_proba:
            threshold = st.slider("Decision threshold (Churn probability)", 0.05, 0.95, 0.50, 0.05)
            proba = float(model.predict_proba(X_one)[:, 1][0])
            pred = proba >= threshold

            cA, cB = st.columns(2)
            cA.metric("Churn probability", f"{proba:.3f}")
            cB.metric("Decision", "Churn" if pred else "No churn")

            st.success("Prediction: **Churn**" if pred else "Prediction: **No churn**")
        else:
            pred = int(model.predict(X_one)[0])
            st.metric("Prediction", "Churn" if pred == 1 else "No churn")
            st.success("Prediction: **Churn**" if pred == 1 else "Prediction: **No churn**")


# ---------------------------
# TAB 3: Model Info
# ---------------------------
with tab3:
    st.markdown("## 🧠 Model Info")

    if "train_result" not in st.session_state or "chosen_model" not in st.session_state:
        st.info("Train models first and choose a model to inspect feature relevance.")
    else:
        tr = st.session_state["train_result"]
        model = st.session_state["chosen_model"]
        chosen_name = st.session_state.get("chosen_model_name", tr.results_df.loc[0, "Model"])

        st.subheader(f"Selected model: {chosen_name}")

        st.markdown(
            """
### Methodology (pipeline)
- **Boolean** → 0/1  
- **Numeric** → standardized for distance-based models  
- **Categorical** → one-hot encoding with `handle_unknown="ignore"` (robust for new categories)

### Feature relevance
- Logistic Regression → coefficients  
- Decision Tree → feature importance  
- KNN / Nearest Centroid → no native importance (optional: permutation importance)
"""
        )

        model_step = model.named_steps["model"]
        model_class = model_step.__class__.__name__

        if model_class == "LogisticRegression":
            df_feat = get_logreg_feature_table(model, top_n=30)
            st.markdown("### Top encoded features (|coefficient|)")
            st.dataframe(df_feat[["feature", "coef", "abs"]], use_container_width=True)

            agg = aggregate_to_base_feature(df_feat, score_col="abs")
            st.markdown("### Aggregated by original column (sum of |coefficients|)")
            st.dataframe(agg, use_container_width=True)

            top10 = agg.head(10)
            fig, ax = plt.subplots()
            bars = ax.bar(top10["base_feature"], top10["abs"])
            ax.set_title("Top drivers (aggregated) – Logistic Regression")
            ax.tick_params(axis="x", rotation=25)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in top10["abs"]], label_type="edge", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        elif model_class == "DecisionTreeClassifier":
            df_feat = get_tree_feature_table(model, top_n=30)
            st.markdown("### Top encoded features (importance)")
            st.dataframe(df_feat[["feature", "importance", "abs"]], use_container_width=True)

            agg = aggregate_to_base_feature(df_feat, score_col="abs")
            st.markdown("### Aggregated by original column (sum of importances)")
            st.dataframe(agg, use_container_width=True)

            top10 = agg.head(10)
            fig, ax = plt.subplots()
            bars = ax.bar(top10["base_feature"], top10["abs"])
            ax.set_title("Top drivers (aggregated) – Decision Tree")
            ax.tick_params(axis="x", rotation=25)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in top10["abs"]], label_type="edge", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning(
                "This model type has no native feature importance."
                "one could add permutation importance (model-agnostic, but slower)."
            )
