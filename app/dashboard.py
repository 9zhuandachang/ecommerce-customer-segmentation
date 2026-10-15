"""
Streamlit Dashboard — E-commerce Customer Segmentation & Repurchase Prediction
Three views: Segment Profiles | Customer Review | Model Comparison
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Analytics",
    page_icon=":material/analytics:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

SNAPSHOTS_PATH = DATA_DIR / "snapshots.parquet"
SCORED_PATH    = DATA_DIR / "scored_customers.parquet"
RESULTS_PATH   = DATA_DIR / "experiment_results.pkl"
SEG_PIPE_PATH  = MODEL_DIR / "segmentation_pipeline.pkl"
K_RESULTS_PATH = MODEL_DIR / "k_selection_results.pkl"
BEST_NAME_PATH = MODEL_DIR / "best_model_name.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_snapshots() -> pd.DataFrame:
    if not SNAPSHOTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SNAPSHOTS_PATH)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df

@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    if not SCORED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SCORED_PATH)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df

@st.cache_data(show_spinner=False)
def load_results():
    return joblib.load(RESULTS_PATH) if RESULTS_PATH.exists() else None

@st.cache_data(show_spinner=False)
def load_k_results():
    return joblib.load(K_RESULTS_PATH) if K_RESULTS_PATH.exists() else None

@st.cache_data(show_spinner=False)
def load_best_name() -> str:
    return joblib.load(BEST_NAME_PATH) if BEST_NAME_PATH.exists() else "—"

def load_model_version() -> str:
    """Identify the saved model artifact used for historical scoring."""
    with BEST_MODEL_PATH.open("rb") as model_file:
        digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    return f"sha256:{digest[:12]}"

# ---------------------------------------------------------------------------
# Check pipeline readiness
# ---------------------------------------------------------------------------

pipeline_ready = all(path.exists() for path in (
    SNAPSHOTS_PATH, SCORED_PATH, RESULTS_PATH, BEST_MODEL_PATH, BEST_NAME_PATH
))

if not pipeline_ready:
    st.title("Pipeline not yet run")
    st.info(
        "Run the training pipeline first, then refresh this page.\n\n"
        "```bash\n"
        "# Build the audited snapshots first, then:\n"
        "python src/train_pipeline.py\n"
        "```",
        icon=":material/info:",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

snapshots      = load_snapshots()
scored_all     = load_scored()
scored         = scored_all.loc[
    scored_all["cutoff_date"].eq(scored_all["cutoff_date"].max())
].copy() if not scored_all.empty else scored_all
results        = load_results()
k_results_data = load_k_results()
best_name      = load_best_name()
model_version  = load_model_version()

SEGMENT_FEATURES = ["recency_days", "frequency", "monetary", "aov", "product_diversity"]

# Compute segment profile from the production scored snapshot
if not scored.empty and "segment_name" in scored.columns:
    agg_dict = dict(
        size        = ("CustomerID", "count"),
        recency     = ("recency_days", "median"),
        frequency   = ("frequency", "median"),
        monetary    = ("monetary", "median"),
        aov         = ("aov", "median"),
        diversity   = ("product_diversity", "median"),
    )
    if "repurchase" in scored.columns:
        agg_dict["repurchase_rate"] = ("repurchase", "mean")
    profile = scored.groupby("segment_name").agg(**agg_dict).reset_index()
else:
    profile = pd.DataFrame()

# Segment colour map (consistent across charts)
SEG_PALETTE = {
    "Champions": "#1a56db",
    "Loyal":     "#0e9f6e",
    "At Risk":   "#e74694",
    "Segment 4": "#7e3af2",
    "Segment 5": "#6b7280",
}

def seg_color(name: str) -> str:
    return SEG_PALETTE.get(name, "#6b7280")

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Customer Analytics")
    st.caption("E-commerce · UCI Online Retail II")
    st.divider()
    page = st.radio(
        "View",
        ["Overview", "Segment profiles", "Customer review", "Model comparison"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Best model: **{best_name}**")
    st.caption(f"Model version: **{model_version}**")

# ============================================================
# PAGE: Overview
# ============================================================

if page == "Overview":
    st.title("Customer analytics", anchor=False)
    st.caption("30-day repurchase prediction · UCI Online Retail II")

    n_customers  = len(scored)
    n_segments   = scored["segment_name"].nunique() if "segment_name" in scored.columns else 0
    avg_score    = scored["repurchase_score"].mean() if "repurchase_score" in scored.columns else 0
    cutoff_str   = str(scored["cutoff_date"].iloc[0]) if "cutoff_date" in scored.columns and len(scored) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{n_customers:,}")
    c2.metric("Segments", n_segments)
    c3.metric("Avg model score", f"{avg_score:.3f}")
    c4.metric("Snapshot date", cutoff_str)
    st.caption("Scores rank customers within a selected snapshot; probability calibration has not been validated.")

    if results and best_name in results:
        m = results[best_name]["test"]
        st.divider()
        st.subheader("Best model — test set performance", anchor=False)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("ROC-AUC", m["roc_auc"])
        mc2.metric("PR-AUC",  m["pr_auc"])
        mc3.metric("F1",      m["f1"])
        mc4.metric("Brier",   m["brier"])

    st.divider()
    st.subheader("How to use this dashboard", anchor=False)
    st.markdown(
        """
        | View | Purpose |
        |------|---------|
        | **Segment profiles** | Explore customer clusters by recency, frequency and spend |
        | **Customer review** | Filter by segment, rank by repurchase score, export CSV |
        | **Model comparison** | Inspect six prototype candidates; the planned A/B feature comparison is pending |
        """
    )

# ============================================================
# PAGE: Segment profiles
# ============================================================

elif page == "Segment profiles":
    st.title("Segment profiles", anchor=False)
    st.caption("K-Means clustering on RFM features. Best k selected by silhouette score.")

    if profile.empty:
        st.warning("No segment data available.")
        st.stop()

    # K-selection chart
    if k_results_data:
        k_df = pd.DataFrame(k_results_data)
        st.subheader("K-Means selection", anchor=False)
        fig_k = px.bar(
            k_df, x="k", y="silhouette",
            text=k_df["silhouette"].round(3),
            color="silhouette",
            color_continuous_scale=[[0, "#d1defa"], [1, "#1a56db"]],
            labels={"k": "Number of clusters (k)", "silhouette": "Silhouette score"},
        )
        fig_k.update_traces(texttemplate="%{text}", textposition="outside", marker_line_width=0)
        fig_k.update_layout(
            showlegend=False, height=300,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[0, k_df["silhouette"].max() * 1.2]),
            xaxis=dict(showgrid=False, tickmode="linear"),
        )
        st.plotly_chart(fig_k)

    # Segment size + recency
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Segment size", anchor=False)
        fig_pie = px.pie(
            profile, names="segment_name", values="size",
            color="segment_name", color_discrete_map=SEG_PALETTE,
            hole=0.45,
        )
        fig_pie.update_traces(
            textposition="outside", textinfo="percent+label",
            marker=dict(line=dict(color="white", width=2)),
        )
        fig_pie.update_layout(
            showlegend=False, height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie)

    with col_b:
        st.subheader("Median recency by segment", anchor=False)
        fig_rec = px.bar(
            profile.sort_values("recency"),
            x="segment_name", y="recency",
            color="segment_name", color_discrete_map=SEG_PALETTE,
            labels={"segment_name": "Segment", "recency": "Median recency (days)"},
        )
        fig_rec.update_layout(
            showlegend=False, height=320,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False),
        )
        fig_rec.update_traces(marker_line_width=0)
        st.plotly_chart(fig_rec)

    # Stats table
    st.subheader("Segment statistics", anchor=False)
    rename = {
        "segment_name": "Segment",
        "size":         "Customers",
        "recency":      "Median recency (days)",
        "frequency":    "Median invoices",
        "monetary":     "Median spend (£)",
        "aov":          "Median AOV (£)",
        "diversity":    "Median products",
    }
    if "repurchase_rate" in profile.columns:
        rename["repurchase_rate"] = "Observed 30-day repurchase rate"

    display_profile = profile.rename(columns=rename).set_index("Segment")
    fmt = {"Median spend (£)": "£{:,.0f}", "Median AOV (£)": "£{:,.2f}"}
    if "Observed 30-day repurchase rate" in display_profile.columns:
        fmt["Observed 30-day repurchase rate"] = "{:.1%}"
    st.dataframe(display_profile.style.format(fmt))

    # Scatter: Recency vs Frequency
    st.subheader("Recency vs frequency", anchor=False)
    plot_df = scored.sample(min(len(scored), 3000), random_state=42) if len(scored) > 3000 else scored
    fig_sc = px.scatter(
        plot_df, x="recency_days", y="frequency",
        color="segment_name", color_discrete_map=SEG_PALETTE,
        opacity=0.55,
        labels={
            "recency_days": "Recency (days)",
            "frequency": "Invoice frequency",
            "segment_name": "Segment",
        },
        hover_data=["CustomerID", "monetary"],
    )
    fig_sc.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig_sc.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(title="Segment"),
    )
    st.plotly_chart(fig_sc)

# ============================================================
# PAGE: Customer review
# ============================================================

elif page == "Customer review":
    st.title("Customer review", anchor=False)
    st.caption("Filter by segment, rank by model score and export a review list. Scores are not validated probabilities.")

    if scored.empty or "repurchase_score" not in scored.columns:
        st.error("Scored customer data not found. Re-run the training pipeline.")
        st.stop()

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])

    with col_ctrl1:
        available_dates = sorted(snapshots["cutoff_date"].unique()) if not snapshots.empty else []
        date_options = [str(d.date()) for d in available_dates]
        selected_date = st.selectbox(
            "Snapshot date",
            options=date_options,
            index=len(date_options) - 1 if date_options else 0,
        )

    date_scored = scored_all.loc[
        scored_all["cutoff_date"].dt.strftime("%Y-%m-%d").eq(selected_date)
    ].copy()

    with col_ctrl2:
        seg_options = ["All segments"] + sorted(date_scored["segment_name"].unique().tolist()) \
            if "segment_name" in date_scored.columns else ["All segments"]
        selected_seg = st.selectbox("Segment", seg_options)

    with col_ctrl3:
        max_n = max(1, min(len(date_scored), 500))
        top_n = st.slider("Top N customers", min_value=1, max_value=max_n,
                          value=min(50, max_n), step=1)

    # Filter & rank
    display = date_scored.copy()
    if selected_seg != "All segments":
        display = display[display["segment_name"] == selected_seg]

    if display.empty:
        st.info("No customers match this date and segment.")
        st.stop()

    display = display.sort_values("repurchase_score", ascending=False).head(top_n).copy()
    display = display.drop(columns="rank", errors="ignore")
    display.insert(0, "rank", range(1, len(display) + 1))

    # Score distribution
    st.subheader("Score distribution", anchor=False)
    fig_hist = px.histogram(
        display, x="repurchase_score", nbins=25,
        labels={"repurchase_score": "Repurchase score", "count": "Customers"},
        color_discrete_sequence=["#1a56db"],
    )
    fig_hist.update_layout(
        height=250, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Customers"),
        xaxis=dict(showgrid=False),
        bargap=0.05,
    )
    fig_hist.update_traces(marker_line_width=0)
    st.plotly_chart(fig_hist)

    # Table
    st.subheader(f"Top {len(display)} customers", anchor=False)
    show_cols = ["rank", "CustomerID", "segment_name", "repurchase_score",
                 "recency_days", "frequency", "monetary", "aov"]
    show_cols = [c for c in show_cols if c in display.columns]
    col_rename = {
        "rank": "Rank", "CustomerID": "Customer ID",
        "segment_name": "Segment", "repurchase_score": "Score",
        "recency_days": "Recency (days)", "frequency": "Invoices",
        "monetary": "Total spend (£)", "aov": "AOV (£)",
    }
    st.dataframe(
        display[show_cols].rename(columns=col_rename).style.format({
            "Score": "{:.3f}",
            "Total spend (£)": "£{:,.0f}",
            "AOV (£)": "£{:,.2f}",
        }),
        height=420,
    )

    # Export
    st.divider()
    export_cols = ["CustomerID", "segment_name", "repurchase_score", "rank",
                   "recency_days", "frequency", "monetary"]
    export_df = display[[c for c in export_cols if c in display.columns]].copy()
    export_df["snapshot_date"]  = selected_date
    export_df["model_version"]  = model_version
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"customer_review_{selected_date}.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption(f"{len(export_df)} customers · model: {best_name} · version: {model_version}")

# ============================================================
# PAGE: Model comparison
# ============================================================

elif page == "Model comparison":
    st.title("Model comparison", anchor=False)
    st.caption("Six prototype candidates on chronological splits; the proposal's two feature sets are still pending.")

    if results is None:
        st.warning("No model results found. Run the training pipeline first.")
        st.stop()

    # Summary dataframe
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":      name,
            "Val AUC":    r["val"]["roc_auc"],
            "Val PR-AUC": r["val"]["pr_auc"],
            "Val Brier":  r["val"]["brier"],
            "Val F1":     r["val"]["f1"],
            "Test AUC":   r["test"]["roc_auc"],
            "Test PR-AUC":r["test"]["pr_auc"],
            "Test Brier": r["test"]["brier"],
            "Test F1":    r["test"]["f1"],
        })
    summary_df = pd.DataFrame(rows)

    st.caption(f"Selected by validation average precision (AP): **{best_name}**")

    # ROC-AUC grouped bar
    st.subheader("ROC-AUC", anchor=False)
    fig_auc = go.Figure([
        go.Bar(name="Validation", x=summary_df["Model"], y=summary_df["Val AUC"],
               marker_color="#93b4f5", marker_line_width=0),
        go.Bar(name="Test",       x=summary_df["Model"], y=summary_df["Test AUC"],
               marker_color="#1a56db", marker_line_width=0),
    ])
    fig_auc.update_layout(
        barmode="group", height=340,
        yaxis=dict(range=[0.45, 1.0], title="ROC-AUC",
                   showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.25, bargroupgap=0.05,
    )
    st.plotly_chart(fig_auc)

    # PR-AUC grouped bar
    st.subheader("PR-AUC", anchor=False)
    fig_pr = go.Figure([
        go.Bar(name="Validation", x=summary_df["Model"], y=summary_df["Val PR-AUC"],
               marker_color="#86efac", marker_line_width=0),
        go.Bar(name="Test",       x=summary_df["Model"], y=summary_df["Test PR-AUC"],
               marker_color="#0e9f6e", marker_line_width=0),
    ])
    fig_pr.update_layout(
        barmode="group", height=340,
        yaxis=dict(title="PR-AUC", showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.25, bargroupgap=0.05,
    )
    st.plotly_chart(fig_pr)

    # Metrics table
    st.subheader("All metrics", anchor=False)
    display_summary = summary_df.set_index("Model")
    higher_better = ["Val AUC", "Val PR-AUC", "Val F1", "Test AUC", "Test PR-AUC", "Test F1"]
    lower_better  = ["Val Brier", "Test Brier"]
    st.dataframe(
        display_summary.style
            .highlight_max(axis=0, subset=higher_better, color="#dbeafe")
            .highlight_min(axis=0, subset=lower_better,  color="#dbeafe")
            .format("{:.4f}"),
    )

    # Ensemble weight sensitivity
    ens_rows = [r for r in rows if "Ensemble" in r["Model"]]
    if ens_rows:
        st.subheader("Ensemble weight sensitivity", anchor=False)
        w_label = {
            "Ensemble_w025": "w=0.25",
            "Ensemble_w050": "w=0.50",
            "Ensemble_w075": "w=0.75",
        }
        ens_df = pd.DataFrame(ens_rows)
        ens_df["LR weight"] = ens_df["Model"].map(w_label)
        melted = ens_df.melt(
            id_vars=["LR weight"],
            value_vars=["Val AUC", "Test AUC"],
            var_name="Split", value_name="ROC-AUC",
        )
        fig_ens = px.line(
            melted, x="LR weight", y="ROC-AUC", color="Split",
            markers=True,
            color_discrete_sequence=["#93b4f5", "#1a56db"],
            labels={"LR weight": "LR weight in ensemble"},
        )
        fig_ens.update_layout(
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig_ens.update_traces(line=dict(width=2), marker=dict(size=7))
        st.plotly_chart(fig_ens)

    # Leakage note
    with st.expander("Leakage control", icon=":material/shield:"):
        st.markdown(
            """
            **Chronological splits** — earlier label windows finish before the next partition begins.

            - **Train (12 monthly cutoffs)** — fitted imputation, scaling, K-Means and prediction models
            - **Validation (3 monthly cutoffs)** — model selection
            - **Test (3 monthly cutoffs)** — held-out prototype metrics reported above

            All transformations are fit on training data only and applied to validation and test.
            """
        )
