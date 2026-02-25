import streamlit as st
import pandas as pd
from src.manager import Manager
from src.analytical_modules.explanation_generator import ExplanationGenerator
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import numpy as np
from src.utils import build_context, explanation_llm, setup_logging, load_custom_data
from matplotlib_venn._common import VennDiagram
import urllib.parse
import re
import uuid
import logging
from pathlib import Path

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# log each new session with a unique ID and timestamp, and log each rerun with the session ID and rerun count
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["rerun_count"] = 0
    logger.info("NEW_SESSION session_id=%s", st.session_state["session_id"])

st.session_state["rerun_count"] += 1
logger.info(
    "RERUN session_id=%s rerun_count=%d",
    st.session_state["session_id"],
    st.session_state["rerun_count"],
)

# Set up streamlit frontend 
st.set_page_config(
    page_title = "Any Table LLM Assistant",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Initialize session state
# Use to track conversation history, last results, pending questions for reruns, and explanations.
if "history" not in st.session_state:
    st.session_state["history"] = []

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "last_explanation" not in st.session_state:
    st.session_state["last_explanation"] = None

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

if "rerun_query" not in st.session_state:
    st.session_state["rerun_query"] = None

# Used to force-reset the upload widget when switching data sources.
if "upload_uploader_key" not in st.session_state:
    st.session_state["upload_uploader_key"] = 0

# Some custom styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4 {
        color: #22303C;
    }
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Set a single global default 
pio.templates.default = "seaborn"

# Set default color scheme
pio.templates["seaborn"].layout.colorway = [
    "rgba(70,120,150,0.8)",  
    "rgba(110,140,170,0.8)", 
    "rgba(150,170,190,0.8)", 
    "rgba(200,140,120,0.8)", 
    "rgba(120,150,110,0.8)",
    "rgba(180,180,100,0.8)",
    "rgba(160,110,160,0.8)",
    "rgba(100,160,180,0.8)",
    "rgba(210,160,100,0.8)",
    "rgba(90,130,100,0.8)"
]

# Helper function for plotly styling
def apply_default_style(fig):
    """Apply a single, consistent style safely across Plotly trace types."""
    # ---- Layout (titles, axes, legend, margins)
    fig.update_layout(
        template="seaborn",
        font=dict(size=14, color="#222222"),
        title_font=dict(size=16, color="#111111"),
        legend=dict(
            title_font=dict(size=14),
            font=dict(size=13),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            title_font=dict(size=14, color="#111111"),
            tickfont=dict(size=12, color="#333333"),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.5)"
        ),
        yaxis=dict(
            title_font=dict(size=14, color="#111111"),
            tickfont=dict(size=12, color="#333333"),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    # ---- Per-trace styling ----
    for tr in fig.data:
        t = (tr.type or "").lower()

        # Scatter / line
        if t in ["scatter", "scattergl", "line"]:
            has_multicolor = (hasattr(tr, "marker") and isinstance(getattr(tr.marker, "color", None), (list, np.ndarray)))
            tr.update(
                marker=dict(
                    size=6,
                    opacity=0.85,
                    color=None if has_multicolor else getattr(tr.marker, "color", "rgba(70,120,150,0.8)"),
                    line=dict(width=0.5, color="black")
                ),
                line=dict(width=2)
            )

        # Bar / box / violin
        elif t in ["bar", "box", "violin"]:
            if hasattr(tr, "marker"):
                if getattr(tr.marker, "color", None) is None:
                    tr.marker.color = "rgba(70,120,150,0.8)"
                tr.marker.line = dict(width=1, color="black")

        # Heatmap / surface
        elif t in ["heatmap", "surface", "contour"]:
            tr.update(colorscale="Viridis", showscale=True)

        # Pie / donut
        elif t in ["pie", "donut"]:
            tr.update(textfont=dict(size=13, color="#222222"))

    return fig


# Matplot lib venn viagram style
plt.rcParams.update({"font.size": 14})


# Helper function to build a compact schema summary for the dashboard
def build_schema_summary(df: pd.DataFrame):
    """Return a compact schema summary table for dashboard display."""
    total_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        unique = int(s.nunique(dropna=True))
        sample_vals = ", ".join(
            map(str, s.dropna().astype(str).drop_duplicates().head(3).tolist())
        )
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null_%": round(non_null / total_rows * 100, 2),
                "missing_%": round((1 - (non_null / total_rows)) * 100, 2),
                "unique": unique,
                "sample_values": sample_vals,
            }
        )
    return pd.DataFrame(rows)


def build_numeric_plot_candidates(
    df: pd.DataFrame,
    min_parse_ratio: float = 0.75,
    max_alpha_ratio: float = 0.8,
):
    """
    Build numeric series candidates for plotting.
    Includes true numeric columns and object/category columns that are mostly parseable to numeric.
    """
    candidates = {}
    parse_info = {}

    # Keep native numeric columns as-is
    for col in df.select_dtypes(include=[np.number]).columns:
        series = pd.to_numeric(df[col], errors="coerce")
        candidates[col] = series
        parse_info[col] = {"converted": False, "parse_ratio": 1.0}

    # Try coercing object-like columns with conservative cleaning.
    # Guard against ID-like fields (e.g., AvatarKey123) by skipping columns
    # where most non-null values contain alphabetic characters.
    object_like_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in object_like_cols:
        s = df[col]
        s_non_null = s.dropna().astype(str).str.strip()
        if s_non_null.empty:
            continue

        alpha_ratio = float(s_non_null.str.contains(r"[A-Za-z]", regex=True).mean())
        if alpha_ratio >= max_alpha_ratio:
            continue

        cleaned = (
            s.astype(str)
            .str.strip()
            .str.replace(r"^\s*$", "", regex=True)
            .str.replace(r"(?i)^(na|n/a|none|null|unknown)$", "", regex=True)
            .str.replace(",", "", regex=False)
        )
        cleaned = cleaned.mask(cleaned.eq(""), np.nan)
        parsed = pd.to_numeric(cleaned, errors="coerce")

        valid_non_null = s.notna().sum()
        if valid_non_null == 0:
            continue
        parse_ratio = float(parsed.notna().sum() / valid_non_null)

        if parse_ratio >= min_parse_ratio:
            label = f"{col}"
            candidates[label] = parsed
            parse_info[label] = {
                "converted": True,
                "parse_ratio": parse_ratio,
                "alpha_ratio": alpha_ratio,
                "source_col": col,
            }

    return candidates, parse_info

#----------------# Main App Code #------------------------

# Setup app layout 
st.title("Any Table LLM Assistant")

# Introduction section
st.markdown("""
Welcome to the **Any Table LLM Assistant**, a natural-language interface to explore tabular data.
          
---
""")

# load example csv from test_data
@st.cache_data
def load_example_data(csv_path):
    return pd.read_csv(csv_path)

#### NEW _ DATASET SELECTION TABS #######
st.header("Dataset Selection")
tab_example, tab_upload = st.tabs(["Use an example CSV file", "Upload CSV"])

# Use example dataset tab
with tab_example:
    st.write("Select one of our example CSV files.")
    example_files = sorted(Path("test_data").glob("*.csv"))
    if not example_files:
        st.warning("No example CSV files found.")
    else:
        example_names = [p.name for p in example_files]
        default_name = st.session_state.get("selected_example_name", example_names[0])
        if default_name not in example_names:
            default_name = example_names[0]
        default_index = example_names.index(default_name)

        selected_example_name = st.selectbox(
            "Choose an example CSV",
            options=example_names,
            index=default_index,
            key="selected_example_name",
        )
        selected_example_path = Path("test_data") / selected_example_name
        st.caption(f"Selected file: `test_data/{selected_example_name}`")

        if st.button("Use selected example CSV"):
            try:
                example_df = load_example_data(str(selected_example_path))
                if example_df.shape[0] == 0 or example_df.shape[1] == 0:
                    st.error("Selected example CSV appears empty. Please choose another file.")
                else:
                    st.success(
                        f"Loaded example CSV with {example_df.shape[0]:,} rows and {example_df.shape[1]:,} columns."
                    )
                    st.session_state["df"] = load_custom_data(example_df)
                    st.session_state["df_source"] = f"example:{selected_example_name}"
                    st.session_state.pop("manager", None)
                    st.session_state["upload_uploader_key"] += 1
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to read example CSV: {e}")

# Upload dataset tab
with tab_upload:
    st.write("Upload your own CSV file.")
    uploader_key = f"upload_csv_{st.session_state['upload_uploader_key']}"
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key=uploader_key)
    if uploaded_file is not None:
        try:
            # Read CSV into dataframe. Allow pandas to infer encoding and separators.
            uploaded_df = pd.read_csv(uploaded_file)
            # Basic sanity checks: require at least 1 column and 1 row
            if uploaded_df.shape[0] == 0 or uploaded_df.shape[1] == 0:
                st.error("Uploaded CSV appears empty. Please check the file and try again.")
            else:
                st.success(f"Loaded uploaded CSV with {uploaded_df.shape[0]:,} rows and {uploaded_df.shape[1]:,} columns.")
                st.session_state["df"] = load_custom_data(uploaded_df)
                st.session_state["df_source"] = "upload"
                # When user uploads a new dataset, force manager re-initialization below
                st.session_state.pop("manager", None)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# Require the user to choose an example CSV or upload a custom CSV.
if "df" not in st.session_state:
    st.info("Choose an example CSV or upload a custom CSV to continue.")
    st.stop()

# Local df variable 
df = st.session_state["df"]

# Instantiate session-specific manager
if "manager" not in st.session_state:
    st.session_state["manager"] = Manager(df.copy(deep=True))

else:
    # If the manager exists but the underlying df changed in session_state, re-init it.
    # We detect this by a simple check: store last_df_shape in session_state
    last_shape = st.session_state.get("manager_df_shape")
    current_shape = (df.shape[0], df.shape[1])
    if last_shape != current_shape:
        st.session_state["manager"] = Manager(df.copy(deep=True))

# store current df shape for future change detection
st.session_state["manager_df_shape"] = (df.shape[0], df.shape[1])

m = st.session_state["manager"]

# Instantiate session-specific explanation agent
if "explanation_agent" not in st.session_state:
    st.session_state["explanation_agent"] = ExplanationGenerator(explanation_llm())
explanation_module = st.session_state["explanation_agent"]

# Data Overview Section
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Records", f"{df.shape[0]:,}")
with col2:
    st.metric("Number of Columns", f"{df.shape[1]}")
with col3:
    source = st.session_state.get("df_source", "unknown")
    st.caption(f"Data source: **{source}**")

if st.session_state["df_source"] == "upload":
    st.caption("Below is a preview of your uploaded dataset.")
elif str(st.session_state["df_source"]).startswith("example:"):
    example_name = str(st.session_state["df_source"]).split(":", 1)[1]
    st.caption(f"Below is a preview of the example dataset `test_data/{example_name}`.")
st.dataframe(df, width='stretch', hide_index=True)

st.markdown("---")

with st.expander("See data summary dashboard"):
    st.header("Data Summary Dashboard")

    # Quick metrics
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / df.shape[0] * 100) if df.shape[0] else 0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with metric_col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with metric_col3:
        st.metric("Missing Cells", f"{missing_pct:.2f}%")
    with metric_col4:
        st.metric("Duplicate Rows", f"{dup_pct:.2f}%")

    # Schema summary
    st.subheader("Schema Overview")
    schema_df = build_schema_summary(df)
    st.dataframe(schema_df, width="stretch", hide_index=True)

    # Basic distributions
    numeric_candidates, parse_info = build_numeric_plot_candidates(df)
    coerced_source_cols = {
        info["source_col"] for info in parse_info.values() if info.get("converted")
    }

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.subheader("Categorical Distribution")
        categorical_cols = [
            col
            for col in df.select_dtypes(include=["object", "category", "bool"]).columns
            if col not in coerced_source_cols
        ]
        if categorical_cols:
            cat_col = st.selectbox("Categorical column", categorical_cols, key="dashboard_cat_col")
            max_categories_to_show = 30
            counts_all = df[cat_col].astype(str).value_counts(dropna=False)
            total_categories = int(counts_all.shape[0])

            if total_categories <= max_categories_to_show:
                counts = counts_all
            else:
                top_n = st.slider(
                    "Top categories",
                    min_value=5,
                    max_value=max_categories_to_show,
                    value=10,
                    key="dashboard_top_n",
                )
                counts = counts_all.head(top_n)

            st.bar_chart(counts)
            st.caption(
                f"Showing {counts.shape[0]} out of {total_categories} total categories."
            )
        else:
            st.caption("No categorical columns detected.")

    with dist_col2:
        st.subheader("Numeric Distribution")
        numeric_labels = list(numeric_candidates.keys())

        if numeric_labels:
            num_col = st.selectbox("Numeric column", numeric_labels, key="dashboard_num_col")
            num_data = numeric_candidates[num_col].dropna()
            if num_data.empty:
                st.caption("Selected numeric column has no non-null values.")
            else:
                hist_fig = go.Figure(data=[go.Histogram(x=num_data, nbinsx=30)])
                hist_fig.update_layout(
                    title=f"Distribution of {num_col}",
                    xaxis_title=num_col,
                    yaxis_title="Count",
                )
                st.plotly_chart(hist_fig, width="stretch")

                info = parse_info.get(num_col, {})
                if info.get("converted"):
                    st.caption(
                        f"Auto-coerced from `{info.get('source_col')}`. "
                        f"Parseable values: {info.get('parse_ratio', 0.0) * 100:.1f}%."
                    )
        else:
            st.caption("No numeric columns detected.")

    st.markdown("---")

# Pre-load question into text box if rerun is scheduled
if st.session_state["pending_question"] is not None:
    st.session_state["question_input"] = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

### Query interface ###
st.header("Ask a Question")

# Input question box with Enter submission
with st.form("query_form", clear_on_submit=False, enter_to_submit=True):
    question = st.text_input("Ask a question:",key="question_input", autocomplete="off")
    submitted = st.form_submit_button("Submit", width='stretch')

# Reset session button
if st.button("Reset Conversation"):
    # Clear session state and start fresh - no conversation context sent to LLMs
    st.session_state["history"] = []
    st.session_state["last_result"] = None
    st.session_state["pending_question"] = None 
    st.session_state["last_explanation"] = None
    st.rerun()

# Handle rerun logic if user clicks "rerun" on a previous question. Reprocess with context. 
if st.session_state["rerun_query"]:
    question = st.session_state["rerun_query"]
    st.session_state["rerun_query"] = None

    # Rebuild context
    context = build_context(st.session_state["history"], max_turns=10)

    with st.spinner("Reprocessing your question..."):
        result = m.process_question(question, context=context)
    
    # Add to history immediately 
    code_str = result.get("code") if isinstance(result, dict) else None
    st.session_state["history"].append({"question": question, "code": code_str})

    explanation = None
    if code_str and result.get("type") != "error" and result.get("type") != "text":
        # Rebuild context including latest run
        updated_context = build_context(st.session_state["history"], max_turns=10)
        explanation = explanation_module.generate_explanation(updated_context)
        st.session_state["last_explanation"] = explanation
    
    else:
        st.session_state["last_explanation"] = None

    st.session_state["last_result"] = result
    st.rerun()  

# Process new question submission 
result = None

if submitted and question:

    # Build LLM memory context from session history
    context = build_context(st.session_state["history"], max_turns=10)

    with st.spinner("Processing your question..."):
        result = m.process_question(question, context=context)

    # Extract LLM generated code for memory
    code_str = result.get("code") if isinstance(result, dict) else None
    
    # Update session history
    st.session_state["history"].append({
            "question": question,
            "code": code_str
        })
    
    # Generate explanation if code was generated
    if code_str and result.get("type") != "error" and result.get("type") != "text":
        # Rebuild context including the latest turn
        updated_context = build_context(st.session_state["history"], max_turns=10)

        explanation = explanation_module.generate_explanation(updated_context)
        st.session_state["last_explanation"] = explanation

    else:
        st.session_state["last_explanation"] = None

# If no new result but session has history, keep showing the last full result
if result is None:
    result = st.session_state.get("last_result", None)

# Save the last real result separately
if result is not None:
    st.session_state["last_result"] = result

# Display Results
if result is not None:
    res_type = result.get('type')
    res_data = result.get('data')
    res_code = result.get('code')

    # Get explanation from session state
    explanation = st.session_state.get("last_explanation", None)

    code_text = None

    if res_code:
        code_text = str(res_code)

    # Tabs for organizing results
    # If code generated, show it in a code tab. If not, just show the result tab.
    if code_text:
        tab_result, tab_code = st.tabs(["Result", "Code"])
    else:
        tab_result = st.tabs(["Result"])[0]
        tab_code = None

    with tab_result:
        # Display explanation if available
        if explanation:
            st.info(f"💡 {explanation}")

        # For plotly plots
        if res_type == "plotly" and isinstance(res_data, go.Figure):

            plotly_config = {"displayModeBar": True,
                                "scrollZoom": True,
                                "responsive": True,
                                "editable": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "irAE_plot",
                                    "scale": 3
                                }}
            
            fig = apply_default_style(res_data)
            st.plotly_chart(fig, config=plotly_config)

        # Handle matplotlib plots like venn diagrams
        elif res_type == "plot" and (isinstance(res_data, plt.Figure)) or hasattr(res_data, "figure") or isinstance(res_data, VennDiagram):
            st.pyplot(res_data, clear_figure=True)

        elif res_type == "dataframe":
            df = res_data.copy()
            for col in df.select_dtypes(include=["float", "float64"]).columns:
                df[col] = df[col].map(
                    # handle floats by formatting to 6 decimal places and stripping trailing zeros, but leave NaNs as-is
                    lambda x: f"{x:.6f}".rstrip("0").rstrip(".") if pd.notna(x) else x
                    )
            st.dataframe(df,
                width="stretch",
                hide_index=True)
            st.write(f"Result has {res_data.shape[0]} rows and {res_data.shape[1]} columns.")

        elif res_type == "number":
            if isinstance(res_data, (float, np.floating)):
                # Format floats to 6 decimal places and strip trailing zeros, but leave integers and other types as-is
                st.metric(label="Result", value = f"{res_data:.6f}".rstrip("0").rstrip("."))
            else:
                st.metric(label="Result", value = res_data)

        elif res_type == "error":
            st.error("There was an issue processing your request. Please try again or rephrase your question.")
            
        else:
            # If the returned data is a string, show it with links for (ASCO), (NCCN), (SITC)
            if isinstance(res_data, str):
                st.markdown(res_data)

            else:
                # non-string results (dict, list, etc): leave as-is
                st.write(res_data)

    if tab_code:
        with tab_code:
            code_text = str(res_code) if res_type != "error" and not isinstance(res_data, str) else f"No code generated."
            st.code(code_text, language="python")

else:
    st.info("Enter a question above to get started.")


# Conversation history display with rerun buttons for each turn
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Session conversation history")

    for i, turn in enumerate(st.session_state["history"], start=1):
        col_q, col_btn = st.columns([4,1])

        with col_q:
            st.write(f"Q{i}: {turn['question']}")

        with col_btn:
            if st.button("Rerun", key=f"rerun_{i}", help="Rerun this query", width='stretch'):
                st.session_state["pending_question"] = turn['question']
                st.session_state["rerun_query"] = turn['question']
                st.rerun()
