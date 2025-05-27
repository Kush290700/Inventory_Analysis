# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path
from datetime import datetime
import logging

import streamlit as st
import pandas as pd

from utils.io_utils import load_sheets
from utils.cleaning import (
    preprocess_data,
    process_inventory_snapshot,
    process_inventory_detail1,
)
from utils.aggregation import (
    aggregate_sales_history,
    merge_data,
    aggregate_final_data,
)
from utils.costing import compute_holding_cost
from tabs import (
    kpis,
    woh,
    movers,
    holding_cost,
    insights,
    bin_scan,
)

# -----------------------
# Logger Configuration
# -----------------------
logger = logging.getLogger("inventory_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# -----------------------
# App & Page Settings
# -----------------------
st.set_page_config(
    page_title="⚙️ Advanced Inventory Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("⚙️ Advanced Inventory Management Dashboard")

# -----------------------
# Session State for File
# -----------------------
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "file_ts" not in st.session_state:
    st.session_state.file_ts = None

# -----------------------
# File Upload / Persistence
# -----------------------
uploaded = st.sidebar.file_uploader("Upload master .xlsx", type="xlsx")
if uploaded:
    # Delete old file
    old_path = st.session_state.file_path
    if old_path and Path(old_path).exists():
        try:
            Path(old_path).unlink()
        except Exception as err:
            logger.error(f"Failed to delete old file: {err}")

    # Save new file to temp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"inv_upload_{timestamp}_{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.session_state.file_path = str(tmp_path)
    st.session_state.file_ts = datetime.now()

# Ensure a file is present
if not st.session_state.file_path:
    st.sidebar.warning("Please upload your master .xlsx to begin.")
    st.stop()

# Upload Info Panel
with st.sidebar.expander("Upload Info", expanded=False):
    st.markdown(f"**File:** {Path(st.session_state.file_path).name}")
    st.markdown(
        f"**Uploaded:** {st.session_state.file_ts.strftime('%Y-%m-%d %H:%M:%S')}"
    )

# -----------------------
# Data Loading & Caching
# -----------------------
@st.cache_data(show_spinner=False)
def load_everything(file_path: str):
    logger.info(f"Loading sheets from {file_path}")
    try:
        with open(file_path, 'rb') as f:
            sheets = load_sheets(f)
    except Exception as e:
        raise RuntimeError(f"Error loading Excel sheets: {e}")

    # Main pipeline
    sales_df, inv_df, prod_df, cost_df = preprocess_data(
        sheets.get("Sales History"),
        sheets.get("Inventory Detail"),
        sheets.get("Production Batch"),
        sheets.get("Cost Value"),
    )
    agg_sales = aggregate_sales_history(sales_df)
    merged = merge_data(agg_sales, inv_df, prod_df, cost_df)
    df_woh = aggregate_final_data(merged, sales_df)

    # Holding-cost pipeline
    snap = process_inventory_snapshot(sheets.get("Inventory Detail"))
    df_hc = compute_holding_cost(snap)

    # Additional sheets
    inv1_df = process_inventory_detail1(sheets.get("Inventory Detail1", pd.DataFrame()))
    mikuni_df = sheets.get("Mikuni", pd.DataFrame())

    return df_woh, df_hc, sales_df, inv1_df, mikuni_df, cost_df, prod_df

# Show loading spinner
with st.spinner("Processing inventory data..."):
    df_woh, df_hc, sales_df, inv1_df, mikuni_df, cost_df, prod_df = load_everything(
        st.session_state.file_path
    )

# -----------------------
# Post-processing & Protein Merge
# -----------------------
# Ensure SKU is string
prod_df["SKU"] = prod_df.get("SKU", pd.Series(dtype=str)).astype(str)
df_woh["SKU"] = df_woh.get("SKU", pd.Series(dtype=str)).astype(str)

# Merge Protein if present
if "Protein" in prod_df.columns:
    df_woh = df_woh.merge(
        prod_df[["SKU", "Protein"]], on="SKU", how="left"
    )
    df_woh["Protein"] = df_woh["Protein"].fillna("Unknown")
else:
    logger.warning("'Protein' column missing; defaulting to 'Unknown'")
    df_woh["Protein"] = "Unknown"

# -----------------------
# Download Processed Data
# -----------------------
st.sidebar.download_button(
    label="Download KPI Data (CSV)",
    data=df_woh.to_csv(index=False).encode("utf-8"),
    file_name="processed_inventory_kpis.csv",
    mime="text/csv",
)

# -----------------------
# Filters
# -----------------------
prot_opts = ["All"] + sorted(df_woh["Protein"].unique())
state_opts = ["All"] + sorted(df_woh.get("ProductState", pd.Series(dtype=str)).unique())
sku_opts = ["All"] + sorted(df_woh.get("SKU_Desc", pd.Series(dtype=str)).unique())

f_p = st.sidebar.selectbox("Protein", prot_opts)
f_s = st.sidebar.selectbox("State", state_opts)
f_k = st.sidebar.selectbox("SKU Desc", sku_opts)

condition = (
    ((f_p == "All") | (df_woh["Protein"] == f_p))
    & ((f_s == "All") | (df_woh["ProductState"] == f_s))
    & ((f_k == "All") | (df_woh["SKU_Desc"] == f_k))
)
filtered = df_woh[condition]

# -----------------------
# Section Navigation
# -----------------------
section = st.sidebar.radio(
    "Select Section",
    ["📈 KPIs","📊 WOH","🚀 Movers","💰 Holding Cost","🔎 Insights","🗺 Bin Scan"],
)

# -----------------------
# Chart Theming
# -----------------------
def apply_theme(chart):
    return chart.configure_axis(labelFontSize=12, titleFontSize=14).configure_legend(
        labelFontSize=12, titleFontSize=14
    )

# -----------------------
# Render Selected Section
# -----------------------
if section == "📈 KPIs":
    kpis.render(filtered, df_hc, apply_theme)
elif section == "📊 WOH":
    woh.render(filtered, df_hc, cost_df, apply_theme)
elif section == "🚀 Movers":
    movers.render(filtered, apply_theme)
elif section == "💰 Holding Cost":
    holding_cost.render(df_hc, apply_theme)
elif section == "🔎 Insights":
    insights.render(filtered, df_hc, apply_theme)
elif section == "🗺 Bin Scan":
    bin_scan.render(inv1_df, mikuni_df, apply_theme)
