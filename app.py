# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path
t from datetime import datetime
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(page_title="⚙️ Advanced Inventory Dashboard", layout="wide")

# Utility: apply chart theme
def apply_theme(chart):
    return (
        chart
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_legend(labelFontSize=12, titleFontSize=14)
    )

# Persistent upload handling
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.temp_path = None

raw_file = st.sidebar.file_uploader("Upload master .xlsx", type="xlsx")
if raw_file:
    # save to temp file
    tmp_dir = Path(tempfile.gettempdir()) / "inventory_uploads"
    tmp_dir.mkdir(exist_ok=True)
    # delete old
    if st.session_state.temp_path and os.path.exists(st.session_state.temp_path):
        try:
            os.remove(st.session_state.temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete old upload: {e}")
    # write new
    tmp_path = tmp_dir / f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.xlsx"
    with open(tmp_path, "wb") as f:
        f.write(raw_file.getbuffer())
    st.session_state.uploaded_file = tmp_path
    st.session_state.temp_path = str(tmp_path)

if not st.session_state.uploaded_file:
    st.sidebar.warning("Please upload your master .xlsx to begin.")
    st.stop()

# Load & cache data
'tdata_key = str(st.session_state.temp_path)
@st.cache_data(show_spinner=False, ttl=3600)
def load_everything(path):
    try:
        with open(path, 'rb') as f:
            sheets = load_sheets(f)
    except Exception as e:
        raise RuntimeError(f"Error loading Excel sheets: {e}")

    # pipeline
    sales_df, inv_df, prod_df, cost_df = preprocess_data(
        sheets.get("Sales History", pd.DataFrame()),
        sheets.get("Inventory Detail", pd.DataFrame()),
        sheets.get("Production Batch", pd.DataFrame()),
        sheets.get("Cost Value", pd.DataFrame()),
    )
    agg_sales = aggregate_sales_history(sales_df)
    merged = merge_data(agg_sales, inv_df, prod_df, cost_df)
    df_woh = aggregate_final_data(merged, sales_df)

    snap = process_inventory_snapshot(sheets.get("Inventory Detail", pd.DataFrame()))
    df_hc = compute_holding_cost(snap)

    inv1_df = process_inventory_detail1(sheets.get("Inventory Detail1", pd.DataFrame()))
    mikuni_df = sheets.get("Mikuni", pd.DataFrame())

    return df_woh, df_hc, sales_df, inv1_df, mikuni_df, cost_df, prod_df

# Load data with spinner
with st.spinner("Processing inventory data..."):
    df_woh, df_hc, sales_df, inv1_df, mikuni_df, cost_df, prod_df = load_everything(st.session_state.uploaded_file)

# Merge Protein safely
df_woh['SKU'] = df_woh['SKU'].astype(str)
if 'SKU' in prod_df.columns:
    prod_df['SKU'] = prod_df['SKU'].astype(str)

if 'Protein' in prod_df.columns:
    df_woh = df_woh.drop(columns=['Protein'], errors='ignore')
    df_woh = df_woh.merge(prod_df[['SKU','Protein']], on='SKU', how='left')
    df_woh['Protein'] = df_woh['Protein'].fillna('Unknown')
else:
    logger.warning("'Protein' column missing in production data; defaulting to 'Unknown'")
    df_woh['Protein'] = 'Unknown'

# Sidebar filters
prot_opts = ['All'] + sorted(df_woh['Protein'].unique())
state_opts = ['All'] + sorted(df_woh.get('ProductState', pd.Series()).unique())
sku_opts = ['All'] + sorted(df_woh.get('SKU_Desc', pd.Series()).unique())

f_p = st.sidebar.selectbox("Protein", prot_opts)
f_s = st.sidebar.selectbox("State", state_opts)
f_k = st.sidebar.selectbox("SKU", sku_opts)

mask = (
    ((f_p == 'All') | (df_woh['Protein'] == f_p)) &
    ((f_s == 'All') | (df_woh.get('ProductState') == f_s)) &
    ((f_k == 'All') | (df_woh.get('SKU_Desc') == f_k))
)
filtered = df_woh[mask].copy()

# Section picker
section = st.sidebar.radio(
    "Select Section",
    ["📈 KPIs","📊 WOH","🚀 Movers","💰 Holding Cost","🔎 Insights","🗺 Bin Scan"],
    index=0
)

# Title
st.title("⚙️ Advanced Inventory Management Dashboard")

# Dispatch
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
