import io
import os
import tempfile
import uuid

import streamlit as st
import pandas as pd

from utils.io_utils       import load_sheets
from utils.cleaning       import (
    preprocess_data,
    process_inventory_snapshot,
    process_inventory_detail1,
)
from utils.aggregation    import (
    aggregate_sales_history,
    merge_data,
    aggregate_final_data,
)
from utils.costing        import compute_holding_cost
from tabs import (
    kpis,
    woh,
    movers,
    holding_cost,
    insights,
    bin_scan,
)

def apply_theme(chart):
    return (
        chart
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_legend(labelFontSize=12, titleFontSize=14)
    )

st.set_page_config(page_title="⚙️ Advanced Inventory Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# 1) Handle upload → save to tmp + cleanup previous
# -----------------------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload master .xlsx", type="xlsx", key="uploader")

if uploaded is not None:
    # delete old temp file if exists
    old_path = st.session_state.get("saved_file_path")
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass

    # write new file to a unique temp path
    tmp_dir   = tempfile.gettempdir()
    filename  = f"inventory_{uuid.uuid4().hex}.xlsx"
    tmp_path  = os.path.join(tmp_dir, filename)

    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.session_state["saved_file_path"] = tmp_path

# if nothing ever uploaded, halt
if "saved_file_path" not in st.session_state:
    st.sidebar.warning("Please upload your master .xlsx to begin.")
    st.stop()

# -----------------------------------------------------------------------------
# 2) Load from the saved temp file (not raw UploadedFile)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_everything_from_path(path: str):
    with open(path, "rb") as fh:
        sheets = load_sheets(fh)

    # original four-sheet pipeline
    sales_df, inv_df, prod_df, cost_df = preprocess_data(
        sheets["Sales History"],
        sheets["Inventory Detail"],
        sheets["Production Batch"],
        sheets["Cost Value"],
    )
    agg_sales = aggregate_sales_history(sales_df)
    merged    = merge_data(agg_sales, inv_df, prod_df, cost_df)
    df_woh    = aggregate_final_data(merged, sales_df)

    # holding-cost pipeline
    snap  = process_inventory_snapshot(sheets["Inventory Detail"])
    df_hc = compute_holding_cost(snap)

    # Inventory Detail1 + Mikuni
    inv1_df   = process_inventory_detail1(sheets["Inventory Detail1"])
    mikuni_df = sheets.get("Mikuni", pd.DataFrame())

    return df_woh, df_hc, sales_df, inv1_df, mikuni_df, cost_df, prod_df

# unpack
df_woh, df_hc, sales_df, inv1_df, mikuni_df, cost_df, prod_df = load_everything_from_path(
    st.session_state["saved_file_path"]
)

# -----------------------------------------------------------------------------
# 3) Merge Protein (with fall-back) + Filters + Tabs (unchanged)
# -----------------------------------------------------------------------------
# ensure typings
prod_df["SKU"] = prod_df["SKU"].astype(str)
df_woh["SKU"] = df_woh["SKU"].astype(str)

if "Protein" in prod_df.columns:
    df_woh = df_woh.merge(
        prod_df[["SKU","Protein"]],
        on="SKU", how="left", suffixes=(None, "_fromProd")
    )
    df_woh["Protein"] = df_woh["Protein"].fillna("Unknown")
else:
    df_woh["Protein"] = "Unknown"

prot_opts  = ["All"] + sorted(df_woh["Protein"].unique())
state_opts = ["All"] + sorted(df_woh["ProductState"].unique())
sku_opts   = ["All"] + sorted(df_woh["SKU_Desc"].unique())

f_p = st.sidebar.selectbox("Protein", prot_opts)
f_s = st.sidebar.selectbox("State",   state_opts)
f_k = st.sidebar.selectbox("SKU",     sku_opts)

mask = (
    ((f_p=="All") | (df_woh["Protein"]      == f_p)) &
    ((f_s=="All") | (df_woh["ProductState"] == f_s)) &
    ((f_k=="All") | (df_woh["SKU_Desc"]      == f_k))
)
df = df_woh[mask].copy()

section = st.sidebar.radio(
    "Select Section",
    ["📈 KPIs","📊 WOH","🚀 Movers","💰 Holding Cost","🔎 Insights","🗺 Bin Scan"],
)

if section == "📈 KPIs":
    kpis.render(df, df_hc, apply_theme)
elif section == "📊 WOH":
    woh.render(df, df_hc, cost_df, apply_theme)
elif section == "🚀 Movers":
    movers.render(df, apply_theme)
elif section == "💰 Holding Cost":
    holding_cost.render(df_hc, apply_theme)
elif section == "🔎 Insights":
    insights.render(df, df_hc, apply_theme)
elif section == "🗺 Bin Scan":
    bin_scan.render(inv1_df, mikuni_df, apply_theme)
