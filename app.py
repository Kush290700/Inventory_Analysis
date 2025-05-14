import io
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

# ─── Theme helper ────────────────────────────────────────────
def apply_theme(chart):
    return (
        chart
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_legend(labelFontSize=12, titleFontSize=14)
    )

# ──────────────────────────────────────────────────────────────
# 1) Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚙️ Advanced Inventory Dashboard",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# 2) Load & preprocess (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_everything(uploaded_xlsx):
    # 1) read all sheets
    sheets = load_sheets(uploaded_xlsx)

    # 2) original four-sheet pipeline
    sales_df, inv_df, prod_df, cost_df = preprocess_data(
        sheets["Sales History"],
        sheets["Inventory Detail"],
        sheets["Production Batch"],
        sheets["Cost Value"],
    )
    agg_sales = aggregate_sales_history(sales_df)
    merged    = merge_data(agg_sales, inv_df, prod_df, cost_df)
    df_woh    = aggregate_final_data(merged, sales_df)

    # 3) holding-cost pipeline
    snap     = process_inventory_snapshot(sheets["Inventory Detail"])
    df_hc    = compute_holding_cost(snap)

    # 4) new “Inventory Detail1” pipeline
    inv1_df  = process_inventory_detail1(sheets["Inventory Detail1"])

    # 5) raw “Mikuni” sheet
    mikuni_df = sheets.get("Mikuni", pd.DataFrame())

    return df_woh, df_hc, sales_df, inv1_df, mikuni_df

# ──────────────────────────────────────────────────────────────
# 3) Sidebar — uploader + filters + nav
# ──────────────────────────────────────────────────────────────
st.title("⚙️ Advanced Inventory Management Dashboard")
raw_file = st.sidebar.file_uploader("Upload master .xlsx", type="xlsx")
if not raw_file:
    st.sidebar.warning("Please upload your master .xlsx to begin.")
    st.stop()

df_woh, df_hc, sales_df, inv1_df, mikuni_df = load_everything(raw_file)

# unified filters for WOH‐based tabs
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

# ──────────────────────────────────────────────────────────────
# 4) Section picker (including “Bin Scan”)
# ──────────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Select Section",
    [
        "📈 KPIs",
        "📊 WOH",
        "🚀 Movers",
        "💰 Holding Cost",
        "🔎 Insights",
        "🗺 Bin Scan",
    ],
)

# ──────────────────────────────────────────────────────────────
# 5) Dispatch to the appropriate tab
# ──────────────────────────────────────────────────────────────
if section == "📈 KPIs":
    kpis.render(df, df_hc, apply_theme)
elif section == "📊 WOH":
    woh.render(df, df_hc, apply_theme)
elif section == "🚀 Movers":
    movers.render(df, apply_theme)
elif section == "💰 Holding Cost":
    holding_cost.render(df_hc, apply_theme)
elif section == "🔎 Insights":
    insights.render(df, df_hc, apply_theme)
elif section == "🗺 Bin Scan":
    # now passing both Inventory Detail1 + Mikuni
    bin_scan.render(inv1_df, mikuni_df, apply_theme)
