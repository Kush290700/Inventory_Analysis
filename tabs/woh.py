import io
import os
import sys
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import compute_threshold_move
from datetime import datetime, timedelta

# ==== DELIVERY SCHEDULE ====
SCHEDULE_ROWS = [
    {"supplier": "RDFA",  "cutoff_day": 0, "cutoff_time": "10:00", "delivery_day": 2, "notes": "Rossdown Chicken/Conv Turkey, delivers Wed."},
    {"supplier": "RDFA",  "cutoff_day": 0, "cutoff_time": "10:00", "delivery_day": 4, "notes": "Rossdown RWA Turkey, delivers Fri."},
    {"supplier": "PROD",  "cutoff_day": 0, "cutoff_time": "10:00", "delivery_day": None, "notes": "RTE Production, broths, 1-1.5w turnaround"},
    {"supplier": "YAMF",  "cutoff_day": 0, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Yarrow Duck, delivers Wed."},
    {"supplier": "BRCP",  "cutoff_day": 0, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Britco Pork/Donald's Fine Foods, delivers Wed."},
    {"supplier": "NDSF",  "cutoff_day": 0, "cutoff_time": "12:00", "delivery_day": 2, "notes": "North Delta Seafood, picked up Wed."},
    {"supplier": "BFMS",  "cutoff_day": 0, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Black Forest, delivers Tue."},
    {"supplier": "GLDV",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 4, "notes": "Golden Valley Eggs, delivers Fri."},
    {"supplier": "FRVP",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 3, "notes": "Johnston's Pork, delivers Thu."},
    {"supplier": "FRVP",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 0, "notes": "Johnston's Bacon/Ham prebook, delivers following Tue."},
    {"supplier": "YAMF",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 3, "notes": "Chicken (Retail & Food Service), delivers Thu."},
    {"supplier": "HALM",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 3, "notes": "Chicken (Retail & Food Service), delivers Thu."},
    {"supplier": "TRSM",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 3, "notes": "Bosa Charcuterie, delivers Wed."},
    {"supplier": "RDFA",  "cutoff_day": 2, "cutoff_time": "10:00", "delivery_day": 4, "notes": "Rossdown Chicken, delivers Fri."},
    {"supplier": "ABBI",  "cutoff_day": 2, "cutoff_time": "10:00", "delivery_day": 2, "notes": "Rangeland Bison, delivers two weeks from Mon."},
    {"supplier": "PROD",  "cutoff_day": 2, "cutoff_time": "10:00", "delivery_day": None, "notes": "RTE Production, ships Fri. or Tue/Wed following week"},
    {"supplier": "NDSF",  "cutoff_day": 2, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Organic Ocean Seafood, delivers Fri."},
    {"supplier": "BFMS",  "cutoff_day": 2, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Black Forest, delivers next Tue."},
    {"supplier": "FRVL",  "cutoff_day": 2, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Tappen Lamb, delivers next Wed."},
    {"supplier": "GDRA",  "cutoff_day": 3, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Gindara FROZEN Sablefish, picked up Wed."},
    {"supplier": "GDRA",  "cutoff_day": 3, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Gindara FRESH Sablefish, picked up Wed."},
    {"supplier": "FRVP",  "cutoff_day": 3, "cutoff_time": "12:00", "delivery_day": 0, "notes": "Johnston's Pork, delivers Mon."},
    {"supplier": "MISC",  "cutoff_day": 3, "cutoff_time": "13:00", "delivery_day": None, "notes": "Turkey Thigh for PROD, delivery varies"},
    {"supplier": "FRVP",  "cutoff_day": 4, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Johnston's Pork, delivers Tue."},
    {"supplier": "TRSM",  "cutoff_day": 4, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Arctic Deli (whole), delivers Tue."},
    {"supplier": "YAMF",  "cutoff_day": 4, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Chicken (Retail & Food Service), delivers Tue."},
    {"supplier": "HALM",  "cutoff_day": 4, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Chicken (Retail & Food Service), delivers Tue."},
    {"supplier": "BRAD",  "cutoff_day": 4, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Bradner Org. Beef Trim for PROD, picked up Wed."},
    {"supplier": "BRAD",  "cutoff_day": 4, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Bradner Org. Chicken, picked up Wed."},
    {"supplier": "Imperial Dade", "cutoff_day": 0, "cutoff_time": "15:00", "delivery_day": 1, "notes": "Boxes & WH supplies, delivers next day"},
    {"supplier": "Imperial Dade", "cutoff_day": 2, "cutoff_time": "15:00", "delivery_day": 3, "notes": "Boxes & WH supplies, delivers next day"},
    {"supplier": "Imperial Dade", "cutoff_day": 4, "cutoff_time": "15:00", "delivery_day": 0, "notes": "Boxes & WH supplies, delivers next day"},
    {"supplier": "Uline",         "cutoff_day": 0, "cutoff_time": "15:00", "delivery_day": 1, "notes": "Retail boxes & WH supplies, delivers next day"},
    {"supplier": "Uline",         "cutoff_day": 2, "cutoff_time": "15:00", "delivery_day": 3, "notes": "Retail boxes & WH supplies, delivers next day"},
    {"supplier": "Uline",         "cutoff_day": 4, "cutoff_time": "15:00", "delivery_day": 0, "notes": "Retail boxes & WH supplies, delivers next day"},
    {"supplier": "PRAN",  "cutoff_day": 0, "cutoff_time": "10:00", "delivery_day": 0, "notes": "Prairie Ranchers Pork, delivers following Mon."},
    {"supplier": "PCCL",  "cutoff_day": 0, "cutoff_time": "12:00", "delivery_day": 2, "notes": "Westfine Lamb, delivers in two weeks"},
    {"supplier": "BRLW",  "cutoff_day": 0, "cutoff_time": "15:00", "delivery_day": None, "notes": "Wagyu Beef, ~1w frozen, 2w fresh"},
    {"supplier": "63AC",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 1, "notes": "Beef (tagged), delivers following Tue."},
    {"supplier": "63AC",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": 3, "notes": "Beef (fill gaps), delivers Thu."},
    {"supplier": "CMDB",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": None, "notes": "Beef (Meadow Valley), delivery timeline varies, ~1w"},
    {"supplier": "CMDB",  "cutoff_day": 1, "cutoff_time": "12:00", "delivery_day": None, "notes": "Beef (Hardy), delivery timeline varies, ~1w"},
    {"supplier": "Sharp Base", "cutoff_day": 2, "cutoff_time": "10:00", "delivery_day": 3, "notes": "Offsite Storage, picked up next day"},
    {"supplier": "IMPL",  "cutoff_day": 2, "cutoff_time": "13:00", "delivery_day": None, "notes": "Lamb & Venison Trim, 1pm cut off"},
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def get_next_delivery(supplier, now=None):
    now = now or datetime.now()
    today_idx = now.weekday()  # Monday=0
    supplier = (supplier or "").strip().upper()
    slots = [row for row in SCHEDULE_ROWS if row["supplier"].strip().upper() == supplier]
    if not slots:
        return {"found": False, "reason": "No delivery info for supplier"}

    soonest = None
    soonest_delta = None
    for row in slots:
        cutoff_day_idx = row["cutoff_day"]
        cutoff_time = datetime.strptime(row["cutoff_time"], "%H:%M").time()
        days_until_cutoff = (cutoff_day_idx - today_idx) % 7
        cutoff_dt = datetime.combine(now.date(), cutoff_time) + timedelta(days=days_until_cutoff)
        if days_until_cutoff == 0 and now.time() > cutoff_time:
            cutoff_dt += timedelta(days=7)
        delivery_date = None
        if row["delivery_day"] is not None:
            delivery_days = (row["delivery_day"] - cutoff_day_idx) % 7
            delivery_date = cutoff_dt.date() + timedelta(days=delivery_days)
        delta = (cutoff_dt - now).total_seconds()
        if soonest is None or delta < soonest_delta:
            soonest = {
                "cutoff_datetime": cutoff_dt,
                "cutoff_day": WEEKDAYS[cutoff_day_idx],
                "cutoff_time": row["cutoff_time"],
                "delivery_date": delivery_date,
                "delivery_day": WEEKDAYS[row["delivery_day"]] if row["delivery_day"] is not None else None,
                "notes": row["notes"],
                "found": True
            }
            soonest_delta = delta
    return soonest


# ------------------- UTILITY FUNCTIONS ------------------- #

def clean_sku(x):
    if pd.isnull(x): return ""
    x = str(x).strip()
    if x.endswith(".0"): x = x[:-2]
    return x

def calculate_weeks_in_data(sales_df):
    """Return number of weeks between min/max date in sales."""
    if sales_df.empty or "DateExpected" not in sales_df:
        return 4  # fallback
    dates = pd.to_datetime(sales_df["DateExpected"], errors="coerce")
    min_d, max_d = dates.min(), dates.max()
    if pd.isnull(min_d) or pd.isnull(max_d):
        return 4
    return max(1, ((max_d - min_d).days + 1) // 7)

# ------------------- AGGREGATION FUNCTION ------------------- #

def aggregate_data(sheets, weeks_override=None):
    # Sheets
    inv_df = sheets.get('Inventory Detail', pd.DataFrame()).copy()
    sales_df = sheets.get('Sales History', pd.DataFrame()).copy()
    prod_df = sheets.get('Production Batch', pd.DataFrame()).copy()
    prod_detail = sheets.get('Product Detail', pd.DataFrame()).copy()
    cost_val = sheets.get('Cost Value', pd.DataFrame()).copy()

    # Clean up
    inv_df = inv_df.rename(columns={'SKU': 'SKU_Full'})
    inv_df['SKU'] = inv_df['SKU_Full'].str.extract(r'(\d+)').fillna(inv_df['SKU_Full'])
    inv_df['SKU'] = inv_df['SKU'].map(clean_sku)
    inv_df['ProductState'] = inv_df.get('ProductState', '').str.upper().fillna('')
    inv_df['ProductName'] = inv_df.get('SKU_Full', '').astype(str)
    inv_df['WeightLb'] = pd.to_numeric(inv_df['WeightLb'], errors='coerce').fillna(0)
    inv_df['CostValue'] = pd.to_numeric(inv_df['CostValue'], errors='coerce').fillna(0)

    # Sales
    sales_df['SKU'] = sales_df['SKU'].map(clean_sku)
    sales_df['ShippedLb'] = pd.to_numeric(sales_df['ShippedLb'], errors='coerce').fillna(0)
    sales_df['QuantityOrdered'] = pd.to_numeric(sales_df['QuantityOrdered'], errors='coerce').fillna(0)
    sales_df['Cost'] = pd.to_numeric(sales_df['Cost'], errors='coerce').fillna(0)
    sales_df['Rev'] = pd.to_numeric(sales_df['Rev'], errors='coerce').fillna(0)

    # Production
    prod_df['SKU'] = prod_df['SKU'].map(clean_sku)
    if 'ProductionShippedLb' in prod_df:
        prod_df['ProductionShippedLb'] = pd.to_numeric(prod_df['ProductionShippedLb'], errors='coerce').fillna(0.0)
    else:
        prod_df['ProductionShippedLb'] = pd.to_numeric(prod_df.get('WeightLb', 0), errors='coerce').fillna(0.0)

    # Product Detail
    prod_detail['SKU'] = prod_detail['Product Code'].map(clean_sku)
    prod_detail['ParentSKU'] = prod_detail['Velocity Parent'].map(clean_sku)
    prod_detail['SKU_Desc'] = prod_detail['Description'].fillna("").astype(str)

    # Cost/Pack Info
    cost_val['SKU'] = cost_val['SKU'].map(clean_sku)
    cost_val['NumPacks'] = pd.to_numeric(cost_val['NumPacks'], errors='coerce')
    cost_val['WeightLb'] = pd.to_numeric(cost_val['WeightLb'], errors='coerce')
    cost_val['PackSize'] = np.where(
        cost_val['NumPacks'] > 0, cost_val['WeightLb'] / cost_val['NumPacks'], np.nan
    )
    packsize_map = dict(zip(cost_val['SKU'], cost_val['PackSize']))

    # ------------------ AGGREGATE SALES ------------------
    agg_sales = (
        sales_df
        .groupby(["SKU", "Supplier", "Protein", "Description"], as_index=False)
        .agg(
            ShippedLb=("ShippedLb", "sum"),
            QuantityOrdered=("QuantityOrdered", "sum"),
            Cost=("Cost", "sum"),
            Rev=("Rev", "sum")
        )
    )

    # ------------------ AGGREGATE INVENTORY ------------------
    inv_agg = (
        inv_df
        .groupby(["SKU", "ProductState", "ProductName"], as_index=False)
        .agg(
            OnHandWeightLb=("WeightLb", "sum"),
            OnHandCost=("CostValue", "sum")
        )
    )

    # Supplier map from Production
    supplier_map = {}
    if "Supplier" in prod_df.columns:
        supplier_map = dict(zip(prod_df['SKU'], prod_df['Supplier']))

    # ------------------ MERGE DATA ------------------
    df = inv_agg.merge(agg_sales, on="SKU", how="left")
    df = df.fillna({
        "Supplier": "",
        "Protein": "",
        "Description": "",
        "ShippedLb": 0.0,
        "QuantityOrdered": 0,
        "Cost": 0.0,
        "Rev": 0.0
    })
    mask_blank = df["Supplier"].astype(str).str.strip() == ""
    df.loc[mask_blank, "Supplier"] = (
        df.loc[mask_blank, "SKU"].map(supplier_map).fillna("")
    )
    df = df.merge(
        prod_df[["SKU", "ProductionShippedLb"]],
        on="SKU",
        how="left"
    )
    df["ProductionShippedLb"] = df["ProductionShippedLb"].fillna(0.0)

    # ------------------ FINAL AGGREGATION ------------------
    try:
        weeks_span = weeks_override or calculate_weeks_in_data(sales_df)
        if weeks_span <= 0: weeks_span = 4
    except Exception:
        weeks_span = 4

    sku_stats = (
        df
        .groupby(
            ["SKU", "Supplier", "Protein", "Description", "ProductState", "ProductName"],
            as_index=False
        )
        .agg(
            OnHandWeightTotal  = ("OnHandWeightLb",      "sum"),
            OnHandCostTotal    = ("OnHandCost",          "sum"),
            TotalShippedLb     = ("ShippedLb",           "sum"),
            TotalProductionLb  = ("ProductionShippedLb", "sum"),
            TotalRevenue       = ("Rev",                 "sum"),
            TotalCost          = ("Cost",                "sum")
        )
    )
    sku_stats["TotalUsage"]     = sku_stats["TotalShippedLb"] + sku_stats["TotalProductionLb"]
    sku_stats["AvgWeeklyUsage"] = sku_stats["TotalUsage"] / weeks_span
    sku_stats["WOH_Flag"] = np.where(sku_stats["TotalUsage"] == 0, "No Usage", "")
    sku_stats["WeeksOnHand"] = (
        sku_stats["OnHandWeightTotal"]
        / sku_stats["AvgWeeklyUsage"].replace({0: np.nan})
    )
    sku_stats["WeeksOnHand"] = sku_stats["WeeksOnHand"].replace([np.inf, -np.inf], np.nan)
    sku_stats["AnnualTurns"] = (
        sku_stats["AvgWeeklyUsage"] * 52.0
        / sku_stats["OnHandWeightTotal"].replace({0: np.nan})
    )
    sku_stats["AnnualTurns"] = sku_stats["AnnualTurns"].replace([np.inf, -np.inf], 0).fillna(0)
    sku_stats["SKU_Desc"] = (
        sku_stats["SKU"]
        + " – "
        + sku_stats["ProductName"].where(
            sku_stats["ProductName"].ne(""),
            sku_stats["Description"]
        )
    )
    # Add PackSize, ParentSKU
    sku_stats["PackSize"] = sku_stats["SKU"].map(packsize_map)
    desc_map = dict(zip(prod_detail['SKU'], prod_detail['SKU_Desc']))
    parent_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    sku_stats['SKU_Desc'] = sku_stats['SKU'].map(desc_map).fillna(sku_stats['SKU_Desc'])
    sku_stats['ParentSKU'] = sku_stats['SKU'].map(parent_map).fillna(sku_stats['SKU'])
    mask = (sku_stats["ParentSKU"].isin(["", "nan", "none", "null"])) | (sku_stats["ParentSKU"].isna())
    sku_stats.loc[mask, "ParentSKU"] = sku_stats.loc[mask, "SKU"]

    # Use Supplier from agg or product
    sku_stats['Supplier'] = sku_stats['Supplier'].replace("", "Unknown").fillna("Unknown")
    sku_stats['ProductState'] = sku_stats['ProductState'].fillna("").str.upper()

    return sku_stats, prod_detail, cost_val

def get_root_parent(sku, parent_map):
    """Follow the parent chain up to the root."""
    seen = set()
    while sku in parent_map and pd.notnull(parent_map[sku]) and parent_map[sku] != "" and parent_map[sku] != sku:
        if sku in seen:   # avoid infinite loop
            break
        seen.add(sku)
        sku = parent_map[sku]
    return sku

def get_best_description(group):
    """Choose the most common non-empty description in the group."""
    descs = group['SKU_Desc'].dropna().unique()
    if len(descs) == 0:
        return ""
    # Use the longest description or most common (your choice)
    return max(descs, key=lambda x: (len(x), x))  # longest, then alpha

def preprocess_for_parents(sku_stats, prod_detail):
    parent_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    desc_map = dict(zip(prod_detail['SKU'], prod_detail['SKU_Desc']))

    # Add TrueParent column by following chain up
    sku_stats['TrueParent'] = sku_stats['SKU'].apply(lambda x: get_root_parent(x, parent_map))
    # Add Description candidate for each SKU
    sku_stats['SKU_Desc'] = sku_stats['SKU'].map(desc_map).fillna(sku_stats['SKU_Desc'])

    # Now group by TrueParent and aggregate everything up
    agg = (
        sku_stats.groupby('TrueParent', as_index=False)
            .agg(
                MeanUse=('AvgWeeklyUsage', 'sum'),
                InvWt=('OnHandWeightTotal', 'sum'),
                InvCost=('OnHandCostTotal', 'sum'),
                Supplier=('Supplier', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
                Protein=('Protein', lambda x: x.mode()[0] if not x.mode().empty else ""),
                PacksToOrder=('PacksToOrder', 'sum') if 'PacksToOrder' in sku_stats.columns else (lambda x: 0),
                OrderWt=('OrderWt', 'sum') if 'OrderWt' in sku_stats.columns else (lambda x: 0),
                EstCost=('EstCost', 'sum') if 'EstCost' in sku_stats.columns else (lambda x: 0),
            )
    )
    # Get the best description for each root parent
    desc_lookup = sku_stats.groupby('TrueParent').apply(get_best_description)
    agg['SKU_Desc'] = agg['TrueParent'].map(desc_lookup)

    agg['SKU'] = agg['TrueParent']  # For consistency downstream
    return agg

# ------------------- PARENT PURCHASE PLAN ------------------- #

@st.cache_data(show_spinner=False)
def compute_parent_purchase_plan(sku_stats, prod_detail, cost_val, desired_woh):
    # Use same logic as before for parent mapping, now with best-agg'd metrics
    child_to_parent = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    parent_desc_map = dict(zip(prod_detail['ParentSKU'], prod_detail['Description'].fillna("").astype(str)))
    sku_stats['ParentSKU'] = sku_stats['SKU'].map(child_to_parent).fillna(sku_stats['SKU'])
    mask = (sku_stats["ParentSKU"].isin(["", "nan", "none", "null"])) | (sku_stats["ParentSKU"].isna())
    sku_stats.loc[mask, "ParentSKU"] = sku_stats.loc[mask, "SKU"]

    parent_stats = (
        sku_stats.groupby("ParentSKU", as_index=False)
            .agg(
                MeanUse=('AvgWeeklyUsage', 'sum'),
                InvWt=('OnHandWeightTotal', 'sum'),
                InvCost=('OnHandCostTotal', 'sum'),
                Supplier=('Supplier', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
            )
    )
    parent_stats['DesiredWt'] = parent_stats['MeanUse'] * desired_woh
    parent_stats['ToBuyWt'] = (parent_stats['DesiredWt'] - parent_stats['InvWt']).clip(lower=0)
    packsize_map = sku_stats.groupby('ParentSKU')['PackSize'].median()
    global_packsize = sku_stats['PackSize'].dropna().mean()
    parent_stats['PackSize'] = parent_stats['ParentSKU'].map(packsize_map).fillna(global_packsize if not np.isnan(global_packsize) else 1.0)
    parent_stats['PacksToOrder'] = np.where(
        parent_stats['PackSize'] > 0, np.ceil(parent_stats['ToBuyWt'] / parent_stats['PackSize']), 0
    ).astype(int)
    parent_stats['OrderWt'] = parent_stats['PacksToOrder'] * parent_stats['PackSize']
    parent_stats['SKU'] = parent_stats['ParentSKU']
    parent_stats['SKU_Desc'] = parent_stats['SKU'].map(parent_desc_map).fillna(parent_stats['SKU'])
    parent_stats['CostPerLb'] = np.where(
        parent_stats['InvWt'] > 0,
        parent_stats['InvCost'] / parent_stats['InvWt'],
        0
    )
    parent_stats['EstCost'] = parent_stats['OrderWt'] * parent_stats['CostPerLb']
    plan = parent_stats[parent_stats['PacksToOrder'] > 0][
        ["SKU", "SKU_Desc", "Supplier", "InvWt", "DesiredWt", "PackSize", "PacksToOrder", "OrderWt", "EstCost"]
    ].copy()
    plan.reset_index(drop=True, inplace=True)
    return plan

# ------------------- MAIN STREAMLIT TAB ------------------- #

def woh_tab(sheets, theme):
    st.header("📦 Advanced Inventory Weeks-On-Hand (WOH) Dashboard")
    sku_stats, prod_detail, cost_val = aggregate_data(sheets)

    # Tabs for each workflow
    tab1, tab2, tab3 = st.tabs(["FZ → EXT Transfer", "EXT → FZ Transfer", "Parent Purchase Plan"])

    # --- FZ → EXT Transfer ---
    with tab1:
        st.subheader("🔄 Move FZ → EXT")
        fz = sku_stats[(sku_stats['ProductState'].str.startswith('FZ')) & (sku_stats['AvgWeeklyUsage'] > 0)].copy()
        ext = sku_stats[(sku_stats['ProductState'].str.startswith('EXT')) & (sku_stats['AvgWeeklyUsage'] > 0)].copy()
        thr1 = st.slider("Desired FZ WOH (weeks)", 0.0, 4.0, 1.5, 0.25)
        fz['DesiredFZ_Weight'] = fz['AvgWeeklyUsage'] * thr1
        fz['WeightToMove'] = (fz['OnHandWeightTotal'] - fz['DesiredFZ_Weight']).clip(lower=0)
        ext_weight_lookup = ext.set_index("SKU")["OnHandWeightTotal"]
        fz['EXT_Weight'] = fz['SKU'].map(ext_weight_lookup).fillna(0)
        fz['TotalOnHand'] = fz['OnHandWeightTotal'] + fz['EXT_Weight']
        move = fz[fz['WeightToMove'] > 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("SKUs to Move", move["SKU"].nunique())
        c2.metric("Total Weight to Move", f"{move['WeightToMove'].sum():,.0f} lb")
        c3.metric("Total Cost to Move", f"${((move['WeightToMove'] / move['OnHandWeightTotal']) * move['OnHandCostTotal']).sum():,.0f}")
        # Download
        if not move.empty:
            buf1 = io.BytesIO()
            move_cols = ['SKU', 'SKU_Desc', 'Supplier', 'ProductState', 'OnHandWeightTotal', 'AvgWeeklyUsage', 'WeeksOnHand', 'DesiredFZ_Weight', 'WeightToMove', 'EXT_Weight', 'TotalOnHand']
            move[move_cols].to_excel(buf1, index=False, sheet_name="FZ2EXT")
            buf1.seek(0)
            st.download_button("Download FZ→EXT List", buf1.getvalue(), file_name="FZ2EXT_list.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # Chart
        if not move.empty:
            chart = alt.Chart(move).mark_bar().encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToMove:Q", title="Weight to Move (lb)"),
                color="Supplier:N",
                tooltip=["SKU_Desc", "Supplier", "WeightToMove", "WeeksOnHand"]
            ).properties(height=alt.Step(25))
            st.altair_chart(theme(chart).interactive(), use_container_width=True)

    # --- EXT → FZ Transfer ---
    with tab2:
        st.subheader("🔄 Move EXT → FZ")
        fz_woh = sku_stats[sku_stats['ProductState'].str.startswith('FZ')].set_index('SKU')['WeeksOnHand']
        ext = sku_stats[(sku_stats['ProductState'].str.startswith('EXT')) & (sku_stats['AvgWeeklyUsage'] > 0)].copy()
        thr2_default = 1.0
        try:
            thr2_default = float(compute_threshold_move(ext, None))
        except Exception:
            pass
        thr2 = st.slider("Desired FZ WOH to achieve", 0.0, float(fz_woh.max() if not fz_woh.empty else 52.0), thr2_default, step=0.25)
        ext['FZ_Weight'] = ext['SKU'].map(fz_woh).fillna(0)
        ext['DesiredFZ_Weight'] = ext['AvgWeeklyUsage'] * thr2
        ext['WeightToReturn'] = ext['DesiredFZ_Weight'].sub(ext['FZ_Weight']).clip(lower=0)
        ext['TotalOnHand'] = ext['OnHandWeightTotal'] + ext['FZ_Weight']
        back = ext[ext['WeightToReturn'] > 0]
        col1, col2, col3 = st.columns(3)
        col1.metric("SKUs to Return", back["SKU"].nunique())
        col2.metric("Total Weight to Return", f"{back['WeightToReturn'].sum():,.0f} lb")
        col3.metric("Total Cost to Return", f"${((back['WeightToReturn'] / back['OnHandWeightTotal']) * back['OnHandCostTotal']).sum():,.0f}")
        # Download
        if not back.empty:
            buf2 = io.BytesIO()
            back_cols = ['SKU', 'SKU_Desc', 'Supplier', 'ProductState', 'OnHandWeightTotal', 'AvgWeeklyUsage', 'WeeksOnHand', 'DesiredFZ_Weight', 'WeightToReturn', 'FZ_Weight', 'TotalOnHand']
            back[back_cols].to_excel(buf2, index=False, sheet_name="EXT2FZ")
            buf2.seek(0)
            st.download_button("Download EXT→FZ List", buf2.getvalue(), file_name="EXT2FZ_list.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # Chart
        if not back.empty:
            chart2 = alt.Chart(back).mark_bar().encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToReturn:Q", title="Weight to Return (lb)"),
                color="Supplier:N",
                tooltip=["SKU_Desc", "Supplier", "WeightToReturn", "WeeksOnHand"]
            ).properties(height=alt.Step(25))
            st.altair_chart(theme(chart2).interactive(), use_container_width=True)

    with tab3:
        st.subheader("🛒 Usage-Based Parent Purchase Plan with Delivery Intelligence")
    
        # --- Desired Weeks-On-Hand (WOH) Slider ---
        desired_woh = st.slider("Desired Weeks On Hand (WOH) for Parent Purchase Plan", 0.0, 12.0, 4.0, 0.5)
    
        # --- Preprocess Parent Chains and Descriptions ---
        parent_plan = preprocess_for_parents(sku_stats, prod_detail)
    
        # --- Calculate Purchase Plan Fields ---
        parent_plan['DesiredWt'] = parent_plan['MeanUse'] * desired_woh
        parent_plan['ToBuyWt'] = (parent_plan['DesiredWt'] - parent_plan['InvWt']).clip(lower=0)
    
        # Calculate pack size (use median from all SKUs mapping to this parent)
        packsize_map = sku_stats.groupby('TrueParent')['PackSize'].median()
        global_packsize = sku_stats['PackSize'].dropna().mean()
        parent_plan['PackSize'] = parent_plan['SKU'].map(packsize_map).fillna(global_packsize if not np.isnan(global_packsize) else 1.0)
    
        parent_plan['PacksToOrder'] = np.where(
            parent_plan['PackSize'] > 0, np.ceil(parent_plan['ToBuyWt'] / parent_plan['PackSize']), 0
        ).astype(int)
        parent_plan['OrderWt'] = parent_plan['PacksToOrder'] * parent_plan['PackSize']
    
        parent_plan['CostPerLb'] = np.where(
            parent_plan['InvWt'] > 0,
            parent_plan['InvCost'] / parent_plan['InvWt'],
            0
        )
        parent_plan['EstCost'] = parent_plan['OrderWt'] * parent_plan['CostPerLb']
    
        # --- Special Order SO01 Flag ---
        parent_plan["SpecialOrderFlag"] = parent_plan["SKU"].astype(str).str.contains("SO01", na=False) | parent_plan["SKU_Desc"].astype(str).str.contains("SO01", na=False)
        parent_plan["SO Note"] = np.where(parent_plan["SpecialOrderFlag"], "SO01 Special Order", "")
    
        # --- Delivery Info ---
        delivery_info = parent_plan["Supplier"].apply(lambda s: get_next_delivery(s))
        parent_plan["NextCutoff"] = delivery_info.apply(lambda d: d["cutoff_datetime"].strftime("%a %H:%M") if d["found"] else "")
        parent_plan["NextDelivery"] = delivery_info.apply(lambda d: d["delivery_date"].strftime("%Y-%m-%d") if d["found"] and d["delivery_date"] else "")
        parent_plan["DeliveryNotes"] = delivery_info.apply(lambda d: d["notes"] if d["found"] else d.get("reason", ""))
    
        # --- Filters (Supplier/Protein) ---
        all_suppliers = sorted(parent_plan["Supplier"].dropna().unique())
        all_proteins = sorted(parent_plan["Protein"].dropna().unique())
        selected_supplier = st.selectbox("Supplier Filter", ["All"] + all_suppliers)
        selected_protein = st.selectbox("Protein Filter", ["All"] + all_proteins)
        plan_df = parent_plan.copy()
        if selected_supplier != "All":
            plan_df = plan_df[plan_df["Supplier"] == selected_supplier]
        if selected_protein != "All":
            plan_df = plan_df[plan_df["Protein"] == selected_protein]
    
        # --- Only show those with packs to order
        plan_df = plan_df[plan_df['PacksToOrder'] > 0]
    
        display_cols = [
            "SKU", "SKU_Desc", "Supplier", "Protein", "InvWt", "DesiredWt",
            "PackSize", "PacksToOrder", "OrderWt", "EstCost",
            "SO Note", "NextCutoff", "NextDelivery", "DeliveryNotes"
        ]
        display = plan_df[display_cols].copy()
        display["InvWt"] = display["InvWt"].map("{:,.0f} lb".format)
        display["DesiredWt"] = display["DesiredWt"].map("{:,.0f} lb".format)
        display["PackSize"] = display["PackSize"].map("{:,.0f} lb".format)
        display["OrderWt"] = display["OrderWt"].map("{:,.0f} lb".format)
        display["EstCost"] = display["EstCost"].map("${:,.2f}".format)
    
        st.dataframe(display, use_container_width=True)
    
        buf = io.BytesIO()
        display.to_excel(buf, index=False, sheet_name="PurchasePlan")
        buf.seek(0)
        st.download_button("📥 Download Purchase Plan", data=buf.getvalue(), file_name="Purchase_Plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
        supplier_summary = (
            plan_df.groupby("Supplier", as_index=False)["EstCost"].sum()
            .sort_values("EstCost", ascending=False)
            .head(10)
        )
        if not supplier_summary.empty:
            pie = alt.Chart(supplier_summary).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="EstCost", type="quantitative", stack=True),
                color=alt.Color("Supplier:N", legend=alt.Legend(title="Supplier")),
                tooltip=[alt.Tooltip("Supplier:N"), alt.Tooltip("EstCost:Q", format="$.0f")]
            ).properties(title="Top Suppliers in Purchase Plan (by Estimated Cost)")
            st.altair_chart(pie, use_container_width=True)
