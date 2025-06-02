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
    if pd.isnull(x):
        return ""
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
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
    """
    Returns:
      sku_stats      → DataFrame with one row per (SKU, Supplier, Protein, Description, 
                         ProductState, ProductName), plus computed metrics, plus NumPacksOnHand
      prod_detail    → raw “Product Detail” (for lookups)
      cost_val       → raw “Cost Value” (unused for pack logic now)
      total_packs    → DataFrame {SKU, Inv_TotalPacks} (sum of packs on hand across all states)
      inv1_pack_sum  → DataFrame {SKU, InventoryDetail1_TotalPacks} (sum of distinct PackId1 in Inventory Detail1)
    """

    inv_df      = sheets.get('Inventory Detail', pd.DataFrame()).copy()
    sales_df    = sheets.get('Sales History', pd.DataFrame()).copy()
    prod_df     = sheets.get('Production Batch', pd.DataFrame()).copy()
    prod_detail = sheets.get('Product Detail', pd.DataFrame()).copy()
    inv1        = sheets.get('Inventory Detail1', pd.DataFrame()).copy()   # <— now using "Inventory Detail1"
    cost_val    = sheets.get('Cost Value', pd.DataFrame()).copy()          # still loaded but not used for packs

    #
    # ── 1) CLEAN & NORMALIZE ALL DATAFRAMES ────────────────────────────────────
    #

    # ---- Inventory Detail (inv_df) ----
    inv_df = inv_df.rename(columns={'SKU': 'SKU_Full'})
    inv_df['SKU'] = inv_df['SKU_Full'].str.extract(r'(\d+)').fillna(inv_df['SKU_Full'])
    inv_df['SKU'] = inv_df['SKU'].map(clean_sku)
    inv_df['ProductState'] = inv_df.get('ProductState', '').str.upper().fillna('')
    inv_df['ProductName']  = inv_df.get('SKU_Full', '').astype(str)
    inv_df['WeightLb']     = pd.to_numeric(inv_df['WeightLb'], errors='coerce').fillna(0)
    inv_df['CostValue']    = pd.to_numeric(inv_df['CostValue'], errors='coerce').fillna(0)

    # ---- Sales History (sales_df) ----
    sales_df['SKU']            = sales_df['SKU'].map(clean_sku)
    sales_df['ShippedLb']      = pd.to_numeric(sales_df['ShippedLb'], errors='coerce').fillna(0)
    sales_df['QuantityOrdered']= pd.to_numeric(sales_df['QuantityOrdered'], errors='coerce').fillna(0)
    sales_df['Cost']           = pd.to_numeric(sales_df['Cost'], errors='coerce').fillna(0)
    sales_df['Rev']            = pd.to_numeric(sales_df['Rev'], errors='coerce').fillna(0)

    # ---- Production Batch (prod_df) ----
    prod_df['SKU'] = prod_df['SKU'].map(clean_sku)
    if 'ProductionShippedLb' in prod_df.columns:
        prod_df['ProductionShippedLb'] = pd.to_numeric(prod_df['ProductionShippedLb'], errors='coerce').fillna(0.0)
    else:
        prod_df['ProductionShippedLb'] = pd.to_numeric(prod_df.get('WeightLb', 0), errors='coerce').fillna(0.0)

    # ---- Product Detail (prod_detail) ----
    prod_detail['SKU']       = prod_detail['Product Code'].map(clean_sku)
    prod_detail['ParentSKU'] = prod_detail['Velocity Parent'].map(clean_sku)
    prod_detail['SKU_Desc']  = prod_detail['Description'].fillna("").astype(str)

    # ---- NEW: Inventory Detail1 (inv1) ----
    #    We only need {SKU, ProductState, PackId1, WeightLb} from this sheet.
    if not inv1.empty and all(col in inv1.columns for col in ["SKU", "ProductState", "PackId1", "WeightLb"]):
        inv1['SKU']         = inv1['SKU'].map(clean_sku)
        inv1['ProductState']= inv1['ProductState'].str.upper().fillna('')
        inv1['PackId1']     = inv1['PackId1'].astype(str).str.strip().fillna('')
        inv1['WeightLb']    = pd.to_numeric(inv1['WeightLb'], errors='coerce').fillna(0)
    else:
        # If "Inventory Detail1" is missing required columns, replace with an empty DataFrame
        inv1 = pd.DataFrame(columns=["SKU", "ProductState", "PackId1", "WeightLb"])

    # ---- Cost Value (cost_val) – still loaded but not used for pack logic now ----
    cost_val['SKU']      = cost_val['SKU'].map(clean_sku) if 'SKU' in cost_val else cost_val.get('SKU', pd.Series()).map(clean_sku)
    if 'NumPacks' in cost_val and 'WeightLb' in cost_val:
        cost_val['NumPacks'] = pd.to_numeric(cost_val['NumPacks'], errors='coerce').fillna(0)
        cost_val['WeightLb'] = pd.to_numeric(cost_val['WeightLb'], errors='coerce').fillna(0)

    #
    # ── 2) AGGREGATE SALES & INVENTORY (weight & cost), exactly as before ─────
    #

    # 2a) AGGREGATE SALES → one row per (SKU, Supplier, Protein, Description)
    agg_sales = (
        sales_df
        .groupby(["SKU", "Supplier", "Protein", "Description"], as_index=False)
        .agg(
            ShippedLb       = ("ShippedLb",          "sum"),
            QuantityOrdered = ("QuantityOrdered",    "sum"),
            Cost            = ("Cost",               "sum"),
            Rev             = ("Rev",                "sum")
        )
    )

    # 2b) AGGREGATE INVENTORY (weight & cost) → one row per (SKU, ProductState, ProductName)
    inv_agg = (
        inv_df
        .groupby(["SKU", "ProductState", "ProductName"], as_index=False)
        .agg(
            OnHandWeightLb = ("WeightLb", "sum"),
            OnHandCost     = ("CostValue", "sum")
        )
    )

    # 2c) SUPPLIER MAP FROM PRODUCTION (if inv_agg has blank Supplier)
    supplier_map = {}
    if "Supplier" in prod_df.columns:
        supplier_map = dict(zip(prod_df['SKU'], prod_df['Supplier']))

    # 2d) MERGE inv_agg + agg_sales
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
    mask_blank = (df["Supplier"].astype(str).str.strip() == "")
    df.loc[mask_blank, "Supplier"] = df.loc[mask_blank, "SKU"].map(supplier_map).fillna("")

    df = df.merge(
        prod_df[["SKU", "ProductionShippedLb"]],
        on="SKU",
        how="left"
    )
    df["ProductionShippedLb"] = df["ProductionShippedLb"].fillna(0.0)

    # 2e) FINAL AGGREGATION → one row per (SKU, Supplier, Protein, Description, ProductState, ProductName)
    try:
        weeks_span = weeks_override or calculate_weeks_in_data(sales_df)
        if weeks_span <= 0:
            weeks_span = 4
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
    sku_stats["WOH_Flag"]       = np.where(sku_stats["TotalUsage"] == 0, "No Usage", "")
    sku_stats["WeeksOnHand"]    = sku_stats["OnHandWeightTotal"] / sku_stats["AvgWeeklyUsage"].replace({0: np.nan})
    sku_stats["WeeksOnHand"]    = sku_stats["WeeksOnHand"].replace([np.inf, -np.inf], np.nan)
    sku_stats["AnnualTurns"]    = (sku_stats["AvgWeeklyUsage"] * 52.0) / sku_stats["OnHandWeightTotal"].replace({0: np.nan})
    sku_stats["AnnualTurns"]    = sku_stats["AnnualTurns"].replace([np.inf, -np.inf], 0).fillna(0)
    sku_stats["SKU_Desc"]       = sku_stats["SKU"] + " – " + sku_stats["ProductName"].where(
                                      sku_stats["ProductName"].ne(""),
                                      sku_stats["Description"]
                                  )

    #
    # ── 3) USE Inventory Detail1 (inv1) TO COUNT EXACT PACKS ON HAND ────────────
    #

    # 3a) Count distinct PackId1 per (SKU, ProductState)
    if not inv1.empty:
        pack_counts = (
            inv1
            .loc[:, ['SKU', 'ProductState', 'PackId1']]
            .dropna(subset=['PackId1'])
            .assign(PackId1=lambda d: d['PackId1'].astype(str).str.strip())
            .groupby(['SKU', 'ProductState'], as_index=False)
            .agg(NumPacksOnHand=("PackId1", "nunique"))
        )
    else:
        pack_counts = pd.DataFrame(columns=["SKU", "ProductState", "NumPacksOnHand"])

    # 3b) Merge pack_counts into sku_stats → any missing (SKU,ProductState) gets 0 packs
    sku_stats = sku_stats.merge(
        pack_counts,
        on=["SKU", "ProductState"],
        how="left"
    )
    sku_stats["NumPacksOnHand"] = sku_stats["NumPacksOnHand"].fillna(0).astype(int)

    # 3c) Compute total packs per SKU across all states
    total_packs = (
        sku_stats
        .groupby("SKU", as_index=False)["NumPacksOnHand"]
        .sum()
        .rename(columns={"NumPacksOnHand": "Inv_TotalPacks"})
    )

    # 3d) Also capture total packs from inv1 by SKU for debug
    inv1_pack_sum = (
        inv1
        .loc[:, ['SKU', 'PackId1']]
        .dropna(subset=['PackId1'])
        .assign(PackId1=lambda d: d['PackId1'].astype(str).str.strip())
        .groupby("SKU", as_index=False)
        .agg(InventoryDetail1_TotalPacks=("PackId1", "nunique"))
    )

    #
    # ── 4) FINISH CLEANUP & PARENT MAPPING ────────────────────────────────────
    #
    desc_map   = dict(zip(prod_detail['SKU'], prod_detail['SKU_Desc']))
    parent_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    sku_stats['SKU_Desc']  = sku_stats['SKU'].map(desc_map).fillna(sku_stats['SKU_Desc'])
    sku_stats['ParentSKU'] = sku_stats['SKU'].map(parent_map).fillna(sku_stats['SKU'])
    mask = (
        sku_stats["ParentSKU"].isin(["", "nan", "none", "null"]) |
        sku_stats["ParentSKU"].isna()
    )
    sku_stats.loc[mask, "ParentSKU"] = sku_stats.loc[mask, "SKU"]

    sku_stats['Supplier']     = sku_stats['Supplier'].replace("", "Unknown").fillna("Unknown")
    sku_stats['ProductState'] = sku_stats['ProductState'].fillna("").str.upper()

    return sku_stats, prod_detail, cost_val, total_packs, inv1_pack_sum


# ------------------- PARENT PREPROCESS & PURCHASE PLAN HELPERS ───────────────── #

def get_root_parent(sku, parent_map):
    """Follow the parent chain up to the root."""
    seen = set()
    while (
        sku in parent_map
        and pd.notnull(parent_map[sku])
        and parent_map[sku] != ""
        and parent_map[sku] != sku
    ):
        if sku in seen:
            break
        seen.add(sku)
        sku = parent_map[sku]
    return sku

def get_best_description(group):
    """Choose the most common non-empty description in the group."""
    descs = group['SKU_Desc'].dropna().unique()
    if len(descs) == 0:
        return ""
    return max(descs, key=lambda x: (len(x), x))  # longest, then alphabetical

def preprocess_for_parents(sku_stats, prod_detail):
    parent_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    desc_map   = dict(zip(prod_detail['SKU'], prod_detail['SKU_Desc']))

    sku_stats['TrueParent'] = sku_stats['SKU'].apply(lambda x: get_root_parent(x, parent_map))
    sku_stats['SKU_Desc']   = sku_stats['SKU'].map(desc_map).fillna(sku_stats['SKU_Desc'])

    agg_dict = {
        'MeanUse':  ('AvgWeeklyUsage',  'sum'),
        'InvWt':    ('OnHandWeightTotal','sum'),
        'InvCost':  ('OnHandCostTotal',  'sum'),
        'Supplier': ('Supplier',         lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
        'Protein':  ('Protein',          lambda x: x.mode()[0] if not x.mode().empty else "")
    }

    if 'PacksToOrder' in sku_stats.columns:
        agg_dict['PacksToOrder'] = ('PacksToOrder', 'sum')
    if 'OrderWt' in sku_stats.columns:
        agg_dict['OrderWt'] = ('OrderWt', 'sum')
    if 'EstCost' in sku_stats.columns:
        agg_dict['EstCost'] = ('EstCost', 'sum')

    agg = (
        sku_stats.groupby('TrueParent', as_index=False)
                 .agg(**agg_dict)
    )

    desc_lookup   = sku_stats.groupby('TrueParent').apply(get_best_description)
    agg['SKU_Desc'] = agg['TrueParent'].map(desc_lookup)
    agg['SKU']     = agg['TrueParent']

    return agg


# ------------------- PARENT PURCHASE PLAN ─────────────────────────────────── #

@st.cache_data(show_spinner=False)
def compute_parent_purchase_plan(sku_stats, prod_detail, cost_val, desired_woh):
    child_to_parent   = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    parent_desc_map   = dict(zip(prod_detail['ParentSKU'], prod_detail['Description'].fillna("").astype(str)))
    sku_stats['ParentSKU'] = sku_stats['SKU'].map(child_to_parent).fillna(sku_stats['SKU'])
    mask = (
        sku_stats["ParentSKU"].isin(["", "nan", "none", "null"]) |
        sku_stats["ParentSKU"].isna()
    )
    sku_stats.loc[mask, "ParentSKU"] = sku_stats.loc[mask, "SKU"]

    parent_stats = (
        sku_stats.groupby("ParentSKU", as_index=False)
                 .agg(
                     MeanUse  = ('AvgWeeklyUsage', 'sum'),
                     InvWt    = ('OnHandWeightTotal', 'sum'),
                     InvCost  = ('OnHandCostTotal',   'sum'),
                     Supplier = ('Supplier', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
                 )
    )
    parent_stats['DesiredWt']   = parent_stats['MeanUse'] * desired_woh
    parent_stats['ToBuyWt']     = (parent_stats['DesiredWt'] - parent_stats['InvWt']).clip(lower=0)

    # Derive packsize per ParentSKU: median of child pack sizes
    child_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
    packsize_map_per_child = dict(
        zip(
            sku_stats['SKU'],
            (sku_stats['OnHandWeightTotal'] / sku_stats['NumPacksOnHand'].replace({0: np.nan}))
        )
    )
    parent_packsize = {}
    for parent in parent_stats['ParentSKU'].unique():
        children = [sku for sku, p in child_map.items() if p == parent]
        psizes   = [
            packsize_map_per_child.get(c)
            for c in children
            if packsize_map_per_child.get(c, np.nan) and not pd.isna(packsize_map_per_child.get(c))
        ]
        if psizes:
            parent_packsize[parent] = float(np.nanmedian(psizes))
        else:
            parent_packsize[parent] = 1.0

    parent_stats['PackSize'] = parent_stats['ParentSKU'].map(parent_packsize).fillna(1.0)
    parent_stats['PacksToOrder'] = np.where(
        parent_stats['PackSize'] > 0,
        np.ceil(parent_stats['ToBuyWt'] / parent_stats['PackSize']),
        0
    ).astype(int)
    parent_stats['OrderWt'] = parent_stats['PacksToOrder'] * parent_stats['PackSize']
    parent_stats['SKU']     = parent_stats['ParentSKU']
    parent_stats['SKU_Desc']= parent_stats['SKU'].map(parent_desc_map).fillna(parent_stats['SKU'])
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


# ------------------- MAIN STREAMLIT TAB FUNCTION ─────────────────────────── #

def woh_tab(sheets, theme):
    """
    Render the “📦 Advanced Inventory Weeks-On-Hand (WOH) Dashboard” tab.
    Expects `sheets` to include:
      - "Inventory Detail"
      - "Sales History"
      - "Production Batch"
      - "Product Detail"
      - "Inventory Detail1"  ← used for exact pack counts
      - "Cost Value"         ← (not used for pack counts in this version)
    """
    st.header("📦 Advanced Inventory Weeks-On-Hand (WOH) Dashboard")

    sku_stats, prod_detail, cost_val, total_packs, inv1_pack_sum = aggregate_data(sheets)

    tab1, tab2, tab3, tab4 = st.tabs([
        "FZ → EXT Transfer",
        "EXT → FZ Transfer",
        "Purchase Plan",
        "Product Lookup"
    ])

    # --- TAB 1: FZ → EXT TRANSFER ---
    with tab1:
        st.subheader("🔄 Move FZ → EXT")

        fz = sku_stats[
            (sku_stats['ProductState'].str.startswith('FZ')) &
            (sku_stats['AvgWeeklyUsage'] > 0)
        ].copy()
        ext = sku_stats[
            (sku_stats['ProductState'].str.startswith('EXT')) &
            (sku_stats['AvgWeeklyUsage'] > 0)
        ].copy()

        # Join EXT’s on-hand weight & pack counts into FZ
        ext_onhand_wt    = ext.set_index("SKU")["OnHandWeightTotal"].rename("EXT_OnHandWeight")
        ext_onhand_packs = ext.set_index("SKU")["NumPacksOnHand"].rename("EXT_NumPacksOnHand")
        fz = fz.join(ext_onhand_wt,    on="SKU")
        fz = fz.join(ext_onhand_packs, on="SKU")
        fz["EXT_OnHandWeight"]   = fz["EXT_OnHandWeight"].fillna(0)
        fz["EXT_NumPacksOnHand"] = fz["EXT_NumPacksOnHand"].fillna(0).astype(int)

        fz["Total_OnHandWeight"] = fz["OnHandWeightTotal"] + fz["EXT_OnHandWeight"]
        fz["Total_PacksOnHand"]  = fz["NumPacksOnHand"] + fz["EXT_NumPacksOnHand"]

        thr1 = st.slider("Desired FZ WOH (weeks)", 0.0, 4.0, 1.5, 0.25)
        fz['DesiredFZ_Weight'] = fz['AvgWeeklyUsage'] * thr1
        fz['WeightToMove']     = (fz['OnHandWeightTotal'] - fz['DesiredFZ_Weight']).clip(lower=0)

        move = fz[fz["WeightToMove"] > 0].copy()

        # Expose relevant columns
        move['FZ_OnHandWeight']    = move['OnHandWeightTotal']
        move['FZ_PacksOnHand']     = move['NumPacksOnHand']
        move['EXT_OnHandWeight']   = move['EXT_OnHandWeight']
        move['EXT_PacksOnHand']    = move['EXT_NumPacksOnHand']
        move['Total_OnHandWeight'] = move['Total_OnHandWeight']
        move['Total_PacksOnHand']  = move['Total_PacksOnHand']
        move['TotalShippedLb']     = move['TotalShippedLb'].fillna(0)
        move['TotalProductionLb']  = move['TotalProductionLb'].fillna(0)

        c1, c2, c3 = st.columns(3)
        c1.metric("SKUs to Move", move["SKU"].nunique())
        c2.metric("Total Weight to Move", f"{move['WeightToMove'].sum():,.0f} lb")
        c3.metric("Total Cost to Move", "$0")  # placeholder; add cost logic if you have it

        # Download button (including pack counts & weights)
        if not move.empty:
            buf1 = io.BytesIO()
            move_cols = [
                'SKU',
                'SKU_Desc',
                'Supplier',
                'ProductState',
                'FZ_OnHandWeight',
                'FZ_PacksOnHand',
                'EXT_OnHandWeight',
                'EXT_PacksOnHand',
                'Total_OnHandWeight',
                'Total_PacksOnHand',
                'AvgWeeklyUsage',
                'WeeksOnHand',
                'DesiredFZ_Weight',
                'WeightToMove'
            ]
            move[move_cols].to_excel(buf1, index=False, sheet_name="FZ2EXT")
            buf1.seek(0)
            st.download_button(
                "Download FZ→EXT List",
                buf1.getvalue(),
                file_name="FZ2EXT_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Chart
        if not move.empty:
            chart = (
                alt.Chart(move)
                   .mark_bar()
                   .encode(
                       y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                       x=alt.X("WeightToMove:Q", title="Weight to Move (lb)"),
                       color="Supplier:N",
                       tooltip=["SKU_Desc", "Supplier", "WeightToMove", "WeeksOnHand"]
                   )
                   .properties(height=alt.Step(25))
            )
            st.altair_chart(theme(chart).interactive(), use_container_width=True)

    # --- TAB 2: EXT → FZ TRANSFER ---
    with tab2:
        st.subheader("🔄 Move EXT → FZ")

        fz  = sku_stats[
            (sku_stats['ProductState'].str.startswith('FZ')) &
            (sku_stats['AvgWeeklyUsage'] > 0)
        ].copy()
        ext = sku_stats[
            (sku_stats['ProductState'].str.startswith('EXT')) &
            (sku_stats['AvgWeeklyUsage'] > 0)
        ].copy()

        # Join FZ’s on-hand weight & pack counts into EXT
        fz_onhand_wt    = fz.set_index("SKU")["OnHandWeightTotal"].rename("FZ_OnHandWeight")
        fz_onhand_packs = fz.set_index("SKU")["NumPacksOnHand"].rename("FZ_NumPacksOnHand")
        ext = ext.join(fz_onhand_wt,    on="SKU")
        ext = ext.join(fz_onhand_packs, on="SKU")
        ext["FZ_OnHandWeight"]   = ext["FZ_OnHandWeight"].fillna(0)
        ext["FZ_NumPacksOnHand"] = ext["FZ_NumPacksOnHand"].fillna(0).astype(int)

        ext["Total_OnHandWeight"] = ext["OnHandWeightTotal"] + ext["FZ_OnHandWeight"]
        ext["Total_PacksOnHand"]  = ext["NumPacksOnHand"] + ext["FZ_NumPacksOnHand"]

        thr2_default = 1.0
        try:
            thr2_default = float(compute_threshold_move(ext, None))
        except Exception:
            pass

        thr2 = st.slider(
            "Desired FZ WOH to achieve",
            0.0,
            float(fz["WeeksOnHand"].max() if not fz.empty else 52.0),
            thr2_default,
            step=0.25
        )

        ext["DesiredFZ_Weight"] = ext["AvgWeeklyUsage"] * thr2
        ext["WeightToReturn"]   = (ext["DesiredFZ_Weight"] - ext["FZ_OnHandWeight"]).clip(lower=0)

        back = ext[ext["WeightToReturn"] > 0].copy()

        back['EXT_OnHandWeight']    = back['OnHandWeightTotal']
        back['EXT_PacksOnHand']     = back['NumPacksOnHand']
        back['FZ_OnHandWeight']     = back['FZ_OnHandWeight']
        back['FZ_PacksOnHand']      = back['FZ_NumPacksOnHand']
        back['Total_OnHandWeight']  = back['Total_OnHandWeight']
        back['Total_PacksOnHand']   = back['Total_PacksOnHand']
        back['TotalShippedLb']      = back['TotalShippedLb'].fillna(0)
        back['TotalProductionLb']   = back['TotalProductionLb'].fillna(0)

        col1, col2, col3 = st.columns(3)
        col1.metric("SKUs to Return", back["SKU"].nunique())
        col2.metric("Total Weight to Return", f"{back['WeightToReturn'].sum():,.0f} lb")
        col3.metric("Total Cost to Return", "$0")  # placeholder

        if not back.empty:
            buf2 = io.BytesIO()
            back_cols = [
                'SKU',
                'SKU_Desc',
                'Supplier',
                'ProductState',
                'EXT_OnHandWeight',
                'EXT_PacksOnHand',
                'FZ_OnHandWeight',
                'FZ_PacksOnHand',
                'Total_OnHandWeight',
                'Total_PacksOnHand',
                'AvgWeeklyUsage',
                'WeeksOnHand',
                'DesiredFZ_Weight',
                'WeightToReturn'
            ]
            back[back_cols].to_excel(buf2, index=False, sheet_name="EXT2FZ")
            buf2.seek(0)
            st.download_button(
                "Download EXT→FZ List",
                buf2.getvalue(),
                file_name="EXT2FZ_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if not back.empty:
            chart2 = (
                alt.Chart(back)
                   .mark_bar()
                   .encode(
                       y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                       x=alt.X("WeightToReturn:Q", title="Weight to Return (lb)"),
                       color="Supplier:N",
                       tooltip=["SKU_Desc", "Supplier", "WeightToReturn", "WeeksOnHand"]
                   )
                   .properties(height=alt.Step(25))
            )
            st.altair_chart(theme(chart2).interactive(), use_container_width=True)

    # --- TAB 3: PURCHASE PLAN ---
    with tab3:
        st.subheader("🛒 Usage-Based Product Purchase Plan with Delivery Intelligence")

        desired_woh = st.slider("Desired Weeks On Hand (WOH) for Product Purchase Plan", 0.0, 12.0, 4.0, 0.5)
        parent_plan = preprocess_for_parents(sku_stats, prod_detail)

        parent_plan['DesiredWt'] = parent_plan['MeanUse'] * desired_woh
        parent_plan['ToBuyWt']   = (parent_plan['DesiredWt'] - parent_plan['InvWt']).clip(lower=0)

        # Derive packsize per ParentSKU: median of child pack sizes
        child_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
        packsize_map_per_child = dict(
            zip(
                sku_stats['SKU'],
                (sku_stats['OnHandWeightTotal'] / sku_stats['NumPacksOnHand'].replace({0: np.nan}))
            )
        )
        parent_packsize = {}
        for parent in parent_plan['SKU']:
            children = [sku for sku, p in child_map.items() if p == parent]
            psizes   = [
                packsize_map_per_child.get(c)
                for c in children
                if packsize_map_per_child.get(c, np.nan) and not pd.isna(packsize_map_per_child.get(c))
            ]
            if psizes:
                parent_packsize[parent] = float(np.nanmedian(psizes))
            else:
                parent_packsize[parent] = 1.0

        parent_plan['PackSize'] = parent_plan['SKU'].map(parent_packsize).fillna(1.0)
        parent_plan['PacksToOrder'] = np.where(
            parent_plan['PackSize'] > 0,
            np.ceil(parent_plan['ToBuyWt'] / parent_plan['PackSize']),
            0
        ).astype(int)
        parent_plan['OrderWt'] = parent_plan['PacksToOrder'] * parent_plan['PackSize']

        parent_plan['CostPerLb'] = np.where(
            parent_plan['InvWt'] > 0,
            parent_plan['InvCost'] / parent_plan['InvWt'],
            0
        )
        parent_plan['EstCost'] = parent_plan['OrderWt'] * parent_plan['CostPerLb']

        parent_plan["SpecialOrderFlag"] = (
            parent_plan["SKU"].astype(str).str.contains("SO01", na=False) |
            parent_plan["SKU_Desc"].astype(str).str.contains("SO01", na=False)
        )
        parent_plan["SO Note"] = np.where(parent_plan["SpecialOrderFlag"], "SO01 Special Order", "")

        delivery_info = parent_plan["Supplier"].apply(lambda s: get_next_delivery(s))
        parent_plan["NextCutoff"]   = delivery_info.apply(lambda d: d["cutoff_datetime"].strftime("%a %H:%M") if d["found"] else "")
        parent_plan["NextDelivery"] = delivery_info.apply(lambda d: d["delivery_date"].strftime("%Y-%m-%d") if d["found"] and d["delivery_date"] else "")
        parent_plan["DeliveryNotes"] = delivery_info.apply(lambda d: d["notes"] if d["found"] else d.get("reason", ""))

        all_suppliers = sorted(parent_plan["Supplier"].dropna().unique())
        all_proteins  = sorted(parent_plan["Protein"].dropna().unique())
        selected_supplier = st.selectbox("Supplier Filter", ["All"] + all_suppliers)
        selected_protein  = st.selectbox("Protein Filter", ["All"] + all_proteins)
        plan_df = parent_plan.copy()
        if selected_supplier != "All":
            plan_df = plan_df[plan_df["Supplier"] == selected_supplier]
        if selected_protein != "All":
            plan_df = plan_df[plan_df["Protein"] == selected_protein]

        plan_df = plan_df[plan_df['PacksToOrder'] > 0]

        display_cols = [
            "SKU", "SKU_Desc", "Supplier", "Protein", "InvWt", "DesiredWt",
            "PackSize", "PacksToOrder", "OrderWt", "EstCost",
            "SO Note", "NextCutoff", "NextDelivery", "DeliveryNotes"
        ]
        display = plan_df[display_cols].copy()
        display["InvWt"]       = display["InvWt"].map("{:,.0f} lb".format)
        display["DesiredWt"]   = display["DesiredWt"].map("{:,.0f} lb".format)
        display["PackSize"]    = display["PackSize"].map("{:,.0f} lb".format)
        display["OrderWt"]     = display["OrderWt"].map("{:,.0f} lb".format)
        display["EstCost"]     = display["EstCost"].map("${:,.2f}".format)

        st.dataframe(display, use_container_width=True)

        buf = io.BytesIO()
        display.to_excel(buf, index=False, sheet_name="PurchasePlan")
        buf.seek(0)
        st.download_button(
            "📥 Download Purchase Plan",
            data=buf.getvalue(),
            file_name="Purchase_Plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        supplier_summary = (
            plan_df
            .groupby("Supplier", as_index=False)["EstCost"].sum()
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

    # --- TAB 4: PRODUCT LOOKUP ---
    with tab4:
        st.subheader("🔍 Product Lookup: Parent + Children Usage & WOH")

        parent_map = dict(zip(prod_detail['SKU'], prod_detail['ParentSKU']))
        desc_map   = dict(zip(prod_detail['SKU'], prod_detail['SKU_Desc']))

        def get_root_parent(sku):
            seen = set()
            while (
                sku in parent_map
                and pd.notnull(parent_map[sku])
                and parent_map[sku] != ""
                and parent_map[sku] != sku
            ):
                if sku in seen:
                    break
                seen.add(sku)
                sku = parent_map[sku]
            return sku

        sku_stats["SearchKey"] = sku_stats["SKU"].astype(str) + " – " + sku_stats["SKU_Desc"].fillna("")
        query = st.text_input("Enter SKU, Product Name, or Partial Description:")

        if query:
            mask = (
                sku_stats["SearchKey"].str.contains(query, case=False, na=False)
                | sku_stats["SKU"].astype(str).str.contains(query, case=False, na=False)
            )
            matched = sku_stats[mask]

            if matched.empty:
                st.warning("No products matched your search.")
            else:
                for idx, row in matched.iterrows():
                    search_sku  = row['SKU']
                    root_parent = get_root_parent(search_sku)
                    child_mask = sku_stats['SKU'].apply(lambda x: get_root_parent(x) == root_parent)
                    fam = sku_stats[child_mask].copy()

                    fam_totals = fam.agg({
                        'AvgWeeklyUsage':       'sum',
                        'OnHandWeightTotal':    'sum'
                    })
                    fam_totals['WeeksOnHand'] = (
                        fam_totals['OnHandWeightTotal'] / fam_totals['AvgWeeklyUsage']
                        if fam_totals['AvgWeeklyUsage'] > 0 else float('nan')
                    )

                    parent_desc = (
                        fam[fam['SKU'] == root_parent]['SKU_Desc'].iloc[0]
                        if not fam[fam['SKU'] == root_parent].empty else ""
                    )
                    st.markdown(f"**Parent Product:** `{root_parent}` – {parent_desc}")
                    st.markdown(f"- **Total Usage (lb/wk):** `{fam_totals['AvgWeeklyUsage']:.2f}`")
                    st.markdown(f"- **Total On Hand (lb):** `{fam_totals['OnHandWeightTotal']:.2f}`")
                    st.markdown(f"- **Combined Weeks On Hand:** `{fam_totals['WeeksOnHand']:.2f}`")

                    display_cols = [
                        "SKU", "SKU_Desc", "Supplier", "Protein", "AvgWeeklyUsage",
                        "OnHandWeightTotal", "WeeksOnHand", "ProductState"
                    ]
                    fam_disp = fam[display_cols].copy()
                    fam_disp["AvgWeeklyUsage"]     = fam_disp["AvgWeeklyUsage"].map("{:,.2f} lb".format)
                    fam_disp["OnHandWeightTotal"]  = fam_disp["OnHandWeightTotal"].map("{:,.2f} lb".format)
                    fam_disp["WeeksOnHand"]        = fam_disp["WeeksOnHand"].map("{:,.2f}".format)
                    st.write("**Breakdown by Child SKU:**")
                    st.dataframe(fam_disp, use_container_width=True)

                    buf = io.BytesIO()
                    fam_disp.to_excel(buf, index=False, sheet_name="Lookup")
                    buf.seek(0)
                    st.download_button(
                        f"Download Breakdown for {root_parent}",
                        buf.getvalue(),
                        file_name=f"{root_parent}_lookup.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    break
        else:
            st.info("Enter a SKU or product description to search.")

    # --- 10) PACK COUNT DEBUG EXPANDER ---
    with st.expander("🔍 Pack Count Tracking (InventoryDetail1 vs Inventory)"):
        inv_packs = total_packs.rename(columns={"Inv_TotalPacks": "Inv_TotalPacks_onSKU"})
        merged_debug = inv1_pack_sum.merge(inv_packs, on="SKU", how="outer").fillna(0)
        merged_debug["Difference"] = merged_debug["InventoryDetail1_TotalPacks"] - merged_debug["Inv_TotalPacks_onSKU"]
        st.dataframe(merged_debug, use_container_width=True)
