import io
import os
import sys
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Ensure the project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import compute_threshold_move

@st.cache_data(show_spinner=False)
def compute_parent_purchase_plan(
    df_inv: pd.DataFrame,
    pd_detail: pd.DataFrame,
    cost_df: pd.DataFrame,
    desired_woh: float
) -> pd.DataFrame:
    """
    Enhanced parent-level purchase logic:
    - Only orders at parent if total (parent + children) inventory is insufficient
    - Never double-orders for both parent and child
    - Shows only true parent codes needing orders
    """
    # --- Input validation ---
    required_cols = {
        'df_inv': ['SKU', 'AvgWeeklyUsage', 'OnHandWeightTotal', 'OnHandCostTotal', 'SKU_Desc'],
        'pd_detail': ['Product Code', 'Velocity Parent', 'Description'],
        'cost_df': ['SKU']
    }
    for df_name, cols in required_cols.items():
        df = locals()[df_name]
        if df_name == 'cost_df' and df.empty:
            continue
        if df.empty:
            st.error(f"{df_name} is empty")
            return pd.DataFrame()
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in {df_name}: {missing_cols}")
            return pd.DataFrame()

    # --- Prepare detail sheet ---
    d = pd_detail.copy()
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns={"Product Code": "SKU", "Velocity Parent": "ParentSKU", "Description": "ParentDesc"})
    for col in ["SKU", "ParentSKU", "ParentDesc"]:
        if col not in d.columns:
            d[col] = ""
    d["SKU"] = d["SKU"].fillna("").astype(str).str.strip()
    d["ParentSKU"] = d["ParentSKU"].fillna("").astype(str).str.strip()
    mask = (d["ParentSKU"].str.lower().isin(["", "nan", "none", "null"])) | (d["ParentSKU"].isna())
    d.loc[mask, "ParentSKU"] = d.loc[mask, "SKU"]

    # --- Map ---
    child_to_parent = dict(zip(d["SKU"], d["ParentSKU"]))
    parent_desc_map = dict(zip(d["ParentSKU"], d["ParentDesc"]))

    # --- Map SKUs to parent, aggregate all stats ---
    inv = df_inv.copy()
    inv["SKU"] = inv["SKU"].astype(str).str.strip()
    inv["SKU_Desc"] = inv["SKU_Desc"].astype(str).str.strip()
    inv["ParentSKU"] = inv["SKU"].map(child_to_parent).fillna(inv["SKU"])

    # Group by parent (includes all children)
    parent_stats = (
        inv.groupby("ParentSKU", as_index=False)
            .agg(
                MeanUse=("AvgWeeklyUsage", "sum"),
                InvWt=("OnHandWeightTotal", "sum"),
                InvCost=("OnHandCostTotal", "sum")
            )
    )
    parent_stats["DesiredWt"] = parent_stats["MeanUse"] * desired_woh
    parent_stats["ToBuyWt"] = (parent_stats["DesiredWt"] - parent_stats["InvWt"]).clip(lower=0)

    # Assign pack count using parent code, fallback to 1
    if not cost_df.empty and "NumPacks" in cost_df.columns:
        pack_map = pd.Series(
            pd.to_numeric(cost_df["NumPacks"], errors="coerce").fillna(1).astype(int).clip(lower=1).values,
            index=cost_df["SKU"].astype(str)
        ).to_dict()
        parent_stats["PackCount"] = parent_stats["ParentSKU"].map(pack_map).fillna(1).astype(int)
    else:
        parent_stats["PackCount"] = 1

    # Order logic
    parent_stats["PackWt"] = np.where(
        (parent_stats["InvWt"] > 0) & (parent_stats["PackCount"] > 0),
        parent_stats["InvWt"] / parent_stats["PackCount"],
        0
    )
    parent_stats["PacksToOrder"] = np.where(
        parent_stats["PackWt"] > 0,
        np.ceil(parent_stats["ToBuyWt"] / parent_stats["PackWt"]),
        0
    ).astype(int)
    parent_stats["OrderWt"] = parent_stats["PacksToOrder"] * parent_stats["PackWt"]

    parent_stats["SKU"] = parent_stats["ParentSKU"]
    parent_stats["SKU_Desc"] = parent_stats["SKU"].map(parent_desc_map).fillna(parent_stats["SKU"])
    parent_stats["CostPerLb"] = np.where(
        parent_stats["InvWt"] > 0,
        parent_stats["InvCost"] / parent_stats["InvWt"],
        0
    )
    parent_stats["EstCost"] = parent_stats["OrderWt"] * parent_stats["CostPerLb"]

    # Only return parent codes that *truly* need purchase
    result = parent_stats[parent_stats["PacksToOrder"] > 0][
        ["SKU", "SKU_Desc", "InvWt", "DesiredWt", "PackCount", "PacksToOrder", "OrderWt", "EstCost"]
    ].copy()
    result.reset_index(drop=True, inplace=True)
    return result

def render_fz_ext_purchasing(df, df_hc, cost_df, theme, sheets):
    st.header("📦 Advanced FZ↔EXT Analysis & Purchase Plan")

    # --- Prep ---
    df = df.copy()
    base_cols = [
        "SKU", "SKU_Desc", "ProductState", "Supplier",
        "OnHandWeightTotal", "AvgWeeklyUsage", "WeeksOnHand",
        "TotalShippedLb", "TotalProductionLb", "TotalUsage",
        "PackCount", "OnHandCostTotal", "Protein"
    ]
    for col in base_cols:
        if col not in df.columns:
            if col in ("SKU", "SKU_Desc", "ProductState", "Supplier", "Protein"):
                df[col] = ""
            else:
                df[col] = 0
    for c in ["OnHandWeightTotal", "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "OnHandCostTotal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["SKU_Desc"] = df["SKU_Desc"].astype(str).str.strip()
    df["ProductState"] = df["ProductState"].astype(str).str.upper().fillna("")

    # Filters
    suppliers = sorted(df["Supplier"].dropna().unique())
    proteins = sorted(df["Protein"].dropna().unique())
    colf1, colf2 = st.columns(2)
    sel_supplier = colf1.multiselect("Filter Suppliers", suppliers, default=suppliers)
    sel_protein = colf2.multiselect("Filter Proteins", proteins, default=proteins)
    filtered = df[df["Supplier"].isin(sel_supplier) & df["Protein"].isin(sel_protein)].copy()

    # States
    states = sorted(set([x for x in filtered["ProductState"].unique() if x]))
    fz_states = [x for x in states if x.startswith("FZ")]
    ext_states = [x for x in states if x.startswith("EXT")]
    fz = filtered[filtered["ProductState"].isin(fz_states) & (filtered["AvgWeeklyUsage"] > 0)].copy()
    ext = filtered[filtered["ProductState"].isin(ext_states) & (filtered["AvgWeeklyUsage"] > 0)].copy()
    ext_weight_lookup = ext.set_index("SKU")["OnHandWeightTotal"]
    fz_weight_lookup = fz.set_index("SKU")["OnHandWeightTotal"]

    # FZ→EXT
    st.subheader("🔄 Move FZ → EXT")
    thr1 = st.slider("Desired FZ Weeks-On-Hand", 0.0, 12.0, 2.0, 0.25, key="fz2ext_thr")
    to_move = fz[fz["WeeksOnHand"] > thr1].copy()
    to_move["DesiredFZ_Weight"] = to_move["AvgWeeklyUsage"] * thr1
    to_move["WeightToMove"] = (to_move["OnHandWeightTotal"] - to_move["DesiredFZ_Weight"]).clip(lower=0)
    to_move["EXT_Weight"] = to_move["SKU"].map(ext_weight_lookup).fillna(0)
    to_move["TotalOnHand"] = to_move["OnHandWeightTotal"] + to_move["EXT_Weight"]
    mv1 = to_move[to_move["WeightToMove"] > 0].copy()
    mv1["OnHandCostTotal"] = pd.to_numeric(mv1.get("OnHandCostTotal", 0), errors="coerce").fillna(0)
    mv1["CostToMove"] = np.where(
        mv1["OnHandWeightTotal"] > 0,
        (mv1["WeightToMove"] / mv1["OnHandWeightTotal"]) * mv1["OnHandCostTotal"],
        0
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs to Move", int(mv1["SKU"].nunique()))
    c2.metric("Total Weight to Move", f"{mv1['WeightToMove'].sum():,.0f} lb")
    c3.metric("Total Cost to Move", f"${mv1['CostToMove'].sum():,.0f}")
    if not mv1.empty:
        sel2 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart1 = (
            alt.Chart(mv1)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToMove:Q", title="Weight to Move (lb)"),
                color="Supplier:N",
                opacity=alt.condition(sel2, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("SKU_Desc:N", title="SKU"),
                    alt.Tooltip("Supplier:N", title="Supplier"),
                    alt.Tooltip("Protein:N", title="Protein"),
                    alt.Tooltip("WeeksOnHand:Q", title="Current FZ WOH", format=".1f"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("EXT_Weight:Q", format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q", format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToMove:Q", format=",.0f", title="Weight to Move"),
                    alt.Tooltip("CostToMove:Q", format="$.0f", title="Move Cost"),
                ]
            )
            .add_selection(sel2)
            .properties(height=alt.Step(25))
        )
        st.altair_chart(theme(chart1).interactive(), use_container_width=True)
        export_cols = [
            "SKU", "SKU_Desc", "Protein", "ProductState", "Supplier",
            "OnHandWeightTotal", "EXT_Weight", "TotalOnHand",
            "AvgWeeklyUsage", "WeeksOnHand", "PackCount",
            "DesiredFZ_Weight", "WeightToMove", "CostToMove",
            "TotalShippedLb", "TotalProductionLb", "TotalUsage"
        ]
        export_cols = [c for c in export_cols if c in mv1.columns]
        buf1 = io.BytesIO()
        mv1[export_cols].to_excel(buf1, index=False, sheet_name="FZ2EXT")
        buf1.seek(0)
        st.download_button(
            "⬇️ Download FZ→EXT List",
            buf1.getvalue(),
            file_name="FZ2EXT_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # EXT→FZ
    st.subheader("🔄 Move EXT → FZ")
    thr2 = st.slider("Desired FZ Weeks-On-Hand to Achieve", 0.0, 12.0, 2.0, 0.25, key="ext2fz_thr")
    back = ext.copy()
    back["FZ_Weight"] = back["SKU"].map(fz_weight_lookup).fillna(0)
    back["DesiredFZ_Weight"] = back["AvgWeeklyUsage"] * thr2
    back["WeightToReturn"] = (back["DesiredFZ_Weight"] - back["FZ_Weight"]).clip(lower=0)
    back["TotalOnHand"] = back["OnHandWeightTotal"] + back["FZ_Weight"]
    mv2 = back[back["WeightToReturn"] > 0].copy()
    mv2["OnHandCostTotal"] = pd.to_numeric(mv2.get("OnHandCostTotal", 0), errors="coerce").fillna(0)
    mv2["CostToReturn"] = np.where(
        mv2["OnHandWeightTotal"] > 0,
        (mv2["WeightToReturn"] / mv2["OnHandWeightTotal"]) * mv2["OnHandCostTotal"],
        0
    )
    c4, c5, c6 = st.columns(3)
    c4.metric("SKUs to Return", int(mv2["SKU"].nunique()))
    c5.metric("Total Weight to Return", f"{mv2['WeightToReturn'].sum():,.0f} lb")
    c6.metric("Total Cost to Return", f"${mv2['CostToReturn'].sum():,.0f}")
    if not mv2.empty:
        sel3 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart2 = (
            alt.Chart(mv2)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToReturn:Q", title="Weight to Return (lb)"),
                color="Supplier:N",
                opacity=alt.condition(sel3, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("SKU_Desc:N", title="SKU"),
                    alt.Tooltip("Supplier:N", title="Supplier"),
                    alt.Tooltip("Protein:N", title="Protein"),
                    alt.Tooltip("WeeksOnHand:Q", title="Current EXT WOH", format=".1f"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("FZ_Weight:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q", format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToReturn:Q", format=",.0f", title="Weight to Return"),
                    alt.Tooltip("CostToReturn:Q", format="$.0f", title="Return Cost"),
                ]
            )
            .add_selection(sel3)
            .properties(height=alt.Step(25))
        )
        st.altair_chart(theme(chart2).interactive(), use_container_width=True)
        export_cols2 = [
            "SKU", "SKU_Desc", "Protein", "ProductState", "Supplier",
            "OnHandWeightTotal", "FZ_Weight", "TotalOnHand",
            "AvgWeeklyUsage", "WeeksOnHand", "PackCount",
            "DesiredFZ_Weight", "WeightToReturn", "CostToReturn",
            "TotalShippedLb", "TotalProductionLb", "TotalUsage"
        ]
        export_cols2 = [c for c in export_cols2 if c in mv2.columns]
        buf2 = io.BytesIO()
        mv2[export_cols2].to_excel(buf2, index=False, sheet_name="EXT2FZ")
        buf2.seek(0)
        st.download_button(
            "⬇️ Download EXT→FZ List",
            buf2.getvalue(),
            file_name="EXT2FZ_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # --- Purchase Recommendations by Desired WOH ---
    st.subheader("🛒 Purchase Recommendations by Desired WOH")
    supplier_opts = ["All"] + sorted(df["Supplier"].astype(str).unique())
    selected_supplier = st.selectbox("Supplier", supplier_opts, key="pr_supplier")
    df_pr = df if selected_supplier == "All" else df[df["Supplier"] == selected_supplier]
    desired_woh = st.slider(
        "Desired Weeks-On-Hand", 0.0, 12.0, 4.0, 0.5,
        help="How many weeks’ worth of stock you want on hand"
    )
    pd_detail = sheets.get("Product Detail", pd.DataFrame())
    plan_df = compute_parent_purchase_plan(df_pr, pd_detail, cost_df, desired_woh)
    if not plan_df.empty:
        display = (
            plan_df[[
                "SKU", "SKU_Desc", "InvWt", "DesiredWt",
                "PackCount", "PacksToOrder", "OrderWt", "EstCost"
            ]]
            .assign(
                InvWt=lambda d: d["InvWt"].map("{:,.0f} lb".format),
                DesiredWt=lambda d: d["DesiredWt"].map("{:,.0f} lb".format),
                OrderWt=lambda d: d["OrderWt"].map("{:,.0f} lb".format),
                EstCost=lambda d: d["EstCost"].map("${:,.2f}".format)
            )
        )
        st.dataframe(display, use_container_width=True)
        buf = io.BytesIO()
        plan_df.to_excel(buf, index=False, sheet_name="PurchasePlan")
        buf.seek(0)
        st.download_button(
            "📥 Download Purchase Plan",
            data=buf.getvalue(),
            file_name="Purchase_Plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No purchase plan generated due to invalid or missing data.")

    st.info("This tab helps optimize inventory between Freezer (FZ) and EXT, and calculates precise parent-level purchase requirements. Parent codes are shown for purchasing only if their total inventory is insufficient—including all child codes—to meet your target weeks-on-hand.")
