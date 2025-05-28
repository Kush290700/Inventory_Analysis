import io
import os
import sys
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

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
    Returns a parent-level purchase plan based on desired weeks-on-hand (WOH).
    """
    # 1. Input validation
    required_cols = {
        'df_inv': ['SKU', 'AvgWeeklyUsage', 'OnHandWeightTotal', 'OnHandCostTotal', 'SKU_Desc'],
        'pd_detail': ['Product Code', 'Velocity Parent', 'Description'],
        'cost_df': ['SKU']
    }
    for df_name, cols in required_cols.items():
        df = locals()[df_name]
        if df_name == 'cost_df' and (df is None or df.empty):
            continue
        if df is None or df.empty:
            st.error(f"{df_name} is empty")
            return pd.DataFrame()
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in {df_name}: {missing_cols}")
            return pd.DataFrame()

    # 2. Prepare product details
    d = pd_detail.copy()
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns={"Product Code": "SKU", "Velocity Parent": "ParentSKU", "Description": "ParentDesc"})
    for col in ["SKU", "ParentSKU", "ParentDesc"]:
        d[col] = d.get(col, "").fillna("").astype(str).str.strip()
    mask = (d["ParentSKU"].str.lower().isin(["", "nan", "none", "null"])) | (d["ParentSKU"].isna())
    d.loc[mask, "ParentSKU"] = d.loc[mask, "SKU"]
    child_to_parent = dict(zip(d["SKU"], d["ParentSKU"]))
    parent_desc_map = dict(zip(d["ParentSKU"], d["ParentDesc"]))

    # 3. Inventory mapping
    inv = df_inv.copy()
    inv["SKU"] = inv["SKU"].astype(str).str.strip()
    inv["SKU_Desc"] = inv["SKU_Desc"].astype(str).str.strip()
    inv["ParentSKU"] = inv["SKU"].map(child_to_parent).fillna(inv["SKU"])

    # 4. Parent supplier mapping (most frequent child supplier or fallback)
    supplier_map = {}
    if "Supplier" in inv.columns:
        supplier_map = (
            inv.groupby("ParentSKU")["Supplier"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
            .to_dict()
        )

    # 5. Aggregate to parent SKU
    parent_stats = (
        inv.groupby("ParentSKU", as_index=False)
            .agg(
                MeanUse=("AvgWeeklyUsage", "sum"),
                InvWt=("OnHandWeightTotal", "sum"),
                InvCost=("OnHandCostTotal", "sum"),
            )
    )
    parent_stats["DesiredWt"] = parent_stats["MeanUse"] * desired_woh
    parent_stats["ToBuyWt"] = (parent_stats["DesiredWt"] - parent_stats["InvWt"]).clip(lower=0)

    # 6. Pack size/Pack count logic
    pack_size_map = {}
    if cost_df is not None and not cost_df.empty:
        if "PackSize" in cost_df.columns:
            pack_size_map = pd.Series(
                pd.to_numeric(cost_df["PackSize"], errors="coerce").fillna(1).values,
                index=cost_df["SKU"].astype(str)
            ).to_dict()
        elif "NumPacks" in cost_df.columns and "SKU" in cost_df.columns:
            # Fallback: infer from inventory
            try:
                avg_pack = inv.groupby("ParentSKU")["OnHandWeightTotal"].sum() / inv.groupby("ParentSKU")["PackCount"].sum()
                pack_size_map = avg_pack.fillna(1).to_dict()
            except Exception:
                pack_size_map = {}
    # Default to 80 lb/case if unknown
    parent_stats["PackSize"] = parent_stats["ParentSKU"].map(pack_size_map).fillna(80).astype(float)

    parent_stats["PacksToOrder"] = np.where(
        parent_stats["PackSize"] > 0,
        np.ceil(parent_stats["ToBuyWt"] / parent_stats["PackSize"]),
        0
    ).astype(int)
    parent_stats["OrderWt"] = parent_stats["PacksToOrder"] * parent_stats["PackSize"]

    parent_stats["SKU"] = parent_stats["ParentSKU"]
    parent_stats["SKU_Desc"] = parent_stats["SKU"].map(parent_desc_map).fillna(parent_stats["SKU"])
    parent_stats["Supplier"] = parent_stats["SKU"].map(supplier_map).fillna("Unknown")
    parent_stats["CostPerLb"] = np.where(
        parent_stats["InvWt"] > 0,
        parent_stats["InvCost"] / parent_stats["InvWt"],
        0
    )
    parent_stats["EstCost"] = parent_stats["OrderWt"] * parent_stats["CostPerLb"]

    # --- Only those parents that need to be ordered ---
    plan = parent_stats[parent_stats["PacksToOrder"] > 0][
        ["SKU", "SKU_Desc", "Supplier", "InvWt", "DesiredWt", "PackSize", "PacksToOrder", "OrderWt", "EstCost"]
    ].copy()
    plan.reset_index(drop=True, inplace=True)
    return plan

def render(df, df_hc, cost_df, theme, sheets):
    core_cols = [
        "SKU", "WeeksOnHand", "AvgWeeklyUsage", "OnHandWeightTotal",
        "OnHandCostTotal", "SKU_Desc", "ProductState", "Supplier"
    ]
    if df.empty:
        st.warning("No inventory data available.")
        return
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        st.error(f"Missing core columns in inventory data: {missing_core}")
        return
    df = df.copy()

    # FZ & EXT logic
    state = df["ProductState"].fillna("").str.upper()
    fz = df[(state.str.startswith("FZ")) & (df["AvgWeeklyUsage"] > 0)].copy()
    ext = df[(state.str.startswith("EXT")) & (df["AvgWeeklyUsage"] > 0)].copy()
    fz_woh = fz.set_index("SKU")["WeeksOnHand"]
    ext_weight_lookup = ext.set_index("SKU")["OnHandWeightTotal"]

    # FZ→EXT
    st.subheader("🔄 Move FZ → EXT")
    thr1 = st.slider("Desired FZ WOH (weeks)", 0.0, 4.0, 1.5, 0.25, key="w2e_thr")
    to_move = fz[fz["WeeksOnHand"] > thr1].copy()
    to_move["DesiredFZ_Weight"] = to_move["AvgWeeklyUsage"] * thr1
    to_move["WeightToMove"] = (to_move["OnHandWeightTotal"] - to_move["DesiredFZ_Weight"]).clip(lower=0)
    to_move["EXT_Weight"] = to_move["SKU"].map(ext_weight_lookup).fillna(0)
    to_move["TotalOnHand"] = to_move["OnHandWeightTotal"] + to_move["EXT_Weight"]
    mv1 = to_move[to_move["WeightToMove"] > 0]
    total_wt_move = mv1["WeightToMove"].sum()
    total_cost_move = np.where(
        mv1["OnHandWeightTotal"] > 0,
        (mv1["WeightToMove"] / mv1["OnHandWeightTotal"]) * mv1["OnHandCostTotal"],
        0
    ).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs to Move", mv1["SKU"].nunique())
    c2.metric("Total Weight to Move", f"{total_wt_move:,.0f} lb")
    c3.metric("Total Cost to Move", f"${total_cost_move:,.0f}")
    suppliers = sorted(mv1["Supplier"].dropna().unique())
    sel_sup = st.multiselect("Filter Suppliers", suppliers, default=suppliers, key="mv1_sups")
    mv1 = mv1[mv1["Supplier"].isin(sel_sup)]
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
                    alt.Tooltip("WeeksOnHand:Q", title="Current FZ WOH", format=".1f"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("EXT_Weight:Q", format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q", format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("PackCount:Q", title="Case or Packs Available"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToMove:Q", format=",.0f", title="Weight to Move"),
                ]
            )
            .add_selection(sel2)
            .properties(height=alt.Step(25))
        )
        st.altair_chart(theme(chart1).interactive(), use_container_width=True)
    if not mv1.empty:
        export_cols = [
            "SKU_Desc", "ProductState", "Supplier", "OnHandWeightTotal",
            "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "DesiredFZ_Weight",
            "WeightToMove", "EXT_Weight"
        ]
        export_cols = [c for c in export_cols if c in mv1.columns]
        buf1 = io.BytesIO()
        mv1[export_cols].to_excel(buf1, index=False, sheet_name="FZ2EXT")
        buf1.seek(0)
        st.download_button(
            "Download FZ→EXT List",
            buf1.getvalue(),
            file_name="FZ2EXT_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # EXT→FZ
    st.subheader("🔄 Move EXT → FZ")
    thr2_default = 1.0
    try:
        if not df_hc.empty:
            thr2_default = float(compute_threshold_move(ext, df_hc))
    except Exception as e:
        st.warning(f"Error computing threshold for EXT → FZ: {str(e)}")
    thr2 = st.slider(
        "Desired FZ WOH to achieve",
        0.0, float(fz_woh.max()) if not fz_woh.empty else 52.0, thr2_default, step=0.25,
        key="e2f_thr"
    )
    back = ext[ext["SKU"].map(fz_woh).fillna(0) < thr2].copy()
    back["FZ_Weight"] = back["SKU"].map(fz_woh).fillna(0)
    back["DesiredFZ_Weight"] = back["AvgWeeklyUsage"] * thr2
    back["WeightToReturn"] = back["DesiredFZ_Weight"].sub(back["FZ_Weight"]).clip(lower=0)
    back["TotalOnHand"] = back["OnHandWeightTotal"] + back["FZ_Weight"]
    total_wt_ret = back["WeightToReturn"].sum()
    total_cost_ret = np.where(
        back["OnHandWeightTotal"] > 0,
        (back["WeightToReturn"] / back["OnHandWeightTotal"]) * back["OnHandCostTotal"],
        0
    ).sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("SKUs to Return", back["SKU"].nunique())
    col2.metric("Total Weight to Return", f"{total_wt_ret:,.0f} lb")
    col3.metric("Total Cost to Return", f"${total_cost_ret:,.0f}")
    sup2 = sorted(back["Supplier"].dropna().unique())
    chosen2 = st.multiselect("Filter Suppliers", sup2, default=sup2, key="mv2_sups")
    back = back[back["Supplier"].isin(chosen2)]
    if not back.empty:
        sel3 = alt.selection_multi(fields=["Supplier"], bind="legend")
        chart2 = (
            alt.Chart(back)
            .mark_bar()
            .encode(
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                x=alt.X("WeightToReturn:Q", title="Weight to Return (lb)"),
                color="Supplier:N",
                opacity=alt.condition(sel3, alt.value(1), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("SKU_Desc:N", title="SKU"),
                    alt.Tooltip("Supplier:N", title="Supplier"),
                    alt.Tooltip("WeeksOnHand:Q", title="Current WOH", format=".1f"),
                    alt.Tooltip("OnHandWeightTotal:Q", format=",.0f", title="EXT On-Hand Wt"),
                    alt.Tooltip("FZ_Weight:Q", format=",.0f", title="FZ On-Hand Wt"),
                    alt.Tooltip("TotalOnHand:Q", format=",.0f", title="Total On-Hand Wt"),
                    alt.Tooltip("PackCount:Q", title="Case or Packs Available"),
                    alt.Tooltip("DesiredFZ_Weight:Q", format=",.0f", title="Desired FZ Wt"),
                    alt.Tooltip("WeightToReturn:Q", format=",.0f", title="Weight to Return"),
                ]
            )
            .add_selection(sel3)
            .properties(height=alt.Step(25))
        )
        st.altair_chart(theme(chart2).interactive(), use_container_width=True)
    if not back.empty:
        export_cols = [
            "SKU_Desc", "ProductState", "Supplier", "OnHandWeightTotal",
            "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "DesiredFZ_Weight",
            "WeightToReturn", "FZ_Weight"
        ]
        export_cols = [c for c in export_cols if c in back.columns]
        buf2 = io.BytesIO()
        back[export_cols].to_excel(buf2, index=False, sheet_name="EXT2FZ")
        buf2.seek(0)
        st.download_button(
            "Download EXT→FZ List",
            buf2.getvalue(),
            file_name="EXT2FZ_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # --- Purchase Recommendations by Desired WOH ---
    st.subheader("🛒 Purchase Recommendations by Desired WOH (Parent SKUs Only)")
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
                "SKU", "SKU_Desc", "Supplier", "InvWt", "DesiredWt",
                "PackSize", "PacksToOrder", "OrderWt", "EstCost"
            ]]
            .assign(
                InvWt=lambda d: d["InvWt"].map("{:,.0f} lb".format),
                DesiredWt=lambda d: d["DesiredWt"].map("{:,.0f} lb".format),
                PackSize=lambda d: d["PackSize"].map("{:,.0f} lb".format),
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
        # Supplier Pie Chart
        supplier_summary = (
            plan_df.groupby("Supplier", as_index=False)["EstCost"].sum()
            .sort_values("EstCost", ascending=False)
            .head(10)
        )
        if not supplier_summary.empty:
            pie = alt.Chart(supplier_summary).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="EstCost", type="quantitative", stack=True),
                color=alt.Color("Supplier:N", legend=alt.Legend(title="Supplier")),
                tooltip=[
                    alt.Tooltip("Supplier:N"),
                    alt.Tooltip("EstCost:Q", format="$.0f")
                ]
            ).properties(title="Top Suppliers in Purchase Plan (by Estimated Cost)")
            st.altair_chart(pie, use_container_width=True)
    else:
        st.warning("No purchase plan generated due to invalid or missing data.")
