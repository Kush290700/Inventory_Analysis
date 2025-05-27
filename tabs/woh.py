import io
import os
import sys
import logging
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.classification import compute_threshold_move

@st.cache_data(show_spinner=False)
def compute_parent_purchase_plan(
    df_inv: pd.DataFrame,
    pd_detail: pd.DataFrame,
    cost_df: pd.DataFrame,
    desired_woh: float
) -> pd.DataFrame:
    try:
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

        # Product Detail
        d = pd_detail.copy()
        d.columns = [str(c).strip() for c in d.columns]
        d = d.rename(columns={"Product Code": "SKU", "Velocity Parent": "ParentSKU", "Description": "ParentDesc"})
        for col in ["SKU", "ParentSKU", "ParentDesc"]:
            d[col] = d.get(col, "").fillna("").astype(str).str.strip()
        mask = (d["ParentSKU"].str.lower().isin(["", "nan", "none", "null"])) | (d["ParentSKU"].isna())
        d.loc[mask, "ParentSKU"] = d.loc[mask, "SKU"]

        child_to_parent = dict(zip(d["SKU"], d["ParentSKU"]))
        parent_desc_map = dict(zip(d["ParentSKU"], d["ParentDesc"]))

        inv = df_inv.copy()
        inv["SKU"] = inv["SKU"].astype(str).str.strip()
        inv["SKU_Desc"] = inv["SKU_Desc"].astype(str).str.strip()
        inv["ParentSKU"] = inv["SKU"].map(child_to_parent).fillna(inv["SKU"])

        # Supplier mapping for parents (most common among children)
        supplier_map = (
            inv.groupby("ParentSKU")["Supplier"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
            .to_dict()
            if "Supplier" in inv.columns else {}
        )

        # Aggregate stats to parent
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

        # Use "PackSize" if you have it, else use OnHandWeightTotal/PackCount fallback
        # Ideally, cost_df should have "PackSize" (e.g. 80 for 80lb/case)
        pack_size_map = {}
        if cost_df is not None and not cost_df.empty:
            if "PackSize" in cost_df.columns:
                pack_size_map = pd.Series(
                    pd.to_numeric(cost_df["PackSize"], errors="coerce").fillna(1).values,
                    index=cost_df["SKU"].astype(str)
                ).to_dict()
        if not pack_size_map and "NumPacks" in cost_df.columns:
            # Fallback: derive average pack size from OnHandWeightTotal / NumPacks
            avg_pack = inv.groupby("ParentSKU")["OnHandWeightTotal"].sum() / inv.groupby("ParentSKU")["PackCount"].sum()
            pack_size_map = avg_pack.fillna(1).to_dict()

        parent_stats["PackSize"] = parent_stats["ParentSKU"].map(pack_size_map).fillna(80).astype(float)  # default to 80lb/case if not available

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

        result = parent_stats[parent_stats["PacksToOrder"] > 0][
            ["SKU", "SKU_Desc", "Supplier", "InvWt", "DesiredWt", "PackSize", "PacksToOrder", "OrderWt", "EstCost"]
        ].copy()
        result.reset_index(drop=True, inplace=True)
        return result

    except Exception as e:
        st.error(f"Error generating purchase plan: {str(e)}")
        return pd.DataFrame()


def render(df, df_hc, cost_df, theme, sheets):
    try:
        for col in ["SKU", "SKU_Desc", "ProductState", "Supplier", "Protein", "OnHandWeightTotal", "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "OnHandCostTotal"]:
            if col not in df.columns:
                df[col] = "" if col in ["SKU", "SKU_Desc", "ProductState", "Supplier", "Protein"] else 0
        for c in ["OnHandWeightTotal", "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "OnHandCostTotal"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["SKU"] = df["SKU"].astype(str).str.strip()
        df["SKU_Desc"] = df["SKU_Desc"].astype(str).str.strip()
        df["ProductState"] = df["ProductState"].astype(str).str.upper().fillna("")

        # --- FZ → EXT ---
        st.subheader("🔄 Move FZ → EXT (Reduce Freezer Overstock)")
        fz = df[df["ProductState"].str.startswith("FZ") & (df["AvgWeeklyUsage"] > 0)].copy()
        ext = df[df["ProductState"].str.startswith("EXT") & (df["AvgWeeklyUsage"] > 0)].copy()
        ext_weight_lookup = ext.set_index("SKU")["OnHandWeightTotal"]
        thr1 = st.slider(
            "Set target max FZ WOH (weeks):", 0.0, 12.0, 2.0, 0.25, key="fz2ext_thr"
        )
        fz["DesiredFZ_Weight"] = fz["AvgWeeklyUsage"] * thr1
        fz["WeightToMove"] = (fz["OnHandWeightTotal"] - fz["DesiredFZ_Weight"]).clip(lower=0)
        fz["EXT_Weight"] = fz["SKU"].map(ext_weight_lookup).fillna(0)
        fz["TotalOnHand"] = fz["OnHandWeightTotal"] + fz["EXT_Weight"]
        mv1 = fz[(fz["WeeksOnHand"] > thr1) & (fz["WeightToMove"] > 0)].copy()
        mv1["CostToMove"] = np.where(
            mv1["OnHandWeightTotal"] > 0,
            (mv1["WeightToMove"] / mv1["OnHandWeightTotal"]) * mv1["OnHandCostTotal"],
            0
        )
        # Chart
        if not mv1.empty:
            mv1_top = mv1.nlargest(10, "WeightToMove")
            chart1 = alt.Chart(mv1_top).mark_bar().encode(
                x=alt.X("WeightToMove:Q", title="Weight to Move (lb)"),
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                color="Supplier:N",
                tooltip=[
                    alt.Tooltip("SKU:N"),
                    alt.Tooltip("SKU_Desc:N", title="SKU Desc"),
                    alt.Tooltip("Supplier:N"),
                    alt.Tooltip("WeightToMove:Q", format=",.0f"),
                    alt.Tooltip("CostToMove:Q", format="$.0f")
                ]
            ).properties(height=320, title="Top SKUs to Move FZ→EXT")
            st.altair_chart(chart1, use_container_width=True)

        st.metric("SKUs to Move", int(mv1["SKU"].nunique()))
        st.metric("Total Weight to Move", f"{mv1['WeightToMove'].sum():,.0f} lb")
        st.metric("Total Cost to Move", f"${mv1['CostToMove'].sum():,.0f}")
        if not mv1.empty:
            buf1 = io.BytesIO()
            mv1.to_excel(buf1, index=False, sheet_name="FZ2EXT")
            buf1.seek(0)
            st.download_button(
                "⬇️ Download FZ→EXT List",
                buf1.getvalue(),
                file_name="FZ2EXT_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --- EXT → FZ ---
        st.subheader("🔄 Move EXT → FZ (Refill Freezer Understock)")
        fz_weight_lookup = fz.set_index("SKU")["OnHandWeightTotal"]
        thr2 = st.slider(
            "Set min FZ WOH to achieve:", 0.0, 12.0, 2.0, 0.25, key="ext2fz_thr"
        )
        ext["FZ_Weight"] = ext["SKU"].map(fz_weight_lookup).fillna(0)
        ext["DesiredFZ_Weight"] = ext["AvgWeeklyUsage"] * thr2
        ext["WeightToReturn"] = (ext["DesiredFZ_Weight"] - ext["FZ_Weight"]).clip(lower=0)
        ext["TotalOnHand"] = ext["OnHandWeightTotal"] + ext["FZ_Weight"]
        mv2 = ext[(ext["FZ_Weight"] < ext["DesiredFZ_Weight"]) & (ext["WeightToReturn"] > 0)].copy()
        mv2["CostToReturn"] = np.where(
            mv2["OnHandWeightTotal"] > 0,
            (mv2["WeightToReturn"] / mv2["OnHandWeightTotal"]) * mv2["OnHandCostTotal"],
            0
        )
        if not mv2.empty:
            mv2_top = mv2.nlargest(10, "WeightToReturn")
            chart2 = alt.Chart(mv2_top).mark_bar().encode(
                x=alt.X("WeightToReturn:Q", title="Weight to Return (lb)"),
                y=alt.Y("SKU_Desc:N", sort='-x', title="SKU"),
                color="Supplier:N",
                tooltip=[
                    alt.Tooltip("SKU:N"),
                    alt.Tooltip("SKU_Desc:N", title="SKU Desc"),
                    alt.Tooltip("Supplier:N"),
                    alt.Tooltip("WeightToReturn:Q", format=",.0f"),
                    alt.Tooltip("CostToReturn:Q", format="$.0f")
                ]
            ).properties(height=320, title="Top SKUs to Return EXT→FZ")
            st.altair_chart(chart2, use_container_width=True)

        st.metric("SKUs to Return", int(mv2["SKU"].nunique()))
        st.metric("Total Weight to Return", f"{mv2['WeightToReturn'].sum():,.0f} lb")
        st.metric("Total Cost to Return", f"${mv2['CostToReturn'].sum():,.0f}")
        if not mv2.empty:
            buf2 = io.BytesIO()
            mv2.to_excel(buf2, index=False, sheet_name="EXT2FZ")
            buf2.seek(0)
            st.download_button(
                "⬇️ Download EXT→FZ List",
                buf2.getvalue(),
                file_name="EXT2FZ_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --- Purchase Plan ---
        st.subheader("🛒 Purchase Recommendations by Desired WOH (Parent SKUs Only)")
        supplier_opts = ["All"] + sorted(df["Supplier"].astype(str).unique())
        selected_supplier = st.selectbox(
            "Filter Purchase Plan by Supplier:", supplier_opts, key="pr_supplier"
        )
        df_pr = df if selected_supplier == "All" else df[df["Supplier"] == selected_supplier]
        desired_woh = st.slider(
            "Desired Weeks-On-Hand for Purchase Plan", 0.0, 12.0, 4.0, 0.5
        )
        pd_detail = sheets.get("Product Detail", pd.DataFrame())
        plan_df = compute_parent_purchase_plan(df_pr, pd_detail, cost_df, desired_woh)
        if not plan_df.empty:
            plan_display = (
                plan_df.assign(
                    InvWt=lambda d: d["InvWt"].map("{:,.0f} lb".format),
                    DesiredWt=lambda d: d["DesiredWt"].map("{:,.0f} lb".format),
                    OrderWt=lambda d: d["OrderWt"].map("{:,.0f} lb".format),
                    EstCost=lambda d: d["EstCost"].map("${:,.2f}".format),
                    PackSize=lambda d: d["PackSize"].map("{:,.0f} lb".format)
                )
            )
            st.dataframe(plan_display, use_container_width=True)
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

    except Exception as e:
        st.error(f"Error rendering the dashboard: {str(e)}")
