import io
import os
import sys
import logging
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Logging setup
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
        # ---- Validation ----
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
                st.error(f"{df_name} is empty (rows: {0 if df is None else len(df)})")
                st.write(f"**DEBUG {df_name}:**", df)
                return pd.DataFrame()
            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in {df_name}: {missing_cols}")
                st.write(f"**DEBUG {df_name} columns:**", df.columns)
                return pd.DataFrame()

        # ---- Product Details ----
        d = pd_detail.copy()
        d.columns = [str(c).strip() for c in d.columns]
        d = d.rename(columns={"Product Code": "SKU", "Velocity Parent": "ParentSKU", "Description": "ParentDesc"})
        for col in ["SKU", "ParentSKU", "ParentDesc"]:
            d[col] = d.get(col, "").fillna("").astype(str).str.strip()
        mask = (d["ParentSKU"].str.lower().isin(["", "nan", "none", "null"])) | (d["ParentSKU"].isna())
        d.loc[mask, "ParentSKU"] = d.loc[mask, "SKU"]

        child_to_parent = dict(zip(d["SKU"], d["ParentSKU"]))
        parent_desc_map = dict(zip(d["ParentSKU"], d["ParentDesc"]))

        # ---- Inventory Data ----
        inv = df_inv.copy()
        inv["SKU"] = inv["SKU"].astype(str).str.strip()
        inv["SKU_Desc"] = inv["SKU_Desc"].astype(str).str.strip()
        inv["ParentSKU"] = inv["SKU"].map(child_to_parent).fillna(inv["SKU"])

        # ---- Supplier map for Parent ----
        supplier_map = {}
        if "Supplier" in inv.columns:
            supplier_map = (
                inv.groupby("ParentSKU")["Supplier"]
                .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
                .to_dict()
            )

        # ---- Parent Aggregation ----
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

        # ---- Pack Size Handling ----
        pack_size_map = {}
        if cost_df is not None and not cost_df.empty:
            if "PackSize" in cost_df.columns:
                pack_size_map = pd.Series(
                    pd.to_numeric(cost_df["PackSize"], errors="coerce").fillna(1).values,
                    index=cost_df["SKU"].astype(str)
                ).to_dict()
            elif "NumPacks" in cost_df.columns and "SKU" in cost_df.columns:
                # Fallback: infer from inventory if possible
                try:
                    avg_pack = inv.groupby("ParentSKU")["OnHandWeightTotal"].sum() / inv.groupby("ParentSKU")["PackCount"].sum()
                    pack_size_map = avg_pack.fillna(1).to_dict()
                except Exception:
                    pack_size_map = {}

        # Default to 80 lb/case if no pack size known (can override as needed)
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

        # ---- Only those parents that need to be ordered ----
        result = parent_stats[parent_stats["PacksToOrder"] > 0][
            ["SKU", "SKU_Desc", "Supplier", "InvWt", "DesiredWt", "PackSize", "PacksToOrder", "OrderWt", "EstCost"]
        ].copy()
        result.reset_index(drop=True, inplace=True)

        # DEBUG: Show result size and head
        st.write("**DEBUG: Purchase Plan Rows:**", result.shape[0])
        st.write(result.head())

        return result

    except Exception as e:
        st.error(f"Error generating purchase plan: {str(e)}")
        st.write("**DEBUG Exception in compute_parent_purchase_plan:**", str(e))
        return pd.DataFrame()

def render(df, df_hc, cost_df, theme, sheets):
    try:
        # Ensure all columns exist and are of the correct type
        for col in ["SKU", "SKU_Desc", "ProductState", "Supplier", "Protein", "OnHandWeightTotal", "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "OnHandCostTotal"]:
            if col not in df.columns:
                df[col] = "" if col in ["SKU", "SKU_Desc", "ProductState", "Supplier", "Protein"] else 0
        for c in ["OnHandWeightTotal", "AvgWeeklyUsage", "WeeksOnHand", "PackCount", "OnHandCostTotal"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["SKU"] = df["SKU"].astype(str).str.strip()
        df["SKU_Desc"] = df["SKU_Desc"].astype(str).str.strip()
        df["ProductState"] = df["ProductState"].astype(str).str.upper().fillna("")

        # === Purchase Plan (Parent) ===
        st.subheader("🛒 Purchase Recommendations by Desired WOH (Parent SKUs Only)")
        supplier_opts = ["All"] + sorted(df["Supplier"].astype(str).unique())
        selected_supplier = st.selectbox(
            "Filter Purchase Plan by Supplier:", supplier_opts, key="pr_supplier"
        )
        df_pr = df if selected_supplier == "All" else df[df["Supplier"] == selected_supplier]

        # DIAGNOSTIC: Show the filtered DataFrame and its columns
        st.write("**DEBUG: df_pr shape:**", df_pr.shape)
        st.write("**DEBUG: df_pr columns:**", df_pr.columns)
        pd_detail = sheets.get("Product Detail", pd.DataFrame())
        st.write("**DEBUG: pd_detail shape:**", pd_detail.shape)
        st.write("**DEBUG: pd_detail columns:**", pd_detail.columns)
        st.write("**DEBUG: cost_df shape:**", (0 if cost_df is None else cost_df.shape))
        if cost_df is not None:
            st.write("**DEBUG: cost_df columns:**", cost_df.columns)

        desired_woh = st.slider(
            "Desired Weeks-On-Hand for Purchase Plan", 0.0, 12.0, 4.0, 0.5
        )
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
        st.write("**DEBUG Exception in render:**", str(e))
